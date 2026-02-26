from __future__ import annotations
import time
from packages.obs.messages import explain
import uuid
from typing import Any, Dict, List
import numpy as np

from packages.feature.pipeline import compute as compute_features
from packages.feature.invariants import check as inv_check, ok as inv_ok
from packages.model.trainer import OnlineTrainer
from packages.features.orderbook_tracker import OrderBookTracker
from packages.features.feature_engine import ohlcv_to_channels, pack_model_b_tensor
from packages.model.model_b_patchtst import ModelBPatchTST
from packages.model.dataset_builder import DatasetBuilder
from packages.model.model_registry import Registry
from packages.model.model_b_trainer import Trainer
from packages.decision.gate import decide as gate_decide
from packages.execution.paper import PaperPortfolio
from packages.obs.telemetry import Telemetry
from packages.modernization import MarketPatternScanner, MultiHorizonForecaster, MarketRegimeDetector


def now_ms() -> int:
    return int(time.time()*1000)

def mk_event(base: Dict[str, Any], stage: str, level: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event_id": str(uuid.uuid4()),
        "ts": now_ms(),
        "run_id": base["run_id"],
        "service": base["service"],
        "exchange": base["exchange"],
        "market": base["market"],
        "symbol": base["symbol"],
        "timeframe": base.get("timeframe"),
        "stage": stage,
        "level": level,
        "payload": payload,
    }

def label_forward(close: np.ndarray, forward: int) -> np.ndarray:
    fut = np.roll(close, -forward)
    y = np.where(fut > close, 1, -1).astype(int)
    y[-forward:] = 0
    return y

class Engine:
    def __init__(self, cfg: Dict[str, Any], market, es, ss, symdb):
        self.cfg=cfg
        base_init={"run_id":"boot","service":"sf-runtime","exchange":cfg['runtime']['exchange'],"market":cfg['runtime']['market'],"timeframe":cfg['runtime']['timeframe'],"symbol":"*"}
        self.market=market
        self.es=es
        self.ss=ss
        self.symdb=symdb
        mc=cfg["model"]
        self.ob_tracker=OrderBookTracker(max_snapshots=120)
        self.es.append(mk_event(base_init, 'MODEL_B_LOAD_START', 'INFO', {'hf_id': str(cfg.get('model_b',{}).get('hf_id',''))}))
        self.model_b = ModelBPatchTST(hf_id=str(cfg.get('model_b',{}).get('hf_id','ibm-research/patchtst-fm-r1')), cache_dir=str(cfg.get('model_b',{}).get('cache_dir','data/hf_cache')), device=str(cfg.get('model_b',{}).get('device','cpu')))
        self.model_b_state = self.model_b.load()
        self.registry = Registry(root=str(cfg.get('storage',{}).get('model_registry','data/model_registry')))
        self.ds_builder = DatasetBuilder(path=str(cfg.get('storage',{}).get('model_b_dataset','data/datasets/model_b_samples.csv')),
                                        horizon_bars=int(cfg.get('model_b',{}).get('horizon_bars',12)),
                                        breakout_atr=float(cfg.get('model_b',{}).get('breakout_atr',0.25)),
                                        hold_bars=int(cfg.get('model_b',{}).get('hold_bars',2)))
        self.model_b_trainer = Trainer(dataset_path=self.ds_builder.path, registry=self.registry,
                                            retrain_interval_sec=float(cfg.get('model_b',{}).get('retrain_interval_sec',1800.0)))
        self.es.append(mk_event(base_init, 'MODEL_B_LOAD_END', 'INFO' if self.model_b_state.loaded else 'ERROR', {'state': self.model_b_state.__dict__}))

        self.trainer=OnlineTrainer(k=int(mc.get('k',35)), min_samples=int(mc.get('min_samples',300)), retrain_interval_sec=float(mc.get('retrain_interval_sec',60.0)), max_train_samples=int(mc.get('max_train_samples',3000)), max_val_samples=int(mc.get('max_val_samples',200)))
        self.portfolio=PaperPortfolio(cfg)
        self.telemetry = Telemetry(path=str(cfg.get('storage',{}).get('data_dir','data')) + '/telemetry_engine.jsonl', service='engine', es=es, base_event={'exchange':cfg['runtime']['exchange'],'market':cfg['runtime']['market'],'timeframe':cfg['runtime']['timeframe']})
        mod_cfg = cfg.get('modernization', {})
        self.pattern_scanner = MarketPatternScanner(
            bins=int(mod_cfg.get('volume_profile_bins', 24)),
            breakout_window=int(mod_cfg.get('breakout_window', 20)),
        )
        self.forecaster = MultiHorizonForecaster(
            horizons=mod_cfg.get('forecast_horizons', [1, 3, 5, 15]),
            mc_samples=int(mod_cfg.get('forecast_mc_samples', 64)),
        )
        self.regime_detector = MarketRegimeDetector(
            vol_threshold=float(mod_cfg.get('regime_volatility_threshold', 0.004)),
            trend_threshold=float(mod_cfg.get('regime_trend_threshold', 0.25)),
        )


    def sync_symbols(self) -> int:
        rows = self.market.fetch_symbols()
        return self.symdb.upsert_many(self.cfg["runtime"]["exchange"], self.cfg["runtime"]["market"], rows)

    def universe(self) -> List[str]:
        u=self.cfg["universe"]; rt=self.cfg["runtime"]
        limit=int(rt.get("max_pairs",50))
        if u.get("mode","db")=="db":
            top=self.symdb.top_symbols(rt["exchange"], rt["market"], limit, quote=u.get("quote"), min_volume=float(u.get("min_volume",0)))
            return top if top else u.get("custom_symbols",[])[:limit]
        return u.get("custom_symbols",[])[:limit]

    def tick(self) -> str:
        price_source=str(self.cfg.get('pricing',{}).get('price_source','last'))

        flags=getattr(self,'runtime_flags',{}) or {}
        force_b=bool(flags.get('force_model_b_retrain',False))
        disable_b=bool(flags.get('disable_model_b_training',False))
        rollback_b=bool(flags.get('rollback_model_b',False))

        rt=self.cfg["runtime"]
        run_id=str(uuid.uuid4())
        base={"run_id":run_id,"service":"sf-runtime","exchange":rt["exchange"],"market":rt["market"],"timeframe":rt["timeframe"],"symbol":"*"}
        symbols=self.universe()
        self.es.append(mk_event(base,"UNIVERSE_SELECTED","INFO",{"selected_n":len(symbols),"selected":symbols[:1000]}))

        prices: Dict[str, float] = {}

        for sym in symbols:
            env=dict(base); env["symbol"]=sym
            try:
                sp1=self.telemetry.span_start('FETCH_OHLCV', trace_id=run_id, symbol=sym)
                ohlcv=self.market.fetch_ohlcv(sym, rt["timeframe"], int(rt["lookback"]))
                self.telemetry.span_end(sp1, status='OK', rows=len(ohlcv))
                sp2=self.telemetry.span_start('FETCH_ORDERBOOK', trace_id=run_id, symbol=sym)
                ob=self.market.fetch_orderbook(sym, depth=50)
                self.telemetry.span_end(sp2, status='OK')
                snap={"symbol":sym,"ohlcv":ohlcv,"orderbook":ob}
                ref=self.ss.put(run_id, sym, snap)

                last_close=float(ohlcv[-1][4]) if ohlcv else 0.0
                prices[sym]=last_close

                self.es.append(mk_event(env,"DATA_FETCHED","INFO",{
                    "ohlcv_rows":len(ohlcv),
                    "snapshot_ref":ref,
                    "price":last_close,
                    "spread":ob.get("spread"),
                    "imb":ob.get("imbalance")
                }))

                feats=compute_features(ohlcv, ob)
                arr=np.array([c for _,_,_,_,c,_ in ohlcv], dtype=float)

                # microstructure/levels features
                try:
                    mid = float((arr[-1] + arr[-1]) / 2.0) if len(arr) else 0.0
                    self.ob_tracker.update(sym, ob, max(1e-12, mid))
                    walls = self.ob_tracker.strongest_walls(sym, max(1e-12, mid), topk=1)
                except Exception:
                    walls = []
                wall = walls[0] if walls else None
                feats["wall_age"] = float(getattr(wall, "age_sec", 0.0) or 0.0)
                feats["wall_touches"] = float(getattr(wall, "touch_count", 0.0) or 0.0)
                feats["wall_dist_bps"] = float(getattr(wall, "dist_bps", 0.0) or 0.0)

                pattern = self.pattern_scanner.scan(ohlcv)
                regime = self.regime_detector.detect(arr)
                forecast = self.forecaster.predict(arr)
                fc3 = forecast.get("m3", {})
                feats["forecast_ret_3m"] = float(fc3.get("ret", 0.0) if isinstance(fc3, dict) else 0.0)
                feats["forecast_uncertainty_3m"] = float(fc3.get("uncertainty", 1.0) if isinstance(fc3, dict) else 1.0)
                feats["trend_strength"] = float(getattr(regime, "trend_strength", 0.0))
                atr = float(feats.get("atr", 0.0) or 0.0)
                rr = (2.0 * atr) / max(1e-12, 1.2 * atr) if atr > 0 else 0.0
                feats["rr_ratio"] = float(rr)

                inv=inv_check(feats, self.cfg["features"]["invariants"])
                self.es.append(mk_event(env,"FEATURES_COMPUTED","INFO",{"features":feats,"invariants":inv}))
                self.es.append(mk_event(env, "FEATURES_SNAPSHOT", "INFO", {
                    "imb": feats.get("ob_imb", 0.0),
                    "spread_bps": feats.get("spread_bps", 0.0),
                    "wall_age": feats.get("wall_age", 0.0),
                    "wall_touches": feats.get("wall_touches", 0.0),
                    "wall_dist_bps": feats.get("wall_dist_bps", 0.0),
                }))
                self.es.append(mk_event(env, "SETUP_EVENT", "INFO", {
                    "regime": regime.name,
                    "forecast_ret_3m": feats.get("forecast_ret_3m", 0.0),
                    "forecast_unc_3m": feats.get("forecast_uncertainty_3m", 1.0),
                    "rr_ratio": feats.get("rr_ratio", 0.0),
                    "poc": pattern.volume_profile_poc,
                }))
                if not inv_ok(inv):
                    self.es.append(mk_event(env,"GATE_DECISION","INFO",{"decision":"FAIL","reasons":["INVARIANTS_OK"]}))
                    continue

                self.es.append(mk_event(env, "MARKET_PATTERN_SCAN", "INFO", pattern.as_dict()))
                self.es.append(mk_event(env, "REGIME_DETECTED", "INFO", regime.as_dict()))
                self.es.append(mk_event(env, "MULTI_HORIZON_FORECAST", "INFO", {"forecast": forecast}))
                fwd=int(self.cfg["model"].get("forward",4))
                y=label_forward(arr, fwd)
                X=np.array([[feats["rsi"],feats["atr"],feats["adx"],feats["ob_imb"],feats["spread"]] for _ in range(len(arr))], dtype=float)

                added=self.trainer.add_samples(X[:-fwd], y[:-fwd])
                pass  # retrain throttled

                x_last=X[-1]
                mout=self.trainer.infer(x_last)
                mhealth=self.trainer.health()
                self.es.append(mk_event(env,"MODEL_INFERRED","INFO",{
                    "pred":mout["pred"],"confidence":mout["confidence"],
                    "metrics":mhealth,"samples_added":added
                }))

                gate=gate_decide(
                    feats, inv, mout, mhealth,
                    {"profile":self.cfg["gate"]["profile"],"profiles":self.cfg["gate"]["profiles"],
                     "min_auc":float(self.cfg["model"].get("min_auc",0.52)),
                     "min_samples":int(self.cfg["model"].get("min_samples",300)),
                     "auc_warmup_samples":int(self.cfg["model"].get("auc_warmup_samples",600))}
                )
                self.es.append(mk_event(env,"GATE_DECISION","INFO",gate))


                # Model-B shadow inference + dataset shadow label
                try:
                    b_feats={
                        "ret_last": float(gate.get("ret_last",0.0) if isinstance(gate,dict) else 0.0),
                        "range_last": float(gate.get("range_last",0.0) if isinstance(gate,dict) else 0.0),
                        "vol_z_last": float(gate.get("vol_z_last",0.0) if isinstance(gate,dict) else 0.0),
                        "imb": float(gate.get("imb",0.0) if isinstance(gate,dict) else 0.0),
                        "spread_bps": float(gate.get("spread_bps",0.0) if isinstance(gate,dict) else 0.0),
                        "wall_dist_bps": float(gate.get("wall_dist_bps",0.0) if isinstance(gate,dict) else 0.0),
                        "wall_age": float(gate.get("wall_age",0.0) if isinstance(gate,dict) else 0.0),
                        "wall_touches": float(gate.get("wall_touches",0.0) if isinstance(gate,dict) else 0.0),
                    }
                    prob=self.model_b_trainer.infer_prob(b_feats)
                    thr=float(self.cfg.get("model_b",{}).get("model_b_prob_min",0.55))
                    decision_b="PASS" if prob>=thr else "FAIL"
                    self.es.append(mk_event(env,"MODEL_B_INFERRED","INFO",{"prob_head":prob,"prob_backbone":0.5,"thr":thr,"decision":decision_b,**b_feats}))
                    h=int(self.ds_builder.horizon_bars)
                    if ohlcv and len(ohlcv)>(h+2):
                        i=len(ohlcv)-(h+2)
                        c0=float(ohlcv[i][4]); c1=float(ohlcv[i+h][4])
                        ret=(c1-c0)/max(1e-12,c0)
                        self.ds_builder.append_shadow(int(time.time()*1000), sym, int(ohlcv[-2][0]) if len(ohlcv)>1 else 0,
                                                     int(time.time()*1000), price_source, "bybit.linear.swap", b_feats, ret)
                except Exception as e:
                    self.es.append(mk_event(env,"ERROR","ERROR",{"where":"MODEL_B_SHADOW","err":str(e)}))

                direction="NEUTRAL"
                entry_th=float(self.cfg["gate"]["profiles"][self.cfg["gate"]["profile"]]["entry_threshold"])
                if mout["pred"]>=entry_th: direction="LONG"
                elif mout["pred"]<=-entry_th: direction="SHORT"

                self.es.append(mk_event(env,"SIGNAL","INFO",{
                    "direction":direction,"pred":mout["pred"],"confidence":mout["confidence"],"gate":gate,
                    "entry": last_close, "sl": feats.get("sl"), "tp1": feats.get("tp1"), "tp2": feats.get("tp2")
                }))

                if gate.get("decision")=="PASS" and direction in ("LONG","SHORT") and sym not in self.portfolio.positions:
                    res=self.portfolio.open(now_ms(), sym, direction, last_close, feats["atr"])
                    self.es.append(mk_event(env,"TRADE_OPEN","INFO",res))

            except Exception as e:
                self.es.append(mk_event(env,"ERROR","ERROR",{"error":str(e)}))
        # retrain model at most once per interval (prevents stalls)

        # retrain Model B head on accumulated labeled samples (scheduled)
        try:
            mB, promoted = (None, False)
            if not disable_b:
                mB, promoted = self.model_b_trainer.maybe_retrain(now_ms(), force=force_b)
            if mB is not None:
                self.es.append(mk_event(base, "MODEL_B_RETRAIN", "INFO", {
                    "auc": getattr(mB,"auc",0.0),
                    "pr_auc": getattr(mB,"pr_auc",0.0),
                    "acc": getattr(mB,"acc",0.0),
                    "samples": getattr(mB,"samples",0),
                    "promoted": bool(promoted)
                }))
            else:
                self.es.append(mk_event(base, "MODEL_B_TRAINING_STATUS", "INFO", {
                    "disabled": bool(disable_b),
                    "dataset_path": self.ds_builder.path,
                    "rows": int(self.ds_builder.stats().get("rows", 0)),
                    "force": bool(force_b)
                }))
        except Exception as e:
            self.es.append(mk_event(base, "ERROR", "ERROR", {"where": "MODEL_B_RETRAIN", "err": str(e)}))



        try:
            m_before = self.trainer.health()
            m_after, did = self.trainer.maybe_retrain(now_ms())
            if did:
                payload = {"before": m_before}
                try:
                    payload["after"] = m_after.__dict__
                except Exception:
                    payload["after"] = m_before
                self.es.append(mk_event(base, "MODEL_RETRAIN", "INFO", payload))
        except Exception as e:
            self.es.append(mk_event(base, "ERROR", "ERROR", {"where": "MODEL_RETRAIN", "err": str(e)}))

        # merge fresh prices for ALL open positions
        try:
            pos_syms = list(self.portfolio.positions.keys())
            if pos_syms:
                pos_prices = self.market.fetch_prices(pos_syms)
                if isinstance(pos_prices, dict):
                    prices.update(pos_prices)
        except Exception:
            pass

        # manage open positions (exit manager)
        try:
            updates, closed = self.portfolio.manage_positions(now_ms(), prices)
            for u in updates:
                self.es.append(mk_event(base, "POSITION_UPDATE", "INFO", u))
            for c in closed:
                self.es.append(mk_event(base, "TRADE_CLOSED", "INFO", c))
                try:
                    self.trainer.on_trade_closed(c)
                except Exception:
                    pass
        except Exception as e:
            self.es.append(mk_event(base, "ERROR", "ERROR", {"where": "manage_positions", "err": str(e)}))

        # account + snapshots for UI/replay
        self.es.append(mk_event(base, "ACCOUNT", "INFO", {
            "deposit": self.portfolio.account.deposit,
            "cash": self.portfolio.account.cash,
            "equity": self.portfolio.account.equity,
            "open_positions": len(self.portfolio.positions),
        }))
        pos_payload = {"positions": {k: {
            "symbol": v.symbol, "side": v.side, "entry": v.entry, "qty": v.qty,
            "sl": v.sl, "tp1": v.tp1, "tp2": v.tp2, "open_ts": v.open_ts
        } for k, v in self.portfolio.positions.items()}}
        self.es.append(mk_event(base, "POSITIONS_SNAPSHOT", "INFO", pos_payload))

        self.es.append(mk_event(base, "RUN_DONE", "INFO", {"run_id": run_id}))
        # Model-B status heartbeat
        try:
            st = self.ds_builder.stats() if hasattr(self,'ds_builder') else {}
            m = getattr(self.model_b_trainer, 'last_metrics', None)
            mp = {
                'dataset_rows': st.get('rows',0),
                'pending': st.get('pending',0),
                'dataset_path': st.get('path',''),
                'training_disabled': bool(disable_b),
            }
            if m is not None:
                mp.update({'auc': getattr(m,'auc',0.0), 'pr_auc': getattr(m,'pr_auc',0.0), 'acc': getattr(m,'accuracy',0.0), 'samples': getattr(m,'samples',0)})
            self.es.append(mk_event(base, 'MODEL_B_STATUS', 'INFO', mp))
        except Exception:
            pass


        # Model-B status heartbeat

        try:

            st=self.ds_builder.stats()

            self.es.append(mk_event(base,

                                    "MODEL_B_STATUS","INFO",{"dataset_rows": st.get("rows",0), "dataset_path": st.get("path",""), "price_source": price_source}))

        except Exception:

            pass


        # Model-B retrain schedule

        try:

            flags=getattr(self,'runtime_flags',{}) or {}

            force_b=bool(flags.get('force_model_b_retrain',False))

            disable_b=bool(flags.get('disable_model_b_training',False))

            if not disable_b:

                mtr,prom=self.model_b_trainer.maybe_retrain(int(time.time()*1000), force=force_b)

                if mtr is not None:

                    self.es.append(mk_event(base,

                                            "MODEL_B_RETRAIN","INFO",{"auc":mtr.auc,"pr_auc":mtr.pr_auc,"acc":mtr.acc,"samples":mtr.samples,"promoted":bool(prom)}))

                    if prom:

                        self.es.append(mk_event(base,

                                                "MODEL_B_PROMOTED","INFO",{"samples":mtr.samples,"auc":mtr.auc}))

        except Exception as e:

            self.es.append(mk_event(base,

                                    "ERROR","ERROR",{"where":"MODEL_B_RETRAIN","err":str(e)}))


        # Accounting reconciliation

        try:

            cash=float(getattr(self.portfolio,'cash',0.0))

            upnl=float(getattr(self.portfolio,'unrealized_pnl',0.0)) if hasattr(self.portfolio,'unrealized_pnl') else 0.0

            equity=float(getattr(self.portfolio,'equity',cash+upnl))

            calc=cash+upnl

            diff=calc-equity

            self.es.append(mk_event(base,

                                    "ACCOUNT","INFO",{"cash":cash,"upnl":upnl,"equity":equity,"equity_calc":calc,"diff":diff,"price_source": price_source}))

            if abs(diff) > max(1e-6, abs(equity)*1e-4):

                self.es.append(mk_event(base,

                                        "ACCOUNT_MISMATCH","ERROR",{"cash":cash,"upnl":upnl,"equity":equity,"equity_calc":calc,"diff":diff}))

        except Exception:

            pass


        return run_id

