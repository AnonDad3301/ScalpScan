from __future__ import annotations
import time
from packages.obs.messages import explain
import uuid
from typing import Any, Dict, List
import numpy as np

from packages.feature.pipeline import compute as compute_features
from packages.feature.invariants import check as inv_check, ok as inv_ok
from packages.model.trainer import OnlineTrainer
from packages.model.ensemble import meta_decide
from packages.model.labeling import triple_barrier_labels
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
        self._open_feature_bank: Dict[str, np.ndarray] = {}
        self._sl_streak: int = 0


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
        market_ctx: Dict[str, Dict[str, Any]] = {}

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
                try:
                    self.ob_tracker.update(sym, ob, last_close)
                    walls = self.ob_tracker.strongest_walls(sym, last_close, topk=3)
                    bid_wall = next((w for w in walls if w.side == "BID"), None)
                    ask_wall = next((w for w in walls if w.side == "ASK"), None)
                    market_ctx[sym] = {
                        "imbalance": float(ob.get("imbalance", 0.0) or 0.0),
                        "spread": float(ob.get("spread", 0.0) or 0.0),
                        "bid_wall": float(bid_wall.price) if bid_wall else 0.0,
                        "ask_wall": float(ask_wall.price) if ask_wall else 0.0,
                        "bid_wall_age": float(bid_wall.age_sec) if bid_wall else 0.0,
                        "ask_wall_age": float(ask_wall.age_sec) if ask_wall else 0.0,
                    }
                except Exception:
                    market_ctx[sym] = {
                        "imbalance": float(ob.get("imbalance", 0.0) or 0.0),
                        "spread": float(ob.get("spread", 0.0) or 0.0),
                    }
                inv=inv_check(feats, self.cfg["features"]["invariants"])
                self.es.append(mk_event(env,"FEATURES_COMPUTED","INFO",{"features":feats,"invariants":inv}))
                if not inv_ok(inv):
                    self.es.append(mk_event(env,"GATE_DECISION","INFO",{"decision":"FAIL","reasons":["INVARIANTS_OK"]}))
                    continue

                arr=np.array([c for _,_,_,_,c,_ in ohlcv], dtype=float)
                pattern = self.pattern_scanner.scan(ohlcv)
                regime = self.regime_detector.detect(arr)
                regime_name_raw = str(getattr(regime, "name", "unknown"))
                regime_name = "mean_reversion" if regime_name_raw == "calm" else ("high_volatility" if regime_name_raw == "volatile" else regime_name_raw)
                # liquidity gate proxy: extreme spread percentile or sparse L2
                bid_n = len(ob.get("bids") or [])
                ask_n = len(ob.get("asks") or [])
                if float(feats.get("spread_pct_rank", 0.0)) > 0.98 or min(bid_n, ask_n) < 3:
                    regime_name = "low_liquidity"
                forecast = self.forecaster.predict(arr)
                self.es.append(mk_event(env, "MARKET_PATTERN_SCAN", "INFO", pattern.as_dict()))
                reg_payload = regime.as_dict()
                reg_payload["name"] = regime_name
                self.es.append(mk_event(env, "REGIME_DETECTED", "INFO", reg_payload))
                self.es.append(mk_event(env, "MULTI_HORIZON_FORECAST", "INFO", {"forecast": forecast}))
                f3 = forecast.get("m3", {}) if isinstance(forecast, dict) else {}
                f5 = forecast.get("m5", {}) if isinstance(forecast, dict) else {}
                p3 = 1.0 / (1.0 + np.exp(-float((f3.get("ret",0.0) or 0.0) / max(1e-9, f3.get("uncertainty",1.0) or 1.0))))
                p5 = 1.0 / (1.0 + np.exp(-float((f5.get("ret",0.0) or 0.0) / max(1e-9, f5.get("uncertainty",1.0) or 1.0))))
                self.es.append(mk_event(env, "SHORT_TERM_LEVEL_PROB", "INFO", {
                    "p_up_3m": float(p3),
                    "p_up_5m": float(p5),
                    "ret_3m": float(f3.get("ret",0.0) or 0.0),
                    "ret_5m": float(f5.get("ret",0.0) or 0.0),
                    "unc_3m": float(f3.get("uncertainty",1.0) or 1.0),
                    "unc_5m": float(f5.get("uncertainty",1.0) or 1.0),
                    "breakout_strength": float(getattr(pattern, "breakout_strength", 0.0)),
                }))

                # enrich features for gate + model-b observability
                f1 = forecast.get("m1", {}) if isinstance(forecast, dict) else {}
                forecast_ret = float(f1.get("ret", 0.0) or 0.0)
                forecast_unc = float(f1.get("uncertainty", 1.0) or 1.0)
                rr_ratio = abs(float(pattern.breakout_strength)) / max(1e-9, float(feats.get("atr", 0.0))) if hasattr(pattern, 'breakout_strength') else 0.0
                ex_cfg = ((self.cfg.get("execution", {}) or {}).get("exits", {}) or {})
                sl_mult = float(ex_cfg.get("sl_atr_mult", 1.2) or 1.2)
                tp1_mult = float(ex_cfg.get("tp1_atr_mult", 1.1) or 1.1)
                tp2_mult = float(ex_cfg.get("tp2_atr_mult", 1.7) or 1.7)
                rr_from_levels = max(tp1_mult, tp2_mult) / max(1e-9, sl_mult)
                rr_ratio = max(float(rr_ratio), float(rr_from_levels))
                trend_strength = abs(float(feats.get("adx", 0.0))) / 100.0
                feats["forecast_uncertainty"] = forecast_unc
                feats["forecast_ret"] = forecast_ret
                feats["rr_ratio"] = float(rr_ratio)
                feats["trend_strength"] = float(trend_strength)
                feats["regime"] = regime_name
                st_live = self.portfolio.stats()
                win_rate = float(st_live.get("win_rate", 0.5))
                sl_rate = float(st_live.get("sl", 0)) / max(1.0, float(st_live.get("total_closed", 0)))
                feats["win_rate"] = win_rate
                feats["sl_rate"] = sl_rate
                feats["latency_ms"] = float(getattr(self.market, "last_latency_ms", 0.0) or 0.0)
                fees_bps = float(self.cfg.get("execution", {}).get("fees_bps", 0.0)) + float(self.cfg.get("execution", {}).get("slippage_bps", 0.0))
                feats["costs_bps"] = fees_bps
                base_risk = max(float(feats.get("atr", 0.0) or 0.0), float(last_close) * 0.0012)
                sl_mult = float(ex_cfg.get("sl_atr_mult", 1.2) or 1.2)
                tp2_mult = float(ex_cfg.get("tp2_atr_mult", 1.7) or 1.7)
                feats["tp_return"] = float((tp2_mult * base_risk) / max(1e-12, last_close))
                feats["sl_return"] = float((sl_mult * base_risk) / max(1e-12, last_close))
                feats["sl_streak"] = float(self._sl_streak)

                try:
                    walls = self.ob_tracker.strongest_walls(sym, last_close, topk=3)
                    if walls:
                        w = walls[0]
                        feats["wall_dist_bps"] = float(w.dist_bps)
                        feats["wall_age"] = float(w.age_sec)
                        feats["wall_touches"] = float(w.touch_count)
                except Exception:
                    pass

                self.es.append(mk_event(env, "FEATURES_SNAPSHOT", "INFO", {"features": feats}))
                if float(feats.get("rr_ratio",0.0)) >= 1.0 and float(feats.get("forecast_uncertainty",1.0)) <= 0.05:
                    self.es.append(mk_event(env, "SETUP_EVENT", "INFO", {
                        "type": "QUALITY_SETUP",
                        "rr_ratio": feats.get("rr_ratio"),
                        "forecast_uncertainty": feats.get("forecast_uncertainty"),
                        "trend_strength": feats.get("trend_strength"),
                    }))

                fwd=int(self.cfg["model"].get("forward",4))
                atr_series = np.full(len(arr), float(feats.get("atr", 0.0)), dtype=float)
                y, y_tp, y_sl = triple_barrier_labels(
                    arr,
                    atr_series,
                    horizon_bars=max(3, min(5, fwd)),
                    tp_atr_mult=float(self.cfg.get("execution", {}).get("exits", {}).get("tp1_atr_mult", 1.1)),
                    sl_atr_mult=float(self.cfg.get("execution", {}).get("exits", {}).get("sl_atr_mult", 1.2)),
                )
                X=np.array([[feats["rsi"],feats["atr"],feats["adx"],feats["ob_imb"],feats["spread"]] for _ in range(len(arr))], dtype=float)

                added=self.trainer.add_samples(X[:-fwd], y[:-fwd])
                pass  # retrain throttled

                x_last=X[-1]
                mout=self.trainer.infer(x_last)
                p_a_up3 = float(max(0.0, min(1.0, 0.5 + 0.5 * mout.get("pred", 0.0))))
                p_a_up5 = float(max(0.0, min(1.0, 0.5 + 0.35 * mout.get("pred", 0.0))))
                mhealth=self.trainer.health()
                self.es.append(mk_event(env,"MODEL_INFERRED","INFO",{
                    "pred":mout["pred"],"confidence":mout["confidence"],
                    "p_up_3m": p_a_up3, "p_up_5m": p_a_up5,
                    "metrics":mhealth,"samples_added":added
                }))

                # Model-B shadow inference + dataset shadow label
                try:
                    closes = np.array([row[4] for row in ohlcv], dtype=float) if ohlcv else np.array([], dtype=float)
                    ret_last = float((closes[-1]-closes[-2]) / max(1e-12, closes[-2])) if len(closes) >= 2 else 0.0
                    range_last = float((ohlcv[-1][2]-ohlcv[-1][3]) / max(1e-12, ohlcv[-1][4])) if ohlcv else 0.0
                    ret_hist = np.diff(closes) / np.maximum(1e-12, closes[:-1]) if len(closes) >= 3 else np.array([], dtype=float)
                    vol_z_last = 0.0
                    if len(ret_hist) >= 10:
                        mu = float(np.mean(ret_hist[-10:]))
                        sd = float(np.std(ret_hist[-10:]))
                        vol_z_last = float((ret_last - mu) / max(1e-12, sd))
                    ctx = market_ctx.get(sym, {})
                    bid_wall = float(ctx.get("bid_wall", 0.0) or 0.0)
                    ask_wall = float(ctx.get("ask_wall", 0.0) or 0.0)
                    wall_dist = 0.0
                    wall_age = 0.0
                    modelb_side = "LONG" if float(mout.get("pred", 0.0)) >= 0 else "SHORT"
                    if bid_wall > 0.0 and ask_wall > 0.0:
                        if modelb_side == "LONG":
                            wall_dist = abs(ask_wall - last_close) / max(1e-12, last_close) * 10000.0
                            wall_age = float(ctx.get("ask_wall_age", 0.0) or 0.0)
                        else:
                            wall_dist = abs(last_close - bid_wall) / max(1e-12, last_close) * 10000.0
                            wall_age = float(ctx.get("bid_wall_age", 0.0) or 0.0)
                    model_b_feats={
                        "ret_last": ret_last,
                        "range_last": range_last,
                        "vol_z_last": vol_z_last,
                        "imb": float(ctx.get("imbalance", ob.get("imbalance", 0.0)) or 0.0),
                        "spread_bps": float(ctx.get("spread", ob.get("spread", 0.0)) or 0.0),
                        "wall_dist_bps": float(wall_dist),
                        "wall_age": float(wall_age),
                        "wall_touches": float(1.0 if wall_dist > 0.0 else 0.0),
                    }
                    prob=self.model_b_trainer.infer_prob(model_b_feats)
                    thr=float(self.cfg.get("model_b",{}).get("model_b_prob_min",0.55))
                    decision_b="PASS" if prob>=thr else "FAIL"
                    self.es.append(mk_event(env,"MODEL_B_INFERRED","INFO",{"prob_head":prob,"prob_backbone":0.5,"thr":thr,"decision":decision_b,**model_b_feats}))
                    h=int(self.ds_builder.horizon_bars)
                    if ohlcv and len(ohlcv)>(h+2):
                        i=len(ohlcv)-(h+2)
                        c0=float(ohlcv[i][4]); c1=float(ohlcv[i+h][4])
                        ret=(c1-c0)/max(1e-12,c0)
                        self.ds_builder.append_shadow(int(time.time()*1000), sym, int(ohlcv[-2][0]) if len(ohlcv)>1 else 0,
                                                     int(time.time()*1000), price_source, "bybit.linear.swap", model_b_feats, ret)
                        self.es.append(mk_event(env, "MODEL_B_DATASET_APPEND", "INFO", {"symbol": sym, "rows": self.ds_builder.stats().get("rows", 0), "ret": ret, "label": 1 if ret > 0 else 0}))
                except Exception as e:
                    self.es.append(mk_event(env,"ERROR","ERROR",{"where":"MODEL_B_SHADOW","err":str(e)}))

                model_b_prob = float(locals().get("prob", 0.5))
                ensemble_cfg = ((self.cfg.get("model", {}) or {}).get("ensemble", {}) or {})
                ens = meta_decide(
                    model_a={"p_up_3m": p_a_up3, "p_up_5m": p_a_up5},
                    model_b={"p_up_3m": model_b_prob, "p_up_5m": model_b_prob},
                    regime=regime_name,
                    cfg=ensemble_cfg,
                )
                ens_d = ens.as_dict()
                self.es.append(mk_event(env, "ENSEMBLE_INFERRED", "INFO", ens_d))
                mout["p_tp_first"] = ens_d.get("p_tp_first", 0.5)
                mout["p_sl_first"] = ens_d.get("p_sl_first", 0.5)
                mout["pred"] = (ens_d.get("p_up_3m", 0.5) - 0.5) * 2.0
                mout["confidence"] = ens_d.get("confidence", mout.get("confidence", 0.5))
                mout["model_a_direction"] = "LONG" if p_a_up3 >= 0.5 else "SHORT"
                b_margin = float(self.cfg.get("model_b", {}).get("agreement_margin", 0.08) or 0.08)
                if abs(model_b_prob - 0.5) < b_margin:
                    mout["model_b_direction"] = "NEUTRAL"
                else:
                    mout["model_b_direction"] = "LONG" if model_b_prob >= 0.5 else "SHORT"

                gate=gate_decide(
                    feats, inv, mout, mhealth,
                    {"profile":self.cfg["gate"]["profile"],"profiles":self.cfg["gate"]["profiles"],
                     "min_auc":float(self.cfg["model"].get("min_auc",0.52)),
                     "min_samples":int(self.cfg["model"].get("min_samples",300)),
                     "auc_warmup_samples":int(self.cfg["model"].get("auc_warmup_samples",600)),
                     "auc_override_confidence_min": float(self.cfg["model"].get("auc_override_confidence_min",0.70))}
                )
                self.es.append(mk_event(env,"GATE_DECISION","INFO",gate))

                direction = str(ens_d.get("direction", "NEUTRAL"))
                if ens_d.get("no_trade_reason"):
                    gate["decision"] = "FAIL"
                    gate.setdefault("reasons", []).append(f"ENSEMBLE_{str(ens_d.get('no_trade_reason', 'no_trade')).upper()}")

                self.es.append(mk_event(env,"SIGNAL","INFO",{
                    "direction":direction,
                    "pred":mout["pred"],
                    "confidence":mout["confidence"],
                    "signal_strength": ens_d.get("signal_strength", 0.0),
                    "p_up_3m": ens_d.get("p_up_3m", 0.5),
                    "p_up_5m": ens_d.get("p_up_5m", 0.5),
                    "p_tp_first": ens_d.get("p_tp_first", 0.5),
                    "p_sl_first": ens_d.get("p_sl_first", 0.5),
                    "market_regime": feats.get("regime"),
                    "decision_reasons": gate.get("reasons", [])[:3],
                    "ev": float(gate.get("expected_value", 0.0)),
                    "top_features": sorted([
                        ("ob_imb_l1", abs(float(feats.get("ob_imb_l1",0.0)))),
                        ("microprice_delta_bps", abs(float(feats.get("microprice_delta_bps",0.0)))),
                        ("vwap_dist", abs(float(feats.get("vwap_dist",0.0)))),
                        ("atr_percentile", abs(float(feats.get("atr_percentile",0.0)))),
                        ("trend_strength", abs(float(feats.get("trend_strength",0.0)))),
                    ], key=lambda x: x[1], reverse=True)[:3],
                    "gate":gate,
                    "entry": last_close, "sl": feats.get("sl"), "tp1": feats.get("tp1"), "tp2": feats.get("tp2")
                }))

                if gate.get("decision")=="PASS" and direction in ("LONG","SHORT") and sym not in self.portfolio.positions:
                    res=self.portfolio.open(now_ms(), sym, direction, last_close, feats["atr"])
                    if res.get("result") == "OK":
                        self._open_feature_bank[sym] = np.asarray(x_last, dtype=float)
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
                st = self.ds_builder.stats() if hasattr(self, "ds_builder") else {}
                cooldown_left_ms = max(0, int(getattr(self.model_b_trainer, "retrain_ms", 0) - (now_ms() - int(getattr(self.model_b_trainer, "last_train_ms", 0)))))
                ds_stats = self.model_b_trainer.dataset_stats() if hasattr(self, 'model_b_trainer') else {}
                self.es.append(mk_event(base, "MODEL_B_RETRAIN_SKIPPED", "INFO", {
                    "rows": st.get("rows", 0),
                    "need_rows": 200,
                    "cooldown_left_ms": cooldown_left_ms,
                    "training_disabled": bool(disable_b),
                    "last_reason": getattr(self.model_b_trainer, 'last_retrain_reason', 'unknown'),
                    "dataset_pos": ds_stats.get('pos', 0),
                    "dataset_neg": ds_stats.get('neg', 0),
                    "dataset_pos_rate": ds_stats.get('pos_rate', 0.0),
                }))
            self.es.append(mk_event(base, "MODEL_B_TRAINING_STATUS", "INFO", {
                "training_disabled": bool(disable_b),
                "reason": getattr(self.model_b_trainer, 'last_retrain_reason', 'unknown'),
                "last_train_ms": int(getattr(self.model_b_trainer, 'last_train_ms', 0)),
                "retrain_ms": int(getattr(self.model_b_trainer, 'retrain_ms', 0)),
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
        try:
            if not did:
                self.es.append(mk_event(base, "MODEL_RETRAIN_SKIPPED", "INFO", {"samples": m_before.get("samples", 0), "auc": m_before.get("auc", 0.0)}))
        except Exception:
            pass

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
            updates, closed = self.portfolio.manage_positions(now_ms(), prices, market_ctx=market_ctx)
            for u in updates:
                self.es.append(mk_event(base, "POSITION_UPDATE", "INFO", u))
            for c in closed:
                self.es.append(mk_event(base, "TRADE_CLOSED", "INFO", c))
                try:
                    reason = str(c.get("reason", ""))
                    if reason == "SL":
                        self._sl_streak += 1
                    elif reason in ("TP2", "TIME_STOP"):
                        self._sl_streak = 0
                    sym = str(c.get("symbol", ""))
                    x = self._open_feature_bank.pop(sym, None)
                    pnl = float(c.get("pnl", 0.0) or 0.0)
                    y = 1 if pnl > 0 else (-1 if pnl < 0 else 0)
                    if x is not None and y != 0:
                        added_trade = self.trainer.add_samples(np.asarray([x], dtype=float), np.asarray([y], dtype=int))
                        self.es.append(mk_event(base, "MODEL_TRADE_SAMPLE", "INFO", {"symbol": sym, "label": y, "pnl": pnl, "added": int(added_trade)}))
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
        try:
            self.es.append(mk_event(base, "TRADE_OUTCOME_STATS", "INFO", self.portfolio.stats()))
        except Exception:
            pass
        pos_payload = {"positions": {k: {
            "symbol": v.symbol, "side": v.side, "entry": v.entry, "qty": v.qty,
            "sl": v.sl, "tp1": v.tp1, "tp2": v.tp2, "open_ts": v.open_ts
        } for k, v in self.portfolio.positions.items()}}
        self.es.append(mk_event(base, "POSITIONS_SNAPSHOT", "INFO", pos_payload))

        self.es.append(mk_event(base, "RUN_DONE", "INFO", {"run_id": run_id}))
        # Model-B status heartbeat
        try:
            st = self.ds_builder.stats() if hasattr(self,'ds_builder') else {}
            ds_stats = self.model_b_trainer.dataset_stats() if hasattr(self, 'model_b_trainer') else {}
            m = getattr(self.model_b_trainer, 'last_metrics', None)
            progress = min(1.0, float(ds_stats.get('rows', 0)) / 400.0)
            mp = {
                'dataset_rows': st.get('rows',0),
                'pending': st.get('pending',0),
                'dataset_path': st.get('path',''),
                'training_disabled': bool(disable_b),
                'dataset_pos': ds_stats.get('pos', 0),
                'dataset_neg': ds_stats.get('neg', 0),
                'dataset_pos_rate': ds_stats.get('pos_rate', 0.0),
                'training_progress': progress,
                'last_retrain_reason': getattr(self.model_b_trainer, 'last_retrain_reason', 'unknown'),
            }
            if m is not None:
                mp.update({'auc': getattr(m,'auc',0.0), 'pr_auc': getattr(m,'pr_auc',0.0), 'acc': getattr(m,'acc',0.0), 'samples': getattr(m,'samples',0)})
            self.es.append(mk_event(base, 'MODEL_B_STATUS', 'INFO', mp))
        except Exception:
            pass


        return run_id
