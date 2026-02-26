from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass
class Account:
    deposit: float
    cash: float
    equity: float

@dataclass
class Position:
    symbol: str
    side: str
    entry: float
    qty: float
    sl: float
    tp1: float
    tp2: float
    open_ts: int
    tp1_done: bool = False
    be_done: bool = False
    last_price: float = 0.0
    realized: float = 0.0
    initial_risk: float = 0.0
    peak_price: float = 0.0
    trough_price: float = 0.0

class PaperPortfolio:
    """
    Paper portfolio with real-time lifecycle:
      - SL close
      - TP1 partial
      - BE move
      - TP2 close
      - time-stop close

    Emits closed trades with final pnl (realized).
    """
    def __init__(self, cfg: Dict[str, Any]):
        ex = cfg.get("execution", {})
        self.fees_bps = float(ex.get("fees_bps", 2.0))
        self.slip_bps = float(ex.get("slippage_bps", 1.0))
        self.risk_usd = float(ex.get("risk_usd", 50.0))
        self.max_positions = int(ex.get("max_positions", 20))
        dep = float(ex.get("deposit_usd", 2000.0))
        self.account = Account(dep, dep, dep)
        self.positions: Dict[str, Position] = {}
        self.exits = ex.get("exits", {})

        self.per_trade_alloc_pct = float(ex.get("per_trade_alloc_pct", 0.10))
        self.total_alloc_pct = float(ex.get("total_alloc_pct", 0.30))
        self.smart_sl = ex.get("smart_sl", {})
        self.trade_stats: Dict[str, int] = {"total_closed": 0, "wins": 0, "sl": 0, "breakeven": 0, "loss": 0}

    def _cost_mult(self) -> float:
        return (self.fees_bps + self.slip_bps) / 10000.0

    def _open_cost_mult(self) -> float:
        return 1.0 + self._cost_mult()

    def _levels(self, price: float, side: str, atr: float):
        base = max(price * 0.0012, atr)
        sl_dist  = float(self.exits.get("sl_atr_mult", 1.2)) * base
        tp1_dist = float(self.exits.get("tp1_atr_mult", 1.1)) * base
        tp2_dist = float(self.exits.get("tp2_atr_mult", 1.7)) * base
        if side == "LONG":
            return price - sl_dist, price + tp1_dist, price + tp2_dist
        else:
            return price + sl_dist, price - tp1_dist, price - tp2_dist

    def exposure_notional(self) -> float:
        return sum(abs(p.entry * p.qty) for p in self.positions.values())

    def can_open(self, notional: float) -> Optional[str]:
        if len(self.positions) >= self.max_positions:
            return "max_positions"
        if notional * self._open_cost_mult() > self.account.cash:
            return "insufficient_cash"
        if notional > self.account.cash * self.per_trade_alloc_pct:
            return "per_trade_alloc_cap"
        if (self.exposure_notional() + notional) > (self.account.deposit * self.total_alloc_pct):
            return "total_alloc_cap"
        return None

    def open(self, ts: int, symbol: str, side: str, price: float, atr: float) -> Dict[str, Any]:
        if symbol in self.positions:
            return {"result":"FAIL","reason":"already_open"}
        sl, tp1, tp2 = self._levels(price, side, atr)

        sl_dist = abs(price - sl) + 1e-12
        qty_risk = self.risk_usd / sl_dist

        notional_cap_trade = self.account.cash * self.per_trade_alloc_pct
        qty_cap_trade = notional_cap_trade / (price + 1e-12)

        notional_cap_total = (self.account.deposit * self.total_alloc_pct) - self.exposure_notional()
        if notional_cap_total <= 0.0:
            return {"result":"FAIL","reason":"total_alloc_cap"}
        notional_cap_total = max(0.0, notional_cap_total)
        qty_cap_total = notional_cap_total / (price + 1e-12)

        qty_cap_cash = self.account.cash / (price * self._open_cost_mult() + 1e-12)

        qty = max(0.0, min(qty_risk, qty_cap_trade, qty_cap_total, qty_cap_cash))
        notional = qty * price
        if qty <= 0.0:
            return {"result":"FAIL","reason":"qty_zero_or_caps"}

        reason = self.can_open(notional)
        if reason:
            return {"result":"FAIL","reason":reason}

        self.account.cash -= notional * self._open_cost_mult()
        self.positions[symbol] = Position(
            symbol, side, price, qty, sl, tp1, tp2, ts,
            last_price=price,
            initial_risk=abs(price-sl),
            peak_price=price,
            trough_price=price,
        )
        return {"result":"OK","qty":qty,"notional":notional,"sl":sl,"tp1":tp1,"tp2":tp2,"cash_after":self.account.cash}

    def _unrealized(self, pos: Position, price: float) -> float:
        return (price - pos.entry) * pos.qty if pos.side == "LONG" else (pos.entry - price) * pos.qty

    def update_equity(self, prices: Dict[str, float]) -> None:
        unreal = 0.0
        for sym, pos in self.positions.items():
            px = float(prices.get(sym, pos.last_price or pos.entry))
            pos.last_price = px
            unreal += self._unrealized(pos, px)
        self.account.equity = self.account.cash + unreal

    def _maybe_move_sl_smart(self, pos: Position, price: float, ctx: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not bool(self.smart_sl.get("enabled", True)):
            return None
        risk = max(1e-12, float(pos.initial_risk or abs(pos.entry-pos.sl)))
        arm_r = float(self.smart_sl.get("arm_r", 0.45))
        min_step_bps = float(self.smart_sl.get("min_step_bps", 2.0))
        wall_buffer_bps = float(self.smart_sl.get("wall_buffer_bps", 6.0))
        fee_lock_bps = float(self.smart_sl.get("fee_lock_bps", 4.0))

        move = (price - pos.entry) if pos.side == "LONG" else (pos.entry - price)
        r_mult = move / risk
        if r_mult < arm_r:
            return None

        old_sl = pos.sl
        reason = "SMART_TRAIL"
        if pos.side == "LONG":
            run = max(0.0, pos.peak_price - pos.entry)
            trail_frac = min(0.85, max(0.0, (r_mult - arm_r) / 2.0))
            candidate = max(old_sl, pos.entry + run * trail_frac)
            if r_mult >= 0.75:
                candidate = max(candidate, pos.entry * (1.0 + fee_lock_bps / 10000.0))
            bid_wall = float((ctx or {}).get("bid_wall", 0.0) or 0.0)
            if bid_wall > 0.0 and bid_wall < price:
                wall_based = bid_wall * (1.0 - wall_buffer_bps / 10000.0)
                candidate = max(candidate, wall_based)
                reason = "SMART_WALL_TRAIL"
            improved = (candidate - old_sl) / max(1e-12, old_sl) * 10000.0
            if improved >= min_step_bps and candidate < price:
                pos.sl = candidate
        else:
            run = max(0.0, pos.entry - pos.trough_price)
            trail_frac = min(0.85, max(0.0, (r_mult - arm_r) / 2.0))
            candidate = min(old_sl, pos.entry - run * trail_frac)
            if r_mult >= 0.75:
                candidate = min(candidate, pos.entry * (1.0 - fee_lock_bps / 10000.0))
            ask_wall = float((ctx or {}).get("ask_wall", 0.0) or 0.0)
            if ask_wall > 0.0 and ask_wall > price:
                wall_based = ask_wall * (1.0 + wall_buffer_bps / 10000.0)
                candidate = min(candidate, wall_based)
                reason = "SMART_WALL_TRAIL"
            improved = (old_sl - candidate) / max(1e-12, old_sl) * 10000.0
            if improved >= min_step_bps and candidate > price:
                pos.sl = candidate

        if pos.sl != old_sl:
            if (pos.side == "LONG" and pos.sl >= pos.entry) or (pos.side == "SHORT" and pos.sl <= pos.entry):
                pos.be_done = True
            return {"event": "SL_MOVE", "from_sl": old_sl, "sl": pos.sl, "reason": reason, "r_mult": r_mult}
        return None

    def manage_positions(self, ts: int, prices: Dict[str, float], market_ctx: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        updates: List[Dict[str, Any]] = []
        closed: List[Dict[str, Any]] = []

        tp1_frac = float(self.exits.get("tp1_fraction", 0.5))
        be_r = float(self.exits.get("breakeven_r", 0.8))
        time_stop_min = int(self.exits.get("time_stop_minutes", 35))

        for sym, pos in list(self.positions.items()):
            px = float(prices.get(sym, pos.last_price or pos.entry))
            pos.last_price = px
            pos.peak_price = max(pos.peak_price, px)
            pos.trough_price = min(pos.trough_price, px)

            if time_stop_min > 0 and (ts - pos.open_ts) >= time_stop_min * 60_000:
                closed.append(self._close(sym, pos, px, ts, "TIME_STOP"))
                continue

            sl_hit = (px <= pos.sl) if pos.side == "LONG" else (px >= pos.sl)
            if sl_hit:
                closed.append(self._close(sym, pos, px, ts, "SL"))
                continue

            sl_update = self._maybe_move_sl_smart(pos, px, (market_ctx or {}).get(sym, {}))
            if sl_update is not None:
                updates.append({"symbol": sym, **sl_update, "price": px})

            tp1_hit = (px >= pos.tp1) if pos.side == "LONG" else (px <= pos.tp1)
            if (not pos.tp1_done) and tp1_hit and tp1_frac > 0:
                qty_close = pos.qty * min(1.0, tp1_frac)
                realized = self._realize_partial(pos, px, qty_close)
                pos.tp1_done = True
                updates.append({"symbol": sym, "event":"TP1", "price": px, "qty_closed": qty_close, "realized": realized})

            if (not pos.be_done) and be_r > 0:
                risk = abs(pos.entry - pos.sl) + 1e-12
                move = (px - pos.entry) if pos.side == "LONG" else (pos.entry - px)
                if move >= be_r * risk:
                    pos.sl = pos.entry
                    pos.be_done = True
                    updates.append({"symbol": sym, "event":"BE", "sl": pos.sl})

            tp2_hit = (px >= pos.tp2) if pos.side == "LONG" else (px <= pos.tp2)
            if tp2_hit:
                closed.append(self._close(sym, pos, px, ts, "TP2"))
                continue

        self.update_equity(prices)
        return updates, closed



    def stats(self) -> Dict[str, float]:
        total = int(self.trade_stats.get("total_closed", 0))
        wins = int(self.trade_stats.get("wins", 0))
        sl = int(self.trade_stats.get("sl", 0))
        be = int(self.trade_stats.get("breakeven", 0))
        loss = int(self.trade_stats.get("loss", 0))
        win_rate = float(wins) / max(1, total)
        return {"total_closed": total, "wins": wins, "sl": sl, "breakeven": be, "loss": loss, "win_rate": win_rate}

    def _realize_partial(self, pos: Position, price: float, qty_close: float) -> float:
        qty_close = max(0.0, min(qty_close, pos.qty))
        if qty_close <= 0:
            return 0.0
        pnl = (price - pos.entry) * qty_close if pos.side == "LONG" else (pos.entry - price) * qty_close
        cost = abs(price * qty_close) * self._cost_mult()
        pnl -= cost
        pos.qty -= qty_close
        pos.realized += pnl
        self.account.cash += abs(price * qty_close) + pnl
        return pnl

    def _close(self, sym: str, pos: Position, price: float, ts: int, reason: str) -> Dict[str, Any]:
        self._realize_partial(pos, price, pos.qty)
        total = pos.realized
        self.trade_stats["total_closed"] = int(self.trade_stats.get("total_closed", 0)) + 1
        if reason == "SL":
            self.trade_stats["sl"] = int(self.trade_stats.get("sl", 0)) + 1
        if abs(float(total)) <= 1e-9:
            self.trade_stats["breakeven"] = int(self.trade_stats.get("breakeven", 0)) + 1
        elif float(total) > 0:
            self.trade_stats["wins"] = int(self.trade_stats.get("wins", 0)) + 1
        else:
            self.trade_stats["loss"] = int(self.trade_stats.get("loss", 0)) + 1
        del self.positions[sym]
        return {
            "symbol": sym, "side": pos.side,
            "entry": pos.entry, "exit": price,
            "open_ts": pos.open_ts, "close_ts": ts,
            "pnl": total, "reason": reason
        }
