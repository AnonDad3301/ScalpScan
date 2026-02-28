from __future__ import annotations

from typing import Any, Dict

from backend.bybit_ws_prices import BybitPublicWS
from backend.bitget_ws_prices import BitgetPublicWS


def build_public_ws_client(runtime_cfg: Dict[str, Any]):
    exchange = str(runtime_cfg.get("exchange", "bybit") or "bybit").lower()
    if exchange == "bitget":
        return BitgetPublicWS(inst_type=str(runtime_cfg.get("bitget_inst_type", "USDT-FUTURES")))
    return BybitPublicWS(channel=str(runtime_cfg.get("bybit_channel", "linear")), testnet=bool(runtime_cfg.get("testnet", False)))
