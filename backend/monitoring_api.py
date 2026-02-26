from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict

try:
    from aiohttp import web
except Exception:  # optional dependency
    web = None


@dataclass
class MonitoringServer:
    runner: Any
    site: Any


async def start_monitoring_api(host: str, port: int, state: Dict[str, Any]) -> MonitoringServer | None:
    if web is None:
        return None

    async def health(_request: web.Request) -> web.Response:
        payload = {
            "ok": True,
            "ts": int(time.time() * 1000),
            "running": bool(state.get("running", False)),
            "preflight_ok": bool(state.get("preflight_ok", False)),
        }
        return web.json_response(payload)

    async def state_handler(_request: web.Request) -> web.Response:
        payload = dict(state)
        payload["ts"] = int(time.time() * 1000)
        return web.json_response(payload)

    app = web.Application()
    app.add_routes([web.get('/healthz', health), web.get('/state', state_handler)])

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    return MonitoringServer(runner=runner, site=site)


async def stop_monitoring_api(server: MonitoringServer | None) -> None:
    if server is None:
        return
    await asyncio.shield(server.runner.cleanup())
