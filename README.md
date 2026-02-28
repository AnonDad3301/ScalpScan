# ScalpForge Desktop (PySide6/Qt) — v1.7.0 (No Docker)

Цель: UI как терминал — плавные таблицы, live PnL, минимальная задержка.
Архитектура: backend hub (WebSocket) + runtime tick loop + optional price polling (ccxt).

## Быстрый старт
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
# Если нужен live с биржи:
pip install ccxt
python backend/hub.py
# В другом терминале:
python desktop_app/main.py
```

## Компоненты
- backend/hub.py — WebSocket сервер + запускает tick loop и (опционально) price polling
- desktop_app/main.py — PySide6 UI (Dashboard / Signals / Positions / Trace)
- data/*.sqlite — локальные базы (events, symbols)

## Конфиг
`config.yaml` — adapter mock/ccxt, exchange, market, timeframe, N пар, интервалы.
- `runtime.ws_port` по умолчанию: `8766` (hub/UI/preflight).


## Предпроверка
Backend больше НЕ запускает сканер автоматически. Сначала выполняется предпроверка (валидность конфига, ccxt, symbols, тест OHLCV). Пока OK не получен — кнопка Старт заблокирована.


## Примечание про `opening handshake failed`
На Windows иногда в консоли backend появляется `opening handshake failed` — это обычно внешний процесс/клиент, который подключился к WS-порту и сразу оборвал соединение. В v1.3.8 эти сообщения подавлены в логах и не влияют на работу.


## Диагностика
В v1.3.8 backend шлёт heartbeat каждую секунду. В панели видно возраст heartbeat. Если heartbeat не приходит — проблема связи/порта.


## WS port check
В v1.3.8 предпроверка не валится, если порт уже занят самим hub (status=listen). FAIL только если порт занят, но не слушает (залипший процесс/антивирус).

## v1.3.8 Fix Pack
- Позиции/SL/TP показываются всегда: движок пишет POSITIONS_SNAPSHOT в events.sqlite.
- UI работает от SQLite (история/сигналы/позиции) и использует 1 постоянное WS-соединение для команд/инструментов.
- Риск: добавлены лимиты per_trade_alloc_pct и total_alloc_pct, чтобы не заходить на весь бюджет.


v1.3.8: Adds POSITIONS_SNAPSHOT events and disables websocket ping keepalive to avoid disconnects.


v1.3.8: POSITIONS_SNAPSHOT persisted; WS ping disabled; websockets logs silenced.


## v1.4.0.1
- Exit manager: TP1 partial, BE, TP2, SL, time-stop
- TRADE_CLOSED events + Closed trades tab
- Trainer learns only on TRADE_CLOSED


## v1.4.1
- Fixed engine tick indentation and price merge for open positions
- CCXTMarket symbol normalization for Bybit perp
- Suppressed noisy Invalid HTTP handshake


## v1.4.4
- Hub: eng.tick and fetch_prices run in background threads (asyncio.to_thread) to prevent stalls.
- Scheduling uses monotonic clock; PRICE_TICK should be ~1s.
- UI Monitor shows last PRICE_TICK age.


## v1.4.5
- Hard real-time: PRICE_TICK is isolated from scanner tick.
- Timeouts: tick_timeout_sec / price_timeout_sec prevent 5-15 minute stalls.
- CCXT timeout set to 10s.


## v1.4.6.6
- SQLiteEventStore is thread-safe (check_same_thread=False + lock). Fixes sqlite3.InterfaceError when engine.tick runs in to_thread.


## v1.4.6.7
- Price loop fallback: missing prices filled from last_prices cache, then 1m OHLCV last close.
- CCXTMarket auto-resets exchange after repeated errors.
- Monitor shows PRICE_TICK age.


## v1.4.6.8
- Price feed isolated in separate process (PriceWorker) to survive ccxt hard-hangs.
- On timeout, worker is terminated/restarted.
- PRICE_WORKER events log errors/timeouts.


## v1.4.6.11
- PriceWorker hardened for Windows spawn: sys.path injected, import time fixed, ready/ping protocol.
- Hub logs PRICE_WORKER_READY at startup.


## v1.5.0
- Real-time prices via Bybit WebSocket V5 tickers topic.
- PriceWorker/REST fallback when WS stale.


## v1.5.2
- Closed trades persisted to data/trades.sqlite (trades_closed). UI reads from this store.


## v1.5.4
- Iteration 1.1: auto resubscribe/reconnect when WS stale or ticker_updates_1s==0.
- Added mark-price fallback via PriceWorker (fetch_mark_prices) to avoid frozen PnL when REST batch times out.
- New config: ws_stale_resub_ms, ws_stale_reconnect_ms, ws_zero_ticks_limit.


## v1.6.0
- Iteration 2: OrderBookTracker (wall age/touches), FEATURES_SNAPSHOT, SETUP_EVENT.
- Iteration 3: PatchTST Model B loader (HF auto-download) + MODEL_B_INFERRED scaffold.
- Iteration 4: Breakeven rule (MOVE_SL_BE) near TP1 in paper execution.

## v1.7.0 (архитектурная модернизация)
- Добавлен `monitoring_api` на `aiohttp` (`/healthz`, `/state`) для real-time мониторинга состояния хаба.
- В runtime добавлен слой market intelligence:
  - `MARKET_PATTERN_SCAN` (Volume Profile POC + breakout detection),
  - `REGIME_DETECTED` (calm/trend/volatile),
  - `MULTI_HORIZON_FORECAST` (1/3/5/15 минут + uncertainty).
- Новые параметры в `config.yaml`:
  - `runtime.monitoring_host`, `runtime.monitoring_port`,
  - секция `modernization` для тюнинга сканера/форкастера/детектора режима.

- PRICE_TICK получает статус `DEGRADED`, если WS подключен, но канал устарел/не даёт тикеров; UI показывает этот режим отдельно.
