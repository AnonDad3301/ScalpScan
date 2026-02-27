from __future__ import annotations
import asyncio, csv, json, os, sys, time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal
from websockets.legacy.client import connect as legacy_connect

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.common.config import Config
from packages.infra.event_store.sqlite_store import SQLiteEventStore
from packages.infra.trades_store.sqlite_trades import TradesStore

def fmt_ts(ts_ms: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(ts_ms)/1000))
    except Exception:
        return ""


def _sparkline(values: List[float], width: int = 24) -> str:
    if not values:
        return "—"
    vals = values[-width:]
    lo = min(vals); hi = max(vals)
    bars = "▁▂▃▄▅▆▇█"
    if hi - lo < 1e-12:
        return bars[0] * len(vals)
    out = []
    for v in vals:
        idx = int((v - lo) / (hi - lo) * (len(bars) - 1))
        out.append(bars[max(0, min(len(bars)-1, idx))])
    return "".join(out)


def write_cmd(cfg: Dict[str, Any], cmd: str, **kwargs) -> None:
    """Write UI command to data/ui_commands.jsonl for hub cmd_loop consumption."""
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    fp = data_dir / "ui_commands.jsonl"
    payload = {"id": int(time.time()*1000), "ts": int(time.time()*1000), "cmd": cmd}
    payload.update(kwargs)
    with fp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

class WSClient(QtCore.QThread):
    """Hybrid transport: receive status via websocket, send commands via file queue."""
    message = Signal(dict)
    status = Signal(str)
    connected = Signal(bool)

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self._stop = False

    def stop(self):
        self._stop = True

    def send_cmd(self, cmd: str, **kwargs):
        # keep robust command path independent of websocket runtime
        write_cmd({}, cmd, **kwargs)

    def run(self):
        asyncio.run(self._main())

    async def _main(self):
        while not self._stop:
            try:
                self.status.emit(f"WS connect: {self.url}")
                async with legacy_connect(self.url, ping_interval=None, ping_timeout=None) as ws:
                    self.connected.emit(True)
                    self.status.emit("WS connected")
                    async for raw in ws:
                        if self._stop:
                            break
                        try:
                            self.message.emit(json.loads(raw))
                        except Exception:
                            continue
            except Exception as e:
                self.connected.emit(False)
                self.status.emit(f"WS reconnect in 1s: {e}")
                await asyncio.sleep(1.0)



class Table(QtWidgets.QTableWidget):
    def __init__(self, headers: List[str]):
        super().__init__(0, len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.setWordWrap(False)
        self.setSortingEnabled(True)

    def set_rows(self, rows: List[List[Any]]):
        self.setSortingEnabled(False)
        self.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QtWidgets.QTableWidgetItem("" if val is None else str(val))
                if c == 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.setItem(r, c, item)
        self.resizeColumnsToContents()
        self.setSortingEnabled(True)

SimpleTable = Table

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: Dict[str, Any], cfg_path: Path):
        super().__init__()
        self.cfg = cfg
        self.cfg_path = cfg_path
        self.root_dir = os.path.dirname(os.path.abspath(cfg_path))
        self.setWindowTitle("ScalpForge Desktop v1.4.3 (No Docker)")
        self.resize(1600, 1000)
        self._apply_theme()

        rt = cfg.get("runtime", {})
        host = rt.get("ws_host", "127.0.0.1")
        port = int(rt.get("ws_port", 8765))
        self.ws_url = f"ws://{host}:{port}"

        self.es = SQLiteEventStore(os.path.join(self.root_dir, cfg["storage"]["events_db"]))
        self.trades = TradesStore(os.path.join(self.root_dir, cfg["storage"].get("trades_db","data/trades.sqlite")))

        # state
        self.ws_connected = False
        self.last_ws_msg_ts = 0
        self.preflight_ok = False
        self.preflight_report: List[Dict[str, Any]] = []
        self.symbols: List[str] = []
        self.events: List[Dict[str, Any]] = []
        self.prices: Dict[str, float] = {}
        self.positions: Dict[str, Any] = {}
        self.account: Dict[str, Any] = {"deposit": None, "cash": None, "equity": None, "open_positions": None}
        self.model_health: Dict[str, Any] = {}
        self.trade_outcome_stats: Dict[str, Any] = {}

        # UI
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self._tab_preflight()
        self._tab_panel()
        self._tab_scanner()
        self._tab_model()
        self._tab_trace()
        self._tab_instruments()
        self._tab_closed()
        self._tab_monitor()
        self._tab_modelb()
        self._tab_dataset()
        self._tab_mlops()
        self._tab_levels()
        self._tab_performance()
        self._tab_analytics()
        self._tab_settings()
        self._tab_logs()

        # WS (commands only)
        self.ws = WSClient(self.ws_url)
        self.ws.message.connect(self.on_message)
        self.ws.status.connect(self.on_status)
        self.ws.connected.connect(self.on_connected)
        self.ws.start()

        # Poll sqlite (source of truth)
        self.poller = QtCore.QTimer()
        self.poller.timeout.connect(self.poll_sqlite)
        self.poller.start(500)

        # Render refresh
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh_ui)
        self.timer.start(250)

        QtCore.QTimer.singleShot(800, lambda: self.ws.send_cmd("run_preflight"))
        QtCore.QTimer.singleShot(1200, lambda: self.ws.send_cmd("get_symbols", limit=2000))


    def _apply_theme(self):
        ui = (self.cfg.get("ui", {}) or {})
        theme = str(ui.get("theme", "light"))
        if theme == "dark_green":
            self.setStyleSheet("""
                QWidget { background-color: #111111; color: #A8FF60; }
                QTableWidget { gridline-color: #2a2a2a; selection-background-color: #1f3d1f; }
                QHeaderView::section { background-color: #1a1a1a; color: #A8FF60; border: 1px solid #2a2a2a; }
                QPushButton { background-color: #1b1b1b; color: #A8FF60; border: 1px solid #355e35; padding: 4px 8px; }
                QLineEdit, QPlainTextEdit, QDoubleSpinBox, QComboBox { background-color: #1a1a1a; color: #A8FF60; border: 1px solid #355e35; }
                QTabWidget::pane { border: 1px solid #2a2a2a; }
            """)

    def closeEvent(self, e):
        try:
            self.ws.stop()
            self.ws.wait(1000)
        except Exception:
            pass
        super().closeEvent(e)

    def _set_header_tips(self, table: QtWidgets.QTableWidget, tips: Dict[str, str]):
        try:
            for i in range(table.columnCount()):
                h = table.horizontalHeaderItem(i)
                if not h:
                    continue
                t = tips.get(h.text())
                if t:
                    h.setToolTip(t)
        except Exception:
            pass

    # Tabs
    def _tab_preflight(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.lbl_status = QtWidgets.QLabel("Статус: —")
        self.lbl_live = QtWidgets.QLabel("Live: —")
        self.lbl_pf = QtWidgets.QLabel("Предпроверка: —")
        self.lbl_pf.setStyleSheet("font-size:16px; font-weight:700;")
        lay.addWidget(self.lbl_status)
        lay.addWidget(self.lbl_live)
        lay.addWidget(self.lbl_pf)
        self.tbl_pf = Table(["Проверка", "OK", "Детали"])
        lay.addWidget(self.tbl_pf)

        row = QtWidgets.QHBoxLayout()
        self.btn_pf = QtWidgets.QPushButton("Запустить предпроверку")
        self.btn_start = QtWidgets.QPushButton("▶ Старт сканера")
        self.btn_stop = QtWidgets.QPushButton("■ Стоп")
        self.btn_syms = QtWidgets.QPushButton("Обновить инструменты")
        self.btn_pf.clicked.connect(lambda: self.ws.send_cmd("run_preflight"))
        self.btn_start.clicked.connect(lambda: self.ws.send_cmd("start"))
        self.btn_stop.clicked.connect(lambda: self.ws.send_cmd("stop"))
        self.btn_syms.clicked.connect(lambda: self.ws.send_cmd("get_symbols", limit=4000))
        row.addWidget(self.btn_pf); row.addWidget(self.btn_start); row.addWidget(self.btn_stop); row.addWidget(self.btn_syms); row.addStretch(1)
        lay.addLayout(row)

        self.tabs.addTab(w, "Предпроверка")

    def _tab_panel(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        grid = QtWidgets.QGridLayout()
        self.kpi_deposit = QtWidgets.QLabel("Депозит: —")
        self.kpi_cash = QtWidgets.QLabel("Cash: —")
        self.kpi_equity = QtWidgets.QLabel("Equity: —")
        self.kpi_open = QtWidgets.QLabel("Позиции: —")
        for i, lab in enumerate([self.kpi_deposit, self.kpi_cash, self.kpi_equity, self.kpi_open]):
            lab.setStyleSheet("font-size:18px; font-weight:700;")
            grid.addWidget(lab, 0, i)
        lay.addLayout(grid)
        self.tbl_positions = Table(["Символ","Сторона","Цена","Вход","Qty","PnL","SL","TP1","TP2"])
        lay.addWidget(self.tbl_positions)
        self.tabs.addTab(w, "Панель")

    def _tab_scanner(self):
        w=QtWidgets.QWidget(); lay=QtWidgets.QVBoxLayout(w)
        self.tbl_signals=Table(["Время","Символ","Направление","Pred","Conf","P3","P5","TP-first","SL-first","Strength","EV","Regime","Gate","Причины"])
        self._set_header_tips(self.tbl_signals, {
            "Pred": "Предсказанное направление/сила сигнала моделью.",
            "Conf": "Уверенность модели в сигнале (0..1).",
            "Gate": "Итог фильтра качества (PASS/FAIL).",
        })
        lay.addWidget(self.tbl_signals)
        self.tabs.addTab(w,"Сканер")

    def _tab_model(self):
        w=QtWidgets.QWidget(); lay=QtWidgets.QVBoxLayout(w)
        grid=QtWidgets.QGridLayout(); lay.addLayout(grid)
        self.m_samples=QtWidgets.QLabel("Samples: —")
        self.m_auc=QtWidgets.QLabel("AUC: —")
        self.m_acc=QtWidgets.QLabel("Accuracy: —")
        self.m_ver=QtWidgets.QLabel("Version: —")
        for i,lab in enumerate([self.m_samples,self.m_auc,self.m_acc,self.m_ver]):
            lab.setStyleSheet("font-size:16px; font-weight:600;"); grid.addWidget(lab,0,i)
        self.tbl_model=Table(["Время","Символ","Pred","Conf","AUC","Samples"])
        self._set_header_tips(self.tbl_model, {
            "AUC": "Качество классификации (чем ближе к 1, тем лучше).",
            "Samples": "Количество обучающих примеров.",
        })
        lay.addWidget(self.tbl_model)
        self.tabs.addTab(w,"Модель/Обучение")

    def _tab_trace(self):
        w=QtWidgets.QWidget(); lay=QtWidgets.QVBoxLayout(w)
        self.tbl_trace=Table(["Время","run_id","Символ","Этап","Уровень","Payload"])
        lay.addWidget(self.tbl_trace)
        self.tabs.addTab(w,"Трассировка")

    def _tab_instruments(self):
        w=QtWidgets.QWidget(); lay=QtWidgets.QVBoxLayout(w)
        self.tbl_syms=Table(["#", "Symbol"]); lay.addWidget(self.tbl_syms)
        self.tabs.addTab(w,"Инструменты")

    def _tab_closed(self):
        w=QtWidgets.QWidget(); lay=QtWidgets.QVBoxLayout(w)
        self.tbl_closed=Table(["CloseTime","Symbol","Side","Entry","Exit","PnL","Reason"])
        lay.addWidget(self.tbl_closed)
        self.tabs.addTab(w,"Завершенные")

    def _tab_monitor(self):
        w=QtWidgets.QWidget(); lay=QtWidgets.QVBoxLayout(w)
        info=QtWidgets.QLabel("Мониторинг: хвост событий PRICE_TICK/ERROR из events.sqlite и telemetry (если включено).")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.txt_mon=QtWidgets.QPlainTextEdit()
        self.txt_mon.setReadOnly(True)
        lay.addWidget(self.txt_mon)
        btn=QtWidgets.QPushButton("Обновить")
        btn.clicked.connect(self._refresh_monitor)
        lay.addWidget(btn)
        self.tabs.addTab(w,"Мониторинг")

    def _tab_performance(self):
        w=QWidget(); lay=QVBoxLayout(w)
        info=QLabel("Эффективность: статистика исходов сделок + вероятности движения 3/5 минут до уровней.")
        info.setWordWrap(True); lay.addWidget(info)
        self.lbl_perf=QLabel("Сделки: пока нет данных")
        lay.addWidget(self.lbl_perf)
        self.tbl_prob=SimpleTable(["Время","Символ","P(up 3m)","P(up 5m)","TP-first","SL-first","Signal","EV","Regime","ret_3m","ret_5m"])
        lay.addWidget(self.tbl_prob)
        self.tabs.addTab(w, "Эффективность")

    def _kpi_card(self, title: str) -> QtWidgets.QLabel:
        lab = QLabel(f"{title}: —")
        lab.setStyleSheet("font-size:16px; font-weight:700; background:#1f2937; color:#e5e7eb; border-radius:10px; padding:10px;")
        return lab

    def _tab_analytics(self):
        w=QWidget(); lay=QVBoxLayout(w)
        info=QLabel("Статистика системы: качество сканирования, сигналы, открытие/закрытие сделок, обучение моделей. Обновляется из events.sqlite в реальном времени.")
        info.setWordWrap(True); lay.addWidget(info)

        cards=QHBoxLayout()
        self.kpi_sig_total=self._kpi_card("Сигналы")
        self.kpi_sig_pass=self._kpi_card("PASS")
        self.kpi_opened=self._kpi_card("Открыто")
        self.kpi_closed=self._kpi_card("Закрыто")
        self.kpi_winrate=self._kpi_card("WinRate")
        for c in (self.kpi_sig_total,self.kpi_sig_pass,self.kpi_opened,self.kpi_closed,self.kpi_winrate):
            cards.addWidget(c)
        lay.addLayout(cards)

        bars=QHBoxLayout()
        self.pb_signal_pass=QtWidgets.QProgressBar(); self.pb_signal_pass.setFormat("Signal PASS rate: %p%")
        self.pb_trade_win=QtWidgets.QProgressBar(); self.pb_trade_win.setFormat("Trade win rate: %p%")
        self.pb_scan_cov=QtWidgets.QProgressBar(); self.pb_scan_cov.setFormat("Scan coverage: %p%")
        self.lbl_graph_scan=QLabel("Scan trend: —")
        self.lbl_graph_signal=QLabel("Signal trend: —")
        for b in (self.pb_signal_pass,self.pb_trade_win,self.pb_scan_cov):
            b.setRange(0,100); b.setValue(0); bars.addWidget(b)
        lay.addLayout(bars)
        lay.addWidget(self.lbl_graph_scan)
        lay.addWidget(self.lbl_graph_signal)

        split=QHBoxLayout()
        left=QVBoxLayout(); right=QVBoxLayout()
        self.tbl_stats_models=SimpleTable(["Метрика","Значение"])
        self.tbl_stats_trades=SimpleTable(["Метрика","Значение"])
        self.tbl_stats_scan=SimpleTable(["Время","Coverage %","Pass %","Error %","Processed","Errors"])
        left.addWidget(QLabel("Модели/обучение")); left.addWidget(self.tbl_stats_models)
        right.addWidget(QLabel("Торговля/сигналы")); right.addWidget(self.tbl_stats_trades)
        split.addLayout(left); split.addLayout(right)
        lay.addLayout(split)
        lay.addWidget(QLabel("История сканера (последние тики)"))
        lay.addWidget(self.tbl_stats_scan)
        self.tabs.addTab(w, "Статистика")

    def _tab_settings(self):
        w=QWidget(); lay=QVBoxLayout(w)
        info=QLabel("Настройки из панели: можно менять фильтры гейта и Telegram-уведомления. Поля на английском, пояснения — на русском.")
        info.setWordWrap(True); lay.addWidget(info)

        form = QtWidgets.QFormLayout()
        self.ed_conf = QtWidgets.QDoubleSpinBox(); self.ed_conf.setRange(0.0, 1.0); self.ed_conf.setSingleStep(0.01)
        self.ed_rr = QtWidgets.QDoubleSpinBox(); self.ed_rr.setRange(0.0, 10.0); self.ed_rr.setSingleStep(0.1)
        self.ed_unc = QtWidgets.QDoubleSpinBox(); self.ed_unc.setRange(0.0, 1.0); self.ed_unc.setSingleStep(0.005)
        self.ed_auc_over = QtWidgets.QDoubleSpinBox(); self.ed_auc_over.setRange(0.0, 1.0); self.ed_auc_over.setSingleStep(0.01)
        self.ed_high_conf = QtWidgets.QDoubleSpinBox(); self.ed_high_conf.setRange(0.0, 1.0); self.ed_high_conf.setSingleStep(0.01)
        self.ed_spread_rank = QtWidgets.QDoubleSpinBox(); self.ed_spread_rank.setRange(0.0, 1.0); self.ed_spread_rank.setSingleStep(0.01)
        self.ed_latency_max = QtWidgets.QSpinBox(); self.ed_latency_max.setRange(0, 20000); self.ed_latency_max.setSingleStep(100)
        self.ed_gray_lo = QtWidgets.QDoubleSpinBox(); self.ed_gray_lo.setRange(0.0, 1.0); self.ed_gray_lo.setSingleStep(0.01)
        self.ed_gray_hi = QtWidgets.QDoubleSpinBox(); self.ed_gray_hi.setRange(0.0, 1.0); self.ed_gray_hi.setSingleStep(0.01)
        self.ed_tg_enabled = QtWidgets.QCheckBox("Telegram enabled")
        self.ed_tg_token = QtWidgets.QLineEdit()
        self.ed_tg_chat = QtWidgets.QLineEdit()
        self.ed_tg_signal = QtWidgets.QCheckBox("send_signal")
        self.ed_tg_trade_open = QtWidgets.QCheckBox("send_trade_open")
        self.ed_tg_trade_closed = QtWidgets.QCheckBox("send_trade_closed")
        self.ed_tg_gate_fail = QtWidgets.QCheckBox("send_on_gate_fail")
        self.ed_tg_min_interval = QtWidgets.QDoubleSpinBox(); self.ed_tg_min_interval.setRange(0.0, 300.0); self.ed_tg_min_interval.setSingleStep(0.5)
        self.ed_tg_inc_vol = QtWidgets.QCheckBox("include_volatility")
        self.ed_tg_inc_pos = QtWidgets.QCheckBox("include_positions")
        self.ed_tg_inc_levels = QtWidgets.QCheckBox("include_levels")
        self.ed_tg_inc_prob = QtWidgets.QCheckBox("include_probabilities")
        self.ed_tg_inc_timing = QtWidgets.QCheckBox("include_timing")
        self.ed_timeframe = QtWidgets.QComboBox(); self.ed_timeframe.addItems(["1m","3m","5m","15m"])
        self.ed_theme = QtWidgets.QComboBox(); self.ed_theme.addItems(["light","dark_green"])

        form.addRow("confidence_min (мин. уверенность)", self.ed_conf)
        form.addRow("rr_min (мин. RR)", self.ed_rr)
        form.addRow("uncertainty_max (макс. неопределенность)", self.ed_unc)
        form.addRow("auc_override_confidence_min", self.ed_auc_over)
        form.addRow("high_confidence_min", self.ed_high_conf)
        form.addRow("spread_rank_max", self.ed_spread_rank)
        form.addRow("latency_ms_max", self.ed_latency_max)
        form.addRow("ensemble.gray_zone_lo", self.ed_gray_lo)
        form.addRow("ensemble.gray_zone_hi", self.ed_gray_hi)
        form.addRow("telegram.enabled", self.ed_tg_enabled)
        form.addRow("telegram.bot_token", self.ed_tg_token)
        form.addRow("telegram.chat_id", self.ed_tg_chat)
        form.addRow("telegram.send_signal", self.ed_tg_signal)
        form.addRow("telegram.send_trade_open", self.ed_tg_trade_open)
        form.addRow("telegram.send_trade_closed", self.ed_tg_trade_closed)
        form.addRow("telegram.send_on_gate_fail", self.ed_tg_gate_fail)
        form.addRow("telegram.min_interval_sec", self.ed_tg_min_interval)
        form.addRow("telegram.include_volatility", self.ed_tg_inc_vol)
        form.addRow("telegram.include_positions", self.ed_tg_inc_pos)
        form.addRow("telegram.include_levels", self.ed_tg_inc_levels)
        form.addRow("telegram.include_probabilities", self.ed_tg_inc_prob)
        form.addRow("telegram.include_timing", self.ed_tg_inc_timing)
        form.addRow("runtime.timeframe", self.ed_timeframe)
        form.addRow("ui.theme", self.ed_theme)
        lay.addLayout(form)

        row=QHBoxLayout()
        b_load=QPushButton("Загрузить из config")
        b_save=QPushButton("Сохранить config")
        b_test=QPushButton("Тест Telegram")
        b_load.clicked.connect(self._settings_load)
        b_save.clicked.connect(self._settings_save)
        b_test.clicked.connect(lambda: write_cmd(self.cfg, "send_test_telegram"))
        row.addWidget(b_load); row.addWidget(b_save); row.addWidget(b_test); row.addStretch(1)
        lay.addLayout(row)

        self.lbl_settings = QLabel("Статус: —")
        lay.addWidget(self.lbl_settings)
        self.tabs.addTab(w, "Настройки")
        self._settings_load()
        self._last_cmd_seen_ts = 0

    def _settings_load(self):
        try:
            c = Config.load(str(self.cfg_path)).raw
            gp = ((c.get("gate", {}) or {}).get("profiles", {}) or {}).get((c.get("gate", {}) or {}).get("profile", "scalp"), {})
            self.ed_conf.setValue(float(gp.get("confidence_min", 0.52)))
            self.ed_rr.setValue(float(gp.get("rr_min", 0.9)))
            self.ed_unc.setValue(float(gp.get("uncertainty_max", 0.02)))
            self.ed_auc_over.setValue(float((c.get("model", {}) or {}).get("auc_override_confidence_min", 0.70)))
            self.ed_high_conf.setValue(float(gp.get("high_confidence_min", 0.62)))
            self.ed_spread_rank.setValue(float(gp.get("spread_rank_max", 0.95)))
            self.ed_latency_max.setValue(int(gp.get("latency_ms_max", 1200)))
            gray = (((c.get("model", {}) or {}).get("ensemble", {}) or {}).get("gray_zone", [0.45, 0.55]))
            self.ed_gray_lo.setValue(float(gray[0] if isinstance(gray, (list, tuple)) and len(gray) > 0 else 0.45))
            self.ed_gray_hi.setValue(float(gray[1] if isinstance(gray, (list, tuple)) and len(gray) > 1 else 0.55))
            tg = ((c.get("notifications", {}) or {}).get("telegram", {}) or {})
            self.ed_tg_enabled.setChecked(bool(tg.get("enabled", False)))
            self.ed_tg_token.setText(str(tg.get("bot_token", "")))
            self.ed_tg_chat.setText(str(tg.get("chat_id", "")))
            self.ed_tg_signal.setChecked(bool(tg.get("send_signal", True)))
            self.ed_tg_trade_open.setChecked(bool(tg.get("send_trade_open", True)))
            self.ed_tg_trade_closed.setChecked(bool(tg.get("send_trade_closed", True)))
            self.ed_tg_gate_fail.setChecked(bool(tg.get("send_on_gate_fail", False)))
            self.ed_tg_min_interval.setValue(float(tg.get("min_interval_sec", 2.0)))
            self.ed_tg_inc_vol.setChecked(bool(tg.get("include_volatility", True)))
            self.ed_tg_inc_pos.setChecked(bool(tg.get("include_positions", True)))
            self.ed_tg_inc_levels.setChecked(bool(tg.get("include_levels", True)))
            self.ed_tg_inc_prob.setChecked(bool(tg.get("include_probabilities", True)))
            self.ed_tg_inc_timing.setChecked(bool(tg.get("include_timing", True)))
            self.ed_timeframe.setCurrentText(str((c.get("runtime", {}) or {}).get("timeframe", "1m")))
            self.ed_theme.setCurrentText(str((c.get("ui", {}) or {}).get("theme", "light")))
            self.lbl_settings.setText("Статус: настройки загружены")
        except Exception as e:
            self.lbl_settings.setText(f"Статус: ошибка загрузки ({e})")

    def _settings_save(self):
        try:
            with open(self.cfg_path, "r", encoding="utf-8") as f:
                c = yaml.safe_load(f) or {}
            gate = c.setdefault("gate", {})
            prof_name = gate.get("profile", "scalp")
            profiles = gate.setdefault("profiles", {})
            gp = profiles.setdefault(prof_name, {})
            gp["confidence_min"] = float(self.ed_conf.value())
            gp["rr_min"] = float(self.ed_rr.value())
            gp["uncertainty_max"] = float(self.ed_unc.value())
            gp["high_confidence_min"] = float(self.ed_high_conf.value())
            gp["spread_rank_max"] = float(self.ed_spread_rank.value())
            gp["latency_ms_max"] = int(self.ed_latency_max.value())
            c.setdefault("model", {})["auc_override_confidence_min"] = float(self.ed_auc_over.value())
            c.setdefault("model", {}).setdefault("ensemble", {})["gray_zone"] = [float(self.ed_gray_lo.value()), float(self.ed_gray_hi.value())]
            tg = c.setdefault("notifications", {}).setdefault("telegram", {})
            tg["enabled"] = bool(self.ed_tg_enabled.isChecked())
            tg["bot_token"] = self.ed_tg_token.text().strip()
            tg["chat_id"] = self.ed_tg_chat.text().strip()
            tg["send_signal"] = bool(self.ed_tg_signal.isChecked())
            tg["send_trade_open"] = bool(self.ed_tg_trade_open.isChecked())
            tg["send_trade_closed"] = bool(self.ed_tg_trade_closed.isChecked())
            tg["send_on_gate_fail"] = bool(self.ed_tg_gate_fail.isChecked())
            tg["min_interval_sec"] = float(self.ed_tg_min_interval.value())
            tg["include_volatility"] = bool(self.ed_tg_inc_vol.isChecked())
            tg["include_positions"] = bool(self.ed_tg_inc_pos.isChecked())
            tg["include_levels"] = bool(self.ed_tg_inc_levels.isChecked())
            tg["include_probabilities"] = bool(self.ed_tg_inc_prob.isChecked())
            tg["include_timing"] = bool(self.ed_tg_inc_timing.isChecked())
            c.setdefault("runtime", {})["timeframe"] = self.ed_timeframe.currentText()
            c.setdefault("ui", {})["theme"] = self.ed_theme.currentText()
            with open(self.cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(c, f, allow_unicode=True, sort_keys=False)
            write_cmd(self.cfg, "reload_config")
            self.lbl_settings.setText("Статус: config сохранен и применен без перезапуска (тема UI может требовать перезапуск)")
        except Exception as e:
            self.lbl_settings.setText(f"Статус: ошибка сохранения ({e})")

    def _tab_logs(self):
        w=QtWidgets.QWidget(); lay=QtWidgets.QVBoxLayout(w)
        row=QtWidgets.QHBoxLayout(); lay.addLayout(row)
        self.btn_export_csv=QtWidgets.QPushButton("Экспорт CSV")
        self.btn_export_json=QtWidgets.QPushButton("Экспорт JSON")
        row.addWidget(self.btn_export_csv); row.addWidget(self.btn_export_json); row.addStretch(1)
        self.txt_log=QtWidgets.QPlainTextEdit(); self.txt_log.setReadOnly(True); lay.addWidget(self.txt_log)
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_json.clicked.connect(self.export_json)
        self.tabs.addTab(w,"Логи/Экспорт")

    # WS handlers
    def on_connected(self, ok: bool):
        self.ws_connected = ok

    def on_status(self, s: str):
        self.lbl_status.setText("Статус: " + s)

    def on_message(self, msg: Dict[str, Any]):
        self.last_ws_msg_ts = int(time.time()*1000)
        t = msg.get("type")
        if t == "preflight":
            self.preflight_ok = bool(msg.get("ok", False))
            self.preflight_report = msg.get("report", []) or []
        elif t == "symbols":
            self.symbols = msg.get("symbols", []) or []
        elif t == "status":
            self.txt_log.appendPlainText(str(msg.get("status")))

    # SQLite polling (IMPORTANT: tail() returns newest first)
    def poll_sqlite(self):
        tail = self.es.tail(12000)  # newest first (wider window for analytics)
        self.events = list(reversed(tail))  # for tables (oldest->newest)

        for e in tail:
            if e.get("stage") == "PRICES_SNAPSHOT":
                self.prices = (e.get("payload", {}).get("prices") or {})
                break
        for e in tail:
            if e.get("stage") == "POSITIONS_SNAPSHOT":
                self.positions = (e.get("payload", {}).get("positions") or {})
                break
        for e in tail:
            if e.get("stage") == "ACCOUNT":
                p = e.get("payload", {})
                self.account = {"deposit": p.get("deposit"), "cash": p.get("cash"), "equity": p.get("equity"), "open_positions": p.get("open_positions")}
                break
        for e in tail:
            if e.get("stage") == "MODEL_INFERRED":
                mh = (e.get("payload", {}).get("metrics") or {})
                if mh:
                    self.model_health = mh
                break
        for e in tail:
            if e.get("stage") == "TRADE_OUTCOME_STATS":
                self.trade_outcome_stats = (e.get("payload") or {})
                break

    def refresh_ui(self):
        now = int(time.time()*1000)
        age = (now - self.last_ws_msg_ts) if self.last_ws_msg_ts else None
        self.lbl_live.setText(f"Live: {'WS OK' if self.ws_connected else 'WS NO'} | last_msg={age}ms" if age is not None else f"Live: {'WS OK' if self.ws_connected else 'WS NO'}")

        self.lbl_pf.setText("Предпроверка: " + ("OK ✅" if self.preflight_ok else "НЕ ПРОЙДЕНА ❌"))
        self.tbl_pf.set_rows([[r.get("name"), "OK" if r.get("ok") else "FAIL", r.get("details","")] for r in (self.preflight_report or [])])
        self.btn_start.setEnabled(bool(self.preflight_ok))

        dep = self.account.get("deposit"); cash = self.account.get("cash"); eq = self.account.get("equity"); op = self.account.get("open_positions")
        self.kpi_deposit.setText(f"Депозит: ${float(dep):.2f}" if dep is not None else "Депозит: —")
        self.kpi_cash.setText(f"Cash: ${float(cash):.2f}" if cash is not None else "Cash: —")

        total_unr = 0.0
        for sym, p in (self.positions or {}).items():
            side = p.get("side")
            entry = float(p.get("entry", 0.0)); qty = float(p.get("qty", 0.0))
            px = float(self.prices.get(sym, entry))
            total_unr += (px-entry)*qty if side == "LONG" else (entry-px)*qty
        self.kpi_equity.setText(f"Equity: ${float(eq):.2f} (uPnL {total_unr:+.2f})" if eq is not None else "Equity: —")
        self.kpi_open.setText(f"Позиции: {op}" if op is not None else "Позиции: —")

        # positions
        pos_rows=[]
        for sym,p in (self.positions or {}).items():
            side=p.get("side")
            entry=float(p.get("entry",0.0)); qty=float(p.get("qty",0.0))
            price=float(self.prices.get(sym, entry))
            pnl=(price-entry)*qty if side=="LONG" else (entry-price)*qty
            pos_rows.append([sym,"ЛОНГ" if side=="LONG" else "ШОРТ",f"{price:.6f}",f"{entry:.6f}",f"{qty:.6f}",f"{pnl:.2f}",
                             f"{float(p.get('sl',0.0)):.6f}",f"{float(p.get('tp1',0.0)):.6f}",f"{float(p.get('tp2',0.0)):.6f}"])
        self.tbl_positions.set_rows(pos_rows)

        # signals/model/trace + Model-B/MLOps/Levels/Dataset
        sig_rows=[]; model_rows=[]; trace_rows=[]
        modelb_rows=[]; mlops_rows=[]; levels_rows=[]; perf_rows=[]
        modelb_status=None
        dataset_rows=None
        for e in self.events[-10000:]:
            ts=fmt_ts(e.get("ts")); sym=e.get("symbol"); stg=e.get("stage"); lvl=e.get("level"); payload=e.get("payload",{})
            if stg=="SIGNAL":
                gate=payload.get("gate",{}) or {}
                sig_rows.append([ts,sym,payload.get("direction"),payload.get("pred"),payload.get("confidence"),payload.get("p_up_3m"),payload.get("p_up_5m"),payload.get("p_tp_first"),payload.get("p_sl_first"),payload.get("signal_strength"),payload.get("ev"),payload.get("market_regime"),gate.get("decision"),";".join(gate.get("reasons",[]) or [])])
            if stg=="MODEL_INFERRED":
                mh=(payload.get("metrics") or {})
                model_rows.append([ts,sym,payload.get("pred"),payload.get("confidence"),mh.get("auc"),mh.get("samples")])
            if stg=="MODEL_B_INFERRED":
                modelb_rows.append([
                    ts, sym, payload.get("prob_head"), payload.get("prob_backbone"),
                    payload.get("wall_age"), payload.get("wall_touches"), payload.get("wall_dist_bps"),
                    payload.get("decision"), payload.get("thr"), payload.get("status", "OK")
                ])
                levels_rows.append([
                    ts, sym, payload.get("imb", 0.0), payload.get("spread_bps", 0.0),
                    payload.get("wall_age", 0.0), payload.get("wall_touches", 0), payload.get("wall_dist_bps", 0.0)
                ])
            if stg=="MODEL_B_STATUS":
                modelb_status=payload
                dataset_rows=payload.get("dataset_rows", dataset_rows)
            if stg=="SIGNAL":
                perf_rows.append([
                    ts, sym,
                    f"{float(payload.get('p_up_3m',0.0)):.2%}" if payload.get('p_up_3m') is not None else "—",
                    f"{float(payload.get('p_up_5m',0.0)):.2%}" if payload.get('p_up_5m') is not None else "—",
                    f"{float(payload.get('p_tp_first',0.0)):.2%}" if payload.get('p_tp_first') is not None else "—",
                    f"{float(payload.get('p_sl_first',0.0)):.2%}" if payload.get('p_sl_first') is not None else "—",
                    payload.get('signal_strength', "—"),
                    payload.get('ev', "—"),
                    payload.get('market_regime', "—"),
                    payload.get('pred', "—"),
                    payload.get('confidence', "—"),
                ])
            if stg=="MODEL_B_DATASET_APPEND":
                dataset_rows=payload.get("rows", dataset_rows)
            if stg in ("MODEL_B_RETRAIN", "MODEL_B_RETRAIN_SKIPPED"):
                mlops_rows.append([
                    ts,
                    payload.get("auc", "—"),
                    payload.get("pr_auc", "—"),
                    payload.get("acc", "—"),
                    payload.get("samples", payload.get("rows", "—")),
                    payload.get("promoted", False) if stg=="MODEL_B_RETRAIN" else "SKIPPED"
                ])
            trace_rows.append([ts,str(e.get("run_id",""))[:8],sym,stg,lvl,json.dumps(payload,ensure_ascii=False)[:800]])
        self.tbl_signals.set_rows(sig_rows[-350:])
        self.tbl_model.set_rows(model_rows[-350:])
        self.tbl_trace.set_rows(trace_rows[-500:])
        self.tbl_modelb.set_rows(modelb_rows[-350:])
        self.tbl_mlops.set_rows(mlops_rows[-350:])
        self.tbl_levels.set_rows(levels_rows[-350:])

        if modelb_status:
            self.lbl_modelb.setText(
                f"Model-B: строк={int(modelb_status.get('dataset_rows',0))} | pending={int(modelb_status.get('pending',0))} | "
                f"auc={float(modelb_status.get('auc',0.0)):.3f} | acc={float(modelb_status.get('acc',0.0)):.3f}"
            )
            self.pb_modelb.setValue(int(max(0,min(100, round(float(modelb_status.get('training_progress',0.0))*100)))))
            self.lbl_mlops.setText(
                f"MLOps: dataset_rows={int(modelb_status.get('dataset_rows',0))}, training_disabled={bool(modelb_status.get('training_disabled',False))}, "
                f"progress={float(modelb_status.get('training_progress',0.0)):.0%}, reason={modelb_status.get('last_retrain_reason','n/a')}"
            )
            self.pb_auc.setValue(int(max(0,min(100, round(float(modelb_status.get('auc',0.0))*100)))))
        else:
            self.lbl_modelb.setText("Model-B: нет событий MODEL_B_STATUS / MODEL_B_INFERRED")
            self.pb_modelb.setValue(0)
            self.pb_auc.setValue(0)

        ds_path = ""
        for e in reversed(self.events[-12000:]):
            if e.get("stage") == "MODEL_B_STATUS":
                ds_path = str((e.get("payload") or {}).get("dataset_path", ""))
                break
        rows_val = int(dataset_rows or 0)
        pos_rate = float((modelb_status or {}).get("dataset_pos_rate", 0.0) or 0.0)
        self.lbl_ds.setText(f"Dataset: {rows_val} rows | positive_rate={pos_rate:.2%}")
        last_ds_ts = ""
        for e in reversed(self.events[-12000:]):
            if e.get("stage") == "MODEL_B_DATASET_APPEND":
                last_ds_ts = fmt_ts(e.get("ts"))
                break
        self.tbl_ds.set_rows([[rows_val, f"{pos_rate:.2%}", last_ds_ts or fmt_ts(self.events[-1].get("ts") if self.events else 0), ds_path]])

        mh=self.model_health or {}
        self.m_samples.setText(f"Samples: {int(mh.get('samples',0))}")
        self.m_auc.setText(f"AUC: {float(mh.get('auc',0.0)):.3f}")
        self.m_acc.setText(f"Accuracy: {float(mh.get('accuracy',0.0)):.3f}")
        self.m_ver.setText(f"Version: {mh.get('version','')}")

        self.tbl_syms.set_rows([[i+1, s] for i,s in enumerate(self.symbols[:4000])])


        st = self.trade_outcome_stats or {}
        total_closed = int(st.get("total_closed", 0) or 0)
        wins = int(st.get("wins", 0) or 0)
        sl = int(st.get("sl", 0) or 0)
        be = int(st.get("breakeven", 0) or 0)
        loss = int(st.get("loss", 0) or 0)
        win_rate = float(st.get("win_rate", 0.0) or 0.0)
        mb_prog = float((modelb_status or {}).get("training_progress", 0.0) or 0.0)
        mb_reason = str((modelb_status or {}).get("last_retrain_reason", "n/a"))
        self.lbl_perf.setText(f"Сделки: всего={total_closed}, прибыльных={wins}, убыточных={loss}, SL={sl}, BE={be}, win_rate={win_rate:.1%} | Model-B progress={mb_prog:.0%}, reason={mb_reason}")

        prob_rows=[]
        for e in self.events[-12000:]:
            if e.get("stage") == "SHORT_TERM_LEVEL_PROB":
                p=e.get("payload",{})
                prob_rows.append([fmt_ts(e.get("ts")), e.get("symbol"), f"{float(p.get('p_up_3m',0.0)):.2%}", f"{float(p.get('p_up_5m',0.0)):.2%}", "—", "—", "—", "—", "—", f"{float(p.get('ret_3m',0.0)):.4f}", f"{float(p.get('ret_5m',0.0)):.4f}"])
        merged = (prob_rows + perf_rows)[-400:]
        self.tbl_prob.set_rows(merged[-250:])

        # analytics tab (aggregated system statistics)
        signal_total = 0
        signal_pass = 0
        signal_fail = 0
        trade_open_total = 0
        trade_closed_total = 0
        close_sl = 0
        close_tp = 0
        for e in self.events[-12000:]:
            stg = e.get("stage")
            p = e.get("payload") or {}
            if stg == "SIGNAL":
                signal_total += 1
                g = p.get("gate") or {}
                if str(g.get("decision", "")).upper() == "PASS":
                    signal_pass += 1
                else:
                    signal_fail += 1
            elif stg == "TRADE_OPEN":
                if str(p.get("result", "OK")) == "OK":
                    trade_open_total += 1
            elif stg == "TRADE_CLOSED":
                trade_closed_total += 1
                r = str(p.get("reason", ""))
                if r == "SL":
                    close_sl += 1
                elif r in ("TP1", "TP2"):
                    close_tp += 1

        signal_pass_rate = (100.0 * signal_pass / max(1, signal_total))
        trade_win_rate = float(win_rate * 100.0)
        scan_cov = 0.0
        scan_pass = 0.0
        scan_err = 0.0
        scan_rows = []
        for e in self.events[-12000:]:
            if e.get("stage") == "SCAN_EFFICIENCY":
                p = e.get("payload") or {}
                scan_cov = float(p.get("coverage_pct", scan_cov) or 0.0)
                scan_pass = float(p.get("pass_rate_pct", scan_pass) or 0.0)
                scan_err = float(p.get("error_rate_pct", scan_err) or 0.0)
                scan_rows.append([
                    fmt_ts(e.get("ts")),
                    f"{float(p.get('coverage_pct',0.0)):.1f}",
                    f"{float(p.get('pass_rate_pct',0.0)):.1f}",
                    f"{float(p.get('error_rate_pct',0.0)):.1f}",
                    int(p.get("processed",0) or 0),
                    int(p.get("errors",0) or 0),
                ])

        self.kpi_sig_total.setText(f"Сигналы: {signal_total}")
        self.kpi_sig_pass.setText(f"PASS: {signal_pass} ({signal_pass_rate:.1f}%)")
        self.kpi_opened.setText(f"Открыто: {trade_open_total}")
        self.kpi_closed.setText(f"Закрыто: {trade_closed_total}")
        self.kpi_winrate.setText(f"WinRate: {trade_win_rate:.1f}%")
        self.pb_signal_pass.setValue(int(max(0,min(100, round(signal_pass_rate)))))
        self.pb_trade_win.setValue(int(max(0,min(100, round(trade_win_rate)))))
        self.pb_scan_cov.setValue(int(max(0,min(100, round(scan_cov)))))

        self.tbl_stats_models.set_rows([
            ["Model-A samples", int(mh.get("samples", 0) or 0)],
            ["Model-A AUC", f"{float(mh.get('auc',0.0) or 0.0):.3f}"],
            ["Model-A accuracy", f"{float(mh.get('accuracy',0.0) or 0.0):.3f}"],
            ["Model-B dataset rows", int((modelb_status or {}).get("dataset_rows",0) or 0)],
            ["Model-B pos rate", f"{float((modelb_status or {}).get('dataset_pos_rate',0.0) or 0.0):.2%}"],
            ["Model-B progress", f"{float((modelb_status or {}).get('training_progress',0.0) or 0.0):.0%}"],
            ["Model-B retrain reason", str((modelb_status or {}).get("last_retrain_reason", "n/a"))],
        ])
        self.tbl_stats_trades.set_rows([
            ["Signals total", signal_total],
            ["Signals PASS", signal_pass],
            ["Signals FAIL", signal_fail],
            ["Trades open", trade_open_total],
            ["Trades closed", trade_closed_total],
            ["Closed by TP", close_tp],
            ["Closed by SL", close_sl],
            ["Win rate", f"{trade_win_rate:.1f}%"],
            ["Scanner pass rate", f"{scan_pass:.1f}%"],
            ["Scanner error rate", f"{scan_err:.1f}%"],
        ])
        self.tbl_stats_scan.set_rows(scan_rows[-120:])
        scan_pass_hist = [float(r[2]) for r in scan_rows[-40:] if isinstance(r[2], str)]
        signal_hist = []
        for e in self.events[-12000:]:
            if e.get("stage") == "SIGNAL":
                g = (e.get("payload") or {}).get("gate") or {}
                signal_hist.append(100.0 if str(g.get("decision","")) == "PASS" else 0.0)
        self.lbl_graph_scan.setText(f"Scan trend: {_sparkline(scan_pass_hist)}")
        self.lbl_graph_signal.setText(f"Signal PASS trend: {_sparkline(signal_hist)}")

        # last command feedback (telegram test / reload config)
        for e in reversed(self.events[-500:]):
            if e.get("stage") == "CMD":
                ts = int(e.get("ts") or 0)
                if ts <= int(getattr(self, "_last_cmd_seen_ts", 0) or 0):
                    break
                p = e.get("payload") or {}
                cmd = str(p.get("cmd", ""))
                if cmd in ("send_test_telegram", "reload_config"):
                    self._last_cmd_seen_ts = ts
                    ok = bool(p.get("ok", False))
                    status = p.get("status", "")
                    reason = p.get("reason", "")
                    msg = f"Статус: CMD {cmd}: {'OK' if ok or status in ('sent','applied') else 'FAIL'} {reason}".strip()
                    self.lbl_settings.setText(msg)
                break

        # closed trades
        closed_rows=[]
        for e in self.events[-4000:]:
            if e.get("stage")=="TRADE_CLOSED":
                p=e.get("payload",{})
                try:
                    closed_rows.append([fmt_ts(p.get("close_ts")), p.get("symbol"), p.get("side"),
                                        f"{float(p.get('entry',0.0)):.6f}", f"{float(p.get('exit',0.0)):.6f}",
                                        f"{float(p.get('pnl',0.0)):.2f}", p.get("reason")])
                except Exception:
                    pass
        self.tbl_closed.set_rows(closed_rows[-700:])

    def _refresh_monitor(self):
        # Show last PRICE_TICK + ERROR events
        buf=[]
        last_pt=None
        for e in reversed(self.events):
            if e.get('stage')=='PRICE_TICK':
                last_pt=e; break
        if last_pt:
            age_ms=int(time.time()*1000)-int(last_pt.get('ts') or 0)
            buf.append(f"LAST PRICE_TICK age_ms={age_ms} payload={json.dumps(last_pt.get('payload',{}), ensure_ascii=False)}")
            buf.append('')

        for e in self.events[-500:]:
            if e.get("stage") in ("PRICE_TICK","ERROR","TRADE_OPEN","TRADE_CLOSED","MODEL_B_STATUS","MODEL_B_RETRAIN","MODEL_B_RETRAIN_SKIPPED","POSITION_UPDATE","TRADE_OUTCOME_STATS","SHORT_TERM_LEVEL_PROB"):
                buf.append(f"{fmt_ts(e.get('ts'))} {e.get('stage')} {e.get('symbol')} {json.dumps(e.get('payload',{}), ensure_ascii=False)[:500]}")
        self.txt_mon.setPlainText("\n".join(buf[-250:]) if buf else "Пока нет событий. Запусти hub и нажми Старт сканера.")

    def export_csv(self):
        out_dir = ROOT / "exports"
        out_dir.mkdir(exist_ok=True)
        fp = out_dir / f"events_{int(time.time())}.csv"
        with fp.open("w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["ts","run_id","symbol","stage","level","payload"])
            for e in self.events:
                wr.writerow([e.get("ts"), e.get("run_id"), e.get("symbol"), e.get("stage"), e.get("level"), json.dumps(e.get("payload",{}), ensure_ascii=False)])
        self.txt_log.appendPlainText(f"exported {fp}")

    def export_json(self):
        out_dir = ROOT / "exports"
        out_dir.mkdir(exist_ok=True)
        fp = out_dir / f"events_{int(time.time())}.json"
        fp.write_text(json.dumps(self.events, ensure_ascii=False, indent=2), encoding="utf-8")
        self.txt_log.appendPlainText(f"exported {fp}")

    def _tab_modelb(self):
        w=QWidget(); lay=QVBoxLayout(w)
        info=QLabel("Model-B (вторая модель): Prob(head)=вероятность от обучаемой головы, Prob(backbone)=вероятность backbone, Decision=пороговый фильтр сигнала.")
        info.setWordWrap(True); lay.addWidget(info)
        self.lbl_modelb=QLabel("Model-B: нет данных"); lay.addWidget(self.lbl_modelb)
        self.pb_modelb=QtWidgets.QProgressBar(); self.pb_modelb.setRange(0,100); self.pb_modelb.setValue(0); self.pb_modelb.setFormat("Готовность Model-B: %p%")
        lay.addWidget(self.pb_modelb)
        btns=QHBoxLayout()
        b1=QPushButton("Force retrain"); b1.clicked.connect(lambda: write_cmd(self.cfg, "force_model_b_retrain"))
        b2=QPushButton("Disable training"); b2.clicked.connect(lambda: write_cmd(self.cfg, "disable_model_b_training"))
        b3=QPushButton("Enable training"); b3.clicked.connect(lambda: write_cmd(self.cfg, "enable_model_b_training"))
        b4=QPushButton("Rollback"); b4.clicked.connect(lambda: write_cmd(self.cfg, "rollback_model_b"))
        for b in (b1,b2,b3,b4): btns.addWidget(b)
        lay.addLayout(btns)
        self.tbl_modelb=SimpleTable(["Time","Symbol","Prob(head)","Prob(backbone)","wall_age","touches","dist_bps","Decision","thr","Status"])
        self._set_header_tips(self.tbl_modelb, {
            "Prob(head)": "Вероятность от обучаемой головы Model-B.",
            "Prob(backbone)": "Вероятность от backbone (базовая).",
            "wall_age": "Возраст сильной стенки стакана (сек).",
            "touches": "Количество касаний стенки.",
            "dist_bps": "Дистанция до стенки в bps.",
            "thr": "Порог вероятности для PASS.",
        })
        lay.addWidget(self.tbl_modelb)
        self.tabs.addTab(w, "Model-B")

    def _tab_dataset(self):
        w=QWidget(); lay=QVBoxLayout(w)
        info=QLabel("Dataset: строки обучающих примеров для Model-B. Label1% показывает долю позитивного класса (рост после горизонта).")
        info.setWordWrap(True); lay.addWidget(info)
        self.lbl_ds=QLabel("Dataset: 0 rows"); lay.addWidget(self.lbl_ds)
        self.tbl_ds=SimpleTable(["Rows","Label1%","Last ts","Path"])
        lay.addWidget(self.tbl_ds)
        self.tabs.addTab(w, "Dataset")

    def _tab_mlops(self):
        w=QWidget(); lay=QVBoxLayout(w)
        info=QLabel("MLOps: метрики переобучения Model-B. AUC/PR_AUC/ACC — качество, Samples — объём train/val, Promoted — принята ли новая версия.")
        info.setWordWrap(True); lay.addWidget(info)
        self.lbl_mlops=QLabel("MLOps: нет retrain"); lay.addWidget(self.lbl_mlops)
        self.pb_auc=QtWidgets.QProgressBar(); self.pb_auc.setRange(0,100); self.pb_auc.setValue(0); self.pb_auc.setFormat("AUC: %p%")
        lay.addWidget(self.pb_auc)
        self.tbl_mlops=SimpleTable(["Time","AUC","PR_AUC","ACC","Samples","Promoted"])
        self._set_header_tips(self.tbl_mlops, {
            "PR_AUC": "Метрика precision-recall (устойчива к дисбалансу классов).",
            "ACC": "Точность классификации.",
            "Promoted": "Новая версия модели принята в прод.",
        })
        lay.addWidget(self.tbl_mlops)
        self.tabs.addTab(w, "MLOps")

    def _tab_levels(self):
        w=QWidget(); lay=QVBoxLayout(w)
        info=QLabel("Levels/OB: параметры стакана. imb=дисбаланс объёма BID/ASK, spread_bps=спред в bps, wall_age=возраст стенки, touches=касания, dist_bps=дистанция до стенки.")
        info.setWordWrap(True); lay.addWidget(info)
        self.tbl_levels=SimpleTable(["Time","Symbol","imb","spread_bps","wall_age","touches","dist_bps"])
        lay.addWidget(self.tbl_levels)
        self.tabs.addTab(w, "Levels/OB")


def main():
    cfg_path = ROOT / "config.yaml"
    cfg = Config.load(str(cfg_path)).raw
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(cfg, cfg_path)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
