from __future__ import annotations
from typing import Dict, Any, Tuple

CATALOG: Dict[str, Tuple[str, str]] = {
    "UNIVERSE_SELECTED": ("Выбран список инструментов", "Universe selected"),
    "DATA_FETCHED": ("Данные загружены (OHLCV/стакан)", "Market data fetched (OHLCV/orderbook)"),
    "FEATURES_SNAPSHOT": ("Снимок фич", "Features snapshot"),
    "SETUP_EVENT": ("Событие сетапа", "Setup event"),
    "MODEL_INFERRED": ("Модель A: инференс", "Model A: inference"),
    "GATE_DECISION": ("Решение гейта", "Gate decision"),
    "SIGNAL": ("Сигнал", "Signal"),
    "ORDER_NEW": ("Создан ордер", "Order created"),
    "ORDER_FILLED": ("Ордер исполнен", "Order filled"),
    "POSITION_OPEN": ("Позиция открыта", "Position opened"),
    "POSITION_UPDATE": ("Обновление позиции", "Position update"),
    "EXIT_ACTION": ("Действие выхода", "Exit action"),
    "TRADE_CLOSED": ("Сделка закрыта", "Trade closed"),
    "ACCOUNT": ("Снимок счёта", "Account snapshot"),
    "SHORT_TERM_LEVEL_PROB": ("Вероятности 3-5м по уровню", "3-5m level probabilities"),
    "TRADE_OUTCOME_STATS": ("Статистика исходов сделок", "Trade outcome stats"),
    "ACCOUNT_MISMATCH": ("Несоответствие учёта", "Accounting mismatch"),
    "MODEL_B_LOAD_START": ("Model-B: загрузка начата", "Model-B: load started"),
    "MODEL_B_LOAD_END": ("Model-B: загрузка завершена", "Model-B: load finished"),
    "MODEL_B_INFERRED": ("Model-B: инференс", "Model-B: inference"),
    "MODEL_B_STATUS": ("Model-B: статус", "Model-B: status"),
    "MODEL_B_RETRAIN": ("Model-B: переобучение", "Model-B: retrain"),
    "MODEL_B_TRAINING_STATUS": ("Model-B: статус обучения", "Model-B: training status"),
    "MODEL_B_RETRAIN_SKIPPED": ("Model-B: переобучение пропущено", "Model-B: retrain skipped"),
    "MODEL_B_DATASET_APPEND": ("Model-B: датасет пополнен", "Model-B: dataset appended"),
    "MODEL_B_PROMOTED": ("Model-B: промоут", "Model-B: promoted"),
    "MODEL_B_ROLLBACK": ("Model-B: откат", "Model-B: rollback"),
    "CMD": ("Команда UI", "UI command"),
    "CMD_ERROR": ("Ошибка команды UI", "UI command error"),
    "ERROR": ("Ошибка", "Error"),
}

def explain(stage: str) -> Dict[str, Any]:
    ru, en = CATALOG.get(stage, ("Событие", "Event"))
    return {"msg_ru": ru, "msg_en": en}
