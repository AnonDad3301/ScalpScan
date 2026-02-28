from packages.modernization import PatternSimilarityEngine, ScenarioSimulationEngine


def test_similarity_summary_has_probabilities():
    eng = PatternSimilarityEngine(top_k=3)
    cur = [0.1, 0.2, 0.3]
    hist = [[0.1, 0.2, 0.31], [-0.2, -0.1, 0.0], [0.1, 0.22, 0.28]]
    outcomes = [
        {"ret_1m": 0.001, "ret_2m": 0.002, "ret_4m": 0.003},
        {"ret_1m": -0.001, "ret_2m": -0.002, "ret_4m": -0.003},
        {"ret_1m": 0.0, "ret_2m": 0.0005, "ret_4m": 0.0001},
    ]
    top = eng.find_topk(cur, hist, outcomes)
    summary = eng.summarize(top)
    assert 0.0 <= summary["p_up_4m"] <= 1.0
    assert 0.0 <= summary["p_down_4m"] <= 1.0


def test_simulation_engine_returns_scenarios():
    sim = ScenarioSimulationEngine()
    out = sim.simulate(
        {"p_up_4m": 0.6, "p_down_4m": 0.2, "avg_ret_4m": 0.003},
        {"ret": 0.004, "uncertainty": 0.001},
    )
    assert out["direction"] in {"bullish", "bearish", "neutral"}
    assert "scenarios" in out
    assert "confidence" in out
