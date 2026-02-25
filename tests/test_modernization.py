import unittest

import numpy as np

from packages.modernization import MarketPatternScanner, MultiHorizonForecaster, MarketRegimeDetector


class ModernizationTests(unittest.TestCase):
    def test_pattern_scanner_outputs(self):
        scanner = MarketPatternScanner()
        ohlcv = []
        price = 100.0
        for i in range(60):
            price += 0.1
            ohlcv.append([i * 60000, price - 0.3, price + 0.5, price - 0.5, price, 100 + i])
        out = scanner.scan(ohlcv)
        self.assertGreater(out.resistance, out.support)
        self.assertGreater(out.volume_profile_poc, 0.0)

    def test_forecaster_multi_horizon(self):
        forecaster = MultiHorizonForecaster(horizons=(1, 3, 5))
        closes = np.linspace(100, 110, 80)
        pred = forecaster.predict(closes)
        self.assertEqual(set(pred.keys()), {"m1", "m3", "m5"})
        self.assertTrue(all("uncertainty" in v for v in pred.values()))

    def test_regime_detector(self):
        detector = MarketRegimeDetector(vol_threshold=0.001, trend_threshold=0.2)
        closes = np.linspace(100, 140, 120)
        regime = detector.detect(closes)
        self.assertIn(regime.name, {"trend", "volatile"})


if __name__ == "__main__":
    unittest.main()
