import unittest

import numpy as np
import pandas as pd

from mean_reversion import StrategyConfig, add_indicators, backtest, build_positions


class StrategyTests(unittest.TestCase):
    def test_zscore_uses_only_current_and_past_values(self):
        index = pd.date_range("2024-01-01", periods=30)
        original = pd.Series(np.arange(100.0, 130.0), index=index)
        changed = original.copy()
        changed.iloc[-1] = 1000
        first = add_indicators(original, 5)
        second = add_indicators(changed, 5)
        pd.testing.assert_series_equal(first["return_z"].iloc[:-1], second["return_z"].iloc[:-1])

    def test_position_rules(self):
        z = pd.Series([np.nan, -2.1, -1.0, -0.4, 2.2, 0.4])
        expected = pd.Series([0, 1, 1, 0, -1, 0], name="signal_position", dtype=int)
        pd.testing.assert_series_equal(build_positions(z, 2.0, 0.5, True), expected)

    def test_execution_is_delayed_one_bar(self):
        close = pd.Series([100, 100, 100, 100, 100, 80, 81], dtype=float)
        results = backtest(close, StrategyConfig(window=3, entry_z=1.0, exit_z=0.2))
        entries = results.index[results["signal_position"] != 0]
        self.assertTrue(len(entries) > 0)
        first = entries[0]
        self.assertEqual(results.loc[first, "position"], 0)


if __name__ == "__main__":
    unittest.main()

