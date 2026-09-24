import unittest
import numpy as np
import pandas as pd
from execution_sim.core import Assumptions, simulate, volume_profile


def day(date, volume=100):
    t = pd.date_range(f"{date} 09:30", periods=390, freq="min", tz="America/New_York")
    return pd.DataFrame({"date": [t[0].date()] * 390, "minute": np.arange(390),
                         "open": [100.0] * 390, "close": [100.0] * 390,
                         "volume": [volume] * 390})


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.day = day("2026-03-24")
        self.profile = volume_profile([day("2026-03-23")])

    def test_flat_prices_without_costs(self):
        assumptions = Assumptions(0, 0)
        for method in ("TWAP", "VWAP", "POV"):
            r = simulate(self.day, method, 1000, .10, self.profile, assumptions)
            self.assertAlmostEqual(r["implementation_shortfall_bps"], 0)
            self.assertAlmostEqual(r["vwap_slippage_bps"], 0)
            self.assertAlmostEqual(r["completion_rate"], 1)

    def test_cap_and_unfilled_order(self):
        r = simulate(self.day, "TWAP", 10000, .05, self.profile)
        self.assertLessEqual(r["filled_shares"], 390 * 100 * .05 + 1e-6)
        self.assertGreater(r["completion_rate"], 0)
        self.assertLess(r["completion_rate"], 1)

    def test_training_profile_normalized(self):
        self.assertAlmostEqual(self.profile.sum(), 1)
        self.assertEqual(len(self.profile), 390)


if __name__ == "__main__":
    unittest.main()

class CostSensitivityTests(unittest.TestCase):
    def test_higher_cost_assumptions_increase_shortfall(self):
        example = day("2026-03-24")
        curve = volume_profile([day("2026-03-23")])
        low = simulate(example, "VWAP", 1000, .10, curve, Assumptions(1, 5))
        high = simulate(example, "VWAP", 1000, .10, curve, Assumptions(4, 20))
        self.assertGreater(high["implementation_shortfall_bps"], low["implementation_shortfall_bps"])
