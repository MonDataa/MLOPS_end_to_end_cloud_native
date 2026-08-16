from __future__ import annotations

import unittest

import pandas as pd

from apps.monitoring.drift import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    evaluate_feature_drift,
    psi_for_categorical,
    psi_for_numeric,
    sanitize_metric_name,
)


def make_frame(rows: int, *, shift: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(index=range(rows))
    for column in NUMERIC_FEATURES:
        frame[column] = 10.0
    for column in CATEGORICAL_FEATURES:
        frame[column] = 'A'

    if shift:
        frame.loc[frame.index[: rows // 2], NUMERIC_FEATURES[0]] = 99.0
        frame.loc[frame.index[: rows // 2], CATEGORICAL_FEATURES[0]] = 'B'

    return frame[FEATURE_COLUMNS]


class DriftTests(unittest.TestCase):
    def test_sanitize_metric_name(self) -> None:
        self.assertEqual(sanitize_metric_name('loan amount%'), 'loan_amount_')

    def test_numeric_psi_is_zero_for_identical_frames(self) -> None:
        reference = make_frame(100)
        current = make_frame(100)
        self.assertAlmostEqual(psi_for_numeric(reference[NUMERIC_FEATURES[0]], current[NUMERIC_FEATURES[0]]), 0.0)

    def test_categorical_psi_increases_after_shift(self) -> None:
        reference = make_frame(100)
        current = make_frame(100, shift=True)
        self.assertGreater(psi_for_categorical(reference[CATEGORICAL_FEATURES[0]], current[CATEGORICAL_FEATURES[0]]), 0.0)

    def test_evaluate_feature_drift_flags_shifted_features(self) -> None:
        reference = make_frame(100)
        current = make_frame(100, shift=True)
        feature_report, metrics = evaluate_feature_drift(reference, current)

        self.assertIn(NUMERIC_FEATURES[0], feature_report)
        self.assertTrue(feature_report[NUMERIC_FEATURES[0]]['drifted'])
        self.assertGreater(metrics['drift_max_psi'], 0.0)


if __name__ == '__main__':
    unittest.main()
