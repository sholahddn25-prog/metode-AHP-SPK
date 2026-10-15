"""
tests/test_engine.py
====================
Regression tests for the AHP engine against the ground-truth numbers of the
MPV Family Fleet case study.

Run:
    python -m pytest tests/ -q
    # or without pytest:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import (  # noqa: E402
    AHPValidationError,
    calculate_consistency,
    calculate_eigenvector,
    default_model,
    run_synthesis,
    synthesize_global,
    validate_matrix,
)

MODEL = default_model()


class TestCriteriaMatrix(unittest.TestCase):
    """Ground truth: matriks kriteria berordo 3x3, urutan [K, S, B]."""

    def setUp(self) -> None:
        self.matrix = MODEL["criteriaMatrix"]
        self.weights = calculate_eigenvector(self.matrix, method="column_sum")

    def test_column_sums(self) -> None:
        import numpy as np

        sums = np.asarray(self.matrix).sum(axis=0)
        self.assertAlmostEqual(sums[0], 1.75, places=6)
        self.assertAlmostEqual(sums[1], 10 / 3, places=6)
        self.assertAlmostEqual(sums[2], 8.0, places=6)

    def test_weights_column_sum(self) -> None:
        w_K, w_S, w_B = self.weights
        self.assertAlmostEqual(w_K, 0.5571, places=4)   # 55.71%
        self.assertAlmostEqual(w_S, 0.3202, places=4)   # 32.02%
        self.assertAlmostEqual(w_B, 0.1226, places=4)   # 12.26%
        self.assertAlmostEqual(float(self.weights.sum()), 1.0, places=9)

    def test_weights_close_to_principal_eigenvector(self) -> None:
        import numpy as np

        arr = np.asarray(self.matrix)
        values, vectors = np.linalg.eig(arr)
        principal = np.abs(vectors[:, values.argmax()].real)
        principal /= principal.sum()
        # For a consistent matrix both methods coincide exactly; for this
        # slightly inconsistent matrix they agree to within ~1.3e-3.
        self.assertTrue(np.allclose(self.weights, principal, atol=2e-3))

    def test_consistency_ground_truth(self) -> None:
        stats = calculate_consistency(self.matrix, self.weights)
        self.assertAlmostEqual(stats["lambdaMax"], 3.0183, places=4)
        self.assertAlmostEqual(stats["ci"], 0.0092, places=4)
        self.assertAlmostEqual(stats["ri"], 0.58, places=6)
        self.assertAlmostEqual(stats["cr"], 0.0158, places=4)
        self.assertTrue(stats["isConsistent"])


class TestAlternativeMatrices(unittest.TestCase):
    """Ground truth: matriks alternatif 4x4 per kriteria."""

    def test_all_orders_are_four(self) -> None:
        for block in MODEL["alternativeMatrices"].values():
            self.assertEqual(len(block), 4)
            for row in block:
                self.assertEqual(len(row), 4)

    def test_local_weights_and_cr(self) -> None:
        expected = {
            # criterion: (weights, cr)
            "Style": ([0.1332, 0.1078, 0.3903, 0.3687], 0.1205),
            "Keandalan": ([0.1243, 0.0772, 0.5256, 0.2728], 0.0426),
            "Konsumsi BB": ([0.3052, 0.2165, 0.3977, 0.0806], 0.0380),
        }
        for criterion, (w_expected, cr_expected) in expected.items():
            with self.subTest(criterion=criterion):
                matrix = MODEL["alternativeMatrices"][criterion]
                w = calculate_eigenvector(matrix, method="column_sum")
                stats = calculate_consistency(matrix, w)
                for got, want in zip(w, w_expected):
                    self.assertAlmostEqual(got, want, places=4)
                self.assertAlmostEqual(stats["cr"], cr_expected, places=4)


class TestSynthesis(unittest.TestCase):
    """Global ranking of the case study."""

    def test_global_scores_and_winner(self) -> None:
        report = run_synthesis(
            criteria_labels=MODEL["criteriaLabels"],
            criteria_matrix=MODEL["criteriaMatrix"],
            alternatives=MODEL["alternatives"],
            alternative_matrices=MODEL["alternativeMatrices"],
        )
        scores = dict(zip(report["alternatives"], report["globalScores"]))
        self.assertAlmostEqual(scores["Avanza"], 0.1493, places=4)
        self.assertAlmostEqual(scores["Xenia"], 0.1041, places=4)
        self.assertAlmostEqual(scores["Ertiga"], 0.4666, places=4)
        self.assertAlmostEqual(scores["Grand Livina"], 0.2800, places=4)
        self.assertEqual(report["winner"], "Ertiga")
        self.assertAlmostEqual(sum(report["globalScores"]), 1.0, places=9)

    def test_ranking_is_descending(self) -> None:
        report = run_synthesis(
            criteria_labels=MODEL["criteriaLabels"],
            criteria_matrix=MODEL["criteriaMatrix"],
            alternatives=MODEL["alternatives"],
            alternative_matrices=MODEL["alternativeMatrices"],
        )
        scores = [row["score"] for row in report["ranking"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_geometric_mean_method(self) -> None:
        report = run_synthesis(
            criteria_labels=MODEL["criteriaLabels"],
            criteria_matrix=MODEL["criteriaMatrix"],
            alternatives=MODEL["alternatives"],
            alternative_matrices=MODEL["alternativeMatrices"],
            method="geometric_mean",
        )
        self.assertEqual(report["winner"], "Ertiga")
        self.assertAlmostEqual(report["criteria"]["weights"][0], 0.5584, places=4)

    def test_contribution_sum_matches_score(self) -> None:
        report = run_synthesis(
            criteria_labels=MODEL["criteriaLabels"],
            criteria_matrix=MODEL["criteriaMatrix"],
            alternatives=MODEL["alternatives"],
            alternative_matrices=MODEL["alternativeMatrices"],
        )
        for row in report["ranking"]:
            self.assertAlmostEqual(
                sum(row["contributions"].values()), row["score"], places=12
            )


class TestValidation(unittest.TestCase):
    """Saaty axioms are enforced."""

    def test_rejects_non_reciprocal(self) -> None:
        with self.assertRaises(AHPValidationError):
            validate_matrix([[1, 3], [2, 1]])

    def test_rejects_non_square(self) -> None:
        with self.assertRaises(AHPValidationError):
            validate_matrix([[1, 2, 3], [1 / 2, 1, 2]])

    def test_rejects_zero_entries(self) -> None:
        with self.assertRaises(AHPValidationError):
            validate_matrix([[1, 0], [0, 1]])

    def test_rejects_bad_diagonal(self) -> None:
        with self.assertRaises(AHPValidationError):
            validate_matrix([[2, 1], [1, 1]])

    def test_synthesize_requires_all_criteria(self) -> None:
        with self.assertRaises(AHPValidationError):
            synthesize_global([0.5, 0.5], {"Only": [1.0]})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
