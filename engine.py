"""
engine.py
=========
AHP (Analytic Hierarchy Process) calculation engine - Thomas L. Saaty.

Pure NumPy, framework agnostic, fully type documented. This module is the
mathematical heart of the AutoAHP Intelligent Decision Support System and is
consumed by ``main.py`` (FastAPI) as well as by the regression tests.

Axioms enforced for every pairwise comparison matrix ``A`` (n x n):

    A_ji = 1 / A_ij      (reciprocal property)
    A_ii = 1             (unit diagonal)
    a_ij > 0             (positive comparisons, Saaty scale 1/9 .. 9)

Public API
----------
    validate_matrix          -> numpy array, raises AHPValidationError
    calculate_eigenvector    -> priority vector w (sums to 1)
    calculate_consistency    -> lambda_max, CI, CR, RI, is_consistent
    synthesize_global        -> global scores + descending ranking
    analyze_matrix           -> single-matrix full report (incl. proof steps)
    run_synthesis            -> full hierarchy report (criteria + alternatives)
    default_model            -> the ground-truth MPV Family Fleet case study
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Saaty Random Index (average random consistency index of a random matrix).
RI_TABLE: Dict[int, float] = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}

#: Default consistency ratio threshold (Saaty: CR < 0.10 is acceptable).
CR_THRESHOLD: float = 0.10

#: Supported priority calculation methods.
METHODS: Tuple[str, ...] = ("column_sum", "geometric_mean")

#: Saaty fundamental scale reference (used by the documentation modal).
SAATY_SCALE: List[Dict[str, Any]] = [
    {"value": 1, "label": "Equal importance", "description": "Two activities contribute equally to the objective."},
    {"value": 2, "label": "Weak", "description": "Experience and judgment slightly favour one over another."},
    {"value": 3, "label": "Moderate importance", "description": "Experience and judgment moderately favour one over another."},
    {"value": 4, "label": "Moderate plus", "description": "Experience and judgement strongly favour one over another."},
    {"value": 5, "label": "Strong importance", "description": "One activity is strongly favoured over another; evidence is very favouring."},
    {"value": 6, "label": "Strong plus", "description": "A very strong dominance is demonstrated in practice."},
    {"value": 7, "label": "Very strong importance", "description": "One activity is very strongly favoured; dominance is demonstrated."},
    {"value": 8, "label": "Very, very strong", "description": "The dominance of one activity over another is affirmed."},
    {"value": 9, "label": "Extreme importance", "description": "The evidence favouring one activity over another is of the highest possible order."},
    {"value": 10, "label": "Reciprocal", "description": "If a_ij = k then a_ji = 1/k (always true in this engine)."},
]


class AHPValidationError(ValueError):
    """Raised when a pairwise comparison matrix violates the Saaty axioms."""


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_matrix(
    matrix: Sequence[Sequence[float]],
    labels: Optional[Sequence[str]] = None,
    name: str = "matrix",
) -> NDArray[np.float64]:
    """
    Validate a pairwise comparison matrix and return it as a float64 array.

    Rules (all Saaty axioms):
        * square shape (n x n), n >= 1
        * every entry finite and strictly positive
        * diagonal entries equal 1
        * a_ji == 1 / a_ij for every pair (reciprocal)
        * labels (if given) must match the matrix order

    Raises:
        AHPValidationError: with a human readable Indonesian/English reason.
    """
    try:
        arr = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AHPValidationError(f"{name}: matrix must contain only numbers ({exc}).") from exc

    if arr.ndim != 2:
        raise AHPValidationError(f"{name}: matrix must be 2-dimensional, got {arr.ndim} dimensions.")
    if arr.shape[0] != arr.shape[1]:
        raise AHPValidationError(f"{name}: matrix must be square, got {arr.shape[0]}x{arr.shape[1]}.")
    if arr.shape[0] < 1:
        raise AHPValidationError(f"{name}: matrix must contain at least one row.")
    if not np.all(np.isfinite(arr)):
        raise AHPValidationError(f"{name}: matrix contains NaN or infinite values.")
    if np.any(arr <= 0):
        raise AHPValidationError(
            f"{name}: every comparison must be > 0 (Saaty scale runs 1/9 .. 9)."
        )

    n = arr.shape[0]
    if not np.allclose(np.diag(arr), 1.0, rtol=1e-6, atol=1e-9):
        raise AHPValidationError(f"{name}: the main diagonal must be exactly 1 (a_ii = 1).")

    # Reciprocal property: A * A == 1 element-wise (A_ji = 1 / A_ij).
    if not np.allclose(arr * arr.T, 1.0, rtol=1e-6, atol=1e-9):
        i, j = np.argwhere(~np.isclose(arr * arr.T, 1.0, rtol=1e-6, atol=1e-9))[0]
        raise AHPValidationError(
            f"{name}: reciprocal property violated at position ({i}, {j}): "
            f"a_ij={arr[i, j]:.6g} but 1/a_ji={1.0 / arr[j, i]:.6g}."
        )

    if labels is not None and len(labels) != n:
        raise AHPValidationError(
            f"{name}: {len(labels)} labels supplied for an {n}x{n} matrix."
        )

    return arr


# --------------------------------------------------------------------------- #
# Priority vector
# --------------------------------------------------------------------------- #

def calculate_eigenvector(
    matrix: Sequence[Sequence[float]],
    method: str = "column_sum",
) -> NDArray[np.float64]:
    """
    Calculate the priority (eigenvector) vector ``w`` of a pairwise matrix.

    Args:
        matrix: square reciprocal matrix (n x n).
        method: ``"column_sum"``      -> normalise each column, average the rows
                 ``"geometric_mean"``  -> nth root of each row product, normalise
                                          (Saaty / Arbel-Crawford approximation,
                                           numerically equal to the principal
                                           eigenvector for consistent matrices).

    Returns:
        1-D float64 array of length n whose entries sum to 1.
    """
    arr = validate_matrix(matrix)

    if method not in METHODS:
        raise AHPValidationError(f"Unknown method '{method}'. Use one of {METHODS}.")

    n = arr.shape[0]

    if method == "column_sum":
        col_sums = arr.sum(axis=0)
        normalized = arr / col_sums            # every column now sums to 1
        weights = normalized.mean(axis=1)      # average across each row
    else:
        row_products = np.prod(arr, axis=1)
        nth_roots = np.power(row_products, 1.0 / n)
        weights = nth_roots / nth_roots.sum()

    # Guard against pathological under/overflow, then renormalise exactly.
    weights = np.asarray(weights, dtype=np.float64)
    total = weights.sum()
    if not math.isfinite(total) or total <= 0:
        raise AHPValidationError("Priority vector could not be normalised (non-positive sum).")
    return weights / total


# --------------------------------------------------------------------------- #
# Consistency
# --------------------------------------------------------------------------- #

def calculate_consistency(
    matrix: Sequence[Sequence[float]],
    weights: Sequence[float],
    threshold: float = CR_THRESHOLD,
) -> Dict[str, Any]:
    """
    Saaty consistency analysis for a single matrix.

    Steps (exactly as specified by the methodology):

        (Aw)_i        = sum_j a_ij * w_j
        lambda_i      = (A w)_i / w_i
        lambda_max    = mean(lambda_i)
        CI            = (lambda_max - n) / (n - 1)
        CR            = CI / RI[n]          (CR = 0 when RI[n] = 0)

    Args:
        matrix: square reciprocal matrix.
        weights: priority vector (same order as the matrix).
        threshold: CR limit under which the matrix counts as consistent.

    Returns:
        dict with lambda_max, ci, cr, ri, is_consistent, lambda_values,
        aw, eigenvalues (full spectrum, real parts) and threshold.
    """
    arr = validate_matrix(matrix)
    w = np.asarray(weights, dtype=np.float64)

    if w.shape != (arr.shape[0],):
        raise AHPValidationError(
            f"weights length {w.shape[0]} does not match matrix order {arr.shape[0]}."
        )
    if np.any(w <= 0):
        raise AHPValidationError("weights must be strictly positive.")

    n = arr.shape[0]

    aw = arr @ w                       # (A . w)
    lambda_values = aw / w             # lambda_i = (A w)_i / w_i
    lambda_max = float(lambda_values.mean())

    if n == 1:
        ci = 0.0
        ri = 0.0
        cr = 0.0
    else:
        ci = (lambda_max - n) / (n - 1)
        ri = RI_TABLE.get(n)
        if ri is None:
            # Saaty's table stops at 10; extrapolate conservatively for n > 10.
            ri = 1.98 * (n - 2.45) / (n - 1) if n > 2 else 0.0
        cr = 0.0 if ri == 0 else ci / ri

    # Full eigenvalue spectrum (real parts) for the scientific audit trail.
    eigenvalues = sorted((float(x.real) for x in np.linalg.eigvals(arr)), reverse=True)

    return {
        "order": n,
        "aw": [float(x) for x in aw],
        "lambdaValues": [float(x) for x in lambda_values],
        "lambdaMax": lambda_max,
        "eigenvalues": eigenvalues,
        "ci": float(ci),
        "ri": float(ri),
        "cr": float(cr),
        "threshold": float(threshold),
        "isConsistent": bool(cr < threshold),
    }


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #

def synthesize_global(
    criterion_weights: Sequence[float],
    alternative_weights_dict: Mapping[str, Sequence[float]],
    criterion_labels: Optional[Sequence[str]] = None,
    alternative_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Global synthesis: P_i = sum_j w_j * s_ij.

    Args:
        criterion_weights: priority vector of the criteria matrix (length m).
        alternative_weights_dict: {criterion label: local priority vector of the
            alternatives under that criterion} (each length n).
        criterion_labels: optional labels aligned with criterion_weights.
        alternative_labels: optional labels aligned with each local vector.

    Returns:
        dict with scores (length n), ranking (descending) and the
        per-criterion contribution matrix used by the stacked bar chart.
    """
    w = np.asarray(criterion_weights, dtype=np.float64)
    m = w.shape[0]

    keys: List[str] = list(alternative_weights_dict.keys())
    if criterion_labels is not None and len(criterion_labels) != m:
        raise AHPValidationError("criterion_labels length does not match criterion_weights.")
    if len(keys) != m:
        raise AHPValidationError(
            f"synthesis needs {m} alternative matrices (one per criterion), got {len(keys)}."
        )

    local = np.asarray(
        [np.asarray(alternative_weights_dict[k], dtype=np.float64) for k in keys]
    )  # shape (m, n)
    n = local.shape[1]

    if alternative_labels is not None and len(alternative_labels) != n:
        raise AHPValidationError("alternative_labels length does not match local vectors.")

    contributions = w[:, None] * local          # (m, n) weighted contribution
    scores = contributions.sum(axis=0)          # P_i

    order = np.argsort(-scores)                 # descending ranking
    names = list(alternative_labels) if alternative_labels else [f"A{i + 1}" for i in range(n)]
    crit_names = list(criterion_labels) if criterion_labels else keys

    ranking = [
        {
            "rank": int(pos + 1),
            "index": int(idx),
            "alternative": names[int(idx)],
            "score": float(scores[idx]),
            "contributions": {
                crit_names[j]: float(contributions[j, int(idx)]) for j in range(m)
            },
        }
        for pos, idx in enumerate(order)
    ]

    return {
        "scores": [float(x) for x in scores],
        "contributions": {
            names[i]: {crit_names[j]: float(contributions[j, i]) for j in range(m)}
            for i in range(n)
        },
        "ranking": ranking,
        "winner": names[int(order[0])],
        "winnerScore": float(scores[order[0]]),
    }


# --------------------------------------------------------------------------- #
# Full reports
# --------------------------------------------------------------------------- #

def analyze_matrix(
    matrix: Sequence[Sequence[float]],
    labels: Optional[Sequence[str]] = None,
    method: str = "column_sum",
    threshold: float = CR_THRESHOLD,
    name: str = "matrix",
) -> Dict[str, Any]:
    """
    Complete single-matrix analysis: validation, priority vector, consistency
    and the step-by-step mathematical proof shown in the Consistency Lab.
    """
    arr = validate_matrix(matrix, labels=labels, name=name)
    n = arr.shape[0]
    weights = calculate_eigenvector(arr, method=method)
    consistency = calculate_consistency(arr, weights, threshold=threshold)

    col_sums = arr.sum(axis=0)
    proof: Dict[str, Any] = {
        "method": method,
        "order": n,
        "columnSums": [float(x) for x in col_sums],
        "normalized": [[float(x) for x in row] for row in (arr / col_sums)],
        "priorityVector": [float(x) for x in weights],
        "aw": consistency["aw"],
        "lambdaValues": consistency["lambdaValues"],
    }

    if method == "geometric_mean":
        row_products = np.prod(arr, axis=1)
        nth_roots = np.power(row_products, 1.0 / n)
        proof["rowProducts"] = [float(x) for x in row_products]
        proof["nthRoots"] = [float(x) for x in nth_roots]

    return {
        "name": name,
        "labels": list(labels) if labels else [f"L{i + 1}" for i in range(n)],
        "matrix": [[float(x) for x in row] for row in arr],
        "method": method,
        "weights": [float(x) for x in weights],
        "lambdaMax": consistency["lambdaMax"],
        "eigenvalues": consistency["eigenvalues"],
        "ci": consistency["ci"],
        "ri": consistency["ri"],
        "cr": consistency["cr"],
        "threshold": threshold,
        "isConsistent": consistency["isConsistent"],
        "proof": proof,
    }


def run_synthesis(
    criteria_labels: Sequence[str],
    criteria_matrix: Sequence[Sequence[float]],
    alternatives: Sequence[str],
    alternative_matrices: Mapping[str, Sequence[Sequence[float]]],
    method: str = "column_sum",
    threshold: float = CR_THRESHOLD,
) -> Dict[str, Any]:
    """
    Full hierarchy analysis: criteria matrix + one alternative matrix per
    criterion -> local weights, consistency, global scores and ranking.
    """
    if method not in METHODS:
        raise AHPValidationError(f"Unknown method '{method}'. Use one of {METHODS}.")
    if len(criteria_labels) != len(set(criteria_labels)):
        raise AHPValidationError("Criteria labels must be unique.")
    if len(alternatives) != len(set(alternatives)):
        raise AHPValidationError("Alternative labels must be unique.")

    criteria_report = analyze_matrix(
        criteria_matrix, labels=criteria_labels, method=method,
        threshold=threshold, name="criteria",
    )

    missing = [c for c in criteria_labels if c not in alternative_matrices]
    if missing:
        raise AHPValidationError(f"Missing alternative matrix for criteria: {', '.join(missing)}.")
    extra = [c for c in alternative_matrices if c not in criteria_labels]
    if extra:
        raise AHPValidationError(f"Unknown alternative matrices: {', '.join(extra)}.")

    alt_reports: Dict[str, Dict[str, Any]] = {}
    for criterion in criteria_labels:
        block = alternative_matrices[criterion]
        if len(block) != len(alternatives):
            raise AHPValidationError(
                f"Matrix '{criterion}' has {len(block)} rows, expected {len(alternatives)}."
            )
        alt_reports[criterion] = analyze_matrix(
            block, labels=alternatives, method=method,
            threshold=threshold, name=f"alternatives[{criterion}]",
        )

    synthesis = synthesize_global(
        criteria_report["weights"],
        {c: alt_reports[c]["weights"] for c in criteria_labels},
        criterion_labels=list(criteria_labels),
        alternative_labels=list(alternatives),
    )

    inconsistent = [
        m["name"] for m in [criteria_report, *alt_reports.values()]
        if not m["isConsistent"]
    ]

    return {
        "method": method,
        "threshold": threshold,
        "criteriaLabels": list(criteria_labels),
        "alternatives": list(alternatives),
        "criteria": criteria_report,
        "alternativesReport": alt_reports,
        "globalScores": synthesis["scores"],
        "contributions": synthesis["contributions"],
        "ranking": synthesis["ranking"],
        "winner": synthesis["winner"],
        "winnerScore": synthesis["winnerScore"],
        "isConsistent": len(inconsistent) == 0,
        "inconsistentMatrices": inconsistent,
    }


# --------------------------------------------------------------------------- #
# Ground truth: MPV Family Fleet case study
# --------------------------------------------------------------------------- #

#: Order of the criteria matrix rows/columns in the case study.
CRITERIA_ORDER: Tuple[str, ...] = ("Keandalan", "Style", "Konsumsi BB")
ALTERNATIVES_ORDER: Tuple[str, ...] = ("Avanza", "Xenia", "Ertiga", "Grand Livina")


def default_model() -> Dict[str, Any]:
    """
    The MPV Family Fleet ground-truth case study.

    Hierarchy:
        Goal  : Pemilihan Mobil Armada Keluarga
        Kriteria (order K, S, B):
            Keandalan : Style          = 2   -> S : K = 1/2
            Konsumsi BB : Style        = 1/3 -> Style : B = 3
            Konsumsi BB : Keandalan    = 1/4 -> K : B = 4
        Alternatif: Avanza, Xenia, Ertiga, Grand Livina.
    """
    criteria_matrix = [
        [1.0,     2.0,        4.0],       # Keandalan vs  K, S, B
        [1 / 2,   1.0,        3.0],       # Style      vs  K, S, B
        [1 / 4,   1 / 3,      1.0],       # Konsumsi BB vs K, S, B
    ]

    alternative_matrices = {
        "Keandalan": [
            [1,     2,     1 / 4, 1 / 3],
            [1 / 2, 1,     1 / 5, 1 / 4],
            [4,     5,     1,     3],
            [3,     4,     1 / 3, 1],
        ],
        "Style": [
            [1,     2,     1 / 3, 1 / 4],
            [1 / 2, 1,     1 / 2, 1 / 5],
            [3,     2,     1,     2],
            [4,     5,     1 / 2, 1],
        ],
        "Konsumsi BB": [
            [1,       2,     1 / 2, 4],
            [1 / 2,   1,     2 / 3, 3],
            [2,       3 / 2, 1,     4],
            [1 / 4,   1 / 3, 1 / 4, 1],
        ],
    }

    return {
        "goal": "Pemilihan Mobil Armada Keluarga (MPV Family Fleet)",
        "criteriaLabels": list(CRITERIA_ORDER),
        "alternatives": list(ALTERNATIVES_ORDER),
        "criteriaMatrix": criteria_matrix,
        "alternativeMatrices": alternative_matrices,
        "riTable": {str(k): v for k, v in RI_TABLE.items()},
        "crThreshold": CR_THRESHOLD,
    }
