"""
main.py
=======
FastAPI application exposing the AHP engine as a JSON API and serving the
AutoAHP Intelligent Decision Support System single-page interface.

Endpoints
---------
    GET   /                          -> SPA (index.html)
    GET   /style.css, /app.js, /cursor3d.js -> SPA assets
    GET   /api/health                -> liveness probe
    GET   /api/ahp/default-model     -> ground-truth MPV Family Fleet case study
    POST  /api/ahp/calculate-matrix  -> weights + consistency for one matrix
    POST  /api/ahp/synthesize        -> full hierarchy synthesis report
    GET   /api/export/csv            -> synthesis result as CSV download
    GET   /api/export/json           -> complete model + results as JSON

Run locally:
    uvicorn main:app --reload --port 8000
    # or:  python main.py
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from engine import (
    AHPValidationError,
    CR_THRESHOLD,
    METHODS,
    RI_TABLE,
    SAATY_SCALE,
    analyze_matrix,
    default_model,
    run_synthesis,
)

ROOT = Path(__file__).resolve().parent

app = FastAPI(
    title="AutoAHP Intelligent Decision Support System",
    description=(
        "Sistem Pendukung Keputusan berbasis Analytic Hierarchy Process "
        "(Thomas L. Saaty) untuk studi kasus Pemilihan Mobil Armada Keluarga."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CalcMethod = Literal["column_sum", "geometric_mean"]


# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #

class MatrixInput(BaseModel):
    """A single n x n pairwise comparison matrix plus its metadata."""

    matrix: List[List[float]] = Field(
        ..., description="Square reciprocal matrix (a_ji = 1/a_ij, a_ii = 1)."
    )
    labels: List[str] = Field(
        default_factory=list, description="Row/column labels, same order as the matrix."
    )
    method: CalcMethod = Field(
        default="column_sum",
        description="Priority method: column_sum (normalised columns) or geometric_mean (nth root).",
    )

    @field_validator("matrix")
    @classmethod
    def _non_empty(cls, v: List[List[float]]) -> List[List[float]]:
        if not v:
            raise ValueError("matrix must not be empty")
        return v


class SynthesisRequest(BaseModel):
    """Full hierarchy: criteria matrix + one alternative matrix per criterion."""

    criteriaLabels: List[str] = Field(..., min_length=1)
    criteriaMatrix: List[List[float]]
    alternatives: List[str] = Field(..., min_length=1)
    alternativeMatrices: Union[Dict[str, List[List[float]]], List[List[float]]] = Field(
        ...,
        description=(
            "Either {criterionLabel: matrix} or a list of matrices aligned "
            "with criteriaLabels."
        ),
    )
    method: CalcMethod = "column_sum"
    crThreshold: float = Field(default=CR_THRESHOLD, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _normalise_alternative_matrices(self) -> "SynthesisRequest":
        if isinstance(self.alternativeMatrices, list):
            if len(self.alternativeMatrices) != len(self.criteriaLabels):
                raise ValueError(
                    "alternativeMatrices list length must match criteriaLabels"
                )
            self.alternativeMatrices = {
                label: block
                for label, block in zip(self.criteriaLabels, self.alternativeMatrices)
            }
        expected = set(self.criteriaLabels)
        got = set(self.alternativeMatrices)
        if expected != got:
            raise ValueError(
                f"alternativeMatrices keys {sorted(got)} do not match "
                f"criteriaLabels {sorted(expected)}"
            )
        return self


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"ok": False, "error": message})


def _build_synthesis(payload: SynthesisRequest) -> Dict[str, Any]:
    report = run_synthesis(
        criteria_labels=payload.criteriaLabels,
        criteria_matrix=payload.criteriaMatrix,
        alternatives=payload.alternatives,
        alternative_matrices=payload.alternativeMatrices,
        method=payload.method,
        threshold=payload.crThreshold,
    )
    return report


def _default_synthesis(method: str = "column_sum", threshold: float = CR_THRESHOLD) -> Dict[str, Any]:
    model = default_model()
    return run_synthesis(
        criteria_labels=model["criteriaLabels"],
        criteria_matrix=model["criteriaMatrix"],
        alternatives=model["alternatives"],
        alternative_matrices=model["alternativeMatrices"],
        method=method,
        threshold=threshold,
    )


# --------------------------------------------------------------------------- #
# SPA assets
# --------------------------------------------------------------------------- #

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/style.css", include_in_schema=False)
def stylesheet() -> FileResponse:
    return FileResponse(ROOT / "style.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def app_js() -> FileResponse:
    return FileResponse(ROOT / "app.js", media_type="application/javascript")


@app.get("/cursor3d.js", include_in_schema=False)
def cursor_js() -> FileResponse:
    return FileResponse(ROOT / "cursor3d.js", media_type="application/javascript")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health() -> Dict[str, Any]:
    """Service liveness probe."""
    return {
        "ok": True,
        "service": "autoahp",
        "method": "Analytic Hierarchy Process (Saaty)",
        "calculationMethods": list(METHODS),
        "crThreshold": CR_THRESHOLD,
        "numpy": np.__version__,
    }


@app.get("/api/ahp/default-model")
def get_default_model(
    method: CalcMethod = Query(default="column_sum"),
    compute: bool = Query(default=True, description="Include the computed synthesis."),
) -> Dict[str, Any]:
    """
    Return the complete ground-truth case study (matrices, labels, RI table)
    and, by default, the fully computed synthesis report for it.
    """
    model = default_model()
    model["saatyScale"] = SAATY_SCALE
    if compute:
        model["result"] = _default_synthesis(method=method)
    return {"ok": True, **model}


@app.post("/api/ahp/calculate-matrix")
def calculate_matrix(payload: MatrixInput) -> Dict[str, Any]:
    """
    Weights (eigenvector) + consistency analysis for a single pairwise matrix.

    Returns lambda_max, CI, CR, RI, is_consistent and the step-by-step proof.
    """
    if payload.method not in METHODS:
        return _error(400, f"method harus salah satu dari {METHODS}")
    try:
        report = analyze_matrix(
            payload.matrix,
            labels=payload.labels or None,
            method=payload.method,
            threshold=CR_THRESHOLD,
            name="matrix",
        )
    except AHPValidationError as exc:
        return _error(400, str(exc))
    except (TypeError, ValueError) as exc:
        return _error(400, f"Data tidak valid: {exc}")
    return {"ok": True, "result": report}


@app.post("/api/ahp/synthesize")
def synthesize(payload: SynthesisRequest) -> Dict[str, Any]:
    """
    Full hierarchy synthesis: criteria weights, per-criterion alternative
    weights, global scores, ranking and consistency of every matrix.
    """
    try:
        report = _build_synthesis(payload)
    except AHPValidationError as exc:
        return _error(400, str(exc))
    except (TypeError, ValueError) as exc:
        return _error(400, f"Data tidak valid: {exc}")
    return {"ok": True, **report}


@app.get("/api/export/csv")
def export_csv(
    method: CalcMethod = Query(default="column_sum"),
) -> PlainTextResponse:
    """Export the ground-truth case study synthesis as CSV (Excel friendly)."""
    try:
        report = _default_synthesis(method=method)
    except AHPValidationError as exc:  # pragma: no cover - defaults are valid
        return _error(400, str(exc))  # type: ignore[return-value]

    model = default_model()
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")

    def _stats_block(title: str, labels: List[str], block: Dict[str, Any]) -> None:
        writer.writerow([title, *labels])
        writer.writerow(["Bobot", *[f"{w:.6f}" for w in block["weights"]]])
        writer.writerow([
            "lambda_max", f"{block['lambdaMax']:.6f}",
            "CI", f"{block['ci']:.6f}",
            "RI", f"{block['ri']:.4f}",
            "CR", f"{block['cr']:.6f}",
            "Konsisten", "YA" if block["isConsistent"] else "TIDAK",
        ])
        writer.writerow([])

    writer.writerow(["AutoAHP - Sistem Pendukung Keputusan AHP"])
    writer.writerow(["Goal", model["goal"]])
    writer.writerow(["Metode prioritas", method])
    writer.writerow(["CR threshold", f"{report['threshold']:.2f}"])
    writer.writerow([])

    _stats_block("Kriteria", report["criteriaLabels"], report["criteria"])
    for label, alt in report["alternativesReport"].items():
        _stats_block(f"Alternatif [{label}]", report["alternatives"], alt)

    writer.writerow(["Rank", "Alternatif", "Skor Global", *report["criteriaLabels"]])
    for row in report["ranking"]:
        writer.writerow(
            [row["rank"], row["alternative"], f"{row['score']:.6f}",
             *[f"{row['contributions'][k]:.6f}" for k in report["criteriaLabels"]]]
        )

    return PlainTextResponse(
        buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="autoahp-synthesis.csv"'
        },
    )


@app.get("/api/export/json")
def export_json(
    method: CalcMethod = Query(default="column_sum"),
) -> PlainTextResponse:
    """Export the complete model (matrices + labels + computed results) as JSON."""
    model = default_model()
    model["saatyScale"] = SAATY_SCALE
    model["result"] = _default_synthesis(method=method)
    payload = {"ok": True, "generatedBy": "AutoAHP v1.0.0", **model}
    return PlainTextResponse(
        json.dumps(payload, ensure_ascii=False, indent=2),
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="autoahp-model.json"'
        },
    )


# --------------------------------------------------------------------------- #
# Error handlers
# --------------------------------------------------------------------------- #

@app.exception_handler(AHPValidationError)
async def ahp_validation_handler(_request: Request, exc: AHPValidationError) -> JSONResponse:
    return _error(400, str(exc))


@app.exception_handler(404)
async def not_found(_request: Request, exc) -> JSONResponse:  # noqa: ANN001
    return _error(404, "Endpoint tidak ditemukan.")


@app.exception_handler(405)
async def method_not_allowed(_request: Request, exc) -> JSONResponse:  # noqa: ANN001
    return _error(405, "Metode HTTP tidak diizinkan.")


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    print("=" * 64)
    print(" AutoAHP - Sistem Pendukung Keputusan (AHP)")
    print(" URL: http://127.0.0.1:8000  (docs: /docs)")
    print("=" * 64)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
