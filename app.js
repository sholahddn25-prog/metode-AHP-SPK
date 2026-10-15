/* ==========================================================================
   app.js — AutoAHP Intelligent Decision Support System
   --------------------------------------------------------------------------
   Responsibilities:
     • Application state (matrices, labels, method, CR threshold, theme).
     • API client (FastAPI backend) with an instant offline fallback engine
       (identical math, used for GitHub Pages / no-backend deployments).
     • Rendering of the 6 dashboard tabs, Chart.js visualisations, toasts,
       the documentation modal and all settings.

   The fallback engine below implements exactly the same methodology as
   engine.py (column_sum / geometric_mean, Saaty consistency, global
   synthesis) so the SPA is fully functional without a server.
   ========================================================================== */

"use strict";

/* ==========================================================================
   0. Constants & ground-truth model
   ========================================================================== */

const RI_TABLE = { 1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49 };
const SAATY_SCALE = [
  [1, "Equal importance"], [2, "Weak"], [3, "Moderate importance"], [4, "Moderate plus"],
  [5, "Strong importance"], [6, "Strong plus"], [7, "Very strong importance"],
  [8, "Very, very strong"], [9, "Extreme importance"],
];

const DEFAULT_MODEL = {
  goal: "Pemilihan Mobil Armada Keluarga (MPV Family Fleet)",
  criteriaLabels: ["Keandalan", "Style", "Konsumsi BB"],
  criteriaShort: { Keandalan: "K", Style: "S", "Konsumsi BB": "B" },
  alternatives: ["Avanza", "Xenia", "Ertiga", "Grand Livina"],
  criteriaMatrix: [
    [1, 2, 4],
    [1 / 2, 1, 3],
    [1 / 4, 1 / 3, 1],
  ],
  alternativeMatrices: {
    Keandalan: [
      [1, 2, 1 / 4, 1 / 3],
      [1 / 2, 1, 1 / 5, 1 / 4],
      [4, 5, 1, 3],
      [3, 4, 1 / 3, 1],
    ],
    Style: [
      [1, 2, 1 / 3, 1 / 4],
      [1 / 2, 1, 1 / 2, 1 / 5],
      [3, 2, 1, 2],
      [4, 5, 1 / 2, 1],
    ],
    "Konsumsi BB": [
      [1, 2, 1 / 2, 4],
      [1 / 2, 1, 2 / 3, 3],
      [2, 3 / 2, 1, 4],
      [1 / 4, 1 / 3, 1 / 4, 1],
    ],
  },
};

const CRIT_COLORS = {
  Keandalan: "#ea1d22", // Marvel red
  Style: "#2e7df7",     // web blue
  "Konsumsi BB": "#b5b6b8", // suit silver
};

const STORAGE_KEY = "autoahp.settings.v1";

/* ==========================================================================
   1. State
   ========================================================================== */

const state = {
  criteriaLabels: [...DEFAULT_MODEL.criteriaLabels],
  alternatives: [...DEFAULT_MODEL.alternatives],
  criteriaMatrix: clone(DEFAULT_MODEL.criteriaMatrix),
  alternativeMatrices: clone(DEFAULT_MODEL.alternativeMatrices),
  method: "column_sum",
  threshold: 0.1,
  activeCriterion: "Keandalan",
  result: null,
  serverOnline: false,
  settings: {
    theme: "dark",
    cursor3d: true,
    reduceMotion: false,
  },
};

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

/* ==========================================================================
   2. Offline fallback engine (mirrors engine.py)
   ========================================================================== */

const Engine = {
  eigenvector(matrix, method) {
    const n = matrix.length;
    let w;
    if (method === "geometric_mean") {
      const roots = matrix.map((row) => Math.pow(row.reduce((a, b) => a * b, 1), 1 / n));
      const sum = roots.reduce((a, b) => a + b, 0);
      w = roots.map((r) => r / sum);
    } else {
      const colSums = Array.from({ length: n }, (_, j) =>
        matrix.reduce((acc, row) => acc + row[j], 0)
      );
      w = matrix.map((row) => row.reduce((acc, v, j) => acc + v / colSums[j], 0) / n);
    }
    const total = w.reduce((a, b) => a + b, 0);
    return w.map((v) => v / total);
  },

  consistency(matrix, weights, threshold) {
    const n = matrix.length;
    const aw = matrix.map((row) => row.reduce((acc, v, j) => acc + v * weights[j], 0));
    const lambdaValues = aw.map((v, i) => v / weights[i]);
    const lambdaMax = lambdaValues.reduce((a, b) => a + b, 0) / n;
    const ci = n === 1 ? 0 : (lambdaMax - n) / (n - 1);
    const ri = RI_TABLE[n] ?? 0;
    const cr = ri === 0 ? 0 : ci / ri;
    return {
      order: n, aw, lambdaValues, lambdaMax, ci, ri, cr,
      threshold: threshold ?? 0.1,
      isConsistent: cr < (threshold ?? 0.1),
    };
  },

  analyze(matrix, labels, method, threshold, name) {
    const weights = this.eigenvector(matrix, method);
    const stats = this.consistency(matrix, weights, threshold);
    const n = matrix.length;
    const colSums = Array.from({ length: n }, (_, j) =>
      matrix.reduce((acc, row) => acc + row[j], 0)
    );
    const proof = {
      method, order: n, columnSums: colSums,
      normalized: matrix.map((row) => row.map((v, j) => v / colSums[j])),
      priorityVector: weights, aw: stats.aw, lambdaValues: stats.lambdaValues,
    };
    if (method === "geometric_mean") {
      proof.rowProducts = matrix.map((row) => row.reduce((a, b) => a * b, 1));
      proof.nthRoots = proof.rowProducts.map((p) => Math.pow(p, 1 / n));
    }
    return {
      name, labels, matrix, method, weights,
      lambdaMax: stats.lambdaMax, ci: stats.ci, ri: stats.ri, cr: stats.cr,
      threshold: threshold ?? 0.1, isConsistent: stats.isConsistent, proof,
    };
  },

  synthesize(criteriaLabels, criteriaMatrix, alternatives, altMatrices, method, threshold) {
    const criteria = this.analyze(criteriaMatrix, criteriaLabels, method, threshold, "criteria");
    const altReport = {};
    for (const c of criteriaLabels) {
      altReport[c] = this.analyze(altMatrices[c], alternatives, method, threshold, `alternatives[${c}]`);
    }
    const m = criteriaLabels.length;
    const n = alternatives.length;
    const scores = new Array(n).fill(0);
    const contributions = {};
    alternatives.forEach((a) => (contributions[a] = {}));
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < m; j++) {
        const part = criteria.weights[j] * altReport[criteriaLabels[j]].weights[i];
        contributions[alternatives[i]][criteriaLabels[j]] = part;
        s += part;
      }
      scores[i] = s;
    }
    const ranking = alternatives
      .map((a, i) => ({ rank: 0, index: i, alternative: a, score: scores[i], contributions: contributions[a] }))
      .sort((x, y) => y.score - x.score)
      .map((row, pos) => ({ ...row, rank: pos + 1 }));
    const inconsistent = [criteria, ...Object.values(altReport)]
      .filter((blk) => !blk.isConsistent)
      .map((blk) => blk.name);
    return {
      method, threshold, criteriaLabels, alternatives,
      criteria, alternativesReport: altReport,
      globalScores: scores, contributions, ranking,
      winner: ranking[0].alternative, winnerScore: ranking[0].score,
      isConsistent: inconsistent.length === 0, inconsistentMatrices: inconsistent,
    };
  },
};

/* ==========================================================================
   3. API client
   ========================================================================== */

const API = {
  base: "", // same origin; on GitHub Pages every call falls back locally
  async post(path, body, timeoutMs = 2500) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(this.base + path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const data = await res.json();
      if (!res.ok || data.ok === false) throw new Error(data.error || `HTTP ${res.status}`);
      markServer(true);
      return data;
    } finally {
      clearTimeout(timer);
    }
  },
  async get(path, timeoutMs = 2500) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(this.base + path, { signal: controller.signal });
      const data = await res.json();
      if (!res.ok || data.ok === false) throw new Error(data.error || `HTTP ${res.status}`);
      markServer(true);
      return data;
    } finally {
      clearTimeout(timer);
    }
  },
};

function markServer(online) {
  if (state.serverOnline === online) {
    setApiStatus(online, false);
    return;
  }
  state.serverOnline = online;
  setApiStatus(online, true);
}

function setApiStatus(online, announce) {
  const el = document.getElementById("api-status");
  el.className = "status-pill " + (online ? "online" : "offline");
  el.textContent = online ? "● API aktif" : "● mode lokal";
  el.title = online
    ? "Terhubung ke backend FastAPI"
    : "Backend tidak terdeteksi — memakai kalkulator lokal JS (ideal untuk GitHub Pages)";
  if (announce) {
    toast(
      online ? "Terhubung ke backend FastAPI — perhitungan di server." : "Backend tidak aktif — beralih ke kalkulator lokal JS.",
      online ? "success" : "warn"
    );
  }
}

/* ==========================================================================
   4. Recalculation pipeline
   ========================================================================== */

let recalcTimer = null;
let recalcSeq = 0; // Monotonic token so out-of-order responses are discarded.

function requestRecalc(immediate = false) {
  clearTimeout(recalcTimer);
  recalcTimer = setTimeout(recalculate, immediate ? 0 : 260);
}

async function recalculate() {
  const seq = ++recalcSeq;
  const payload = {
    criteriaLabels: state.criteriaLabels,
    criteriaMatrix: state.criteriaMatrix,
    alternatives: state.alternatives,
    alternativeMatrices: state.alternativeMatrices,
    method: state.method,
    crThreshold: state.threshold,
  };

  let result;
  try {
    const data = await API.post("/api/ahp/synthesize", payload);
    result = data.result || data;
    result._source = "server";
  } catch (err) {
    markServer(false);
    result = Engine.synthesize(
      payload.criteriaLabels, payload.criteriaMatrix,
      payload.alternatives, payload.alternativeMatrices,
      payload.method, payload.crThreshold
    );
    result._source = "local";
  }

  // A newer request already superseded this one — drop the stale result.
  if (seq !== recalcSeq) return;

  state.result = result;
  renderAll();
}

/* ==========================================================================
   5. Rendering
   ========================================================================== */

function renderAll() {
  renderDashboard();
  renderCriteriaTab();
  renderAlternativesTab();
  renderConsistencyTab();
  renderReportTab();
}

/* ------------------------------ Dashboard -------------------------------- */

let doughnutChart = null;

function renderDashboard() {
  const r = state.result;
  if (!r) return;

  // Winner card
  document.getElementById("winner-name").textContent = r.winner;
  document.getElementById("winner-score").textContent = (r.winnerScore * 100).toFixed(2) + "%";
  document.getElementById("winner-bar-fill").style.width = (r.winnerScore * 100).toFixed(2) + "%";

  // KPI cards
  const anyInconsistent = r.inconsistentMatrices.length > 0;
  const kpis = [
    {
      label: "Rekomendasi",
      value: r.winner,
      cls: "good",
      note: `Skor global ${(r.winnerScore * 100).toFixed(2)}%`,
    },
    {
      label: "CR Matriks Kriteria",
      value: (r.criteria.cr * 100).toFixed(2) + "%",
      cls: r.criteria.isConsistent ? "good" : "bad",
      note: `Ambang ${(r.threshold * 100).toFixed(0)}% · λmax ${r.criteria.lambdaMax.toFixed(4)}`,
    },
    {
      label: "Status Validasi",
      value: r.isConsistent ? "VALID" : "PERIKSA",
      cls: r.isConsistent ? "good" : "bad",
      note: r.isConsistent
        ? "Semua matriks konsisten"
        : `${r.inconsistentMatrices.length} matriks tidak konsisten`,
    },
    {
      label: "Metode Bobot",
      value: state.method === "column_sum" ? "Column Sum" : "Geometric Mean",
      cls: "",
      note: r._source === "server" ? "Dihitung server (NumPy)" : "Dihitung lokal (JS)",
    },
  ];

  document.getElementById("kpi-grid").innerHTML = kpis
    .map(
      (k) => `
      <div class="kpi" data-magnetic>
        <span class="kpi-label">${k.label}</span>
        <span class="kpi-value ${k.cls}">${k.value}</span>
        <span class="kpi-note">${k.note}</span>
      </div>`
    )
    .join("");

  // Ranking list
  document.getElementById("ranking-list").innerHTML = r.ranking
    .map(
      (row) => `
      <div class="rank-row ${row.rank === 1 ? "top" : ""}">
        <span class="rank-no">${row.rank}</span>
        <span class="rank-name">${row.alternative}</span>
        <span class="rank-score">${(row.score * 100).toFixed(2)}%</span>
      </div>`
    )
    .join("");

  // Doughnut chart
  const ctx = document.getElementById("chart-doughnut");
  const labels = r.criteriaLabels;
  const data = r.criteria.weights.map((w) => +(w * 100).toFixed(2));
  const colors = labels.map((l) => CRIT_COLORS[l] || "#64748b");

  if (doughnutChart) {
    doughnutChart.data.labels = labels;
    doughnutChart.data.datasets[0].data = data;
    doughnutChart.update();
  } else if (window.Chart) {
    doughnutChart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels,
        datasets: [{ data, backgroundColor: colors, borderWidth: 0, hoverOffset: 10 }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "64%",
        plugins: {
          legend: { position: "bottom", labels: { color: chartTextColor(), usePointStyle: true, padding: 16 } },
          tooltip: { callbacks: { label: (c) => ` ${c.label}: ${c.parsed}%` } },
        },
        animation: { animateRotate: true, duration: 900 },
      },
    });
  }
}

/* ------------------------------ Criteria tab ----------------------------- */

function renderCriteriaTab() {
  buildMatrix("criteria-matrix", state.criteriaMatrix, state.criteriaLabels, state.criteriaLabels, (i, j, value) => {
    state.criteriaMatrix[i][j] = value;
    // Keep reciprocity: A_ji = 1 / A_ij.
    state.criteriaMatrix[j][i] = 1 / value;
    refreshMatrixCells("criteria-matrix");
    requestRecalc();
  });

  const r = state.result;
  if (!r) return;
  renderWeightBars("criteria-weights", r.criteriaLabels, r.criteria.weights, r.criteriaLabels.map((l) => CRIT_COLORS[l]));

  const c = r.criteria;
  document.getElementById("criteria-stats").innerHTML = `
    ${statRow("λ max", c.lambdaMax.toFixed(6))}
    ${statRow("CI (Consistency Index)", c.ci.toFixed(6))}
    ${statRow("RI (Random Index)", c.ri.toFixed(4))}
    ${statRow("CR (Consistency Ratio)", (c.cr * 100).toFixed(2) + "%", c.isConsistent ? "good" : "bad")}
    ${statRow("Status", c.isConsistent ? "KONSISTEN ✓" : "TIDAK KONSISTEN ✗", c.isConsistent ? "good" : "bad")}
    ${statRow("Sumber perhitungan", r._source === "server" ? "FastAPI · NumPy" : "Fallback JS")}
  `;
}

/* ------------------------------ Alternatives tab ------------------------- */

function renderAlternativesTab() {
  const subtabs = document.getElementById("criteria-subtabs");
  subtabs.innerHTML = state.criteriaLabels
    .map(
      (c) => `
      <button class="sub-tab ${c === state.activeCriterion ? "active" : ""}" data-criterion="${c}">
        ${c} <span style="opacity:.6">(${DEFAULT_MODEL.criteriaShort[c] || c[0]})</span>
      </button>`
    )
    .join("");
  subtabs.querySelectorAll(".sub-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.activeCriterion = btn.dataset.criterion;
      renderAlternativesTab();
    });
  });

  const c = state.activeCriterion;
  const matrix = state.alternativeMatrices[c];
  document.getElementById("alt-criterion-name").textContent = c;

  buildMatrix("alt-matrix", matrix, state.alternatives, state.alternatives, (i, j, value) => {
    matrix[i][j] = value;
    matrix[j][i] = 1 / value;
    refreshMatrixCells("alt-matrix");
    requestRecalc();
  });

  const r = state.result;
  if (!r) return;
  const block = r.alternativesReport[c];
  renderWeightBars(
    "alt-weights", r.alternatives, block.weights,
    r.alternatives.map((_, i) => mixColor("#ea1d22", "#2e7df7", i / (r.alternatives.length - 1)))
  );

  const badge = document.getElementById("alt-cr-badge");
  badge.textContent = `CR ${(block.cr * 100).toFixed(2)}% · λmax ${block.lambdaMax.toFixed(4)}`;
  badge.className = "cr-badge " + (block.isConsistent ? "good" : "bad");
}

/* ------------------------------ Consistency tab -------------------------- */

let gaugeDrawn = false;

function renderConsistencyTab() {
  const r = state.result;
  if (!r) return;

  drawGauge(r.criteria.cr, r.threshold);
  gaugeDrawn = true;

  document.getElementById("gauge-legend").innerHTML = `
    <span class="chip ${r.criteria.isConsistent ? "good" : "bad"}">CR ${(r.criteria.cr * 100).toFixed(2)}%</span>
    <span class="chip">Ambang ${(r.threshold * 100).toFixed(0)}%</span>
    <span class="chip">λmax ${r.criteria.lambdaMax.toFixed(4)}</span>
    <span class="chip">CI ${r.criteria.ci.toFixed(4)}</span>
    <span class="chip">RI ${r.criteria.ri.toFixed(2)}</span>
  `;

  // Consistency table (all matrices)
  const rows = [
    { label: "Kriteria (S, K, B)", blk: r.criteria },
    ...r.criteriaLabels.map((c) => ({ label: `Alternatif [${c}]`, blk: r.alternativesReport[c] })),
  ];
  document.querySelector("#consistency-table tbody").innerHTML = rows
    .map(
      ({ label, blk }) => `
      <tr>
        <td class="name">${label}</td>
        <td class="mono">${blk.labels ? blk.labels.length : blk.proof.order}</td>
        <td class="mono">${blk.lambdaMax.toFixed(4)}</td>
        <td class="mono">${blk.ci.toFixed(4)}</td>
        <td class="mono">${blk.ri.toFixed(2)}</td>
        <td class="mono">${blk.cr.toFixed(4)}</td>
        <td><span class="chip ${blk.isConsistent ? "good" : "bad"}">${blk.isConsistent ? "VALID" : "REVIEW"}</span></td>
      </tr>`
    )
    .join("");

  renderProof(r.criteria);
  updateSensitivityUI();
}

function renderProof(block) {
  const p = block.proof;
  const labels = block.labels;
  const shortLabels = labels.map((l) => DEFAULT_MODEL.criteriaShort[l] || l.slice(0, 3));

  const head = `<tr><th></th>${shortLabels.map((s) => `<th>${s}</th>`).join("")}<th>Σ</th></tr>`;

  const colSumsRow = `<tr><td style="color:var(--text-2)">Σ kolom</td>${p.columnSums
    .map((v) => `<td class="mono">${v.toFixed(4)}</td>`).join("")}<td class="mono">${p.columnSums.reduce((a, b) => a + b, 0).toFixed(4)}</td></tr>`;

  const normalizedRows = p.normalized
    .map(
      (row, i) =>
        `<tr><td class="name" style="text-align:left">${labels[i]}</td>${row
          .map((v) => `<td class="mono">${v.toFixed(4)}</td>`).join("")}<td class="mono hl">${p.priorityVector[i].toFixed(4)}</td></tr>`
    )
    .join("");

  const lambdaRows = p.lambdaValues
    .map(
      (v, i) =>
        `<tr><td class="name" style="text-align:left">${labels[i]}</td>
           <td class="mono">w=${p.priorityVector[i].toFixed(4)}</td>
           <td class="mono">A·w=${p.aw[i].toFixed(4)}</td>
           <td class="mono hl">λ=${v.toFixed(6)}</td></tr>`
    )
    .join("");

  document.getElementById("proof-steps").innerHTML = `
    <div class="proof-step">
      <h4>1 · Matriks Perbandingan Berpasangan</h4>
      <div class="formula">A_ji = 1 / A_ij   ·   A_ii = 1   ·   skala Saaty 1/9–9   ·   n = ${p.order}</div>
      <table class="mini-table"><thead>${head}</thead><tbody>
        ${block.matrix.map((row, i) => `<tr><td class="name" style="text-align:left">${labels[i]}</td>${row.map((v) => `<td class="mono">${formatFraction(v)}</td>`).join("")}<td></td></tr>`).join("")}
      </tbody></table>
    </div>

    <div class="proof-step">
      <h4>2 · Normalisasi Kolom ${p.method === "geometric_mean" ? "(untuk verifikasi visual; bobot memakai akar ke-n)" : ""}</h4>
      <div class="formula">a'_ij = a_ij / Σ_i a_ij     →     w_i = rata-rata baris i ${p.method === "geometric_mean" ? "\n[geometric_mean] w_i = (∏_j a_ij)^(1/n) / Σ" : ""}</div>
      <table class="mini-table"><thead>${head}</thead><tbody>
        ${colSumsRow}
        ${normalizedRows}
      </tbody></table>
    </div>

    <div class="proof-step">
      <h4>3 · Nilai Eigen &amp; λ<sub>max</sub></h4>
      <div class="formula">λ_i = (A · w)_i / w_i          λ_max = (1/n) · Σ λ_i = ${block.lambdaMax.toFixed(6)}</div>
      <table class="mini-table"><tbody>${lambdaRows}</tbody></table>
    </div>

    <div class="proof-step">
      <h4>4 · Indeks Konsistensi</h4>
      <div class="formula">CI  = (λ_max − n) / (n − 1) = (${block.lambdaMax.toFixed(6)} − ${p.order}) / ${p.order - 1} = ${block.ci.toFixed(6)}
RI  = ${block.ri.toFixed(2)}   (tabel Saaty, n = ${p.order})
CR  = CI / RI = ${block.ci.toFixed(6)} / ${block.ri.toFixed(2)} = ${(block.cr * 100).toFixed(2)}%
Status: ${block.isConsistent ? `CR < ${(block.threshold * 100).toFixed(0)}% → matriks KONSISTEN / VALID ✓` : `CR ≥ ambang → matriks perlu ditinjau ulang ✗`}</div>
    </div>
  `;
}

function drawGauge(cr, threshold) {
  const canvas = document.getElementById("cr-gauge");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H - 30, radius = 130;
  const start = Math.PI, end = 2 * Math.PI; // upper half arc
  const maxCR = 0.3; // gauge spans CR 0 .. 0.3

  // Background arc
  ctx.lineWidth = 22;
  ctx.lineCap = "round";
  ctx.strokeStyle = getCssVar("--chart-grid");
  ctx.beginPath();
  ctx.arc(cx, cy, radius, start, end);
  ctx.stroke();

  // Threshold marker
  const thAngle = start + (Math.min(threshold, maxCR) / maxCR) * (end - start);
  ctx.strokeStyle = "rgba(245,158,11,0.85)";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(cx + Math.cos(thAngle) * (radius - 20), cy + Math.sin(thAngle) * (radius - 20));
  ctx.lineTo(cx + Math.cos(thAngle) * (radius + 20), cy + Math.sin(thAngle) * (radius + 20));
  ctx.stroke();

  // Value arc
  const valueAngle = start + (Math.min(cr, maxCR) / maxCR) * (end - start);
  const gradient = ctx.createLinearGradient(cx - radius, cy, cx + radius, cy);
  gradient.addColorStop(0, "#2e7df7");
  gradient.addColorStop(0.5, "#ea1d22");
  gradient.addColorStop(1, "#ef4444");
  ctx.strokeStyle = gradient;
  ctx.lineWidth = 22;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, start, valueAngle);
  ctx.stroke();

  // Needle
  const nx = cx + Math.cos(valueAngle) * (radius - 26);
  const ny = cy + Math.sin(valueAngle) * (radius - 26);
  ctx.strokeStyle = getCssVar("--text-0");
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(nx, ny);
  ctx.stroke();
  ctx.fillStyle = getCssVar("--text-0");
  ctx.beginPath();
  ctx.arc(cx, cy, 7, 0, Math.PI * 2);
  ctx.fill();

  // Readout
  ctx.fillStyle = getCssVar("--text-0");
  ctx.font = "700 34px 'Segoe UI', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText((cr * 100).toFixed(2) + "%", cx, cy - 40);
  ctx.fillStyle = getCssVar("--text-2");
  ctx.font = "500 13px 'Segoe UI', sans-serif";
  ctx.fillText("Consistency Ratio", cx, cy - 18);
  ctx.fillText(`ambang ${(threshold * 100).toFixed(0)}%`, cx, cy + 2);
}

/* ------------------------------ Sensitivity lab -------------------------- */

function updateSensitivityUI() {
  const r = state.result;
  if (!r) return;
  const slider = document.getElementById("sens-slider");
  const target = parseFloat(slider.value);
  document.getElementById("sens-value").textContent = target.toFixed(2) + "%";

  const base = r.criteria.weights;
  const idx = r.criteriaLabels.indexOf("Keandalan");
  if (idx < 0) {
    document.getElementById("sens-result").innerHTML = "";
    return;
  }

  // Rescale the remaining weights proportionally so the vector sums to 1.
  const t = target / 100;
  const restSum = base.reduce((a, w, i) => (i === idx ? a : a + w), 0);
  const newWeights = base.map((w, i) => (i === idx ? t : (w / restSum) * (1 - t)));

  const scores = r.alternatives.map((_, ai) =>
    r.criteriaLabels.reduce(
      (acc, c, j) => acc + newWeights[j] * r.alternativesReport[c].weights[ai], 0
    )
  );
  const order = scores
    .map((s, i) => ({ s, name: r.alternatives[i] }))
    .sort((a, b) => b.s - a.s);

  const original = r.ranking.map((x) => x.alternative).join(" → ");
  const now = order.map((x) => x.name).join(" → ");
  const flipped = original !== now;

  document.getElementById("sens-result").innerHTML = `
    <div class="stat-row"><span class="k">Bobot baru (K, S, B)</span>
      <span class="v">${newWeights.map((w) => (w * 100).toFixed(2) + "%").join(" · ")}</span></div>
    <div class="stat-row"><span class="k">Peringkat terekspos</span>
      <span class="v">${now}</span></div>
    <div class="stat-row"><span class="k">Pemenang</span>
      <span class="v ${flipped ? "bad" : "good"}">${order[0].name}${flipped ? " (BERUBAH!)" : " (tetap)"}</span></div>
  `;
}

/* ------------------------------ Report tab -------------------------------- */

let stackedChart = null;

function renderReportTab() {
  const r = state.result;
  if (!r) return;

  // Stacked horizontal bar: per-criterion contributions.
  const ctx = document.getElementById("chart-stacked");
  const labels = r.alternatives;
  const datasets = r.criteriaLabels.map((c) => ({
    label: c,
    data: labels.map((a) => +(r.contributions[a][c] * 100).toFixed(2)),
    backgroundColor: CRIT_COLORS[c] || "#64748b",
    borderRadius: 6,
    borderSkipped: false,
  }));

  if (stackedChart) {
    stackedChart.data.labels = labels;
    stackedChart.data.datasets = datasets;
    stackedChart.update();
  } else if (window.Chart) {
    stackedChart = new Chart(ctx, {
      type: "bar",
      data: { labels, datasets },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            stacked: true,
            max: 100,
            grid: { color: chartGridColor() },
            ticks: { color: chartTextColor(), callback: (v) => v + "%" },
          },
          y: {
            stacked: true,
            grid: { display: false },
            ticks: { color: chartTextColor(), font: { weight: "600" } },
          },
        },
        plugins: {
          legend: { position: "bottom", labels: { color: chartTextColor(), usePointStyle: true, padding: 16 } },
          tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.parsed.x}%` } },
        },
        animation: { duration: 900 },
      },
    });
  }

  // Synthesis table
  document.querySelector("#synthesis-table tbody").innerHTML = r.ranking
    .map(
      (row) => `
      <tr>
        <td class="mono">${row.rank}</td>
        <td class="name">${row.alternative}</td>
        <td class="mono" style="color:var(--emerald);font-weight:800">${(row.score * 100).toFixed(2)}%</td>
        ${r.criteriaLabels.map((c) => `<td class="mono">${(row.contributions[c] * 100).toFixed(2)}%</td>`).join("")}
        <td>${row.rank === 1 ? '<span class="chip gold">★ PILIHAN</span>' : `<span class="chip">${row.rank}</span>`}</td>
      </tr>`
    )
    .join("");

  const second = r.ranking[1];
  const margin = ((r.winnerScore - second.score) * 100).toFixed(2);
  document.getElementById("report-verdict").innerHTML = r.isConsistent
    ? `<strong>${r.winner}</strong> unggul dengan skor global <strong>${(r.winnerScore * 100).toFixed(2)}%</strong>, 
       unggul <strong>${margin} poin</strong> atas ${second.alternative}. Seluruh matriks lolos uji konsistensi (CR &lt; ${(r.threshold * 100).toFixed(0)}%) sehingga ranking ini <em>layak dijadikan dasar keputusan</em>.`
    : `Peringatan: <strong>${r.inconsistentMatrices.length} matriks</strong> melebihi ambang CR. Ranking tetap ditampilkan untuk analisis, tetapi <em>belum layak dijadikan keputusan final</em> sebelum penilaian ulang.`;
}

/* ==========================================================================
   6. Matrix building (interactive)
   ========================================================================== */

const cellRefs = {}; // containerId -> [{i, j, el}]

function buildMatrix(containerId, matrix, rowLabels, colLabels, onEdit) {
  const container = document.getElementById(containerId);
  const n = matrix.length;
  const refs = [];
  cellRefs[containerId] = refs;

  const table = document.createElement("table");
  table.className = "matrix";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.innerHTML = `<th></th>` + colLabels
    .map((c, j) => `<th>${c}</th>`)
    .join("");
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (let i = 0; i < n; i++) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.className = "row-head";
    th.textContent = rowLabels[i];
    tr.appendChild(th);

    for (let j = 0; j < n; j++) {
      const td = document.createElement("td");
      if (i === j) {
        const fixed = document.createElement("div");
        fixed.className = "cell-fixed diagonal";
        fixed.textContent = "1";
        td.appendChild(fixed);
      } else if (i < j) {
        const input = document.createElement("input");
        input.type = "text";
        input.className = "cell-input";
        input.value = formatFraction(matrix[i][j]);
        input.setAttribute("aria-label", `${rowLabels[i]} terhadap ${colLabels[j]}`);
        input.addEventListener("change", () => handleCellEdit(input, i, j, matrix, onEdit));
        input.addEventListener("keydown", (e) => {
          if (e.key === "Enter") input.blur();
        });
        td.appendChild(input);
        refs.push({ i, j, el: input, kind: "input" });
      } else {
        const fixed = document.createElement("div");
        fixed.className = "cell-fixed reciprocal";
        fixed.textContent = formatFraction(matrix[i][j]);
        td.appendChild(fixed);
        refs.push({ i, j, el: fixed, kind: "reciprocal" });
      }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  container.innerHTML = "";
  container.appendChild(table);
}

function handleCellEdit(input, i, j, matrix, onEdit) {
  const raw = input.value.trim().replace(",", ".");
  let value;

  if (raw.includes("/")) {
    const [a, b] = raw.split("/").map(Number);
    value = b ? a / b : NaN;
  } else {
    value = Number(raw);
  }

  if (!isFinite(value) || value <= 0 || value > 9) {
    input.classList.add("invalid");
    toast(`Nilai harus antara 1/9 (0.111) dan 9. Anda memasukkan "${raw}".`, "error");
    input.value = formatFraction(matrix[i][j]);
    setTimeout(() => input.classList.remove("invalid"), 1200);
    return;
  }

  input.classList.remove("invalid");
  onEdit(i, j, value);
}

function refreshMatrixCells(containerId) {
  const refs = cellRefs[containerId] || [];
  const n = refs.length ? Math.max(...refs.map((r) => Math.max(r.i, r.j))) + 1 : 0;
  refs.forEach(({ i, j, el, kind }) => {
    if (kind === "input") {
      // Avoid clobbering the field the user is actively typing in.
      if (document.activeElement !== el) {
        el.value = formatFraction(getMatrixValue(containerId, i, j));
      }
    } else if (kind === "reciprocal") {
      el.textContent = formatFraction(getMatrixValue(containerId, i, j));
    }
  });
  void n;
}

function getMatrixValue(containerId, i, j) {
  if (containerId === "criteria-matrix") return state.criteriaMatrix[i][j];
  return state.alternativeMatrices[state.activeCriterion][i][j];
}

/* ==========================================================================
   7. Formatting helpers
   ========================================================================== */

const FRACTIONS = [
  [1 / 9, "1/9"], [1 / 8, "1/8"], [1 / 7, "1/7"], [1 / 6, "1/6"], [1 / 5, "1/5"],
  [1 / 4, "1/4"], [1 / 3, "1/3"], [1 / 2, "1/2"], [1, "1"], [1.5, "3/2"],
  [2, "2"], [3, "3"], [4, "4"], [5, "5"], [6, "6"], [7, "7"], [8, "8"], [9, "9"],
];

function formatFraction(value) {
  const found = FRACTIONS.find(([v]) => Math.abs(v - value) < 1e-9);
  if (found) return found[1];
  return String(+value.toFixed(4));
}

function renderWeightBars(containerId, labels, weights, colors) {
  const container = document.getElementById(containerId);
  container.innerHTML = labels
    .map(
      (label, i) => `
      <div class="wb-row">
        <span class="wb-label">${label}</span>
        <div class="wb-track"><span class="wb-fill" style="width:0;background:linear-gradient(90deg,${colors[i] || "#ea1d22"},${colors[(i + 1) % colors.length] || "#2e7df7"})"></span></div>
        <span class="wb-value">${(weights[i] * 100).toFixed(2)}%</span>
      </div>`
    )
    .join("");
  // Animate after paint.
  requestAnimationFrame(() => {
    container.querySelectorAll(".wb-fill").forEach((fill, i) => {
      fill.style.width = (weights[i] * 100).toFixed(2) + "%";
    });
  });
}

function statRow(key, value, cls = "") {
  return `<div class="stat-row"><span class="k">${key}</span><span class="v ${cls}">${value}</span></div>`;
}

function mixColor(a, b, t) {
  const pa = hexToRgb(a), pb = hexToRgb(b);
  const r = Math.round(pa.r + (pb.r - pa.r) * t);
  const g = Math.round(pa.g + (pb.g - pa.g) * t);
  const bl = Math.round(pa.b + (pb.b - pa.b) * t);
  return `rgb(${r},${g},${bl})`;
}

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return {
    r: parseInt(h.slice(0, 2), 16),
    g: parseInt(h.slice(2, 4), 16),
    b: parseInt(h.slice(4, 6), 16),
  };
}

function getCssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function chartTextColor() { return getCssVar("--text-1") || "#a7b3cf"; }
function chartGridColor() { return getCssVar("--chart-grid") || "rgba(140,160,200,0.14)"; }

/* ==========================================================================
   8. Toasts & modal
   ========================================================================== */

function toast(message, kind = "info", duration = 3600) {
  const container = document.getElementById("toasts");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => {
    el.classList.add("out");
    setTimeout(() => el.remove(), 300);
  }, duration);
}

function openModal(title, html) {
  document.getElementById("modal-title").textContent = title;
  document.getElementById("modal-body").innerHTML = html;
  document.getElementById("modal").classList.remove("hidden");
}

function closeModal() {
  document.getElementById("modal").classList.add("hidden");
}

function docsHtml() {
  return `
    <h4>Apa itu AHP?</h4>
    <p>Analytic Hierarchy Process (Thomas L. Saaty, 1980) memecah keputusan kompleks menjadi hierarki: <strong>tujuan → kriteria → alternatif</strong>. Setiap tingkat dinilai dengan perbandingan berpasangan pada skala 1–9, lalu bobot diekstrak sebagai eigenvector prioritas.</p>

    <h4>Aksioma matriks berpasangan</h4>
    <ul>
      <li><code>A_ji = 1 / A_ij</code> (sifat resiprokal)</li>
      <li><code>A_ii = 1</code> (diagonal satuan)</li>
      <li>Semua entri positif pada skala Saaty 1/9 – 9</li>
    </ul>

    <h4>Metode ekstraksi bobot</h4>
    <ul>
      <li><strong>Column Sum</strong> — normalisasi tiap kolom, lalu rata-rata baris.</li>
      <li><strong>Geometric Mean</strong> — akar ke-n dari perkalian baris, dinormalisasi (pendekatan eigenvector utama).</li>
    </ul>

    <h4>Uji konsistensi</h4>
    <ul>
      <li><code>λ_i = (A·w)_i / w_i</code> dan <code>λ_max = rata-rata λ_i</code></li>
      <li><code>CI = (λ_max − n) / (n − 1)</code></li>
      <li><code>CR = CI / RI</code> — valid bila <strong>CR &lt; 10%</strong></li>
    </ul>

    <h4>Skala fundamental Saaty</h4>
    <ul class="scale-list">
      ${SAATY_SCALE.map(([v, label]) => `<li><strong>${v}</strong> — ${label}</li>`).join("")}
      <li><strong>1/2 … 1/9</strong> — nilai resiprocal (bila A lebih rendah dari B)</li>
    </ul>

    <h4>Sintesis global</h4>
    <p>Skor akhir tiap alternatif: <code>P_i = Σ_j w_j · s_ij</code>, diurutkan menurun untuk memperoleh peringkat.</p>

    <h4>Studi kasus ini</h4>
    <p>Pemilihan mobil armada keluarga atas 3 kriteria (Keandalan, Style, Konsumsi BB) dan 4 alternatif (Avanza, Xenia, Ertiga, Grand Livina). Matriks kriteria memiliki CR ≈ 1.58% (konsisten).</p>
  `;
}

/* ==========================================================================
   9. Settings persistence
   ========================================================================== */

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    Object.assign(state.settings, saved);
    if (saved.method) state.method = saved.method;
    if (saved.threshold) state.threshold = saved.threshold;
  } catch (_) { /* corrupted storage — keep defaults */ }
  applySettings();
}

function saveSettings() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({ ...state.settings, method: state.method, threshold: state.threshold })
  );
}

function applySettings() {
  document.documentElement.dataset.theme = state.settings.theme;
  document.getElementById("theme-toggle").textContent = state.settings.theme === "dark" ? "☾" : "☀";
  document.body.classList.toggle("reduce-motion", state.settings.reduceMotion);
  document.body.classList.toggle("no-cursor-3d", !state.settings.cursor3d);

  document.getElementById("setting-threshold").value = state.threshold;
  document.getElementById("setting-method").value = state.method;
  document.getElementById("setting-cursor").checked = state.settings.cursor3d;
  document.getElementById("setting-reduce-motion").checked = state.settings.reduceMotion;

  document.querySelectorAll("#theme-segmented button").forEach((b) =>
    b.classList.toggle("active", b.dataset.themeSet === state.settings.theme)
  );

  if (window.Cursor3D) {
    window.Cursor3D.setEnabled(state.settings.cursor3d);
    window.Cursor3D.setReducedMotion(state.settings.reduceMotion);
  }
}

/* ==========================================================================
   10. Tab navigation
   ========================================================================== */

function activateTab(name) {
  document.querySelectorAll(".tab-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name)
  );
  document.querySelectorAll(".tab-panel").forEach((p) =>
    p.classList.toggle("active", p.id === `tab-${name}`)
  );

  // Charts/gauge need a resize nudge when their tab becomes visible.
  requestAnimationFrame(() => {
    if (name === "consistency" && state.result) {
      drawGauge(state.result.criteria.cr, state.threshold);
    }
    // Chart.js cannot measure a container while its tab is display:none;
    // force a re-measure once the tab is visible.
    if (name === "report" && stackedChart) stackedChart.resize();
    if (name === "dashboard" && doughnutChart) doughnutChart.resize();
  });
}

/* ==========================================================================
   11. Export helpers
   ========================================================================== */

function downloadFile(filename, content, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function buildCsv() {
  const r = state.result;
  const lines = [];
  const esc = (v) => `"${String(v).replace(/"/g, '""')}"`;

  lines.push([esc("AutoAHP - Sistem Pendukung Keputusan AHP")].join(","));
  lines.push([esc("Goal"), esc(DEFAULT_MODEL.goal)].join(","));
  lines.push([esc("Metode prioritas"), esc(state.method)].join(","));
  lines.push([esc("CR threshold"), state.threshold.toFixed(2)].join(","));
  lines.push("");

  const block = (title, labels, blk) => {
    lines.push([esc(title), ...labels.map(esc)].join(","));
    lines.push([esc("Bobot"), ...blk.weights.map((w) => w.toFixed(6))].join(","));
    lines.push([
      esc("Statistik"), esc(`lambda_max=${blk.lambdaMax.toFixed(6)}`),
      esc(`CI=${blk.ci.toFixed(6)}`), esc(`RI=${blk.ri.toFixed(4)}`),
      esc(`CR=${blk.cr.toFixed(6)}`), esc(blk.isConsistent ? "KONSISTEN" : "TIDAK KONSISTEN"),
    ].join(","));
    lines.push("");
  };

  block("Kriteria", r.criteriaLabels, r.criteria);
  r.criteriaLabels.forEach((c) => block(`Alternatif [${c}]`, r.alternatives, r.alternativesReport[c]));

  lines.push([esc("Rank"), esc("Alternatif"), esc("Skor Global"), ...r.criteriaLabels.map(esc)].join(","));
  r.ranking.forEach((row) => {
    lines.push([
      row.rank, esc(row.alternative), row.score.toFixed(6),
      ...r.criteriaLabels.map((c) => row.contributions[c].toFixed(6)),
    ].join(","));
  });

  return lines.join("\n");
}

function buildJson() {
  const r = state.result;
  return JSON.stringify(
    {
      generatedBy: "AutoAHP v1.0.0",
      generatedAt: new Date().toISOString(),
      goal: DEFAULT_MODEL.goal,
      method: state.method,
      crThreshold: state.threshold,
      criteriaLabels: state.criteriaLabels,
      alternatives: state.alternatives,
      criteriaMatrix: state.criteriaMatrix,
      alternativeMatrices: state.alternativeMatrices,
      result: {
        criteriaWeights: r.criteria.weights,
        criteriaCR: r.criteria.cr,
        lambdaMax: r.criteria.lambdaMax,
        ci: r.criteria.ci,
        ri: r.criteria.ri,
        isConsistent: r.isConsistent,
        globalScores: r.globalScores,
        ranking: r.ranking,
        winner: r.winner,
        winnerScore: r.winnerScore,
      },
    },
    null,
    2
  );
}

/* ==========================================================================
   12. Bootstrapping & event wiring
   ========================================================================== */

function wireEvents() {
  // Tabs
  document.querySelectorAll(".tab-btn").forEach((btn) =>
    btn.addEventListener("click", () => activateTab(btn.dataset.tab))
  );

  // Theme toggle (header)
  document.getElementById("theme-toggle").addEventListener("click", () => {
    state.settings.theme = state.settings.theme === "dark" ? "light" : "dark";
    applySettings();
    saveSettings();
    // Re-render so charts pick up the new palette.
    if (state.result) {
      if (doughnutChart) { doughnutChart.destroy(); doughnutChart = null; }
      if (stackedChart) { stackedChart.destroy(); stackedChart = null; }
      renderDashboard();
      renderReportTab();
      drawGauge(state.result.criteria.cr, state.threshold);
    }
  });

  // Docs modal
  document.getElementById("btn-docs").addEventListener("click", () =>
    openModal("Dokumentasi Ilmiah — AHP (Saaty)", docsHtml())
  );
  document.querySelectorAll("#modal [data-close]").forEach((el) =>
    el.addEventListener("click", closeModal)
  );
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });

  // Reset buttons
  document.getElementById("btn-reset-criteria").addEventListener("click", () => {
    state.criteriaMatrix = clone(DEFAULT_MODEL.criteriaMatrix);
    renderCriteriaTab();
    requestRecalc(true);
    toast("Matriks kriteria dikembalikan ke nilai default.", "success");
  });

  document.getElementById("btn-reset-alt").addEventListener("click", () => {
    state.alternativeMatrices[state.activeCriterion] = clone(
      DEFAULT_MODEL.alternativeMatrices[state.activeCriterion]
    );
    renderAlternativesTab();
    requestRecalc(true);
    toast(`Matriks ${state.activeCriterion} dikembalikan ke default.`, "success");
  });

  document.getElementById("btn-reset-all").addEventListener("click", () => {
    state.criteriaMatrix = clone(DEFAULT_MODEL.criteriaMatrix);
    state.alternativeMatrices = clone(DEFAULT_MODEL.alternativeMatrices);
    state.method = "column_sum";
    state.threshold = 0.1;
    state.activeCriterion = "Keandalan";
    applySettings();
    saveSettings();
    renderCriteriaTab();
    renderAlternativesTab();
    requestRecalc(true);
    toast("Semua matriks dikembalikan ke studi kasus default.", "success");
  });

  // Settings
  document.querySelectorAll("#theme-segmented button").forEach((b) =>
    b.addEventListener("click", () => {
      state.settings.theme = b.dataset.themeSet;
      applySettings();
      saveSettings();
      if (state.result) {
        if (doughnutChart) { doughnutChart.destroy(); doughnutChart = null; }
        if (stackedChart) { stackedChart.destroy(); stackedChart = null; }
        renderDashboard();
        renderReportTab();
        drawGauge(state.result.criteria.cr, state.threshold);
      }
    })
  );

  document.getElementById("setting-threshold").addEventListener("change", (e) => {
    const v = parseFloat(e.target.value);
    if (!isFinite(v) || v <= 0 || v > 1) {
      toast("Ambang CR harus antara 0.01 dan 0.5.", "error");
      e.target.value = state.threshold;
      return;
    }
    state.threshold = v;
    saveSettings();
    requestRecalc(true);
    toast(`Ambang CR diatur ke ${(v * 100).toFixed(0)}%.`, "success");
  });

  document.getElementById("setting-method").addEventListener("change", (e) => {
    state.method = e.target.value;
    saveSettings();
    requestRecalc(true);
    toast(`Metode kalkulasi: ${e.target.value === "column_sum" ? "Column Sum" : "Geometric Mean"}.`, "success");
  });

  document.getElementById("setting-cursor").addEventListener("change", (e) => {
    state.settings.cursor3d = e.target.checked;
    applySettings();
    saveSettings();
  });

  document.getElementById("setting-reduce-motion").addEventListener("change", (e) => {
    state.settings.reduceMotion = e.target.checked;
    applySettings();
    saveSettings();
  });

  // Sensitivity slider
  document.getElementById("sens-slider").addEventListener("input", updateSensitivityUI);

  // Report actions
  document.getElementById("btn-print").addEventListener("click", () => window.print());

  document.getElementById("btn-export-csv").addEventListener("click", async () => {
    // Prefer the backend export (guaranteed identical to the engine)…
    try {
      const res = await fetch("/api/export/csv?method=" + state.method);
      if (!res.ok) throw new Error("no backend");
      const text = await res.text();
      downloadFile("autoahp-synthesis.csv", text, "text/csv;charset=utf-8");
      toast("CSV diunduh dari backend.", "success");
    } catch (_) {
      // …otherwise generate client-side so GitHub Pages still works.
      downloadFile("autoahp-synthesis.csv", "\uFEFF" + buildCsv(), "text/csv;charset=utf-8");
      toast("CSV dihasilkan lokal (mode offline).", "success");
    }
  });

  document.getElementById("btn-export-json").addEventListener("click", async () => {
    try {
      const res = await fetch("/api/export/json?method=" + state.method);
      if (!res.ok) throw new Error("no backend");
      const text = await res.text();
      downloadFile("autoahp-model.json", text, "application/json");
      toast("JSON diunduh dari backend.", "success");
    } catch (_) {
      downloadFile("autoahp-model.json", buildJson(), "application/json");
      toast("JSON dihasilkan lokal (mode offline).", "success");
    }
  });
}

async function probeBackend() {
  try {
    const data = await API.get("/api/health", 1500);
    if (data && data.ok) {
      markServer(true);
      return true;
    }
  } catch (_) { /* offline / Pages */ }
  markServer(false);
  return false;
}

async function boot() {
  loadSettings();
  wireEvents();
  renderCriteriaTab();
  renderAlternativesTab();

  await probeBackend();
  await recalculate();

  // Small greeting so the UX feedback loop is visible immediately.
  const r = state.result;
  toast(
    `Analisis AHP siap — rekomendasi: ${r.winner} (${(r.winnerScore * 100).toFixed(2)}%).`,
    "success",
    4200
  );
  if (!r.isConsistent) {
    toast(
      `${r.inconsistentMatrices.length} matriks melebihi ambang CR — perhatikan tab Lab Konsistensi.`,
      "warn",
      5200
    );
  }
}

document.addEventListener("DOMContentLoaded", boot);
