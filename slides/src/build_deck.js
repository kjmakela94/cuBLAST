// Build the 5-slide results deck from an exported run (summary.json + data/*.csv + data/B1_structure.json).
// Usage: python3 make_matrix_figs.py <results_dir> figs && node build_deck.js <results_dir> <out.pptx>
// Requires: npm install (pptxgenjs, react, react-dom, react-icons, sharp); Python 3 with matplotlib + Pillow.
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const pptxgen = require("pptxgenjs");
const React = require("react");
const RDS = require("react-dom/server");
const sharp = require("sharp");
const md = require("react-icons/md");
const bs = require("react-icons/bs");
const tb = require("react-icons/tb");

const RES = process.argv[2] || "results";
const OUT = process.argv[3] || "GPU_kernels_hackathon.pptx";
const FIGS = path.join(__dirname, "figs");
const PY = process.env.PYTHON || "python3";

// ───────────────────────────── data ─────────────────────────────
const S = JSON.parse(fs.readFileSync(path.join(RES, "summary.json"), "utf8").replace(/\bNaN\b/g, "null"));
function csv(name) {
  const [head, ...rows] = fs.readFileSync(path.join(RES, "data", name), "utf8").trim().split(/\r?\n/);
  const keys = head.split(",");
  return rows.map(r => Object.fromEntries(r.split(",").map((v, i) => [keys[i], isNaN(+v) || v === "" ? v : +v])));
}
const GEMM = csv("A2_gemm_sweep.csv"), CG = csv("C1_cg_sweep.csv");
const SPMV = S.spmv, REAL = S.cg_real, ENV = S.env, BEST = S.gemm_best_tflops;
const ACC = Object.fromEntries(S.gemm_backends.map(r => [r.precision, r.rel_error]));
const STRUCT = JSON.parse(fs.readFileSync(path.join(RES, "data", "B1_structure.json"), "utf8"));
const SYNTH = SPMV.some(r => /SYNTH/.test(r.source));
const st = m => (/SYNTH/.test(m.source || "") ? "*" : "");   // per-matrix footnote marker

const fmt = (v, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const nlabel = n => n >= 1e6 ? `${+(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}k` : `${n}`;
const fx = v => (v < 10 ? `${v.toFixed(1)}×` : `${Math.round(v)}×`);
const sci = v => { const e = Math.floor(Math.log10(v)); return `${(v / 10 ** e).toFixed(1)}×10${String(e).replace("-", "⁻").replace(/\d/g, d => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d])}`; };
const range = (arr, f = Math.round) => `${f(Math.min(...arr))}–${f(Math.max(...arr))}`;

const peak64 = S.roofline.peak64_gflops, BW = S.roofline.stream_bw, ridge = S.roofline.ridge_flop_per_byte;
const xk = S.cg_crossover.kernel_only.n, big = CG[CG.length - 1];
const small = CG.filter(r => r.n <= 262144);
const tau0 = small.reduce((a, r) => a + r.gpu_us_it, 0) / small.length;
const pctOfDgemm = SPMV.map(r => 100 * r.gpu_GFLOPs / peak64);
const g64 = GEMM.filter(r => r.precision === "FP64");

// ───────────────────────────── theme & layouts ─────────────────────────────
const THEME = {
  name: "GPU Kernels", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "16202E", lt1: "FFFFFF", dk2: "2B3A4F", lt2: "F1F4F8", accent1: "76B900", accent2: "3B82F6", accent3: "E6A817",
    accent4: "D64545", accent5: "9AA5B4", accent6: "5B6577", hlink: "3B82F6", folHlink: "2B3A4F" },
};
const H = THEME.colors;
const PREC_HEX = { FP64: H.dk2, FP32: H.accent2, TF32: H.accent3, FP16: H.accent1 };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "GPU kernels on real matrices";
pres.author = "Numerical Libraries Mini-Hackathon";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: H.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.75, w: 7.6, h: 1.7, fontFace: "Cambria", fontSize: 46, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.65, w: 7.3, h: 1.2, fontSize: 20, color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: H.lt1 },
  objects: [
    { text: { text: "Numerical Libraries Mini-Hackathon  ·  GPU kernels on real matrices", options: { x: 0.6, y: 7.02, w: 8, h: 0.28, fontSize: 10, color: C.accent6, margin: 0 } } },
    { placeholder: { options: { name: "label", type: "body", x: 0.6, y: 0.32, w: 8, h: 0.3, fontSize: 12, bold: true, color: C.accent1, charSpacing: 2, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.62, w: 12.1, h: 0.72, fontFace: "Cambria", fontSize: 34, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "kicker", type: "body", x: 0.6, y: 1.36, w: 12.1, h: 0.42, fontSize: 16, color: C.accent6, valign: "top", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.2, y: 7.02, w: 0.5, h: 0.28, fontSize: 10, color: "5B6577", align: "right" },
});

// ───────────────────────────── helpers ─────────────────────────────
async function iconPng(Icon) {
  const svg = RDS.renderToStaticMarkup(React.createElement(Icon, { color: "#FFFFFF", size: "256" }));
  return "image/png;base64," + (await sharp(Buffer.from(svg)).png().toBuffer()).toString("base64");
}
function iconCircle(s, data, x, y, d, fill, name) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" }, objectName: name + " circle" });
  const p = d * 0.22;
  s.addImage({ data, x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: name });
}
function card(s, x, y, w, h, fill, name) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { type: "none" }, objectName: name });
}
function txt(s, text, o) { s.addText(text, { isTextBox: true, margin: 0, color: C.text1, valign: "top", ...o }); }
function content(label, title, kicker, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(label, { placeholder: "label" });
  s.addText(title, { placeholder: "title" });
  s.addText(kicker, { placeholder: "kicker" });
  return s;
}
const AX = () => ({
  catAxisLabelColor: "5B6577", valAxisLabelColor: "5B6577", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 12, valGridLine: { color: "E3E8F0", size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineColor: "C9D1DE", valAxisLineShow: false,
  showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: "2B3A4F",
  valAxisTitleColor: "5B6577", catAxisTitleColor: "5B6577", valAxisTitleFontFace: "+mn-lt", catAxisTitleFontFace: "+mn-lt",
  valAxisTitleFontSize: 12, catAxisTitleFontSize: 12, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: "2B3A4F",
});

(async () => {
  const I = { grid: await iconPng(bs.BsGrid3X3Gap), sparse: await iconPng(tb.TbGridDots), loop: await iconPng(md.MdLoop) };

  // ════════════════ 1 · Title ════════════════
  pres.addSection({ title: "Introduction" });
  {
    const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
    s.addText("GPU kernels on real matrices", { placeholder: "title" });
    s.addText("What limits dense GEMM, sparse SpMV and conjugate gradient on a GPU — and when the GPU actually pays off", { placeholder: "body" });
    txt(s, `${ENV.gpu.replace("NVIDIA ", "")}  ·  CUDA ${ENV.cuda_runtime}  ·  CuPy ${ENV.cupy}     |     Numerical Libraries Mini-Hackathon`,
      { x: 0.7, y: 5.55, w: 7.6, h: 0.35, fontSize: 14, color: C.accent5 });
    [[I.grid, "A", "cuBLAS GEMM", "FP64 · FP32 · TF32 · FP16 on tensor cores"],
     [I.sparse, "B", "cuSPARSE SpMV", "three sparse matrices vs one CPU core"],
     [I.loop, "C", "Conjugate gradient", "SpMV + dot + axpy · where the GPU wins"]].forEach(([ic, k, h1, h2], i) => {
      const y = 1.75 + i * 1.3;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.75, y, w: 3.95, h: 1.05, rectRadius: 0.12, fill: { color: "223049" }, line: { type: "none" }, objectName: `part ${k} card` });
      iconCircle(s, ic, 8.95, y + 0.2, 0.65, C.accent1, `part ${k} icon`);
      txt(s, [{ text: `${k}  `, options: { color: C.accent1, bold: true } }, { text: h1, options: { bold: true, color: C.background1 } }], { x: 9.8, y: y + 0.17, w: 2.8, h: 0.36, fontSize: 18 });
      txt(s, h2, { x: 9.8, y: y + 0.55, w: 2.8, h: 0.4, fontSize: 12, color: C.accent5 });
    });
    s.addNotes("Three kernels with very different amounts of arithmetic per byte, one GPU, one CPU core as the baseline. The thread through the talk: arithmetic intensity tells you which hardware ceiling applies, and therefore what to optimise. All timings are warmed up, synchronised with CUDA events, medians, and say whether host-device copies are included.");
  }

  // ════════════════ 2 · Problem: what we multiply ════════════════
  {
    const s = content("PROBLEM", "What we multiply, and what could limit it",
      "Five inputs, three kernels — is each one limited by arithmetic, by memory bandwidth, or by latency?", "Introduction");
    const info = Object.fromEntries(STRUCT.map(m => [m.matrix, m]));
    const notes0 = null; const notes = { cant: "banded, FEM-like · SPD", ecology2: "2-D grid stencil · SPD", "webbase-1M": "web graph · power-law rows",
      s3dkq4m2: "FEM cylindrical shell · SPD", memplus: "memory circuit · irregular" };
    const tiles = [
      { part: "A · GEMM", fill: C.accent1, img: "dense", name: "Dense n × n", l1: `n = ${g64[0].n} … ${fmt(GEMM[GEMM.length - 1].n)}`, l2: "every entry stored" },
      ...STRUCT.map(m => ({ part: "B · SpMV", fill: C.accent3, img: m.matrix, zoom: `${m.matrix}_zoom`, name: m.matrix + st(m),
        l1: `${nlabel(m.n)} rows · ${fmt(m.nnz_per_row, m.nnz_per_row < 10 ? 1 : 0)} nnz/row`, l2: notes[m.matrix] || (m.spd ? "SPD" : "unsymmetric") })),
      { part: "C · CG", fill: C.accent2, img: "poisson", name: "2-D Poisson", l1: `n = ${nlabel(CG[0].n)} … ${nlabel(big.n)}`, l2: "5-point stencil · SPD" },
    ];
    tiles.forEach((t, i) => {
      const x = 0.6 + i * 2.45, y = 1.95, w = 2.3;
      card(s, x, y, w, 3.7, C.background2, `${t.name} card`);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.2, y: y + 0.18, w: 1.15, h: 0.3, rectRadius: 0.08, fill: { color: t.fill }, line: { type: "none" }, objectName: `${t.name} tag` });
      txt(s, t.part, { x: x + 0.2, y: y + 0.18, w: 1.15, h: 0.3, fontSize: 11, bold: true, color: C.background1, align: "center", valign: "middle" });
      s.addImage({ path: path.join(FIGS, `${t.img}.png`), x: x + 0.2, y: y + 0.6, w: 1.9, h: 1.9, objectName: `${t.name} structure` });
      if (t.zoom) {
        s.addImage({ path: path.join(FIGS, `${t.zoom}.png`), x: x + 1.27, y: y + 0.66, w: 0.78, h: 0.78, objectName: `${t.name} zoom` });
      }
      txt(s, t.name, { x: x + 0.2, y: y + 2.62, w: w - 0.35, h: 0.34, fontSize: 16, bold: true, fontFace: "Cambria" });
      txt(s, t.l1, { x: x + 0.2, y: y + 2.98, w: w - 0.3, h: 0.28, fontSize: 13 });
      txt(s, t.l2, { x: x + 0.2, y: y + 3.27, w: w - 0.3, h: 0.28, fontSize: 12, color: C.accent6 });
    });
    txt(s, (SYNTH ? "* synthetic stand-in with the same size and sparsity class (real file unavailable on the run node)  ·  " : "") +
      `insets: first ${fmt(STRUCT[0].zoom_rows)} rows`, { x: 0.6, y: 5.72, w: 12.1, h: 0.26, fontSize: 11, italic: true, color: C.accent6 });
    card(s, 0.6, 6.08, 12.1, 0.72, "EAF4D9", "intensity band");
    txt(s, [
      { text: "FLOP per byte decides the limit:   ", options: { bold: true } },
      { text: "GEMM " }, { text: "grows with n", options: { bold: true, color: "4E7D00" } },
      { text: "  ·  SpMV " }, { text: "≤ 1/6", options: { bold: true, color: "B07A00" } },
      { text: "  ·  CG ≈ 0.11 + latency  ·  ridge of this GPU " },
      { text: `≈ ${ridge.toFixed(1)}`, options: { bold: true } },
    ], { x: 0.85, y: 6.08, w: 11.7, h: 0.72, fontSize: 15, valign: "middle" });
    s.addNotes(`Left to right: the dense matrices for GEMM; the three sparse matrices for SpMV, drawn as non-zero density maps with an inset of the first rows so you can see the local structure — a wide band, a five-point stencil, and an irregular power-law pattern; and the 2-D Poisson problem we scale up for CG. The question for each: how many floating-point operations per byte does it need? GEMM reuses every element n times, SpMV uses each stored value once — 2 FLOP for 12 bytes. This GPU needs about ${ridge.toFixed(1)} FLOP per byte before arithmetic becomes the limit, because its FP64 units run at 1/64 of FP32.${SYNTH ? " Note: the sparse matrices are synthetic stand-ins because the SuiteSparse download failed on the run node." : ""}`);
  }

  // ════════════════ 3 · Part A: GEMM ════════════════
  pres.addSection({ title: "Results" });
  {
    const s = content("PART A · cuBLAS", "GEMM is compute-bound: precision is the lever",
      `cublasGemmEx, n × n × n, data resident on the GPU — ${fmt(BEST.FP16, 0)} TFLOP/s in FP16 vs ${fmt(BEST.FP64, 1)} TFLOP/s in FP64`, "Results");
    const precs = ["FP16", "TF32", "FP32", "FP64"], sizes = [...new Set(GEMM.map(r => r.n))];
    s.addChart(pres.charts.LINE, precs.map(p => ({ name: p, labels: sizes.map(nlabel),
      values: sizes.map(n => { const r = GEMM.find(g => g.precision === p && g.n === n); return r ? r.gflops / 1e3 : null; }) })), {
      x: 0.6, y: 1.95, w: 7.5, h: 4.85, ...AX(), chartColors: precs.map(p => PREC_HEX[p]), lineSize: 2.75, lineDataSymbol: "circle", lineDataSymbolSize: 8,
      valAxisLogScaleBase: 10, valAxisMinVal: 0.1, valAxisMaxVal: 1000, valAxisLabelFormatCode: "General",
      showValAxisTitle: true, valAxisTitle: "TFLOP/s  (log scale)", showCatAxisTitle: true, catAxisTitle: "matrix size n" });
    txt(s, `Precision ladder at n = ${fmt(sizes[sizes.length - 1])}`, { x: 8.45, y: 1.98, w: 4.2, h: 0.3, fontSize: 14, bold: true, color: C.accent6 });
    const hdr = t => ({ text: t, options: { bold: true, color: "5B6577", fontSize: 12 } });
    const rows = [[hdr("Precision"), hdr("TFLOP/s"), hdr("vs FP64"), hdr("rel. error")]];
    for (const p of ["FP64", "FP32", "TF32", "FP16"]) rows.push([
      { text: p, options: { bold: true, color: PREC_HEX[p], fontSize: 15 } },
      { text: fmt(BEST[p], BEST[p] < 10 ? 2 : 0), options: { bold: true, fontSize: 15 } },
      { text: p === "FP64" ? "1×" : `${Math.round(BEST[p] / BEST.FP64)}×`, options: { fontSize: 15 } },
      { text: ACC[p] > 0 ? sci(ACC[p]) : "reference", options: { fontSize: 15, color: ACC[p] > 0 ? "16202E" : "5B6577" } }]);
    s.addTable(rows, { x: 8.45, y: 2.35, w: 4.25, colW: [1.05, 1.0, 0.9, 1.3], rowH: 0.42, fontFace: "Calibri", color: "16202E",
      border: { type: "solid", pt: 0.75, color: "E3E8F0" }, fill: { color: "FFFFFF" }, valign: "middle", margin: [0, 0.06, 0, 0.06] });
    const floor = GEMM.filter(r => (r.precision === "TF32" || r.precision === "FP16") && r.n <= 512).map(r => r.ms * 1e3);
    const fl = Math.round(floor.reduce((a, b) => a + b) / floor.length);
    const bul = [
      `Small n is launch-bound: TF32/FP16 calls with n\u00A0≤\u00A0512 all take ≈ ${fl} µs`,
      `Tensor cores: TF32 ${(BEST.TF32 / BEST.FP32).toFixed(1)}× and FP16 ${(BEST.FP16 / BEST.FP32).toFixed(1)}× over FP32, for ≈ 3 fewer correct digits`,
      `FP64 runs at the consumer 1:64 rate, so SGEMM is ${Math.round(BEST.FP32 / BEST.FP64)}× faster than DGEMM`,
    ];
    txt(s, bul.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < bul.length - 1 } })), { x: 8.45, y: 4.65, w: 4.25, h: 2.1, fontSize: 14, paraSpaceAfter: 8 });
    s.addNotes(`Throughput rises with n until the tensor cores saturate around n = 4096. Below n ≈ 512 every call costs about ${fl} µs regardless of size — launch latency, the same effect that later decides CG. Precision is the big lever: FP16 with FP32 accumulation reaches ${fmt(BEST.FP16, 0)} TFLOP/s, ${Math.round(BEST.FP16 / BEST.FP64)}× DGEMM, at a relative error of about 4e-4 instead of 1e-7 in FP32. A timing pitfall worth mentioning: without synchronising, the same 4096³ GEMM appeared to run at ${fmt(S.pitfalls.rows[1]["TFLOP/s"], 0)} TFLOP/s — the clock had only measured the launch.`);
  }

  // ════════════════ 4 · Part B: SpMV + roofline ════════════════
  {
    const s = content("PART B · cuSPARSE  +  ROOFLINE", "SpMV is bandwidth-bound, GEMM compute-bound",
      `cuSPARSE SpMV reaches ${fmt(Math.min(...SPMV.map(r => r.gpu_GBs)) / 1e3, 2)}–${fmt(Math.max(...SPMV.map(r => r.gpu_GBs)) / 1e3, 2)} TB/s (${range(SPMV.map(r => r.pct_stream))}% of measured bandwidth) and is ${range(SPMV.map(r => r.speedup))}× faster than one CPU core`, "Results");
    const labels = SPMV.map(r => r.matrix + st(r));
    s.addChart([
      { type: pres.charts.BAR, data: [
        { name: "CPU 1 core", labels, values: SPMV.map(r => r.cpu_GBs) },
        { name: "GPU cold", labels, values: SPMV.map(r => r.gpu_GBs) },
        { name: "GPU warm", labels, values: SPMV.map(r => r.gpu_warm_GBs) }],
        options: { chartColors: [H.accent5, H.accent1, "B9DC6E"], barGrouping: "clustered", barGapWidthPct: 60, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0" } },
      { type: pres.charts.LINE, data: [{ name: `STREAM ${fmt(BW, 0)} GB/s`, labels, values: SPMV.map(() => BW) }],
        options: { chartColors: [H.dk2], lineSize: 2, lineDash: "dash", lineDataSymbol: "none" } },
    ], { x: 0.6, y: 1.95, w: 6.0, h: 4.45, ...AX(), barDir: "col", valAxisMinVal: 0, valAxisMaxVal: 2000, valAxisMajorUnit: 500, valAxisLabelFormatCode: "#,##0",
      showValAxisTitle: true, valAxisTitle: "achieved bandwidth  [GB/s]", catAxisLabelFontSize: 13, legendFontSize: 11,
      showTitle: true, title: "SpMV bandwidth", titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: "2B3A4F" });
    // roofline (scatter, post-processed to log-log)
    const xs = [0.01, ridge, 1e4, ...g64.map(r => r.n / 12), ...SPMV.map(r => 2 * r.nnz / (r.bytes_MB * 2 ** 20))];
    const N = xs.length, blank = () => Array(N).fill(null);
    const roof = blank(); roof[0] = 0.01 * BW; roof[1] = peak64; roof[2] = peak64;
    const dg = blank(); g64.forEach((r, i) => { dg[3 + i] = r.gflops; });
    const spm = SPMV.map((r, j) => { const v = blank(); v[3 + g64.length + j] = r.gpu_GFLOPs; return { name: `SpMV ${r.matrix}${st(r)}`, values: v }; });
    s.addChart(pres.charts.SCATTER, [{ name: "X", values: xs }, { name: "Roof (measured)", values: roof }, { name: "DGEMM", values: dg }, ...spm], {
      x: 6.85, y: 1.95, w: 5.85, h: 4.45, ...AX(), chartColors: [H.dk2, H.accent1, H.accent3, "C77700", "F2C14E"],
      lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 10,
      valAxisLogScaleBase: 10, valAxisMinVal: 10, valAxisMaxVal: 10000, valAxisLabelFormatCode: "General", catAxisMinVal: 0.01, catAxisMaxVal: 10000,
      showValAxisTitle: true, valAxisTitle: "GFLOP/s (FP64, log)", showCatAxisTitle: true, catAxisTitle: "FLOP per byte (log)",
      showTitle: true, title: `Roofline: min(${fmt(peak64 / 1e3, 2)} TFLOP/s, I × ${fmt(BW / 1e3, 2)} TB/s)`, titleFontFace: "+mn-lt", titleFontSize: 14, titleColor: "2B3A4F" });
    txt(s, [
      { text: "Reading: ", options: { bold: true } },
      { text: `SpMV sits at ≈ 0.1 FLOP/B, left of the ridge (${ridge.toFixed(1)}): most of the bandwidth, only ${range(pctOfDgemm)}% of DGEMM's FLOP/s — fewer bytes help, faster arithmetic does not.` },
    ], { x: 0.6, y: 6.5, w: 12.1, h: 0.3, fontSize: 12, color: C.accent6 });
    s.addNotes(`Left: with the L2 cache flushed, cuSPARSE moves ${range(SPMV.map(r => r.gpu_GBs))} GB/s; one CPU core manages ${range(SPMV.map(r => r.cpu_GBs))}. The irregular web-graph matrix hurts the CPU most, so the speed-up is largest there. Warm-cache bars exceed DRAM speed because the matrices fit in the ${Math.round(ENV.l2_mb)} MB L2 — the cold numbers are the honest ones. Right: the roofline puts both parts of the talk on one chart. DGEMM points climb onto the flat compute roof as n grows; SpMV points sit far left on the sloped bandwidth roof. So the lever for GEMM is precision, and the lever for SpMV is bytes: compressed indices, lower-precision values, or multiplying several vectors at once.${SYNTH ? " Matrices marked with a star are synthetic stand-ins." : ""}`);
  }

  // ════════════════ 5 · Part C: CG + takeaway ════════════════
  {
    const s = content("PART C · CONJUGATE GRADIENT", "CG: the GPU wins above n ≈ 10⁴",
      `2-D Poisson, ${S.cg_crossover.iters} iterations · cuSPARSE SpMV + cuBLAS dot/axpy vs SciPy/NumPy on one core`, "Results");
    const labels = CG.map(r => nlabel(r.n));
    s.addChart(pres.charts.LINE, [
      { name: "CPU, 1 core", labels, values: CG.map(r => r.cpu_us_it) },
      { name: "GPU, kernels only", labels, values: CG.map(r => r.gpu_us_it) },
      { name: "GPU, with copies", labels, values: CG.map(r => r.e2e_us_it) },
    ], { x: 0.6, y: 1.95, w: 7.1, h: 3.9, ...AX(), chartColors: [H.dk2, H.accent1, H.accent3], lineSize: 2.75, lineDataSymbol: "circle", lineDataSymbolSize: 8,
      valAxisLogScaleBase: 10, valAxisMinVal: 1, valAxisMaxVal: 1000000, valAxisLabelFormatCode: "#,##0",
      showValAxisTitle: true, valAxisTitle: "µs per iteration (log)", showCatAxisTitle: true, catAxisTitle: "unknowns n" });
    card(s, 8.0, 1.95, 4.7, 1.6, C.background2, "crossover card");
    txt(s, `n* ≈ ${(xk / 1e4).toFixed(2)} × 10⁴`, { x: 8.25, y: 2.08, w: 4.3, h: 0.6, fontSize: 30, bold: true, fontFace: "Cambria", color: C.accent1 });
    txt(s, `GPU iterations cost a flat ≈ ${Math.round(tau0)} µs (launch + sync latency) until n ≈ 250k; one core streams ≈ 24 GB/s. Equating the two predicts 1.1 × 10⁴.`,
      { x: 8.25, y: 2.7, w: 4.3, h: 0.8, fontSize: 13 });
    txt(s, "Copies: speed-up, kernels only → with copies", { x: 8.0, y: 3.72, w: 4.7, h: 0.3, fontSize: 13, bold: true, color: C.accent6 });
    REAL.forEach((r, i) => {
      const x = 8.0 + i * 2.4;
      card(s, x, 4.08, 2.3, 1.77, C.background2, `${r.matrix} card`);
      txt(s, [{ text: r.matrix + st(r), options: { bold: true } }, { text: `  ${r.iters} its`, options: { color: C.accent6 } }], { x: x + 0.18, y: 4.2, w: 2.0, h: 0.3, fontSize: 14 });
      txt(s, [{ text: fx(r.speedup), options: { color: C.accent1 } }, { text: " → ", options: { color: C.accent5, fontFace: "Calibri" } }, { text: fx(r.speedup_e2e), options: { color: C.accent3 } }],
        { x: x + 0.18, y: 4.55, w: 2.0, h: 0.55, fontSize: 24, bold: true, fontFace: "Cambria", valign: "middle" });
      txt(s, `copies = ${Math.round(100 * r.xfer_share)}% of GPU time`, { x: x + 0.18, y: 5.15, w: 2.0, h: 0.5, fontSize: 12, color: C.accent6 });
    });
    // takeaway band
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 6.05, w: 12.1, h: 0.78, rectRadius: 0.12, fill: { color: C.text1 }, line: { type: "none" }, objectName: "takeaway band" });
    txt(s, [
      { text: "TAKEAWAY   ", options: { bold: true, color: C.accent1, charSpacing: 2 } },
      { text: "FLOP per byte decides what to optimise:  ", options: { bold: true, color: C.background1 } },
      { text: "GEMM → precision  ·  SpMV → bytes  ·  CG → latency, keep data on GPU", options: { color: C.background2 } },
    ], { x: 0.9, y: 6.05, w: 11.6, h: 0.78, fontSize: 15, valign: "middle" });
    s.addNotes(`The green line is flat at about ${Math.round(tau0)} µs per iteration from n = 256 to a quarter of a million unknowns: pure latency from about nine GPU operations and two host syncs per iteration. The CPU line grows with n at roughly 24 GB/s. Where they cross — about ${fmt(xk, 0)} unknowns — the GPU starts to win; a back-of-envelope model predicts 11,000. Copies matter when there are few iterations: ${REAL[0].matrix} converges in ${REAL[0].iters} iterations, so copies are ${Math.round(100 * REAL[0].xfer_share)}% of the GPU time and the speed-up drops from ${fx(REAL[0].speedup)} to ${fx(REAL[0].speedup_e2e)}; ${REAL[1].matrix} runs ${REAL[1].iters} iterations and keeps ${fx(REAL[1].speedup_e2e)}. Closing message: compute-bound kernels — pick the precision; bandwidth-bound — count bytes; latency-bound — fewer launches and keep data resident.`);
  }

  await pres.writeFile({ fileName: OUT });
  execFileSync(PY, [path.join(__dirname, "apply_theme.py"), OUT], { stdio: "inherit" });
  execFileSync(PY, [path.join(__dirname, "postprocess_roofline.py"), OUT], { stdio: "inherit" });
  console.log("wrote", OUT);
})();
