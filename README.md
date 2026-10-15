# cuBLAST

**Open `presentation.ipynb` → Run All → present six concise pages.**

Four-mode GEMM, three real SuiteSparse matrices, a Roofline explanation, and
FP64 conjugate gradient with transfer costs. Charts and conclusions update from
the measured data. GPU slots stay blank when CUDA is unavailable.

## Install & run

```bash
python -m pip install -r requirements.txt
python -m jupyterlab presentation.ipynb
```

On an NVIDIA machine with CUDA 12.x, also install the precompiled GPU package:

```bash
python -m pip install "cupy-cuda12x>=13.4,<14" --only-binary=:all:
python experiment.py --run
```

Outputs: **`results/presentation.html`** (standalone, arrow-key navigation) and
**`results/report.json`** (all data, environment, residuals and timing samples).
Return the JSON from the GPU machine for analysis; put it at that same path locally
and rerun the notebook with `RECOMPUTE = False`.

## What is measured

- **A:** FP64, pedantic FP32, TF32, FP16 inputs with FP32 accumulation/output.
  Configure the actual card's **official dense peaks + source URL** in the notebook
  to enable peak utilization. TF32 requires compute capability 8.0+.
- **B:** `HB/bcsstk14`, `HB/bcsstk18`, `Williams/cant`, downloaded once into `.cache/`.
  CPU/GPU resident SpMV and effective GB/s; byte model = CSR + one x read + one y write.
- **C:** A controlled SPD Poisson size sweep, identical FP64 tolerance `1e-8`.
  GPU CG uses cuSPARSE SpMV and cuBLAS dot/axpy/scal. End-to-end includes upload/setup,
  solve and download; host scalar reductions synchronize each iteration.

Warm-up, repeated samples, GPU synchronization and correctness checks are built in.
GPU Event times cover resident **library calls**, including possible Python submission
gaps; CPU SpMV allocates its returned vector, while GPU output is preallocated.
Effective GB/s is not measured DRAM traffic. Crossover is a sampled observation,
not an exact threshold. Theoretical expectations are labeled separately from evidence.

Only six source files are needed: this README, the notebook, one engine, one dependency
file, `.gitignore`, and the license. Cache/results are generated and ignored by Git.
Datasets: [SuiteSparse, CC-BY 4.0](https://sparse.tamu.edu/about), Davis & Hu (2011),
[doi:10.1145/2049662.2049663](https://doi.org/10.1145/2049662.2049663).
The report records source links, CSR hashes and preprocessing; cached arrays are
derived from the original Matrix Market files and are not committed.
