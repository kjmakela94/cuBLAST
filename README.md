# cuBLAST

A basic Python / Jupyter benchmark for exploring CPU and NVIDIA GPU linear algebra.
Start with **`basic_benchmark.ipynb`**, which includes step-by-step explanations,
correctness checks, and visualizations in English.

The notebook retains outputs from a local CPU run for preview. No GPU measurements
are included in those saved outputs. After changing machines or parameters, restart
the kernel and run all cells to collect your own results.

## Features

- FP64 / FP32 / TF32 / FP16 square GEMM: size sweeps, elapsed time, GFLOP/s,
  numerical error, and speedups against documented CPU baselines.
- CSR SpMV: three default synthetic sizes and optional real Matrix Market files (`.mtx` / `.mtx.gz`).
- CuPy GPU operations, with SpMV explicitly using generic cuSPARSE through `cupyx.cusparse.spmv`.
- Upload / compute / download breakdowns, warm-up, GPU synchronization, repeated measurements, medians, and IQR.
- Raw measurements as CSV, environment metadata as JSON, and figures as PNG.
- Automatic CPU-only execution when CUDA is unavailable, without simulated GPU results.

This baseline does not yet include comparisons with theoretical GPU peak performance
or conjugate gradient (CG). Synthetic matrices demonstrate the
workflow; the formal Task B still requires three real matrices from SuiteSparse.

### GEMM modes

GPU GEMM calls `cublasGemmEx` through CuPy's low-level binding with an explicit
compute type for each mode. Inputs and outputs are column-major and preallocated.
Each operation owns a dedicated cuBLAS handle, so pointer/math settings do not
affect other CuPy operations.

| Mode | Input storage | Accumulation / output | cuBLAS compute type |
| --- | --- | --- | --- |
| FP64 | FP64 | FP64 / FP64 | `CUBLAS_COMPUTE_64F` |
| FP32 | FP32 | FP32 / FP32 | `CUBLAS_COMPUTE_32F_PEDANTIC` |
| TF32 | FP32 | FP32 / FP32 | `CUBLAS_COMPUTE_32F_FAST_TF32` |
| FP16 | FP16 | FP32 / FP32 | `CUBLAS_COMPUTE_32F` |

FP16 here is a mixed-precision GEMM with half-precision inputs and single-precision
accumulation/output. TF32 is a compute mode, not a NumPy storage dtype. The benchmark
requests TF32 on compute capability 8.0+ GPUs; older GPUs record an explicit skip.
FP16 requires compute capability 5.3+ in this baseline. A requested mode does not
prove which kernel or Tensor Core instructions the library selected; use a profiler
to verify Tensor Core utilization. See the
[NVIDIA cuBLAS reference](https://docs.nvidia.com/cuda/cublas/index.html).

CPU timings cover native FP64 and FP32 only. TF32/FP16 GPU speedups use **CPU FP32**
as their baseline, explicitly recorded in `cpu_reference_mode`; they do not imply
identical arithmetic precision. Select FP32 in `gemm_modes` to collect that CPU
baseline. With no GPU, TF32/FP16 are skipped rather than emulated or mislabeled.

## Run locally or on a course server

Python 3.10+ is recommended. Create an isolated environment from the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab basic_benchmark.ipynb
```

On Windows, activate the environment with `.venv\Scripts\activate`. In Jupyter,
select the Python kernel for this environment, then choose
**Restart Kernel and Run All Cells**. If your editor selects another kernel,
register the project environment explicitly:

```bash
python -m ipykernel install --prefix .venv --name cublast --display-name "Python (cuBLAST)"
```

Apple Silicon and other machines without an NVIDIA GPU can run the CPU experiments.
The CUDA experiments require an NVIDIA GPU.

### Enable the NVIDIA GPU experiments

In an environment with an NVIDIA driver and usable CUDA libraries, install **one**
CuPy distribution that matches the environment. For CUDA 12.x:

```bash
python -m pip install -r requirements-gpu.txt
```

For CUDA 13.x, install the base requirements, then use
`python -m pip install cupy-cuda13x`. Do not install `cupy`, `cupy-cuda12x`, and
`cupy-cuda13x` together. A Python package does not replace the NVIDIA driver.
Consult the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
for CUDA library requirements and installation options.

The notebook reports the actual device, CuPy/CUDA versions, and any GPU skip reason.
Restart the kernel after installation or configuration changes, then run from the first cell.

### Run in Colab

1. Select a GPU runtime.
2. Clone your repository using its actual URL, or upload and extract the entire repository.
3. Switch to the repository with `%cd /content/cuBLAST`, adjusting the path as needed.
4. Install the base dependencies with `%pip install -r requirements.txt`. Check whether
   the existing CuPy distribution matches the environment; install a matching one if needed.
   Restart the runtime afterward and switch back to the repository directory.
5. Open a copy of `basic_benchmark.ipynb` and run from the first cell. Both `benchmark.py`
   and `plots.py` must be in the current directory; uploading only the notebook is insufficient.

## Configure the experiments

Edit the configuration in Section 2 of the notebook:

| Parameter | Default / purpose |
| --- | --- |
| `PROFILE` | `quick`; also supports `full` and the very small `smoke` profile |
| `gemm_sizes` | quick: 256, 512, 1024, 2048 |
| `cpu_max_n` | quick: 1024; larger cases skip CPU timing and have no inferred speedup |
| `sparse_sizes` | quick: 1000, 10000, 100000 |
| `nnz_per_row` | 16; merging duplicate random columns may slightly reduce the actual nnz |
| `dtypes` | SpMV storage types: float32, float64 |
| `gemm_modes` | FP64, FP32, TF32, FP16; independent of SpMV dtypes |
| `warmup` / `repeats` | 3 warm-up calls / 7 measurement samples |
| `gpu_gemm_inner` | Average 5 GEMM calls within each CUDA Event sample |
| `spmv_inner` | Average 20 SpMV calls within each sample |
| `cpu_threads` | 1; threadpoolctl limits BLAS threads, and environment metadata records the libraries |
| `max_host_bytes` | Conservative 1 GiB working-set budget, not a hard process memory limit |
| `gpu_memory_fraction` | Limit the estimated working set to 60% of currently free GPU memory |
| `measure_transfers` | True; also measure upload, compute, and download using preallocated buffers |

One BLAS thread is the default for reproducibility; it does not represent peak
multicore CPU performance. SciPy CSR SpMV threading depends on its implementation;
threadpoolctl does not automatically make it multithreaded.

The memory budget skips clearly oversized synthetic cases. External `.mtx` files
must be parsed before their working set can be checked, so this budget cannot
enforce a hard limit on parsing memory. Check large files before loading them.
GPU experiment errors are raised rather than silently hidden.

### Add real matrices

Place files in `data/`, then configure the notebook, for example:

```python
MTX_PATHS = [
    ROOT / "data" / "matrix_1.mtx",
    ROOT / "data" / "matrix_2.mtx",
    ROOT / "data" / "matrix_3.mtx.gz",
]
```

Replace these examples with actual filenames and use distinct names for different
files. Loading merges duplicate entries, removes explicit zeros, and sorts column
indices. SciPy expands symmetric Matrix Market storage. Real-valued rectangular
matrices are supported; this baseline uses 32-bit CSR indices. Results record the
source paths, matrix dimensions, and nnz.

## Interpret the timings

| `scope` | Meaning |
| --- | --- |
| `compute` | Inputs already reside on the device; CPU wall time / GPU CUDA Events; averaged repeated calls |
| `upload` | Copy all inputs from NumPy / SciPy host buffers into preallocated GPU buffers, including synchronization |
| `compute_wall` | Wall time for one operation in the transfer pipeline, including the call and synchronization |
| `download` | Copy the complete output into a preallocated host buffer, including synchronization |
| `end_to_end` | Total wall time for upload + compute_wall + download in the same trial |

These timings exclude input generation, file I/O, CSR conversion, correctness
checks, and first-time CUDA context initialization. The end-to-end experiment uses
ordinary NumPy host buffers, without pinned-memory asynchronous overlap or initial
buffer allocation. Each phase is synchronized, defining a serial pipeline.

GEMM preallocates CPU/GPU outputs. CPU SpMV uses SciPy `A @ x`, including allocation
of the returned vector. GPU SpMV preallocates its output, although the wrapper may
still handle descriptors and workspace. CUDA Event timing may also include Python
submission gaps. These are baseline measurements of actual library calls, not
isolated CUDA kernel execution times.

- GEMM: `GFLOP/s = 2*n**3 / seconds / 1e9`.
- SpMV: `GFLOP/s = 2*nnz / seconds / 1e9`.
- Effective bandwidth: `(CSR array bytes + x bytes + y bytes) / compute_seconds / 1e9`.
  This assumes one access to each stored input/output and is not measured DRAM traffic.
  Repeated calls may benefit from caching.
- Speedup: matching matrix sizes and documented CPU reference modes, using
  `CPU compute / GPU compute` or `CPU compute / GPU end_to_end`. TF32/FP16 use
  CPU FP32; FP64 and FP32 use the corresponding CPU precision.
- GFLOP/s is populated only for compute / end_to_end rows. Effective GB/s is populated
  only for SpMV compute rows.
- Tables report medians; GEMM time error bars show the interquartile range (IQR).
  Transfer stacks use phase means from the same trials, rather than adding phase medians.

Before importing CuPy, the baseline sets `CUPY_TF32=0` and disables optional
accelerators for unrelated high-level calls. GEMM uses explicit per-call compute
types, so its TF32 mode does not rely on that global CuPy switch. The benchmark
does not force `NVIDIA_TF32_OVERRIDE=0`: if the environment already sets it to `0`,
TF32 is explicitly skipped. Unset that variable and restart the kernel to enable
TF32. FP32 remains pedantic FP32 independently. Do not import CuPy before the
environment-check cell.

## Data and figures

Each run creates `results/<UTC_timestamp>/`:

```text
environment.json  # Hardware, software, BLAS threads, parameters, and timing scope
correctness.csv   # Complete correctness checks on small examples
results.csv       # Medians, IQR, throughput, math modes, storage/compute types, and errors
samples.csv       # Seconds per trial; batched measurements already divided by inner
skipped.csv       # Cases skipped because of memory budgets or CPU size limits
speedups.csv      # Comparisons with both CPU and GPU measurements; named CPU reference
gemm.png
spmv.png
transfers.png
```

All GEMM modes start from identical seeded FP64 input values before storage
conversion. Up to 16x16 output entries per size are checked against a common CPU
FP64 reference computed from those original values, so `relative_error` includes
input quantization. Thresholds are 1e-12 for FP64, 1e-5 for FP32, and 5e-3 for
TF32/FP16 on these random test inputs. The tolerances are experiment checks, not
universal error guarantees for arbitrary matrices.

GEMM also records `arithmetic_error` against an FP64 product of the stored inputs
and `input_quantization_error` between the two FP64 references. These separate
storage rounding from arithmetic effects; the errors need not add linearly.
Small GEMM correctness checks cover complete outputs of every available mode;
unsupported modes have `status=skipped`, not a passing result. SpMV still validates
the complete vector with its original FP32/FP64 tolerances. Failed checks stop execution.

Results, environments, and datasets are ignored by Git by default. Select the
figures and data you need when preparing a report.

## Verify the implementation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Enable real CUDA validation explicitly on an NVIDIA GPU:

```bash
RUN_CUDA_TESTS=1 python -m pytest -q
```

Tests cover numerical correctness, Matrix Market loading, sample statistics,
transfer-phase bookkeeping, exports, figures, mode eligibility, CPU reference joins,
and raw cuBLAS argument/layout/scalar contracts. CPU contract tests do not execute
GPU arithmetic or produce GPU performance measurements. Tests do not enforce fixed speed
thresholds. Skipped GPU tests are not reported as validated on machines without a GPU.

## References

- [CuPy performance and timing](https://docs.cupy.dev/en/stable/user_guide/performance.html)
- [CuPy environment variables](https://docs.cupy.dev/en/stable/reference/environment.html)
- [CuPy installation](https://docs.cupy.dev/en/stable/install.html)
- [CuPy sparse matrices](https://docs.cupy.dev/en/stable/reference/scipy_sparse.html)
- [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)
