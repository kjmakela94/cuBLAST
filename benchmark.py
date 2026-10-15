"""Small, reproducible CPU/CUDA linear-algebra experiments for the notebook.

CUDA is optional. No simulated GPU results are produced. Timing is at the Python
library-operation level, not a standalone, profiler-isolated CUDA kernel.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from scipy import io, sparse
from threadpoolctl import threadpool_info, threadpool_limits


@dataclass
class Config:
    gemm_sizes: tuple[int, ...] = (256, 512, 1024, 2048)
    cpu_max_n: int = 1024
    sparse_sizes: tuple[int, ...] = (1000, 10000, 100000)
    nnz_per_row: int = 16
    dtypes: tuple[str, ...] = ("float32", "float64")
    warmup: int = 3
    repeats: int = 7
    gpu_gemm_inner: int = 5
    spmv_inner: int = 20
    cpu_threads: int = 1
    seed: int = 42
    # Conservative working-set guard, including validation/temporary headroom.
    max_host_bytes: int = 1024**3
    gpu_memory_fraction: float = 0.6
    enable_gpu: bool = True
    measure_transfers: bool = True

    def __post_init__(self):
        for field in ("gemm_sizes", "sparse_sizes"):
            values = getattr(self, field)
            if not values or any(not isinstance(n, int) or n <= 0 for n in values):
                raise ValueError(f"{field} must contain positive integers")
            if len(set(values)) != len(values):
                raise ValueError(f"{field} must not contain duplicates")
        if not self.dtypes or any(d not in ("float32", "float64") for d in self.dtypes):
            raise ValueError("This baseline supports float32 and float64 only")
        if len(set(self.dtypes)) != len(self.dtypes):
            raise ValueError("dtypes must not contain duplicates")
        for field in ("repeats", "gpu_gemm_inner", "spmv_inner", "cpu_threads",
                      "nnz_per_row", "max_host_bytes"):
            if not isinstance(getattr(self, field), int) or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.warmup < 0 or self.cpu_max_n < 0:
            raise ValueError("warmup and cpu_max_n must be nonnegative")
        if not 0 < self.gpu_memory_fraction < 1:
            raise ValueError("gpu_memory_fraction must be between 0 and 1")


@dataclass
class GPU:
    cp: object = None
    sparse: object = None
    cusparse: object = None
    reason: str = ""
    name: str = ""

    @property
    def available(self):
        return self.cp is not None


def detect_gpu(enabled: bool = True) -> GPU:
    """Set baseline policy BEFORE importing CuPy; require restart if imported.

    Disabling CuPy accelerators avoids alternate GEMM backends. SpMV below calls
    the generic cuSPARSE wrapper explicitly rather than relying on CSR dispatch.
    """
    if not enabled:
        return GPU(reason="GPU disabled by configuration; CPU-only run")
    if "cupy" in sys.modules:
        raise RuntimeError("CuPy is already imported. Restart the kernel and Run All "
                           "so baseline settings take effect before CUDA initialization.")
    os.environ["CUPY_TF32"] = "0"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["CUPY_ACCELERATORS"] = ""
    try:
        import cupy as cp
        import cupyx.scipy.sparse as gpu_sparse
        from cupyx import cusparse
    except ImportError as exc:
        return GPU(reason=f"CuPy unavailable: {exc}")
    try:
        if cp.cuda.runtime.is_hip:
            return GPU(reason="This notebook targets NVIDIA CUDA, not ROCm")
        if cp.cuda.runtime.getDeviceCount() < 1:
            return GPU(reason="No CUDA device found")
        probe = cp.zeros(1)
        cp.cuda.get_current_stream().synchronize()
        del probe
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props["name"]
        name = name.decode() if isinstance(name, bytes) else str(name)
        return GPU(cp=cp, sparse=gpu_sparse, cusparse=cusparse, name=name)
    except Exception as exc:
        # Only device discovery degrades gracefully. Experiment errors must surface.
        return GPU(reason=f"CUDA initialization failed: {type(exc).__name__}: {exc}")


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=3,
                stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def environment(gpu: GPU, cfg: Config) -> dict:
    versions = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "threadpoolctl"):
        versions[name] = importlib.metadata.version(name)
    with threadpool_limits(limits=cfg.cpu_threads, user_api="blas"):
        pools = threadpool_info()
    result = {
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "cpu": _cpu_name(),
        "logical_cpu_count": os.cpu_count(), "versions": versions,
        "blas_pools_during_benchmark": pools,
        "requested_blas_threads": cfg.cpu_threads,
        "gpu_available": gpu.available, "gpu_skip_reason": gpu.reason,
        "config": asdict(cfg),
        "timing_scope": {
            "cpu": "perf_counter; BLAS threads limited; SpMV output allocation included",
            "gpu": "CUDA Events on current stream; preallocated output; warm operation calls",
            "end_to_end": "wall time: upload + compute + download, sync each phase; "
                          "preallocated buffers; no context init, input generation or file I/O",
        },
        "gpu_policy": {k: os.environ.get(k) for k in
                       ("CUPY_TF32", "NVIDIA_TF32_OVERRIDE", "CUPY_ACCELERATORS")},
    }
    if gpu.available:
        cp = gpu.cp
        free, total = cp.cuda.runtime.memGetInfo()
        result["gpu"] = {
            "name": gpu.name, "device_id": cp.cuda.Device().id,
            "free_memory_bytes": free, "total_memory_bytes": total,
            "cupy": cp.__version__, "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "spmv_backend": "cupyx.cusparse.spmv (generic cuSPARSE)",
            "fp32_mode": "TF32 disabled before import; CuPy default FP32 GEMM",
        }
    return result


def time_cpu(fn: Callable, cfg: Config, inner: int = 1) -> list[float]:
    for _ in range(cfg.warmup):
        fn()
    times = []
    for _ in range(cfg.repeats):
        start = perf_counter()
        for _ in range(inner):
            fn()
        times.append((perf_counter() - start) / inner)
    return times


def time_gpu(fn: Callable, gpu: GPU, cfg: Config, inner: int) -> list[float]:
    cp = gpu.cp
    for _ in range(cfg.warmup):
        fn()
    cp.cuda.get_current_stream().synchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    times = []
    for _ in range(cfg.repeats):
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end) / (1000 * inner))
    return times


def time_transfer_pipeline(upload: Callable, compute: Callable, download: Callable,
                           gpu: GPU, cfg: Config) -> dict[str, list[float]]:
    """Measure phases in each SAME pipeline trial; totals are not summed medians."""
    sync = gpu.cp.cuda.get_current_stream().synchronize
    def trial():
        sync()
        t0 = perf_counter()
        upload()
        sync()
        t1 = perf_counter()
        compute()
        sync()
        t2 = perf_counter()
        download()
        sync()
        t3 = perf_counter()
        return t1 - t0, t2 - t1, t3 - t2, t3 - t0
    for _ in range(cfg.warmup):
        trial()
    trials = np.asarray([trial() for _ in range(cfg.repeats)])
    return {name: trials[:, i].tolist() for i, name in enumerate(
        ("upload", "compute_wall", "download", "end_to_end"))}


def relative_error(actual, expected) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError("Validation shapes differ")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise AssertionError("Non-finite result in correctness check")
    denominator = np.linalg.norm(expected.ravel())
    difference = np.linalg.norm((actual - expected).ravel())
    return float(difference / denominator) if denominator else float(difference)


def check_error(actual, expected, dtype) -> float:
    error = relative_error(actual, expected)
    tolerance = 1e-5 if np.dtype(dtype) == np.dtype("float32") else 1e-12
    if error > tolerance:
        raise AssertionError(f"Correctness failed: relative error {error:.3g} > {tolerance}")
    return error


def correctness_demo(gpu: GPU, cfg: Config) -> pd.DataFrame:
    records = []
    rng = np.random.default_rng(cfg.seed)
    with threadpool_limits(limits=cfg.cpu_threads, user_api="blas"):
        for dtype in cfg.dtypes:
            a = rng.standard_normal((64, 64)).astype(dtype)
            b = rng.standard_normal((64, 64)).astype(dtype)
            reference = a.astype("float64") @ b.astype("float64")
            records.append(dict(operation="GEMM", backend="CPU", dtype=dtype,
                                relative_error=check_error(a @ b, reference, dtype)))
            matrix = sparse.csr_matrix([[4, 0, 1], [0, 3, 0], [1, 0, 2]], dtype=dtype)
            x = np.array([1, 2, 3], dtype=dtype)
            expected = np.array([7, 6, 7])
            records.append(dict(operation="SpMV", backend="CPU", dtype=dtype,
                                relative_error=check_error(matrix @ x, expected, dtype)))
            if gpu.available:
                cp = gpu.cp
                result = cp.asnumpy(cp.asarray(a) @ cp.asarray(b))
                records.append(dict(operation="GEMM", backend="GPU", dtype=dtype,
                                    relative_error=check_error(result, reference, dtype)))
                result = gpu.cusparse.spmv(gpu.sparse.csr_matrix(matrix), cp.asarray(x))
                records.append(dict(operation="SpMV", backend="GPU", dtype=dtype,
                                    relative_error=check_error(cp.asnumpy(result), expected, dtype)))
    return pd.DataFrame(records).assign(passed=True)


def make_csr(n: int, nnz_per_row: int, dtype: str, seed: int):
    """O(n*k) construction: no dense n*n intermediate. Duplicates are merged."""
    rng = np.random.default_rng(seed)
    k = min(nnz_per_row, n)
    if n * k >= np.iinfo(np.int32).max:
        raise ValueError("Synthetic case exceeds this baseline's int32 index limit")
    indices = rng.integers(0, n, size=n * k, dtype=np.int32)
    indptr = np.arange(0, n * k + 1, k, dtype=np.int32)
    data = rng.standard_normal(n * k).astype(dtype)
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(n, n))
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def load_mtx(path: Path | str, dtype: str = "float64"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Matrix Market file not found: {path}")
    # New SciPy defaults to sparse arrays; normalize to CSR explicitly below.
    kwargs = {"spmatrix": False} if "spmatrix" in inspect.signature(io.mmread).parameters else {}
    raw = io.mmread(path, **kwargs)
    if np.iscomplexobj(raw.data if sparse.issparse(raw) else raw):
        raise ValueError("This baseline accepts real-valued matrices only")
    matrix = sparse.csr_matrix(raw, dtype=dtype)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if min(matrix.shape) <= 0 or not np.all(np.isfinite(matrix.data)):
        raise ValueError("Matrix must have positive dimensions and finite values")
    if max(*matrix.shape, matrix.nnz) >= np.iinfo(np.int32).max:
        raise ValueError("This baseline requires 32-bit CSR indices")
    matrix.indices = matrix.indices.astype(np.int32, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int32, copy=False)
    return matrix


def estimated_spmv_bytes(matrix) -> int:
    """One read of CSR and x, one write of y; NOT measured DRAM traffic."""
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
               + sum(matrix.shape) * matrix.dtype.itemsize)


def _record(rows, samples, *, times, inner=1, **metadata):
    times = np.asarray(times, dtype=float)
    if np.any(times <= 0) or not np.all(np.isfinite(times)):
        raise RuntimeError("Nonpositive or non-finite timing sample; increase batch size")
    median = float(np.median(times))
    row = dict(metadata, median_s=median, q25_s=float(np.quantile(times, .25)),
               q75_s=float(np.quantile(times, .75)), repeats=len(times), inner=inner)
    is_operation = metadata.get("scope") in ("compute", "end_to_end")
    row["gflops"] = metadata["flops"] / median / 1e9 if is_operation else np.nan
    row["effective_gb_s"] = (metadata["estimated_bytes"] / median / 1e9
                             if metadata.get("operation") == "SpMV" and
                             metadata.get("scope") == "compute" else np.nan)
    rows.append(row)
    for i, seconds in enumerate(times):
        samples.append(dict(metadata, trial=i, seconds=float(seconds), inner=inner))


def _gpu_fits(gpu: GPU, cfg: Config, needed: int) -> bool:
    free, _ = gpu.cp.cuda.runtime.memGetInfo()
    return needed <= free * cfg.gpu_memory_fraction


def _release_gpu(gpu):
    if gpu.available:
        gpu.cp.cuda.get_current_stream().synchronize()
        gpu.cp.get_default_memory_pool().free_all_blocks()


def _gemm_case(n, dtype, gpu, cfg, rows, samples, skips):
    case = f"gemm_{n}"
    itemsize = np.dtype(dtype).itemsize
    # Includes random-generation, CPU output, host download, and validation headroom.
    if 8 * n * n * itemsize > cfg.max_host_bytes:
        skips.append(dict(case=case, dtype=dtype, backend="ALL", reason="host memory budget"))
        return
    rng = np.random.default_rng(cfg.seed + n)
    # Match the underlying random inputs across precisions before rounding.
    a = rng.standard_normal((n, n)).astype(dtype, copy=False)
    b = rng.standard_normal((n, n)).astype(dtype, copy=False)
    ix = np.linspace(0, n - 1, min(16, n), dtype=int)
    reference = a[ix, :].astype("float64") @ b[:, ix].astype("float64")
    common = dict(operation="GEMM", case=case, source="synthetic", n=n, m=n,
                  nnz=n*n, dtype=dtype, flops=2*n**3, estimated_bytes=0,
                  validation_scope="up to 16x16 sampled output entries vs FP64")
    if n <= cfg.cpu_max_n:
        c = np.empty_like(a)
        fn = lambda: np.matmul(a, b, out=c)
        fn()
        error = check_error(c[np.ix_(ix, ix)], reference, dtype)
        _record(rows, samples, times=time_cpu(fn, cfg), **common, backend="CPU",
                scope="compute", relative_error=error)
        del fn, c
    else:
        skips.append(dict(case=case, dtype=dtype, backend="CPU", reason="cpu_max_n"))
    if not gpu.available:
        return
    if not _gpu_fits(gpu, cfg, 6*n*n*itemsize):
        skips.append(dict(case=case, dtype=dtype, backend="GPU", reason="GPU memory budget"))
        return
    cp = gpu.cp
    da, db = cp.asarray(a), cp.asarray(b)
    dc = cp.empty_like(da)
    fn = lambda: cp.matmul(da, db, out=dc)
    fn()
    actual = cp.asnumpy(dc[cp.ix_(cp.asarray(ix), cp.asarray(ix))])
    error = check_error(actual, reference, dtype)
    _record(rows, samples, times=time_gpu(fn, gpu, cfg, cfg.gpu_gemm_inner),
            inner=cfg.gpu_gemm_inner, **common, backend="GPU", scope="compute",
            relative_error=error)
    if cfg.measure_transfers:
        host_out = np.empty_like(a)
        def upload():
            da.set(a)
            db.set(b)
        phases = time_transfer_pipeline(upload, fn, lambda: dc.get(out=host_out), gpu, cfg)
        check_error(host_out[np.ix_(ix, ix)], reference, dtype)
        for scope, times in phases.items():
            _record(rows, samples, times=times, **common, backend="GPU", scope=scope,
                    relative_error=error)


def run_gemm(cfg: Config, gpu: GPU):
    rows, samples, skips = [], [], []
    with threadpool_limits(limits=cfg.cpu_threads, user_api="blas"):
        for dtype in cfg.dtypes:
            for n in sorted(cfg.gemm_sizes):
                print(f"GEMM {dtype} n={n}", flush=True)
                try:
                    _gemm_case(n, dtype, gpu, cfg, rows, samples, skips)
                finally:
                    _release_gpu(gpu)
    return pd.DataFrame(rows), pd.DataFrame(samples), pd.DataFrame(skips)


def _spmv_case(matrix, case, source, gpu, cfg, rows, samples, skips):
    m, n = matrix.shape
    dtype = str(matrix.dtype)
    estimated_bytes = estimated_spmv_bytes(matrix)
    common = dict(operation="SpMV", case=case, source=source, m=m, n=n,
                  nnz=matrix.nnz, dtype=dtype, flops=2*matrix.nnz,
                  estimated_bytes=estimated_bytes,
                  validation_scope="full vector vs CPU FP64 reference")
    x = np.random.default_rng(cfg.seed).standard_normal(n).astype(dtype)
    reference = matrix.astype("float64") @ x.astype("float64")
    fn = lambda: matrix @ x
    error = check_error(fn(), reference, dtype)
    _record(rows, samples, times=time_cpu(fn, cfg, cfg.spmv_inner),
            inner=cfg.spmv_inner, **common, backend="CPU", scope="compute", relative_error=error)
    if not gpu.available:
        return
    if not _gpu_fits(gpu, cfg, estimated_bytes*4):
        skips.append(dict(case=case, dtype=dtype, backend="GPU", reason="GPU memory budget"))
        return
    cp = gpu.cp
    da, dx = gpu.sparse.csr_matrix(matrix), cp.asarray(x)
    dy = cp.empty(m, dtype=dtype)
    fn = lambda: gpu.cusparse.spmv(da, dx, y=dy, alpha=1, beta=0)
    fn()
    error = check_error(cp.asnumpy(dy), reference, dtype)
    _record(rows, samples, times=time_gpu(fn, gpu, cfg, cfg.spmv_inner),
            inner=cfg.spmv_inner, **common, backend="GPU", scope="compute", relative_error=error)
    if cfg.measure_transfers:
        host_out = np.empty(m, dtype=dtype)
        def upload():
            da.data.set(matrix.data)
            da.indices.set(matrix.indices)
            da.indptr.set(matrix.indptr)
            dx.set(x)
        phases = time_transfer_pipeline(upload, fn, lambda: dy.get(out=host_out), gpu, cfg)
        check_error(host_out, reference, dtype)
        for scope, times in phases.items():
            _record(rows, samples, times=times, **common, backend="GPU", scope=scope,
                    relative_error=error)


def run_spmv(cfg: Config, gpu: GPU, mtx_paths=()):
    paths = [Path(p) for p in mtx_paths]
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Use distinct Matrix Market filenames to identify cases")
    rows, samples, skips = [], [], []
    with threadpool_limits(limits=cfg.cpu_threads, user_api="blas"):
        for dtype in cfg.dtypes:
            cases = [(f"synthetic_{n}", n) for n in sorted(cfg.sparse_sizes)]
            cases += [(f"mtx_{p.stem}", p) for p in paths]
            for case, spec in cases:
                print(f"SpMV {dtype} {case}", flush=True)
                if isinstance(spec, int):
                    # Construction and FP64 validation both need O(n*k) temporaries.
                    needed = 64 * spec * min(cfg.nnz_per_row, spec) + 64 * spec
                    if needed > cfg.max_host_bytes:
                        skips.append(dict(case=case, dtype=dtype, backend="ALL",
                                          reason="host memory budget"))
                        continue
                    matrix = make_csr(spec, cfg.nnz_per_row, dtype, cfg.seed + spec)
                    source = "synthetic"
                else:
                    # mmread must parse the file first; huge external files need manual care.
                    matrix = load_mtx(spec, dtype)
                    source = str(spec)
                    if 8 * estimated_spmv_bytes(matrix) > cfg.max_host_bytes:
                        skips.append(dict(case=case, dtype=dtype, backend="ALL",
                                          reason="loaded .mtx exceeds host working-set budget"))
                        del matrix
                        continue
                try:
                    _spmv_case(matrix, case, source, gpu, cfg, rows, samples, skips)
                finally:
                    del matrix
                    _release_gpu(gpu)
    return pd.DataFrame(rows), pd.DataFrame(samples), pd.DataFrame(skips)


def speedups(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    keys = ["operation", "case", "dtype", "m", "n", "nnz"]
    cpu = results[(results.backend == "CPU") & (results.scope == "compute")]
    cpu = cpu[keys + ["median_s"]].rename(columns={"median_s": "cpu_s"})
    gpu = results[(results.backend == "GPU") & results.scope.isin(["compute", "end_to_end"])]
    merged = gpu.merge(cpu, on=keys, how="inner")
    merged["speedup"] = merged.cpu_s / merged.median_s
    return merged


def export_results(folder, cfg, env, results, samples, skips, checks):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    schemas = {
        "results": ["operation", "case", "dtype", "backend", "scope", "median_s", "gflops"],
        "samples": ["operation", "case", "dtype", "backend", "scope", "trial", "seconds"],
        "skipped": ["case", "dtype", "backend", "reason"],
        "correctness": ["operation", "backend", "dtype", "relative_error", "passed"],
        "speedups": ["operation", "case", "dtype", "scope", "speedup"],
    }
    for name, frame in (("results", results), ("samples", samples),
                        ("skipped", skips), ("correctness", checks),
                        ("speedups", speedups(results))):
        if frame.empty and not len(frame.columns):
            frame = pd.DataFrame(columns=schemas[name])
        frame.to_csv(folder / f"{name}.csv", index=False)
    metadata = dict(env, config=asdict(cfg))
    (folder / "environment.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return folder
