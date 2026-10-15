"""cuBLAST: one experiment engine for the six-page presentation.

Run: python experiment.py --run. GPU results remain null without NVIDIA CUDA.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import argparse
import base64
import hashlib
import html
import io as _io
import json
import os
import platform
import subprocess
import tarfile
import urllib.request
import numpy as np
import scipy
from scipy import sparse, io
from threadpoolctl import threadpool_limits, threadpool_info

GEMM_MODES = {
    "FP64": dict(input_dtype="float64", output_dtype="float64", accumulation_dtype="float64",
                 compute_type="CUBLAS_COMPUTE_64F", tolerance=1e-12),
    "FP32": dict(input_dtype="float32", output_dtype="float32", accumulation_dtype="float32",
                 compute_type="CUBLAS_COMPUTE_32F_PEDANTIC", tolerance=1e-5),
    "TF32": dict(input_dtype="float32", output_dtype="float32", accumulation_dtype="float32",
                 compute_type="CUBLAS_COMPUTE_32F_FAST_TF32", tolerance=5e-3),
    "FP16": dict(input_dtype="float16", output_dtype="float32", accumulation_dtype="float32",
                 compute_type="CUBLAS_COMPUTE_32F", tolerance=5e-3),
}

@dataclass
class GPU:
    cp: object = None
    sparse: object = None
    cusparse: object = None
    reason: str = ""
    name: str = ""
    cublas: object = None
    compute_capability: tuple[int, int] = (0, 0)

    @property
    def available(self):
        return self.cp is not None

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

def check_gemm_error(actual, expected, mode) -> float:
    error = relative_error(actual, expected)
    tolerance = GEMM_MODES[mode]["tolerance"]
    if error > tolerance:
        raise AssertionError(f"{mode} correctness failed: relative error {error:.3g} > {tolerance}")
    return error

def gemm_skip_reason(gpu: GPU, mode: str) -> str:
    """Do not label CPU emulation or unsupported hardware as native GPU modes."""
    if mode not in GEMM_MODES:
        raise ValueError(f"Unknown GEMM mode: {mode}")
    if not gpu.available:
        return gpu.reason or "No NVIDIA CUDA GPU available"
    capability = gpu.compute_capability[0]*10 + gpu.compute_capability[1]
    if mode == "TF32":
        if capability < 80:
            return "TF32 requires NVIDIA compute capability 8.0+ (Ampere or newer)"
        if os.environ.get("NVIDIA_TF32_OVERRIDE") == "0":
            return "NVIDIA_TF32_OVERRIDE=0 disables TF32; unset it and restart the kernel"
    if mode == "FP16" and capability < 53:
        return "This FP16 baseline requires NVIDIA compute capability 5.3+"
    if gpu.cublas is None or not hasattr(gpu.cublas, GEMM_MODES[mode]["compute_type"]):
        return "Required cuBLAS compute type is unavailable in this CuPy build"
    return ""

@contextmanager
def gemm_operation(gpu, a, b, c, mode):
    """Yield a cuBLAS GEMM callable with column-major storage and owned scalars.

    Handle creation/configuration is outside timing. A private handle avoids leaking
    pointer/math mode changes into other CuPy operations. gemmEx binds the current
    CuPy stream on every call, matching the events used by time_gpu.
    """
    spec = GEMM_MODES[mode]
    reason = gemm_skip_reason(gpu, mode)
    if reason:
        raise ValueError(reason)
    if any(len(array.shape) != 2 or not array.flags.f_contiguous for array in (a, b, c)):
        raise ValueError("cuBLAS GEMM buffers must be two-dimensional and Fortran-contiguous")
    m, k = a.shape
    kb, n = b.shape
    if kb != k or c.shape != (m, n) or min(m, n, k) <= 0:
        raise ValueError("Incompatible GEMM buffer shapes")
    if (a.dtype != np.dtype(spec["input_dtype"]) or b.dtype != a.dtype or
            c.dtype != np.dtype(spec["output_dtype"])):
        raise ValueError("GEMM buffer dtypes do not match the selected mode")
    cb = gpu.cublas
    runtime = gpu.cp.cuda.runtime
    input_type = getattr(runtime, {"float16": "CUDA_R_16F", "float32": "CUDA_R_32F",
                                   "float64": "CUDA_R_64F"}[spec["input_dtype"]])
    output_type = getattr(runtime, "CUDA_R_64F" if mode == "FP64" else "CUDA_R_32F")
    compute_type = getattr(cb, spec["compute_type"])
    alpha = np.array(1, dtype=spec["accumulation_dtype"])
    beta = np.array(0, dtype=spec["accumulation_dtype"])
    handle = cb.create()
    try:
        cb.setPointerMode(handle, cb.CUBLAS_POINTER_MODE_HOST)
        cb.setMathMode(handle, cb.CUBLAS_DEFAULT_MATH)
        def call():
            cb.gemmEx(handle, cb.CUBLAS_OP_N, cb.CUBLAS_OP_N, m, n, k,
                      alpha.ctypes.data, a.data.ptr, input_type, m,
                      b.data.ptr, input_type, k, beta.ctypes.data,
                      c.data.ptr, output_type, m, compute_type, cb.CUBLAS_GEMM_DEFAULT)
        yield call
    finally:
        try:
            gpu.cp.cuda.get_current_stream().synchronize()
        finally:
            cb.destroy(handle)


@dataclass
class Settings:
    gemm_sizes: tuple = (256, 512, 1024, 2048, 4096)
    cpu_max_n: int = 2048
    cg_sides: tuple = (24, 48, 96, 192)
    repeats: int = 5
    warmup: int = 2
    seed: int = 42
    cpu_threads: int = 1
    enable_gpu: bool = True
    # Official dense peaks for the ACTUAL card. Never guess from a GPU name.
    peaks_tflops: dict = field(default_factory=dict)
    peak_source: str = ""
    peak_bandwidth_gbs: float | None = None

    def __post_init__(self):
        if self.repeats < 1 or self.warmup < 1 or min(self.gemm_sizes) < 1:
            raise ValueError("Positive sizes/repeats/warmup are required")
        supplied = [v for v in self.peaks_tflops.values() if v is not None]
        if supplied and (not self.peak_source or any(
                not np.isfinite(v) or v <= 0 for v in supplied)):
            raise ValueError("Peak comparisons require positive dense peaks and an official source URL")


MATRICES = [
    dict(group="HB", name="bcsstk14", domain="Structural mechanics"),
    dict(group="HB", name="bcsstk18", domain="Structural mechanics"),
    dict(group="Williams", name="cant", domain="Cantilever structure"),
]


def discover_gpu(enabled=True):
    if not enabled:
        return GPU(reason="CPU-only run selected")
    try:
        import cupy as cp
        from cupyx import cusparse
        from cupy_backends.cuda.libs import cublas
        import cupyx.scipy.sparse as cs
        if cp.cuda.runtime.is_hip or cp.cuda.runtime.getDeviceCount() == 0:
            return GPU(reason="No NVIDIA CUDA device")
        cp.zeros(1).sum().item()
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props["name"]
        return GPU(cp=cp, sparse=cs, cusparse=cusparse, cublas=cublas,
                   name=name.decode() if isinstance(name, bytes) else str(name),
                   compute_capability=(props["major"], props["minor"]))
    except Exception as exc:
        # Only discovery degrades gracefully; actual experiment failures propagate.
        return GPU(reason=f"{type(exc).__name__}: {exc}")


def _median_cpu(call, s, inner=1):
    for _ in range(s.warmup):
        call()
    samples = []
    for _ in range(s.repeats):
        start = perf_counter()
        for _ in range(inner):
            call()
        samples.append((perf_counter()-start)*1000/inner)
    return float(np.median(samples)), samples


def _median_gpu(call, gpu, s, inner=10):
    cp = gpu.cp
    for _ in range(s.warmup):
        call()
    cp.cuda.get_current_stream().synchronize()
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    samples = []
    for _ in range(s.repeats):
        begin.record()
        for _ in range(inner):
            call()
        end.record(); end.synchronize()
        samples.append(cp.cuda.get_elapsed_time(begin, end)/inner)
    return float(np.median(samples)), samples


def _gemm_one(n, mode, gpu, s):
    spec = GEMM_MODES[mode]
    row = dict(mode=mode, n=n, cpu_ms=None, gpu_ms=None, gflops=None,
               cpu_gflops=None, cpu_error=None, error=None, peak_utilization=None,
               **spec, gpu_status=gemm_skip_reason(gpu, mode) or "ready")
    if 64*n*n > 1024**3:
        row["gpu_status"] = "Skipped: conservative 1 GiB host working-set budget"
        return row
    if row["gpu_status"] != "ready" and (mode in ("TF32", "FP16") or n>s.cpu_max_n):
        return row
    rng = np.random.default_rng(s.seed+n)
    aa, bb = rng.standard_normal((n,n)), rng.standard_normal((n,n))
    ix = np.linspace(0,n-1,min(16,n),dtype=int)
    reference = aa[ix,:] @ bb[:,ix]
    a, b = np.asfortranarray(aa,dtype=spec["input_dtype"]), np.asfortranarray(bb,dtype=spec["input_dtype"])
    del aa, bb
    if mode in ("FP64","FP32") and n <= s.cpu_max_n:
        c = np.empty_like(a)
        def call_cpu():
            np.matmul(a,b,out=c)
        call_cpu()
        row["cpu_error"] = check_gemm_error(c[np.ix_(ix,ix)],reference,mode)
        row["cpu_ms"], row["cpu_samples_ms"] = _median_cpu(call_cpu,s)
        row["cpu_gflops"] = 2*n**3/(row["cpu_ms"]*1e6)
        del c, call_cpu
    if row["gpu_status"] != "ready":
        return row
    cp = gpu.cp
    free,_ = cp.cuda.runtime.memGetInfo()
    if 2*(a.nbytes+b.nbytes+n*n*np.dtype(spec["output_dtype"]).itemsize) > .6*free:
        row["gpu_status"] = "Skipped: GPU working-set budget"
        return row
    da, db = cp.asarray(a,order="F"), cp.asarray(b,order="F")
    dc = cp.empty((n,n),dtype=spec["output_dtype"],order="F")
    with gemm_operation(gpu,da,db,dc,mode) as call_gpu:
        call_gpu()
        measured = cp.asnumpy(dc[cp.ix_(cp.asarray(ix),cp.asarray(ix))])
        row["error"] = check_gemm_error(measured,reference,mode)
        row["gpu_ms"], row["gpu_samples_ms"] = _median_gpu(call_gpu,gpu,s,5)
    row["gflops"] = 2*n**3/(row["gpu_ms"]*1e6)
    peak = s.peaks_tflops.get(mode)
    row["peak_utilization"] = row["gflops"]/(1000*peak) if peak else None
    row["gpu_status"] = "measured"
    return row


def _load_real_matrix(spec, cache):
    cache.mkdir(parents=True,exist_ok=True)
    path = cache/f'{spec["name"]}.npz'
    if path.exists():
        a = sparse.load_npz(path)
    else:
        archive = cache/f'{spec["name"]}.tar.gz'
        errors = []
        for base in ("https://www.cise.ufl.edu/research/sparse/MM", "https://sparse.tamu.edu/MM", "https://sparse-files.engr.tamu.edu/MM"):
            url = f'{base}/{spec["group"]}/{spec["name"]}.tar.gz'
            try:
                request = urllib.request.Request(url,headers={"User-Agent":"cuBLAST research benchmark"})
                with urllib.request.urlopen(request,timeout=90) as src, archive.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src,dst)
                with tarfile.open(archive) as tar:
                    member = next(m for m in tar.getmembers() if m.name.endswith(f'/{spec["name"]}.mtx'))
                    # Read only the matrix member; never extract arbitrary archive paths.
                    import inspect
                    kwargs = {'spmatrix':False} if 'spmatrix' in inspect.signature(io.mmread).parameters else {}
                    a = sparse.csr_matrix(io.mmread(tar.extractfile(member),**kwargs),dtype=np.float64)
                break
            except Exception as exc:
                errors.append(f"{url}: {exc}")
        else:
            raise RuntimeError("Could not download a required real matrix. " + " | ".join(errors))
        a.sum_duplicates(); a.eliminate_zeros(); a.sort_indices()
        a.indices = a.indices.astype(np.int32,copy=False)
        a.indptr = a.indptr.astype(np.int32,copy=False)
        sparse.save_npz(path,a)
        archive.unlink(missing_ok=True)
    a = a.astype(np.float64,copy=False)
    if not np.isfinite(a.data).all():
        raise ValueError("Non-finite real matrix values")
    digest = hashlib.sha256()
    for array in (a.data,a.indices,a.indptr):
        digest.update(array.tobytes())
    info = dict(spec,source=f'https://sparse.tamu.edu/{spec["group"]}/{spec["name"]}',
                m=a.shape[0],n=a.shape[1],nnz=a.nnz,csr_sha256=digest.hexdigest(),
                license='CC-BY-4.0',citation='Davis & Hu (2011), doi:10.1145/2049662.2049663',
                processing='Matrix Market symmetric expansion; sum duplicates; remove explicit zeros; sorted int32 CSR')
    return a, info


def _spmv_one(a, info, gpu, s):
    m,n = a.shape
    x = np.random.default_rng(s.seed).standard_normal(n)
    reference = a @ x
    # One CSR pass, one x read, one y write: effective bytes, not measured DRAM traffic.
    byte_count = int(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes+(m+n)*8)
    row = dict(matrix=info["name"],m=m,n=n,nnz=a.nnz,estimated_bytes=byte_count,
               intensity=2*a.nnz/byte_count,gpu_ms=None,gb_s=None,gpu_error=None)
    row["cpu_ms"],row["cpu_samples_ms"] = _median_cpu(lambda:a@x,s,20)
    row["cpu_gb_s"] = byte_count/(row["cpu_ms"]*1e6)
    if gpu.available:
        cp = gpu.cp
        da, dx = gpu.sparse.csr_matrix(a), cp.asarray(x)
        dy = cp.empty(m,dtype=np.float64)
        def call():
            gpu.cusparse.spmv(da,dx,y=dy,alpha=1,beta=0)
        call()
        error = relative_error(cp.asnumpy(dy),reference)
        if error > 1e-12:
            raise AssertionError(f"SpMV validation failed for {info['name']}: {error}")
        row["gpu_error"] = error
        row["gpu_ms"],row["gpu_samples_ms"] = _median_gpu(call,gpu,s,30)
        row["gb_s"] = byte_count/(row["gpu_ms"]*1e6)
    return row


def run(settings=None, root=None):
    s,root = settings or Settings(), Path(root or Path(__file__).parent)
    gpu = discover_gpu(s.enable_gpu)
    try:
        cpu = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True,
                                      stderr=subprocess.DEVNULL).strip() if platform.system()=='Darwin' else platform.processor()
    except (OSError,subprocess.SubprocessError):
        cpu = platform.machine()
    meta = dict(utc=datetime.now(timezone.utc).isoformat(),cpu=cpu,platform=platform.platform(),
                numpy=np.__version__,scipy=scipy.__version__,gpu=gpu.name or None,
                gpu_status="measured" if gpu.available else "pending",gpu_reason=gpu.reason,
                cublas=None,cuda=None,compute_capability=gpu.compute_capability,
                timing="Warm calls; medians; GPU events for kernels, synchronized wall clock for CG",
                cg_tolerance=1e-8,tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'))
    if gpu.available:
        meta.update(cupy=gpu.cp.__version__,cuda=gpu.cp.cuda.runtime.runtimeGetVersion(),
                    cublas=gpu.cublas.getVersion(gpu.cp.cuda.device.get_cublas_handle()))
    report = dict(schema=1,meta=meta,settings=asdict(s),gemm=[],spmv=[],matrices=[],cg=[])
    with threadpool_limits(limits=s.cpu_threads,user_api="blas"):
        meta['blas'] = threadpool_info()
        for mode in GEMM_MODES:
            for n in s.gemm_sizes:
                report['gemm'].append(_gemm_one(n,mode,gpu,s))
                if gpu.available:
                    gpu.cp.get_default_memory_pool().free_all_blocks()
        for spec in MATRICES:
            a,info = _load_real_matrix(spec,root/'.cache')
            report['matrices'].append(info)
            report['spmv'].append(_spmv_one(a,info,gpu,s))
            del a
            if gpu.available:
                gpu.cp.get_default_memory_pool().free_all_blocks()
        report['cg'] = run_cg_family(gpu if gpu.available else None,s.cg_sides,
                                     repeats=min(3,s.repeats),rtol=1e-8)
    if not all(r['converged'] and (r['gpu_converged'] is not False) for r in report['cg']):
        raise AssertionError("A CG case failed true-residual validation; do not present it as converged")
    result = root/'results'
    result.mkdir(exist_ok=True)
    (result/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


"""Compact FP64 CG benchmark; GPU timings are wall time including scalar syncs."""
from time import perf_counter
import numpy as np
import scipy.sparse as sparse
from scipy.linalg.blas import ddot, daxpy, dscal


def poisson_problem(side):
    """SPD five-point Dirichlet Laplacian; known, deterministic solution."""
    if side < 2:
        raise ValueError("grid side must be >= 2")
    t = sparse.diags([-np.ones(side-1), 2*np.ones(side), -np.ones(side-1)],
                     [-1, 0, 1], format="csr")
    a = sparse.kronsum(t, t, format="csr")
    a.sort_indices()
    exact = np.random.default_rng(42).standard_normal(a.shape[0])
    return a, a @ exact, exact


def _cg_loop(b, vectors, spmv, dot, axpy, scal, tol, maxiter):
    x, r, p, q = vectors
    x.fill(0)
    r[...] = b
    p[...] = r
    rr = dot(r, r)
    norm_b = np.sqrt(rr)
    trace = [1.0 if norm_b else 0.0]
    if not norm_b:
        return 0, trace
    for iteration in range(1, maxiter + 1):
        spmv(p, q)
        curvature = dot(p, q)
        if not np.isfinite(curvature) or curvature <= 0:
            raise ArithmeticError("CG requires a positive definite matrix")
        alpha = rr / curvature
        axpy(alpha, p, x)
        axpy(-alpha, q, r)
        new_rr = dot(r, r)
        trace.append(np.sqrt(max(0., new_rr)) / norm_b)
        if trace[-1] <= tol:
            return iteration, trace
        scal(new_rr / rr, p)
        axpy(1., r, p)
        rr = new_rr
    return maxiter, trace


def cg_cpu(a, b, tol=1e-8, maxiter=2000, vectors=None):
    """Return solution, iteration count and recursive relative-residual trace."""
    vectors = vectors or [np.empty_like(b) for _ in range(4)]
    def spmv(p, q):
        q[:] = a @ p
    iterations, trace = _cg_loop(
        b, vectors, spmv, ddot,
        lambda alpha, x, y: daxpy(x, y, a=alpha),
        lambda alpha, x: dscal(alpha, x), tol, maxiter)
    return vectors[0], iterations, trace


class CGGPU:
    """CuSPARSE SpMV + explicit cuBLAS dot/axpy/scal, FP64 throughout.

    Host-pointer dot results synchronize twice per iteration. This overhead is
    included in solve_ms; no claim of fully asynchronous or optimal CG is made.
    The private handle avoids changing CuPy's shared pointer/math mode.
    """
    def __init__(self, a, b, cp):
        from cupyx.scipy.sparse import csr_matrix
        from cupyx import cusparse
        from cupy.cuda import cublas
        self.cp, self.cb, self.cs = cp, cublas, cusparse
        self.a, self.b = csr_matrix(a), cp.asarray(b)
        self.vectors = [cp.empty_like(self.b) for _ in range(4)]
        self.handle = cublas.create()
        cublas.setPointerMode(self.handle, cublas.CUBLAS_POINTER_MODE_HOST)
        self.scalar = np.empty((), dtype=np.float64)
        self.result = np.empty((), dtype=np.float64)

    def close(self):
        if self.handle is not None:
            self.cb.destroy(self.handle)
            self.handle = None

    def dot(self, x, y):
        self.cb.ddot(self.handle, x.size, x.data.ptr, 1, y.data.ptr, 1,
                     self.result.ctypes.data)
        return float(self.result)

    def axpy(self, alpha, x, y):
        self.scalar[...] = alpha
        self.cb.daxpy(self.handle, x.size, self.scalar.ctypes.data,
                      x.data.ptr, 1, y.data.ptr, 1)

    def scal(self, alpha, x):
        self.scalar[...] = alpha
        self.cb.dscal(self.handle, x.size, self.scalar.ctypes.data, x.data.ptr, 1)

    def solve(self, tol=1e-8, maxiter=2000):
        def spmv(p, q):
            self.cs.spmv(self.a, p, y=q, alpha=1., beta=0.)
        it, trace = _cg_loop(self.b, self.vectors, spmv, self.dot,
                            self.axpy, self.scal, tol, maxiter)
        return self.vectors[0], it, trace


def run_cg_family(gpu=None, sides=(24, 48, 96, 192), repeats=3,
                  rtol=1e-8, maxiter=2000):
    """Return compact per-problem rows; missing GPU observations are None.

    Solve/total times are medians after one full warm-up solve. Pipeline phase
    means and total mean use the same end-to-end trials, so their sum matches.
    GPU end-to-end includes CSR/RHS upload, workspace/handle setup, CG and the
    solution download; excludes matrix generation, CUDA startup and validation.
    CPU solve and resident GPU solve reuse preallocated work vectors.
    """
    if repeats < 1 or not 0 < rtol < 1:
        raise ValueError("repeats >= 1 and 0 < rtol < 1 are required")
    rows, cp, tol = [], (gpu.cp if gpu is not None else None), rtol
    for side in sides:
        a, b, exact = poisson_problem(side)
        common = dict(label=f"Poisson {side}×{side}", grid=side,
                      n=a.shape[0], nnz=a.nnz, precision="FP64")
        vectors = [np.empty_like(b) for _ in range(4)]
        cg_cpu(a, b, tol, maxiter, vectors)
        timings = []
        for _ in range(repeats):
            start = perf_counter()
            x, iterations, trace = cg_cpu(a, b, tol, maxiter, vectors)
            timings.append((perf_counter() - start) * 1000)
        residual = float(np.linalg.norm(b - a @ x) / np.linalg.norm(b))
        row = dict(common, cpu_ms=float(np.median(timings)),
                   iterations=iterations, residual=residual,
                   converged=residual <= tol * 1.05, cpu_trace=trace,
                   solution_error=float(np.linalg.norm(x-exact)/np.linalg.norm(exact)),
                   gpu_ms=None, gpu_total_ms=None, gpu_total_mean_ms=None,
                   upload_ms=None, download_ms=None, pipeline_solve_ms=None,
                   phase_sampling="not run",
                   gpu_iterations=None, gpu_residual=None, gpu_converged=None,
                   gpu_solution_error=None, gpu_trace=None)
        rows.append(row)
        if cp is None:
            continue
        cp.cuda.get_current_stream().synchronize()
        device = CGGPU(a, b, cp)
        try:
            device.solve(tol, maxiter)
            cp.cuda.get_current_stream().synchronize()
            timings, totals, phases = [], [], []
            for _ in range(repeats):
                cp.cuda.get_current_stream().synchronize()
                start = perf_counter()
                dx, iterations, trace = device.solve(tol, maxiter)
                cp.cuda.get_current_stream().synchronize()
                timings.append((perf_counter()-start)*1000)
            x = cp.asnumpy(dx)
            residual = float(np.linalg.norm(b-a@x)/np.linalg.norm(b))
            for _ in range(repeats):
                cp.cuda.get_current_stream().synchronize()
                t0 = perf_counter()
                transient = CGGPU(a, b, cp)
                try:
                    cp.cuda.get_current_stream().synchronize()
                    t1 = perf_counter()
                    dx, _, _ = transient.solve(tol, maxiter)
                    cp.cuda.get_current_stream().synchronize()
                    t2 = perf_counter()
                    cp.asnumpy(dx)
                    cp.cuda.get_current_stream().synchronize()
                    t3 = perf_counter()
                    totals.append((t3-t0)*1000)
                    phases.append(np.diff([t0, t1, t2, t3])*1000)
                finally:
                    transient.close()
            upload_ms, pipeline_ms, download_ms = np.mean(phases, axis=0)
            row.update(gpu_ms=float(np.median(timings)),
                       gpu_total_ms=float(np.median(totals)),
                       gpu_total_mean_ms=float(np.mean(totals)),
                       upload_ms=float(upload_ms), download_ms=float(download_ms),
                       pipeline_solve_ms=float(pipeline_ms),
                       phase_sampling="means from the same end-to-end trials",
                       gpu_iterations=iterations,
                       gpu_residual=residual, gpu_converged=residual <= tol*1.05,
                       gpu_solution_error=float(np.linalg.norm(x-exact)/np.linalg.norm(exact)),
                       gpu_trace=trace)
        finally:
            device.close()
    return rows


"""Presentation charts. Accept records or pandas DataFrames; return matplotlib Figures.

GEMM: mode,n,cpu_ms,gpu_ms,gflops,error. peaks: {mode: peak_TFLOP_s}.
SpMV: matrix,n,nnz,cpu_ms,gpu_ms,gb_s.
CG: label,n,cpu_ms,gpu_ms,upload_ms,download_ms,cpu_history,gpu_history.
Missing/None GPU values always appear as pending, never fabricated measurements.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

INK, MUTED, GRID = "#172337", "#6E7D90", "#E6EBF0"
GREEN, CPU = "#58A600", "#7B899B"
MODES = ("FP64", "FP32", "TF32", "FP16")
COLORS = dict(zip(MODES, ("#364A70", "#2577CB", "#835AC5", GREEN)))


def _records(data):
    return data.to_dict("records") if hasattr(data, "to_dict") else list(data)


def _num(row, key):
    try:
        value = float(row.get(key, np.nan))
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _figure(title, subtitle, titles, widths=None):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, len(titles), figsize=(12.8, 4.7),
                                 gridspec_kw={"width_ratios": widths or [1]*len(titles)})
    fig.patch.set_facecolor("#FBFCFE")
    fig.subplots_adjust(left=.065, right=.975, bottom=.22, top=.71, wspace=.37)
    fig.text(.065, .92, title, fontsize=24, weight="bold", color=INK)
    fig.text(.065, .84, subtitle, fontsize=11.5, color=MUTED)
    for ax, label in zip(np.atleast_1d(axes), titles):
        ax.set_facecolor("#FBFCFE")
        ax.set_title(label, fontsize=13, weight="bold", color=INK, loc="left", pad=15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=10, length=0, pad=7)
        ax.grid(axis="y", color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
    return fig, np.atleast_1d(axes)


def _pending(ax, text="GPU results pending"):
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.text(.5, .53, "—", transform=ax.transAxes, ha="center", fontsize=34, color="#BCC6D1")
    ax.text(.5, .31, text, transform=ax.transAxes, ha="center", va="center",
            color=MUTED, fontsize=11, linespacing=1.6)


def _legend(ax):
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="best")


def _foot(fig, text):
    fig.text(.065, .065, text, fontsize=9.5, color=MUTED)


def plot_gemm(data, peaks=None):
    rows, peaks = _records(data), peaks or {}
    fig, axes = _figure("A  /  Dense GEMM", "Four compute modes · throughput, accuracy and peak utilization",
                         ["Throughput", "Relative error", "GPU peak utilization"], [1.4, 1, 1])
    ax = axes[0]
    gpu = any(np.isfinite(_num(r, "gpu_ms")) for r in rows)
    for mode in MODES:
        subset = sorted([r for r in rows if r.get("mode") == mode], key=lambda r:r["n"])
        for backend, style in (("gpu", "-"), ("cpu", "--")):
            part = [r for r in subset if _num(r, backend+"_ms") > 0]
            if part:
                y = [2*r["n"]**3/(_num(r, backend+"_ms")*1e6) for r in part]
                ax.plot([r["n"] for r in part], y, style, color=COLORS[mode], marker="o",
                        ms=4, lw=2.2, label=f"{mode} {'GPU' if backend=='gpu' else 'CPU'}")
    ax.set(xlabel="Matrix dimension  n", ylabel="GFLOP/s")
    if gpu:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    _legend(ax)
    errors = [(m, max([_num(r,"error") for r in rows if r.get("mode")==m and
                      np.isfinite(_num(r,"gpu_ms")) and _num(r,"error")>=0], default=np.nan)) for m in MODES]
    if any(np.isfinite(e) for _, e in errors):
        axes[1].grid(False)
        for i, (mode, error) in enumerate(errors):
            axes[1].text(0, .9-i*.24, mode, color=COLORS[mode], weight="bold", transform=axes[1].transAxes)
            axes[1].text(.95, .9-i*.24, f"{error:.1e}" if np.isfinite(error) else "pending",
                         ha="right", color=INK if np.isfinite(error) else MUTED, transform=axes[1].transAxes)
        axes[1].set_axis_off()
    else:
        _pending(axes[1], "FP64 · FP32 · TF32 · FP16\nGPU validation pending")
    util = []
    for mode in MODES:
        measured = [2*r["n"]**3/(_num(r,"gpu_ms")*1e6) for r in rows
                    if r.get("mode")==mode and _num(r,"gpu_ms")>0]
        peak = peaks.get(mode)
        util.append(100*max(measured)/(peak*1000) if measured and peak and peak>0 else np.nan)
    if any(np.isfinite(util)):
        axes[2].barh(MODES, util, color=[COLORS[m] for m in MODES], height=.5)
        axes[2].invert_yaxis(); axes[2].set(xlabel="% of matching dense peak", xlim=(0, max(100, np.nanmax(util)*1.15)))
        for i, v in enumerate(util):
            if np.isfinite(v):
                axes[2].text(v+2, i, f"{v:.1f}%", va="center", fontsize=10, color=MUTED)
    else:
        _pending(axes[2], "Add measured GPU data\nand official card peaks")
    _foot(fig, "Dashed = CPU baseline. Peaks: dense, boost-clock specifications; FP16 uses FP32 accumulation. FP64 peak unspecified.")
    return fig


def plot_spmv(data):
    rows = _records(data)
    fig, axes = _figure("B  /  Real sparse matrices", "SuiteSparse · CSR · FP64 · resident library time excludes transfers",
                         ["SpMV latency", "GPU effective bandwidth", "GPU speedup"], [1.2, 1, 1])
    labels = [r["matrix"] for r in rows]; y = np.arange(len(rows))
    for backend, offset, color in (("cpu", -.17, CPU), ("gpu", .17, GREEN)):
        values = [_num(r, backend+"_ms") for r in rows]
        if any(np.isfinite(values)):
            axes[0].scatter(values,y+offset,s=55,label=backend.upper(),color=color,zorder=3)
    axes[0].set(yticks=y, yticklabels=labels, xlabel="Time (ms, log scale)",xscale="log")
    axes[0].invert_yaxis(); _legend(axes[0])
    if any(_num(r,"gb_s")>0 for r in rows):
        axes[1].barh(y, [_num(r,"gb_s") for r in rows], .5, color=GREEN)
        axes[1].set(yticks=y, yticklabels=labels, xlabel="Effective GB/s")
        axes[1].invert_yaxis()
    else:
        _pending(axes[1])
    speed = [_num(r,"cpu_ms")/_num(r,"gpu_ms") if _num(r,"gpu_ms")>0 else np.nan for r in rows]
    if any(np.isfinite(speed)):
        axes[2].barh(y, speed, .5, color=GREEN)
        axes[2].axvline(1, color=CPU, lw=1, ls="--")
        axes[2].set(yticks=y, yticklabels=labels, xlabel="CPU time / GPU time")
        axes[2].invert_yaxis()
    else:
        _pending(axes[2], "CPU baseline measured\nGPU comparison pending")
    _foot(fig, "Effective GB/s is estimated from CSR arrays + one input/output vector pass; it is not measured DRAM traffic.")
    return fig


def plot_cg(data):
    rows = sorted(_records(data), key=lambda r:r["n"])
    fig, axes = _figure("C  /  Conjugate gradient", "SPD problems · same tolerance · GPU end-to-end time includes transfers",
                         ["Time to solution", "GPU time breakdown", "Residual convergence"], [1.15, 1, 1.1])
    for key, label, color, style in (("cpu_ms","CPU",CPU,"--"),("gpu_ms","GPU solve",GREEN,"-"),
                                    ("gpu_total_ms","GPU + transfer","#2577CB","-")):
        part = [r for r in rows if _num(r,key)>0]
        if part:
            axes[0].plot([r["n"] for r in part], [_num(r,key) for r in part], style,
                         color=color, marker="o", ms=4, lw=2.2, label=label)
    axes[0].set(xlabel="Unknowns  n", ylabel="Time (ms)", xscale="log", yscale="log")
    _legend(axes[0])
    measured = [r for r in rows if _num(r,"gpu_ms")>0]
    if measured:
        left = np.zeros(len(measured)); y = np.arange(len(measured))
        for r in measured:
            r["pipeline_solve_ms"] = r.get("pipeline_solve_ms") or r["gpu_ms"]
        for key,label,color in (("upload_ms","Upload","#ADC9E6"),("pipeline_solve_ms","Solve",GREEN),("download_ms","Download","#835AC5")):
            vals = np.array([max(0,_num(r,key)) if np.isfinite(_num(r,key)) else 0 for r in measured])
            axes[1].barh(y,vals,.5,left=left,label=label,color=color); left += vals
        axes[1].set(yticks=y, yticklabels=[r.get("label",f'{r["n"]:,}') for r in measured], xlabel="Time (ms)")
        axes[1].invert_yaxis(); _legend(axes[1])
    else:
        _pending(axes[1], "Upload · solve · download\nGPU measurements pending")
    for r in rows:
        r["cpu_history"] = r.get("cpu_history") or r.get("cpu_trace") or r.get("trace") or []
        r["gpu_history"] = r.get("gpu_history") or r.get("gpu_trace") or []
    histories = [(r, key, label, color) for r in rows[-1:] for key,label,color in
                 (("cpu_history","CPU",CPU),("gpu_history","GPU",GREEN)) if len(r.get(key,[]))]
    for r,key,label,color in histories:
        axes[2].semilogy(np.maximum(r[key],np.finfo(float).tiny), color=color, lw=4 if label=="CPU" else 1.5, linestyle="--" if label=="CPU" else "-", label=label)
    if histories:
        axes[2].set(xlabel="Iteration", ylabel="Relative residual")
        _legend(axes[2])
    else:
        _pending(axes[2], "Residual history pending")
    _foot(fig, "Lines: median solve times. Phase bars: means from the same end-to-end trials. GPU crossover includes transfers.")
    return fig


def plot_roofline():
    """Qualitative model only: no measured hardware coordinates or peak values."""
    fig, (ax,) = _figure("Why the bottlenecks differ", "Roofline model · conceptual, not measured hardware performance",
                          ["Attainable performance ≤ min(compute peak, bandwidth × arithmetic intensity)"])
    fig.set_size_inches(10.8, 4.7)
    ax.plot([.06,.54,.94], [.08,.8,.8], color=INK, lw=3)
    ax.text(.15,.45,"Memory-bound",color="#2577CB",fontsize=13,weight="bold",rotation=32)
    ax.text(.68,.85,"Compute-bound",color=GREEN,fontsize=13,weight="bold")
    ax.annotate("SpMV\nFew FLOPs per byte",xy=(.18,.24),xytext=(.30,.07),color="#2577CB",
                fontsize=12,arrowprops={"arrowstyle":"->","color":"#2577CB"})
    ax.annotate("Large GEMM\nReuse raises intensity",xy=(.80,.80),xytext=(.64,.35),color=GREEN,
                fontsize=12,arrowprops={"arrowstyle":"->","color":GREEN})
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Arithmetic intensity  →  FLOP/byte",
           ylabel="Attainable FLOP/s  →",xticks=[],yticks=[])
    ax.grid(False)
    _foot(fig, "Expected mechanism, not a benchmark conclusion: caches, matrix structure, size and hardware affect the observed regime.")
    return fig


def apply_peaks(report, peaks, source):
    supplied = {m:v for m,v in peaks.items() if v is not None}
    if not supplied:
        return
    Settings(peaks_tflops=supplied,peak_source=source)
    report['settings']['peaks_tflops'] = supplied
    report['settings']['peak_source'] = source
    for row in report['gemm']:
        peak = supplied.get(row['mode'])
        row['peak_utilization'] = row['gflops']/(1000*peak) if peak and row.get('gflops') else None


def analyse(report):
    """Evidence first; unmeasured GPU outcomes never become conclusions."""
    gemm,spmv,cg = report['gemm'],report['spmv'],report['cg']
    cpu = {(r['mode'],r['n']):r for r in gemm if r.get('cpu_ms')}
    shared = sorted(n for mode,n in cpu if mode=='FP32' and ('FP64',n) in cpu)
    if shared:
        n = shared[-1]
        ratio = cpu['FP64',n]['cpu_ms']/cpu['FP32',n]['cpu_ms']
        dense = f'At n = {n:,}, CPU FP32 is {ratio:.2f}× faster than FP64.'
    else:
        dense = 'No shared CPU GEMM size was measured.'
    native = [r for r in gemm if r.get('gpu_ms')]
    if native:
        best = max(native,key=lambda r:r['gflops'])
        dense_gpu = f"GPU best: {best['mode']} at {best['n']:,}, {best['gflops']/1000:.2f} TFLOP/s."
        peaks = [r for r in native if r.get('peak_utilization') is not None]
        if peaks:
            peak = max(peaks,key=lambda r:r['peak_utilization'])
            dense_gpu += f" Best configured-peak utilization: {peak['mode']} {peak['peak_utilization']:.1%}."
        else:
            dense_gpu += ' Official card peaks are still required for utilization.'
    else:
        dense_gpu = 'GPU throughput, precision error and card-peak utilization: not measured.'
    low,high = min(r['nnz'] for r in spmv),max(r['nnz'] for r in spmv)
    sparse_note = f"Three real matrices span {low:,}–{high:,} nonzeros; CPU SpMV takes {min(r['cpu_ms'] for r in spmv):.3f}–{max(r['cpu_ms'] for r in spmv):.3f} ms."
    observed = [r for r in spmv if r.get('gpu_ms')]
    sparse_gpu = (f"GPU wins {sum(r['gpu_ms']<r['cpu_ms'] for r in observed)}/{len(observed)} cases; peak effective bandwidth {max(r['gb_s'] for r in observed):.1f} GB/s."
                  if observed else 'GPU latency, effective GB/s and speedup: not measured.')
    last = cg[-1]
    cg_note = f"All {len(cg)} CPU solves reach 10⁻⁸. Largest: {last['n']:,} unknowns, {last['iterations']} iterations, {last['cpu_ms']:.1f} ms."
    tested = sorted([r for r in cg if r.get('gpu_total_ms')],key=lambda r:r['n'])
    wins = [r for r in tested if r['gpu_total_ms'] < r['cpu_ms']]
    if wins:
        cg_gpu = f"First sampled GPU end-to-end win: n = {wins[0]['n']:,}. This is an observed sample, not an exact crossover threshold."
    elif tested:
        cg_gpu = 'No GPU end-to-end crossover observed within this tested size range.'
    else:
        cg_gpu = 'GPU end-to-end crossover: not measured; transfers and scalar synchronization must be included.'
    intensities = [r['intensity'] for r in spmv]
    theory = f"FP64 GEMM intensity grows as n/12; these SpMV cases are only {min(intensities):.2f}–{max(intensities):.2f} FLOP/byte."
    return dict(gemm=dense,gemm_gpu=dense_gpu,spmv=sparse_note,spmv_gpu=sparse_gpu,
                cg=cg_note,cg_gpu=cg_gpu,roofline=theory,
                status='CPU evidence complete; GPU conclusions pending' if not native else 'Measured GPU results available')


_STYLE = '''
<style>
.cublast{box-sizing:border-box;width:100%;max-width:1280px;min-height:710px;margin:12px auto;padding:30px 46px 24px;background:#fbfcfe;color:#172337;font:16px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;border:1px solid #e6ebf0;border-radius:8px;display:flex;flex-direction:column;position:relative;overflow:hidden}
.cublast *{box-sizing:border-box}.cublast h1{font-size:54px;line-height:1.08;letter-spacing:-2px;margin:26px 0 18px;font-weight:750}.cublast h2{font-size:31px;margin:20px 0 16px;letter-spacing:-.8px}.cublast h3{font-size:21px;margin:0 0 10px}.cublast p{margin:8px 0}.cublast .eyebrow{color:#708097;font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;display:flex;justify-content:space-between}.cublast .green{color:#58a600}.cublast .muted{color:#6e7d90}.cublast .lead{font-size:23px;max-width:900px;line-height:1.4}.cublast .grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:24px 0}.cublast .card{padding:23px 24px;background:#fff;border:1px solid #e6ebf0;border-radius:10px}.cublast .card p{font-size:15px;color:#6e7d90}.cublast .tag{display:inline-block;border-radius:16px;padding:5px 12px;background:#eaf4de;color:#4b850b;font-size:12px;font-weight:700}.cublast .tag.pending{background:#eef1f6;color:#6e7d90}.cublast .figure{width:100%;height:auto;display:block;margin:5px 0 10px}.cublast .finding{border-left:4px solid #58a600;padding:8px 15px;margin:9px 0;background:#f0f6e9;font-size:17px}.cublast .pending-note{color:#6e7d90;font-size:14px;margin:8px 0}.cublast .footer{margin-top:auto;display:flex;gap:18px;justify-content:space-between;color:#8793a3;font-size:11px;padding-top:15px}.cublast .chips{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0}.cublast .chip{background:#edf1f6;border-radius:6px;padding:7px 11px;font-size:13px;color:#526177}.cublast table{width:100%;border-collapse:collapse;font-size:12px;margin:5px 0 7px}.cublast th,.cublast td{text-align:left;padding:5px 9px;border-bottom:1px solid #e6ebf0}.cublast th{color:#708097;font-weight:600}.cublast a{color:#2577cb;text-decoration:none}.cublast .steps{border-radius:8px;background:#172337;color:white;padding:18px 22px;margin:14px 0;font:14px/1.7 ui-monospace,SFMono-Regular,Consolas,monospace}.cublast .summary{font-size:18px;line-height:1.5;margin-bottom:17px;max-width:1040px}.cublast .summary b{display:block;font-size:13px;text-transform:uppercase;letter-spacing:1.3px;color:#58a600;margin-bottom:4px}
@media(max-width:800px){.cublast{padding:20px;min-height:0}.cublast h1{font-size:36px}.cublast .grid3{grid-template-columns:1fr}.cublast .lead{font-size:19px}}
.cublast{border-radius:18px;border-top:5px solid #58a600;background:radial-gradient(ellipse at 100% 0%,#f0f6e7,transparent 45%),#fbfcfe}.cublast .card{box-shadow:0 6px 22px #17233706;border-radius:16px;padding:20px}.cublast .method{font-size:12px;color:#267c78;font-weight:650;padding-top:10px;border-top:1px solid #e6ebf0}.cublast .metric{font-size:58px;letter-spacing:-3px;font-weight:750;color:#267c78;line-height:1.2;margin-bottom:10px}.cublast .gallery .figure{max-height:180px;object-fit:contain;margin:0}.cublast .compact-chart .figure{max-height:355px;object-fit:contain}.cublast .compact-chart{margin-top:-12px}.cublast .finding{border-radius:0 10px 10px 0}.cublast .grid3{margin:18px 0}.cublast .summary{font-size:17px;margin-bottom:12px}
</style>''' 


def _png(figure):
    import matplotlib.pyplot as plt
    buffer = _io.BytesIO()
    figure.savefig(buffer,format='png',dpi=145,facecolor=figure.get_facecolor())
    plt.close(figure)
    return '<img class="figure" alt="Measured benchmark results" src="data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()+'">'


SPEC_URL = 'https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf'


def prepare_visuals(report):
    """Embed verified matrix patterns so saved reports render without downloads."""
    if 'RTX 5090' in (report['meta'].get('gpu') or ''):
        if not report['settings'].get('peaks_tflops'):
            apply_peaks(report, {'FP32':104.8, 'TF32':104.8, 'FP16':209.5}, SPEC_URL)
        report['hardware_reference'] = dict(bandwidth_gbs=1792, l2_bytes=98304*1024, source=SPEC_URL)
    for info in report['matrices']:
        path = Path(__file__).parent/'.cache'/f'{info["name"]}.npz'
        if info.get('pattern') or not path.exists():
            continue
        matrix = sparse.load_npz(path)
        digest = hashlib.sha256()
        for array in (matrix.data, matrix.indices, matrix.indptr):
            digest.update(array.tobytes())
        if digest.hexdigest() != info.get('csr_sha256'):
            continue
        coo = matrix.tocoo(); size = 144
        bins = (coo.row.astype(np.int64)*size//matrix.shape[0])*size + coo.col.astype(np.int64)*size//matrix.shape[1]
        occupied = np.bincount(bins, minlength=size*size)>0
        info['pattern'] = base64.b64encode(np.packbits(occupied).tobytes()).decode()
        info['pattern_size'] = size


def matrix_gallery(report):
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(1,3,figsize=(12.8,2.15))
    fig.patch.set_facecolor('#FBFCFE')
    fig.subplots_adjust(left=.04,right=.98,top=.84,bottom=.12,wspace=.12)
    for ax,m in zip(axes,report['matrices']):
        if m.get('pattern'):
            size=m['pattern_size']
            grid=np.unpackbits(np.frombuffer(base64.b64decode(m['pattern']),dtype=np.uint8))[:size*size].reshape(size,size)
            ax.imshow(grid,cmap=ListedColormap(['#EDF2F6','#267C78']),interpolation='nearest')
        ax.set_axis_off()
        ax.set_title(f"{m['name']}  /  {m['m']:,} × {m['n']:,}",fontsize=11,color=INK,pad=8)
        ax.text(.5,-.10,f"{m['nnz']:,} nonzeros",ha='center',transform=ax.transAxes,fontsize=10,color=MUTED)
    return fig


def method_svg(kind):
    # Schematics illustrate operations; empirical patterns appear on the sparse slide.
    cells=''
    if kind=='dense':
        for block,(x,color) in enumerate([(18,'#2577CB'),(142,'#835AC5'),(266,'#58A600')]):
            for i in range(5):
                for j in range(5):
                    cells+=f'<rect x="{x+j*13}" y="{25+i*13}" width="10" height="10" rx="2" fill="{color}" opacity="{.35+.12*((i+j)%5)}"/>'
        cells+='<text x="104" y="65">×</text><text x="230" y="65">=</text>'
        cells+='<text x="40" y="116" class="sm">A</text><text x="164" y="116" class="sm">B</text><text x="288" y="116" class="sm">C</text>'
    elif kind=='sparse':
        for i in range(7):
            for j in range(7):
                fill='#267C78' if i==j or abs(i-j)==1 else '#E9EFF3'
                cells+=f'<rect x="{18+j*10}" y="{22+i*10}" width="8" height="8" rx="1" fill="{fill}"/>'
        cells+='<text x="113" y="64">→</text><rect x="157" y="32" width="154" height="49" rx="10" fill="#E1F0EB"/><text x="234" y="62" text-anchor="middle" class="sm">CSR · SpMV</text><text x="20" y="116" class="sm">Nonzeros only → y = Ax</text>'
    else:
        for x,y in [(53,23),(53,49),(53,75),(27,49),(79,49)]:
            cells+=f'<path d="M53 49L{x} {y}" stroke="#B8C7D8" stroke-width="3"/><circle cx="{x}" cy="{y}" r="8" fill="#835AC5"/>'
        cells+='<text x="124" y="44" class="sm">SpMV → dot → axpy</text><path d="M140 63H300V85H140Z" fill="none" stroke="#835AC5" stroke-width="2"/><text x="220" y="80" text-anchor="middle" class="sm">repeat until converged</text><text x="20" y="116" class="sm">5-point Poisson · SPD</text>'
    return '<svg viewBox="0 0 350 135" role="img" aria-label="'+kind+' method schematic" style="width:100%;font:28px Arial;fill:#172337"><style>.sm{font:13px Arial}</style>'+cells+'</svg>'


def plot_interpretation(report):
    fig, axes = _figure('Data reuse changes the limit', 'A model explains the trend; cache behavior needs separate evidence.', ['Arithmetic intensity', 'Sparse working set vs L2'], [1.1,1])
    ax=axes[0]
    ns=np.array(sorted(set(r['n'] for r in report['gemm'])))
    ax.plot(ns,ns/12,color=COLORS['FP64'],lw=2.5,marker='o',label='FP64 GEMM: n/12')
    for r in report['spmv']:
        ax.scatter([r['n']],[r['intensity']],s=55,color='#267C78',zorder=4)
    ax.axhspan(.14,.18,color='#267C78',alpha=.12)
    ax.text(350,.28,'SpMV ≈ 0.15 FLOP/byte',color='#267C78',fontsize=11)
    ax.set(xscale='log',yscale='log',xlabel='Matrix dimension',ylabel='FLOP / estimated byte')
    _legend(ax)
    ref=report.get('hardware_reference',{}); cache=ref.get('l2_bytes')
    rows=report['spmv']; sizes=[r['estimated_bytes']/2**20 for r in rows]
    axes[1].barh([r['matrix'] for r in rows],sizes,color='#267C78',height=.48)
    for i,v in enumerate(sizes):
        axes[1].text(v+1,i,f'{v:.2f} MiB',va='center',fontsize=10,color=INK)
    if cache:
        axes[1].axvline(cache/2**20,color=COLORS['TF32'],ls='--',lw=2)
        axes[1].text(cache/2**20-1,-.58,f'L2 {cache/2**20:.0f} MiB',ha='right',color=COLORS['TF32'],fontsize=11)
        axes[1].set_xlim(0,cache/2**20*1.12)
    axes[1].set_xlabel('CSR + input + output (MiB)'); axes[1].invert_yaxis()
    _foot(fig,'GEMM assumes ideal reuse; SpMV uses a byte model. Working-set fit suggests possible cache reuse, not a measured hit rate.')
    return fig


def pages(report):
    prepare_visuals(report)
    a=analyse(report); report['analysis']=a
    esc=html.escape; meta=report['meta']; s=report['settings']; gpu=bool(meta.get('gpu'))
    native=[r for r in report['gemm'] if r.get('gpu_ms')]
    sparse_gpu=[r for r in report['spmv'] if r.get('gpu_ms')]
    tested=sorted([r for r in report['cg'] if r.get('gpu_total_ms')],key=lambda r:r['n'])
    best=max(native,key=lambda r:r['gflops']) if native else None
    last=tested[-1] if tested else report['cg'][-1]
    wins=[r for r in tested if r['gpu_total_ms']<r['cpu_ms']]
    dense_note=a['gemm_gpu']
    if best:
        fp32=next(r for r in native if r['mode']=='FP32' and r['n']==best['n'])
        dense_note=f"{best['mode']} reaches {best['gflops']/1000:.1f} TFLOP/s — {fp32['gpu_ms']/best['gpu_ms']:.2f}× FP32 at n = {best['n']:,}."
    sparse_note=a['spmv_gpu']
    cache=report.get('hardware_reference',{}).get('l2_bytes')
    cache_note='SpMV has little arithmetic reuse; latency and cache behavior matter.'
    if sparse_gpu and cache:
        largest=max(sparse_gpu,key=lambda r:r['estimated_bytes'])
        cache_note=f"Largest working set: {largest['estimated_bytes']/2**20:.1f} MiB < {cache/2**20:.0f} MiB L2. Warm repetitions may reuse cache; effective GB/s does not establish DRAM saturation."
    cg_note=a['cg_gpu']
    if wins:
        cg_note=f"First sampled GPU win: {wins[0]['n']:,} unknowns. Largest solve is {last['cpu_ms']/last['gpu_total_ms']:.2f}× faster including transfers."
    report['analysis'].update(gemm_gpu=dense_note,cache=cache_note,cg_gpu=cg_note)
    def wrap(i,label,body,foot=''):
        return f'<section class="cublast"><div class="eyebrow"><span>cuBLAST / {label}</span><span>{i:02d} / 06</span></div>{body}<div class="footer"><span>Numerical Libraries Mini-Hackathon · {esc(meta.get("gpu") or "CPU baseline / GPU pending")}</span><span>{foot}</span></div></section>'
    def finding(t): return '<div class="finding">'+esc(t)+'</div>'
    first='<h1>Dense. Sparse.<br><span class="green">Time to solution.</span></h1><p class="lead">Three experiments connect matrix structure to GPU performance.</p><div class="grid3">'
    for title,kind,desc,method in [('01 / Dense GEMM','dense','Random square matrices · n = 256–4,096','cuBLAS gemmEx · four precision modes'),('02 / Real sparse SpMV','sparse','3 SuiteSparse structural matrices','Matrix Market → CSR → cuSPARSE'),('03 / Iterative solve','cg','2D Poisson · 576–36,864 unknowns','CG · SpMV + cuBLAS vector operations')]:
        first+=f'<div class="card"><h3>{title}</h3>{method_svg(kind)}<p>{desc}</p><div class="method">{method}</div></div>'
    first+='</div><div class="chips">'+''.join(f'<span class="chip">{t}</span>' for t in [f"{s['warmup']} warm-ups",f"{s['repeats']} timing samples",'1 CPU BLAS thread','Synchronized GPU timing','Numerical checks'])+'</div>'
    out=[wrap(1,'Design',first,'Schematics illustrate the methods; measured matrix structures follow')]
    out.append(wrap(2,'A / Precision',_png(plot_gemm(report['gemm'],s['peaks_tflops']))+finding(dense_note)+'<div class="pending-note">Error: maximum sampled relative error vs common FP64 reference, including input rounding. FP16 inputs accumulate/output in FP32.</div>', 'Dense nominal peaks · NVIDIA Blackwell whitepaper, Table 3'))
    gallery=_png(matrix_gallery(report)); chart=_png(plot_spmv(report['spmv']))
    # Shorter subplot image keeps the pattern gallery and metrics on one slide.
    out.append(wrap(3,'B / Structure → performance','<h2>Real structure. Measured speed.</h2><div class="gallery">'+gallery+'</div><div class="compact-chart">'+chart+'</div>'+finding(sparse_note),'Patterns: occupied 144 × 144 bins of the verified CSR matrices · not synthetic'))
    out.append(wrap(4,'Interpretation',_png(plot_interpretation(report))+finding(a['roofline'])+'<div class="pending-note">'+esc(cache_note)+'</div>','Bandwidth and cache reference: NVIDIA Blackwell whitepaper, Table 3'))
    solve_note=''
    if tested:
        overhead=(last['upload_ms']+last['download_ms'])/last['gpu_total_mean_ms']*100
        solve_note=f"Largest case: setup + transfers are {overhead:.1f}% of mean end-to-end time. CPU/GPU residual traces overlap; both reach 10⁻⁸. Small-case costs plausibly include launches and host-scalar synchronization."
    out.append(wrap(5,'C / Convergence',_png(plot_cg(report['cg']))+finding(cg_note)+'<div class="pending-note">'+esc(solve_note or a['cg'])+'</div>','Unpreconditioned CG · FP64 · SPD five-point stencil · equal tolerance'))
    metrics=[(f"{best['gflops']/1000:.0f}" if best else '—','TFLOP/s','FP16 → FP32 accumulation'),(f"{max(r['cpu_ms']/r['gpu_ms'] for r in sparse_gpu):.1f}×" if sparse_gpu else '—','SpMV speedup','Largest real matrix · resident'),(f"{last['cpu_ms']/last['gpu_total_ms']:.2f}×" if tested else '—','CG speedup','Largest solve · transfers included')]
    body='<h1>Performance has<br><span class="green">three conditions.</span></h1><div class="grid3">'
    for value,label,desc in metrics:
        body+=f'<div class="card"><div class="metric">{value}</div><h3>{label}</h3><p>{desc}</p></div>'
    body+='</div><div class="summary"><b>01 / Choose precision</b>FP16/TF32 trade input precision for throughput. Validate error against the application’s needs.</div><div class="summary"><b>02 / Account for data reuse</b>SpMV has low arithmetic intensity. This warm dataset fits L2; DRAM saturation remains unproven.</div><div class="summary"><b>03 / Measure the whole solve</b>'+esc(cg_note)+'</div>'
    body+='<div class="pending-note">Scope: one GPU, one CPU BLAS thread, five timing samples; no profiler. Peaks use matching dense modes and nominal boost clocks.</div>'
    body+=f'<div class="pending-note">Sources: <a href="{SPEC_URL}">NVIDIA architecture / dense peaks / cache</a> · <a href="https://sparse.tamu.edu/">SuiteSparse / Davis & Hu (2011)</a> · <a href="https://docs.nvidia.com/cuda/cublas/index.html">cuBLAS</a></div>'
    out.append(wrap(6,'Takeaways',body,esc(meta['utc'][:19].replace('T',' '))+' UTC'))
    return out


def export_html(report, root=None, rendered=None):
    root = Path(root or Path(__file__).parent)
    rendered = rendered or pages(report)
    target = root/'results'
    target.mkdir(exist_ok=True)
    (target/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    controls = '''<div id="nav"><button onclick="move(-1)">←</button><span id="count"></span><button onclick="move(1)">→</button></div><script>let current=0;const slides=[...document.querySelectorAll('.deck>.cublast')];function move(d){current=Math.max(0,Math.min(slides.length-1,current+d));slides.forEach((s,i)=>s.classList.toggle('active',i===current));document.getElementById('count').textContent=(current+1)+' / '+slides.length}document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();move(1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();move(-1)}});move(0)</script>'''
    style = '''<style>body{margin:0;background:#e9edf2}.deck{padding:20px 0 70px}.deck>.cublast{display:none;box-shadow:0 8px 40px #17233712}.deck>.active{display:flex}#nav{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);display:flex;align-items:center;gap:24px;border-radius:30px;background:#172337;color:white;padding:8px 16px;font:13px Arial}button{border:0;background:transparent;color:white;font-size:20px;cursor:pointer}@media print{body{background:white}.deck{padding:0}.deck>.cublast{display:flex;break-after:page;margin:0;box-shadow:none;border:0}#nav{display:none}@page{size:landscape;margin:0}}</style>'''
    content = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>cuBLAST · Two kernels, two limits</title>'+_STYLE+style+'</head><body><main class="deck">'+''.join(rendered)+'</main>'+controls+'</body></html>'
    (target/'presentation.html').write_text(content,encoding='utf-8')
    return target/'presentation.html'


def show(rendered,index):
    from IPython.display import HTML, display
    display(HTML(_STYLE+rendered[index-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the benchmark or rerender a saved report.')
    parser.add_argument('--run',action='store_true',help='Run CPU and available GPU experiments')
    parser.add_argument('--input',type=Path,help='Render an existing results JSON without rerunning')
    parser.add_argument('--peaks',type=json.loads,default={},help='Official dense TFLOP/s as a JSON mode:value map')
    parser.add_argument('--peak-source',default='',help='Official card specification URL')
    args = parser.parse_args()
    root = Path(__file__).parent
    if args.input:
        report = json.loads(args.input.read_text())
        apply_peaks(report,args.peaks,args.peak_source)
    elif args.run:
        report = run(Settings(peaks_tflops=args.peaks,peak_source=args.peak_source),root)
    else:
        parser.error('Use --run or --input results/report.json')
    print(export_html(report,root))
