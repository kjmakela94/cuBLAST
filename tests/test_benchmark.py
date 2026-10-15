"""Verify numerical correctness and measurement bookkeeping, not speed thresholds."""
import os
import ctypes
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from scipy import io, sparse

from benchmark import (Config, GPU, check_error, correctness_demo, detect_gpu,
                       environment, estimated_spmv_bytes, export_results, load_mtx,
                       make_csr, relative_error, run_gemm, run_spmv, speedups,
                       GEMM_MODES, check_gemm_error, gemm_operation, gemm_skip_reason)
from plots import plot_gemm, plot_spmv, plot_transfers


@pytest.fixture
def config():
    return Config(gemm_sizes=(16, 32), cpu_max_n=32, sparse_sizes=(20, 40),
                  gemm_modes=("FP32", "FP64"),
                  nnz_per_row=4, warmup=1, repeats=3, spmv_inner=2,
                  gpu_gemm_inner=2, enable_gpu=False)


def test_cpu_notebook_pipeline(config, tmp_path):
    """Exercise the same data, plotting, and export path as the notebook."""
    import matplotlib.pyplot as plt
    gpu = detect_gpu(False)
    checks = correctness_demo(gpu, config)
    assert checks[checks.status == "passed"].passed.all()
    assert len(checks[checks.status == "passed"]) == 4
    assert set(checks[checks.status == "skipped"].backend) == {"GPU"}
    gemm, gs, gskip = run_gemm(config, gpu)
    spmv, ss, sskip = run_spmv(config, gpu)
    results = pd.concat([gemm, spmv], ignore_index=True)
    samples = pd.concat([gs, ss], ignore_index=True)
    assert len(results) == 8
    assert set(results.backend) == {"CPU"}
    assert np.allclose(results.gflops, results.flops / results.median_s / 1e9)
    for _, row in results.iterrows():
        subset = samples[(samples.case == row["case"]) & (samples.dtype == row["dtype"])]
        assert len(subset) == config.repeats
        assert row.median_s == pytest.approx(subset.seconds.median())
    assert speedups(results).empty
    out = export_results(tmp_path / "run", config, environment(gpu, config), results,
                         samples, pd.concat([gskip, sskip]), checks)
    assert len(pd.read_csv(out / "results.csv")) == 8
    for name, figure in (("gemm", plot_gemm(results)), ("spmv", plot_spmv(results)),
                         ("transfers", plot_transfers(samples))):
        figure.savefig(out / f"{name}.png")
        plt.close(figure)
        assert (out / f"{name}.png").stat().st_size > 1000


def test_csr_generation_reproducible_and_linear_storage():
    a = make_csr(1000, 8, "float32", 42)
    b = make_csr(1000, 8, "float32", 42)
    assert a.has_canonical_format
    assert (a != b).nnz == 0
    assert a.nnz <= 8000
    assert a.dtype == np.float32
    assert estimated_spmv_bytes(a) == (a.data.nbytes + a.indices.nbytes +
                                       a.indptr.nbytes + 2*1000*4)


def test_matrix_market_symmetric_duplicates_and_rectangular(tmp_path, config):
    symmetric = tmp_path / "symmetric.mtx"
    symmetric.write_text("%%MatrixMarket matrix coordinate real symmetric\n"
                         "3 3 5\n1 1 4\n2 2 3\n3 3 2\n3 1 0.5\n3 1 0.5\n")
    a = load_mtx(symmetric, "float32")
    assert np.array_equal(a @ np.array([1, 2, 3]), [7, 6, 7])
    assert a.has_canonical_format and a.indices.dtype == np.int32
    rectangular = tmp_path / "rectangular.mtx"
    io.mmwrite(rectangular, sparse.coo_matrix([[1., 0, 2], [0, 3, 0]]))
    frame, _, _ = run_spmv(config, GPU(), [rectangular])
    rows = frame[frame.case == "mtx_rectangular"]
    assert len(rows) == 2 and (rows.m == 2).all() and (rows.n == 3).all()
    assert (rows.relative_error < 1e-5).all()
    complex_path = tmp_path / "complex.mtx"
    io.mmwrite(complex_path, sparse.coo_matrix([[1+2j]]))
    with pytest.raises(ValueError, match="real-valued"):
        load_mtx(complex_path)


def test_correctness_rejects_bad_results():
    with pytest.raises(AssertionError):
        check_error([1, 100], [1, 2], "float32")
    with pytest.raises(AssertionError, match="Non-finite"):
        relative_error([np.nan], [1])
    assert relative_error([0, 0], [0, 0]) == 0


def test_budget_skips_without_fake_data(config):
    config.max_host_bytes = 1
    frame, samples, skips = run_gemm(config, GPU())
    assert frame.empty and samples.empty and len(skips) == 4
    frame, samples, skips = run_spmv(config, GPU())
    assert frame.empty and samples.empty and len(skips) == 4


def test_speedups_only_join_matching_cpu_cases():
    base = dict(operation="GEMM", dtype="float32", m=16, n=16, nnz=256)
    data = pd.DataFrame([
        dict(base, case="a", backend="CPU", scope="compute", median_s=4.),
        dict(base, case="a", backend="GPU", scope="compute", median_s=1.),
        dict(base, case="a", backend="GPU", scope="end_to_end", median_s=8.),
        dict(base, case="unmatched", backend="GPU", scope="compute", median_s=1.),
    ])
    result = speedups(data)
    assert list(result.speedup) == [4., .5]


@pytest.mark.parametrize("kwargs", [dict(repeats=0), dict(dtypes=("float16",)),
                                    dict(gemm_sizes=(16, 16)), dict(gpu_memory_fraction=1.1),
                                    dict(gemm_modes=("BF16",)), dict(gemm_modes=("TF32", "TF32"))])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


@pytest.mark.skipif(os.environ.get("RUN_CUDA_TESTS") != "1", reason="Set RUN_CUDA_TESTS=1 on NVIDIA GPU")
def test_real_cuda_pipeline(config):
    gpu = detect_gpu()
    assert gpu.available, gpu.reason
    config.gemm_modes = ("FP64", "FP32", "TF32", "FP16")
    checks = correctness_demo(gpu, config)
    assert checks[checks.status == "passed"].passed.all()
    for runner in (run_gemm, run_spmv):
        frame, samples, _ = runner(config, gpu)
        assert set(frame.backend) == {"CPU", "GPU"}
        assert (speedups(frame).speedup > 0).all()
        pipeline = samples[(samples.backend == "GPU") & (samples.scope != "compute")]
        trials = pipeline.pivot(index=["case", "mode", "trial"], columns="scope", values="seconds")
        assert np.allclose(trials.upload + trials.compute_wall + trials.download,
                           trials.end_to_end, rtol=1e-10, atol=1e-12)
        if runner == run_gemm:
            supported = {mode for mode in config.gemm_modes if not gemm_skip_reason(gpu, mode)}
            assert set(frame[frame.backend == "GPU"]["mode"]) == supported


def test_four_mode_cpu_only_run_reports_gpu_skips(config):
    config.gemm_modes = Config().gemm_modes
    assert set(config.gemm_modes) == {"FP64", "FP32", "TF32", "FP16"}
    frame, _, skips = run_gemm(config, GPU(reason="CPU-only test"))
    assert set(frame["mode"]) == {"FP32", "FP64"}
    assert set(skips["mode"]) == set(config.gemm_modes)
    assert set(skips.backend) == {"GPU"}
    assert len(skips) == len(config.gemm_modes)*len(config.gemm_sizes)
    checks = correctness_demo(GPU(), config)
    assert len(checks[checks.status == "skipped"]) == 4
    assert checks[checks.status == "skipped"].passed.isna().all()


def test_low_precision_speedups_use_cpu_fp32_without_merging_modes():
    base = dict(operation="GEMM", case="same", m=16, n=16, nnz=256, scope="compute")
    data = pd.DataFrame([
        dict(base, backend="CPU", mode="FP32", dtype="float32", median_s=8.),
        dict(base, backend="CPU", mode="FP64", dtype="float64", median_s=20.),
        dict(base, backend="GPU", mode="FP32", dtype="float32", median_s=4.),
        dict(base, backend="GPU", mode="TF32", dtype="float32", median_s=2.),
        dict(base, backend="GPU", mode="FP16", dtype="float16", median_s=1.),
        dict(base, backend="GPU", mode="FP64", dtype="float64", median_s=5.),
    ])
    ratios = speedups(data).set_index("mode")
    assert len(ratios) == 4
    assert ratios.loc["TF32", "speedup"] == 4
    assert ratios.loc["FP16", "speedup"] == 8
    assert ratios.loc["FP64", "speedup"] == 4
    assert ratios.loc["TF32", "cpu_reference_mode"] == "FP32"
    assert ratios.loc["FP16", "cpu_reference_mode"] == "FP32"


class RecordingCublas:
    """CPU-only contract test: record pointers and enums, never simulate GPU timings."""
    CUBLAS_POINTER_MODE_HOST = 0
    CUBLAS_DEFAULT_MATH = 0
    CUBLAS_OP_N = 0
    CUBLAS_GEMM_DEFAULT = -1
    CUBLAS_COMPUTE_64F = 70
    CUBLAS_COMPUTE_32F_PEDANTIC = 69
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77
    CUBLAS_COMPUTE_32F = 68

    def __init__(self):
        self.calls = []
        self.destroyed = []

    def create(self):
        return 123

    def destroy(self, handle):
        self.destroyed.append(handle)

    def setPointerMode(self, handle, mode):
        assert (handle, mode) == (123, 0)

    def setMathMode(self, handle, mode):
        assert (handle, mode) == (123, 0)

    def gemmEx(self, *args):
        scalar_type = ctypes.c_double if args[-2] == 70 else ctypes.c_float
        assert scalar_type.from_address(args[6]).value == 1
        assert scalar_type.from_address(args[13]).value == 0
        self.calls.append(args)


def contract_gpu():
    cp = SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(CUDA_R_16F=2, CUDA_R_32F=0, CUDA_R_64F=1),
        get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))
    return GPU(cp=cp, cublas=RecordingCublas(), compute_capability=(8, 0))


def buffer(array):
    return SimpleNamespace(shape=array.shape, dtype=array.dtype, flags=array.flags,
                           data=SimpleNamespace(ptr=array.ctypes.data))


@pytest.mark.parametrize("mode,input_type,output_type,compute_type", [
    ("FP64", 1, 1, 70), ("FP32", 0, 0, 69),
    ("TF32", 0, 0, 77), ("FP16", 2, 0, 68),
])
def test_cublas_contract_rectangular_layout_scalars_and_compute_type(
        mode, input_type, output_type, compute_type, monkeypatch):
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    gpu = contract_gpu()
    spec = GEMM_MODES[mode]
    a = np.ones((3, 7), dtype=spec["input_dtype"], order="F")
    b = np.ones((7, 5), dtype=spec["input_dtype"], order="F")
    c = np.empty((3, 5), dtype=spec["output_dtype"], order="F")
    with gemm_operation(gpu, buffer(a), buffer(b), buffer(c), mode) as call:
        call()
    args = gpu.cublas.calls[0]
    assert args[:6] == (123, 0, 0, 3, 5, 7)
    assert (args[7], args[8], args[9]) == (a.ctypes.data, input_type, 3)
    assert (args[10], args[11], args[12]) == (b.ctypes.data, input_type, 7)
    assert (args[14], args[15], args[16]) == (c.ctypes.data, output_type, 3)
    assert args[-2:] == (compute_type, -1)
    assert gpu.cublas.destroyed == [123]


def test_tf32_hardware_and_environment_gates(monkeypatch):
    gpu = contract_gpu()
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    assert gemm_skip_reason(gpu, "TF32") == ""
    gpu.compute_capability = (7, 5)
    assert "8.0+" in gemm_skip_reason(gpu, "TF32")
    assert gemm_skip_reason(gpu, "FP16") == ""
    gpu.compute_capability = (8, 0)
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    assert "NVIDIA_TF32_OVERRIDE=0" in gemm_skip_reason(gpu, "TF32")
    assert gemm_skip_reason(gpu, "FP32") == ""


def test_gemm_buffer_validation_and_handle_cleanup():
    gpu = contract_gpu()
    a = np.ones((3, 7), dtype="float32", order="F")
    b = np.ones((7, 5), dtype="float32", order="F")
    c = np.empty((3, 5), dtype="float32", order="F")
    with pytest.raises(RuntimeError, match="experiment failed"):
        with gemm_operation(gpu, buffer(a), buffer(b), buffer(c), "FP32"):
            raise RuntimeError("experiment failed")
    assert gpu.cublas.destroyed == [123]
    with pytest.raises(ValueError, match="Fortran-contiguous"):
        with gemm_operation(gpu, buffer(a.copy(order="C")), buffer(b), buffer(c), "FP32"):
            pass


def test_gemm_precision_error_thresholds():
    assert check_gemm_error([1.001], [1.], "TF32") == pytest.approx(.001)
    assert check_gemm_error([1.001], [1.], "FP16") == pytest.approx(.001)
    with pytest.raises(AssertionError):
        check_gemm_error([1.001], [1.], "FP32")
    with pytest.raises(AssertionError):
        check_gemm_error([np.inf], [1.], "FP16")


def test_figures_keep_tf32_distinct_from_fp32():
    import matplotlib.pyplot as plt
    # Synthetic fixture for plotting bookkeeping, not exported benchmark data.
    rows = []
    samples = []
    for mode in ("FP64", "FP32", "TF32", "FP16"):
        dtype = GEMM_MODES[mode]["input_dtype"]
        base = dict(operation="GEMM", case="fixture", m=32, n=32, nnz=1024,
                    mode=mode, dtype=dtype, backend="GPU")
        rows.append(dict(base, scope="compute", median_s=.001, q25_s=.0009,
                         q75_s=.0011, gflops=1., relative_error=.0001))
        for phase, seconds in (("upload", .001), ("compute_wall", .002), ("download", .001)):
            samples.append(dict(base, scope=phase, trial=0, seconds=seconds))
    results = pd.DataFrame(rows)
    figure = plot_gemm(results)
    labels = figure.axes[1].get_legend_handles_labels()[1]
    assert set(labels) == {"GPU FP64", "GPU FP32", "GPU TF32", "GPU FP16"}
    plt.close(figure)
    figure = plot_transfers(pd.DataFrame(samples))
    labels = [label.get_text() for label in figure.axes[0].get_xticklabels()]
    assert len(labels) == 4 and any("TF32" in label for label in labels)
    plt.close(figure)
