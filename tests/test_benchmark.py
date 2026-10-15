"""Verify numerical correctness and measurement bookkeeping, not speed thresholds."""
import os
import numpy as np
import pandas as pd
import pytest
from scipy import io, sparse

from benchmark import (Config, GPU, check_error, correctness_demo, detect_gpu,
                       environment, estimated_spmv_bytes, export_results, load_mtx,
                       make_csr, relative_error, run_gemm, run_spmv, speedups)
from plots import plot_gemm, plot_spmv, plot_transfers


@pytest.fixture
def config():
    return Config(gemm_sizes=(16, 32), cpu_max_n=32, sparse_sizes=(20, 40),
                  nnz_per_row=4, warmup=1, repeats=3, spmv_inner=2,
                  gpu_gemm_inner=2, enable_gpu=False)


def test_cpu_notebook_pipeline(config, tmp_path):
    """Exercise the same data, plotting, and export path as the notebook."""
    import matplotlib.pyplot as plt
    gpu = detect_gpu(False)
    checks = correctness_demo(gpu, config)
    assert checks.passed.all() and len(checks) == 4
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
                                    dict(gemm_sizes=(16, 16)), dict(gpu_memory_fraction=1.1)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


@pytest.mark.skipif(os.environ.get("RUN_CUDA_TESTS") != "1", reason="Set RUN_CUDA_TESTS=1 on NVIDIA GPU")
def test_real_cuda_pipeline(config):
    gpu = detect_gpu()
    assert gpu.available, gpu.reason
    assert correctness_demo(gpu, config).passed.all()
    for runner in (run_gemm, run_spmv):
        frame, samples, _ = runner(config, gpu)
        assert set(frame.backend) == {"CPU", "GPU"}
        assert (speedups(frame).speedup > 0).all()
        pipeline = samples[(samples.backend == "GPU") & (samples.scope != "compute")]
        trials = pipeline.pivot(index=["case", "dtype", "trial"], columns="scope", values="seconds")
        assert np.allclose(trials.upload + trials.compute_wall + trials.download,
                           trials.end_to_end, rtol=1e-10, atol=1e-12)
