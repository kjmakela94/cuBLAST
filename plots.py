"""Notebook-friendly figures; every plotted point comes from measured data."""
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

from benchmark import speedups


COLORS = {("CPU", "float32"): "#52677f", ("CPU", "float64"): "#a3afbd",
          ("GPU", "float32"): "#4f9900", ("GPU", "float64"): "#99c94c"}
MODE_COLORS = {"FP64": "#99c94c", "FP32": "#4f9900", "TF32": "#2583b8", "FP16": "#9461c9"}


def _gemm_color(backend, mode):
    if backend == "CPU":
        return COLORS[("CPU", "float64" if mode == "FP64" else "float32")]
    return MODE_COLORS[mode]


def _style():
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 12,
                         "axes.labelsize": 10, "font.size": 10})


def _empty(ax, message):
    ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes,
            color="#657287", wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_gemm(results):
    _style()
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), layout="constrained")
    fig.suptitle("GEMM | FP64, FP32, TF32 and FP16 (FP32 accumulation)", fontsize=16, fontweight="bold")
    data = results[(results.operation == "GEMM") & (results.scope == "compute")] if not results.empty else results
    if data.empty:
        for ax in axes:
            _empty(ax, "No GEMM measurements")
        return fig
    data = data.copy()
    if "mode" not in data:
        data["mode"] = data.dtype.map({"float32": "FP32", "float64": "FP64"})
    for (backend, mode), group in data.groupby(["backend", "mode"]):
        group = group.sort_values("n")
        x = group.n.to_numpy()
        color = _gemm_color(backend, mode)
        label = f"{backend} {mode}"
        low = (group.median_s - group.q25_s).to_numpy() * 1000
        high = (group.q75_s - group.median_s).to_numpy() * 1000
        axes[0].errorbar(x, group.median_s * 1000, yerr=[low, high],
                         marker="o", capsize=3, color=color, label=label)
        axes[1].plot(x, group.gflops, "o-", color=color, label=label)
        axes[3].plot(x, np.maximum(group.relative_error, 1e-16), "o-", color=color, label=label)
    axes[0].set(ylabel="Median time (ms), whiskers = IQR", yscale="log", title="Operation time")
    axes[1].set(ylabel="GFLOP/s", title="Achieved throughput")
    for ax in axes[:2]:
        ax.set(xlabel="Square matrix size n", xscale="log", xticks=sorted(data.n.unique()))
        ax.set_xticklabels(sorted(data.n.unique()))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    ratios = speedups(results)
    ratios = ratios[ratios.operation == "GEMM"] if not ratios.empty else ratios
    if ratios.empty:
        _empty(axes[2], "GPU speedup unavailable\nRequires matched CPU and GPU measurements")
    else:
        for (mode, scope), group in ratios.groupby(["mode", "scope"]):
            group = group.sort_values("n")
            axes[2].plot(group.n, group.speedup, marker="o",
                         linestyle="-" if scope == "compute" else "--",
                         color=MODE_COLORS[mode], label=f"{mode} {scope}")
        axes[2].axhline(1, color="#52677f", linestyle=":", label="Equal time")
        axes[2].set(xlabel="Square matrix size n", ylabel="CPU time / GPU time",
                    xscale="log", yscale="log")
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=.2)
    axes[2].set_title("GPU speedup (>1 means faster)")
    axes[2].set_ylabel("CPU baseline / GPU time\nTF32 and FP16 use CPU FP32")
    axes[3].set(title="Sampled error vs common FP64 inputs", xlabel="Square matrix size n",
                ylabel="Relative norm error (floor 1e-16)", xscale="log", yscale="log")
    axes[3].grid(alpha=.2)
    axes[3].legend(fontsize=8)
    return fig


def plot_spmv(results):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), layout="constrained")
    fig.suptitle("CSR SpMV | measured operations on resident data", fontsize=16, fontweight="bold")
    data = results[(results.operation == "SpMV") & (results.scope == "compute")] if not results.empty else results
    if data.empty:
        for ax in axes:
            _empty(ax, "No SpMV measurements")
        return fig
    cases = list(data.case.drop_duplicates())
    x = np.arange(len(cases))
    groups = list(data.groupby(["backend", "dtype"]))
    width = .8 / len(groups)
    for index, ((backend, dtype), group) in enumerate(groups):
        group = group.set_index("case").reindex(cases)
        positions = x + (index - (len(groups)-1)/2)*width
        color = COLORS[(backend, dtype)]
        label = f"{backend} {dtype}"
        axes[0].bar(positions, group.median_s * 1000, width, color=color, label=label)
        axes[2].bar(positions, group.effective_gb_s, width, color=color, label=label)
    axes[0].set(ylabel="Median time (ms)", yscale="log", title="Operation time")
    axes[2].set(ylabel="Estimated bytes / time (GB/s)", title="Effective bandwidth estimate")
    for ax in (axes[0], axes[2]):
        ax.set_xticks(x, cases, rotation=35, ha="right")
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    ratios = speedups(results)
    ratios = ratios[ratios.operation == "SpMV"] if not ratios.empty else ratios
    if ratios.empty:
        _empty(axes[1], "GPU speedup unavailable\nNo GPU measurements in this run")
    else:
        ratio_groups = list(ratios.groupby(["dtype", "scope"]))
        width = .8 / len(ratio_groups)
        for i, ((dtype, scope), group) in enumerate(ratio_groups):
            group = group.set_index("case").reindex(cases)
            axes[1].bar(x + (i - (len(ratio_groups)-1)/2)*width, group.speedup,
                        width, color=COLORS[("GPU", dtype)],
                        hatch="//" if scope == "end_to_end" else None,
                        label=f"{dtype} {scope}")
        axes[1].axhline(1, color="#52677f", linestyle=":")
        axes[1].set(yscale="log", ylabel="CPU time / GPU time")
        axes[1].set_xticks(x, cases, rotation=35, ha="right")
        axes[1].legend(fontsize=8)
    axes[1].set_title("GPU speedup (>1 means faster)")
    return fig


def plot_transfers(samples):
    """Stack arithmetic MEANS of phases from the same trials, not medians."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained")
    fig.suptitle("GPU transfer cost | preallocated buffers, synchronized phases",
                 fontsize=15, fontweight="bold")
    for ax, operation in zip(axes, ("GEMM", "SpMV")):
        data = samples[(samples.operation == operation) & (samples.backend == "GPU") &
                       samples.scope.isin(["upload", "compute_wall", "download"])] if not samples.empty else samples
        if data.empty:
            _empty(ax, "No GPU transfer measurements\nRun on NVIDIA CUDA with transfers enabled")
            ax.set_title(operation)
            continue
        label_column = "mode" if operation == "GEMM" and "mode" in data else "dtype"
        means = data.groupby(["case", label_column, "scope"], sort=False).seconds.mean().unstack("scope") * 1000
        positions = np.arange(len(means))
        bottom = np.zeros(len(means))
        for phase, color in (("upload", "#52677f"), ("compute_wall", "#4f9900"),
                             ("download", "#bac4d1")):
            values = means[phase].to_numpy()
            ax.bar(positions, values, bottom=bottom, label=phase, color=color)
            bottom += values
        ax.set_xticks(positions, [f"{case}\n{dtype}" for case, dtype in means.index],
                      rotation=45, ha="right", fontsize=8)
        ax.set(ylabel="Mean synchronized wall time (ms)", title=operation)
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    return fig
