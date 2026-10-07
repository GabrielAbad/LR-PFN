"""Re-render the metrics-only figures (sweep, certificate, beyond_window,
chunks) with a layout tuned for the single-column NeurIPS text width, so the
legends and axes stay readable after \\includegraphics scales them to
\\linewidth. Reads results/metrics.json only -- no torch, no model, no
segfault risk. Style/colors mirror long_range_pfn/evaluate.py.

    .venv/bin/python paper/regen_figs.py   # (plain python3 also works)
"""

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "results")
METRICS = os.path.join(RESULTS_DIR, "metrics.json")

# Fonts sized so they survive the down-scale to the NeurIPS column (~5.5in):
# figures are rendered close to that width, so these are near their final pt.
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.0,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

C_TEACHER_FULL = "#1a1a1a"
C_TEACHER_LOCAL = "#e08214"
C_STUDENT = "#2166ac"
C_STREAM = "#7b3294"
C_ABLATION = "#878787"
C_GP = "#1b7837"

SWEEP_STYLES = {
    "gp_exact_full": ("Exact GP (full)", C_GP, "--", "s"),
    "gp_exact_local": ("Exact GP ($D$ only), Prop. 1 bound", C_GP, ":", "s"),
    "teacher_full": ("Teacher (full)", C_TEACHER_FULL, "--", "o"),
    "teacher_local": ("Teacher (truncated)", C_TEACHER_LOCAL, "-", "o"),
    "student": ("LongRange-PFN", C_STUDENT, "-", "o"),
    "student_stream": ("LongRange-PFN (3 chunks)", C_STREAM, "-.", "^"),
    "student_null_z": ("Ablation (null $z$)", C_ABLATION, ":", "v"),
}


def _save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


def _line_with_band(ax, xs, results, key, m_key, se_key):
    label, color, ls, marker = SWEEP_STYLES[key]
    means = [results[x][key][m_key] for x in xs]
    ax.plot(xs, means, color=color, ls=ls, marker=marker, label=label, zorder=3)
    ses = [results[x][key][se_key] for x in xs]
    if all(s is not None for s in ses):
        lo = [m - 1.96 * s for m, s in zip(means, ses)]
        hi = [m + 1.96 * s for m, s in zip(means, ses)]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0, zorder=2)


def sweep_figure(results):
    xs = sorted(results.keys())
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
        for key in SWEEP_STYLES:
            _line_with_band(axes[0], xs, results, key, "nll", "nll_se")
        axes[0].set_xlabel(r"Direct context size $|D|$")
        axes[0].set_ylabel("Test NLL (nats)")
        axes[0].set_title("(a) Predictive quality")
        axes[0].set_xticks(xs)

        for key in ["teacher_local", "student", "student_stream", "student_null_z"]:
            _line_with_band(axes[1], xs, results, key, "kl", "kl_se")
        axes[1].set_xlabel(r"Direct context size $|D|$")
        axes[1].set_ylabel(r"KL(teacher $\Vert$ method) (nats)")
        axes[1].set_title("(b) Divergence from teacher")
        axes[1].set_xticks(xs)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        _save(fig, "sweep.png")


def certificate_figure(results):
    xs = sorted(results.keys())
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 2.9))
        for key in ["teacher_local", "gp_exact_local", "student"]:
            _line_with_band(ax, xs, results, key, "nll", "nll_se")
        bound = [results[x]["gp_exact_local"]["nll"] for x in xs]
        ymin = min(results[x]["student"]["nll"] for x in xs) - 0.25
        ax.fill_between(xs, [ymin] * len(xs), bound, color=C_GP, alpha=0.07, lw=0)
        mid = xs[len(xs) // 2 - 1]
        ax.annotate(
            "unreachable from $D$\nalone (Prop. 1)",
            xy=(mid, results[mid]["gp_exact_local"]["nll"]),
            xytext=(mid + 5, results[mid]["gp_exact_local"]["nll"] - 0.34),
            fontsize=9, color=C_GP,
            arrowprops=dict(arrowstyle="->", color=C_GP, lw=0.9),
        )
        ax.set_xlabel(r"Direct context size $|D|$")
        ax.set_ylabel("Test NLL (nats)")
        ax.set_xticks(xs)
        ax.legend(loc="upper right", fontsize=8.5)
        fig.tight_layout()
        _save(fig, "certificate.png")


def beyond_window_figure(results, max_context=80):
    totals = sorted(results.keys())
    styles = {
        "gp_exact_full": ("Exact GP (full stream)", C_GP, "--", "s"),
        "gp_exact_window": ("Exact GP (window)", C_GP, ":", "s"),
        "teacher_window": ("Teacher (window)", C_TEACHER_LOCAL, "-", "o"),
        "student_stream": ("LongRange-PFN (recursive $z$)", C_STUDENT, "-", "o"),
    }
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
        ax = axes[0]
        for m, (label, color, ls, marker) in styles.items():
            means = [results[N][m]["nll"] for N in totals]
            ses = [results[N][m]["nll_se"] for N in totals]
            ax.plot(totals, means, color=color, ls=ls, marker=marker, label=label, zorder=3)
            ax.fill_between(
                totals,
                [m_ - 1.96 * s for m_, s in zip(means, ses)],
                [m_ + 1.96 * s for m_, s in zip(means, ses)],
                color=color, alpha=0.15, lw=0, zorder=2,
            )
        ax.axvline(max_context, color="#999999", lw=0.9, ls="-")
        ax.text(
            max_context + 5, ax.get_ylim()[0] + 0.02, "pre-training window",
            fontsize=8.5, color="#777777", rotation=90, va="bottom",
        )
        ax.set_xlabel(r"Total context size $|D_{\mathrm{tot}}|$")
        ax.set_ylabel("Test NLL (nats)")
        ax.set_title("(a) Streaming past the window")
        ax.set_xticks(totals)
        ax.legend(loc="lower left", fontsize=8.5)

        ax = axes[1]
        gains = [
            results[N]["teacher_window"]["nll"] - results[N]["student_stream"]["nll"]
            for N in totals
        ]
        ses = [
            math.hypot(results[N]["teacher_window"]["nll_se"], results[N]["student_stream"]["nll_se"])
            for N in totals
        ]
        headroom = [
            results[N]["gp_exact_window"]["nll"] - results[N]["gp_exact_full"]["nll"]
            for N in totals
        ]
        width = 22
        ax.bar(totals, headroom, width=width, color=C_GP, alpha=0.25,
               label=r"Headroom (window $\to$ full)")
        ax.bar(totals, gains, width=width * 0.55, color=C_STUDENT,
               yerr=[1.96 * s for s in ses], capsize=3, error_kw={"lw": 0.9},
               label=r"Realized by recursive $z$")
        ax.axhline(0, color="#1a1a1a", lw=0.8)
        ax.set_xlabel(r"Total context size $|D_{\mathrm{tot}}|$")
        ax.set_ylabel("NLL gain over teacher (nats)")
        ax.set_title("(b) Gain from the summary")
        ax.set_xticks(totals)
        ax.legend(loc="upper left", fontsize=8.5)
        fig.tight_layout()
        _save(fig, "beyond_window.png")


def chunks_figure(results):
    Ts = sorted(results.keys())
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 2.9))
        means = [results[T]["kl"] for T in Ts]
        ses = [results[T]["kl_se"] for T in Ts]
        ax.errorbar(Ts, means, yerr=[1.96 * s for s in ses], color=C_STREAM,
                    marker="o", capsize=3, lw=2.0, label="LongRange-PFN, $|D|=10$")
        ax.axvspan(0.8, 3.2, color=C_STREAM, alpha=0.06, lw=0)
        ax.text(1.9, ax.get_ylim()[1], "seen in training", fontsize=8.5,
                color=C_STREAM, ha="center", va="top")
        ax.set_xlabel(r"Recursive chunks $T$ over the same $D'$")
        ax.set_ylabel(r"KL(teacher $\Vert$ student) (nats)")
        ax.set_xticks(Ts)
        ax.legend(loc="lower right", fontsize=8.5)
        fig.tight_layout()
        _save(fig, "chunks.png")


def main():
    with open(METRICS) as f:
        metrics = json.load(f)
    sweep = {int(k): v for k, v in metrics["sweep"].items()}
    beyond = {int(k): v for k, v in metrics["beyond_window"].items()}
    chunks = {int(k): v for k, v in metrics["chunks"].items()}
    sweep_figure(sweep)
    certificate_figure(sweep)
    beyond_window_figure(beyond, max_context=80)
    chunks_figure(chunks)


if __name__ == "__main__":
    main()
