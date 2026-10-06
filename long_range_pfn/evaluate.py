"""Evaluates the Long-Range PFN POC and produces publication-quality figures.

Experiments (fresh prior datasets, exact GP posterior as gold standard):
  sweep          |D_total| = 80 fixed, |D| swept: teacher-full / teacher-local
                 / student (1 and 3 chunks) / no-z ablation / exact GP
  beyond_window  |D_total| up to 4x the teacher's window, matched window
                 budgets: student = 70 recent + recursive z; teacher = 80 recent
  chunks         NLL vs number of recursive chunks at a fixed split

Run:  python -m long_range_pfn.evaluate
"""

from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from pfns.priors import fast_gp

from .config import Config, pick_device, RESULTS_DIR
from .data import make_criterion, sample_prior_batch
from .models import build_teacher, LongRangePFN

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

C_TEACHER_FULL = "#1a1a1a"
C_TEACHER_LOCAL = "#e08214"
C_STUDENT = "#2166ac"
C_STREAM = "#7b3294"
C_ABLATION = "#878787"
C_GP = "#1b7837"


def _save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def kl_per_point(t_logits: torch.Tensor, s_logits: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(t_logits, dim=-1)
    log_q = F.log_softmax(s_logits, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


@torch.no_grad()
def gp_exact_nll(cfg, x_ctx, y_ctx, x_test, y_test) -> float:
    """Mean pointwise NLL of the exact GP posterior (analytic gold standard)."""
    x_ctx, y_ctx = x_ctx.cpu().double(), y_ctx.cpu().double()
    x_test, y_test = x_test.cpu().double(), y_test.cpu().double()
    model, likelihood = fast_gp.get_model(x_ctx, y_ctx, dict(cfg.gp_hyperparameters))
    model.double().eval()
    likelihood.double().eval()
    with gpytorch.settings.fast_pred_var(False), gpytorch.settings.fast_computations(
        False, False, False
    ):
        pred = likelihood(model(x_test))
    mean, var = pred.mean, pred.variance
    nll = 0.5 * math.log(2 * math.pi) + 0.5 * var.log() + (y_test - mean) ** 2 / (2 * var)
    return nll.mean().item()


def _mean_se(values: list[float]) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "se": float(a.std(ddof=1) / np.sqrt(len(a)))}


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_sweep(cfg, device, teacher, student, criterion, n_batches=100, batch_size=32):
    seq_len = cfg.max_context + cfg.num_test
    sweep = [5, 10, 20, 40]
    methods = ["teacher_full", "teacher_local", "student", "student_stream", "student_null_z"]
    raw = {
        n: {
            **{m: {"nll": [], "kl": []} for m in methods},
            "gp_exact_full": {"nll": [], "kl": None},
            "gp_exact_local": {"nll": [], "kl": None},
        }
        for n in sweep
    }

    for _ in range(n_batches):
        x, y = sample_prior_batch(cfg, batch_size, seq_len, device)
        y_test = y[:, cfg.max_context :]
        t_full = teacher(x, y[:, : cfg.max_context])
        gp_full = gp_exact_nll(
            cfg, x[:, : cfg.max_context], y[:, : cfg.max_context],
            x[:, cfg.max_context :], y_test,
        )

        for n_local in sweep:
            n_far = cfg.max_context - n_local
            x_far, y_far = x[:, :n_far], y[:, :n_far]
            x_local = x[:, n_far : cfg.max_context]
            y_local = y[:, n_far : cfg.max_context]
            x_test = x[:, cfg.max_context :]
            x_local_test = torch.cat((x_local, x_test), dim=1)

            logits = {
                "teacher_full": t_full,
                "teacher_local": teacher(x_local_test, y_local),
                "student": student.predict(x_far, y_far, x_local, y_local, x_test),
                "student_stream": student.predict(
                    x_far, y_far, x_local, y_local, x_test, n_chunks=3
                ),
                "student_null_z": student(
                    x_local_test, y_local, student.summarizer.null_summary(x.shape[0])
                ),
            }
            for m in methods:
                raw[n_local][m]["nll"].append(criterion(logits[m], y_test).mean().item())
                raw[n_local][m]["kl"].append(kl_per_point(t_full, logits[m]).mean().item())
            raw[n_local]["gp_exact_full"]["nll"].append(gp_full)
            raw[n_local]["gp_exact_local"]["nll"].append(
                gp_exact_nll(cfg, x_local, y_local, x_test, y_test)
            )

    results = {}
    for n in sweep:
        results[n] = {}
        for m, d in raw[n].items():
            results[n][m] = {
                "nll": _mean_se(d["nll"])["mean"],
                "nll_se": _mean_se(d["nll"])["se"],
                "kl": _mean_se(d["kl"])["mean"] if d["kl"] else None,
                "kl_se": _mean_se(d["kl"])["se"] if d["kl"] else None,
            }
    return results


@torch.no_grad()
def evaluate_beyond_window(
    cfg, device, teacher, student, criterion,
    n_batches=40, batch_size=32, chunk_size=70, n_local=70,
):
    totals = [80, 160, 240, 320]
    methods = ["student_stream", "teacher_window", "gp_exact_full", "gp_exact_window"]
    raw = {N: {m: [] for m in methods} for N in totals}

    for _ in range(n_batches):
        for N in totals:
            x, y = sample_prior_batch(cfg, batch_size, N + cfg.num_test, device)
            n_far = N - n_local
            x_far, y_far = x[:, :n_far], y[:, :n_far]
            x_local, y_local = x[:, n_far:N], y[:, n_far:N]
            x_test, y_test = x[:, N:], y[:, N:]
            n_chunks = max(1, math.ceil(max(n_far, 1) / chunk_size))

            s_logits = student.predict(
                x_far if n_far > 0 else None,
                y_far if n_far > 0 else None,
                x_local, y_local, x_test, n_chunks=n_chunks,
            )
            x_win = x[:, N - cfg.max_context : N]
            y_win = y[:, N - cfg.max_context : N]
            t_logits = teacher(torch.cat((x_win, x_test), dim=1), y_win)

            raw[N]["student_stream"].append(criterion(s_logits, y_test).mean().item())
            raw[N]["teacher_window"].append(criterion(t_logits, y_test).mean().item())
            raw[N]["gp_exact_full"].append(
                gp_exact_nll(cfg, x[:, :N], y[:, :N], x_test, y_test)
            )
            raw[N]["gp_exact_window"].append(gp_exact_nll(cfg, x_win, y_win, x_test, y_test))

    return {
        N: {m: {"nll": _mean_se(v)["mean"], "nll_se": _mean_se(v)["se"]} for m, v in d.items()}
        for N, d in raw.items()
    }


@torch.no_grad()
def evaluate_chunks(cfg, device, student, teacher, criterion, n_batches=40, batch_size=32):
    """NLL and KL-to-teacher-full vs number of recursive chunks, |D| = 10 fixed."""
    seq_len = cfg.max_context + cfg.num_test
    n_local = 10
    n_far = cfg.max_context - n_local
    chunk_counts = [1, 2, 3, 4, 5]
    raw = {T: {"nll": [], "kl": []} for T in chunk_counts}

    for _ in range(n_batches):
        x, y = sample_prior_batch(cfg, batch_size, seq_len, device)
        y_test = y[:, cfg.max_context :]
        t_full = teacher(x, y[:, : cfg.max_context])
        x_far, y_far = x[:, :n_far], y[:, :n_far]
        x_local = x[:, n_far : cfg.max_context]
        y_local = y[:, n_far : cfg.max_context]
        x_test = x[:, cfg.max_context :]
        for T in chunk_counts:
            logits = student.predict(x_far, y_far, x_local, y_local, x_test, n_chunks=T)
            raw[T]["nll"].append(criterion(logits, y_test).mean().item())
            raw[T]["kl"].append(kl_per_point(t_full, logits).mean().item())

    return {
        T: {
            "nll": _mean_se(d["nll"])["mean"], "nll_se": _mean_se(d["nll"])["se"],
            "kl": _mean_se(d["kl"])["mean"], "kl_se": _mean_se(d["kl"])["se"],
        }
        for T, d in raw.items()
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

SWEEP_STYLES = {
    "gp_exact_full": ("Exact GP (full context)", C_GP, "--", "s"),
    "gp_exact_local": ("Exact GP ($D$ only) — Prop. 1 bound", C_GP, ":", "s"),
    "teacher_full": ("Teacher (full context)", C_TEACHER_FULL, "--", "o"),
    "teacher_local": ("Teacher (truncated to $D$)", C_TEACHER_LOCAL, "-", "o"),
    "student": ("LongRange-PFN", C_STUDENT, "-", "o"),
    "student_stream": ("LongRange-PFN (3 chunks)", C_STREAM, "-.", "^"),
    "student_null_z": ("Ablation (null $z$)", C_ABLATION, ":", "v"),
}


def _line_with_band(ax, xs, results, key, metric, se_key=None):
    label, color, ls, marker = SWEEP_STYLES[key]
    means = [results[x][key][metric] for x in xs]
    if any(v is None for v in means):
        return
    ax.plot(xs, means, color=color, ls=ls, marker=marker, label=label, zorder=3)
    if se_key:
        ses = [results[x][key][se_key] for x in xs]
        lo = [m - 1.96 * s for m, s in zip(means, ses)]
        hi = [m + 1.96 * s for m, s in zip(means, ses)]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0, zorder=2)


def sweep_figure(results):
    xs = sorted(results.keys())
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
        for key in SWEEP_STYLES:
            _line_with_band(axes[0], xs, results, key, "nll", "nll_se")
        axes[0].set_xlabel(r"Direct context size $|D|$   ($|D_{\mathrm{tot}}| = 80$)")
        axes[0].set_ylabel("Test NLL (nats)")
        axes[0].set_title("(a) Predictive quality")
        axes[0].set_xticks(xs)

        for key in ["teacher_local", "student", "student_stream", "student_null_z"]:
            _line_with_band(axes[1], xs, results, key, "kl", "kl_se")
        axes[1].set_xlabel(r"Direct context size $|D|$   ($|D_{\mathrm{tot}}| = 80$)")
        axes[1].set_ylabel("KL(teacher-full $\\Vert$ method) (nats)")
        axes[1].set_title("(b) Divergence from full-context teacher")
        axes[1].set_xticks(xs)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4)
        fig.tight_layout()
        _save(fig, "sweep.png")


def certificate_figure(results):
    """Zoom on the Prop. 1 information bound: exact GP on D alone vs student."""
    xs = sorted(results.keys())
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for key in ["teacher_local", "gp_exact_local", "student"]:
            _line_with_band(ax, xs, results, key, "nll", "nll_se")
        # shade the region unreachable by any D-only method
        bound = [results[x]["gp_exact_local"]["nll"] for x in xs]
        ymin = min(results[x]["student"]["nll"] for x in xs) - 0.25
        ax.fill_between(xs, [ymin] * len(xs), bound, color=C_GP, alpha=0.07, lw=0)
        mid = xs[len(xs) // 2 - 1]
        ax.annotate(
            "unreachable from $D$ alone\n(Proposition 1)",
            xy=(mid, results[mid]["gp_exact_local"]["nll"]),
            xytext=(mid + 6, results[mid]["gp_exact_local"]["nll"] - 0.32),
            fontsize=8.5, color=C_GP,
            arrowprops=dict(arrowstyle="->", color=C_GP, lw=0.9),
        )
        ax.set_xlabel(r"Direct context size $|D|$   ($|D_{\mathrm{tot}}| = 80$)")
        ax.set_ylabel("Test NLL (nats)")
        ax.set_xticks(xs)
        ax.legend(loc="upper right")
        fig.tight_layout()
        _save(fig, "certificate.png")


def beyond_window_figure(results, max_context):
    totals = sorted(results.keys())
    styles = {
        "gp_exact_full": ("Exact GP (full stream)", C_GP, "--", "s"),
        "gp_exact_window": ("Exact GP (80-pt window)", C_GP, ":", "s"),
        "teacher_window": ("Teacher (80-pt window)", C_TEACHER_LOCAL, "-", "o"),
        "student_stream": ("LongRange-PFN (recursive $z$)", C_STUDENT, "-", "o"),
    }
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
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
            max_context + 4, ax.get_ylim()[0] + 0.02, "teacher pre-training window",
            fontsize=8, color="#777777", rotation=90, va="bottom",
        )
        ax.set_xlabel(r"Total context size $|D_{\mathrm{tot}}|$")
        ax.set_ylabel("Test NLL (nats)")
        ax.set_title("(a) Streaming past the window")
        ax.set_xticks(totals)
        ax.legend(loc="best")

        # panel (b): NLL improvement over the window-limited teacher
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
        ax.bar(
            totals, headroom, width=width, color=C_GP, alpha=0.25,
            label="Headroom (exact GP, window $\\to$ full)",
        )
        ax.bar(
            totals, gains, width=width * 0.55, color=C_STUDENT,
            yerr=[1.96 * s for s in ses], capsize=3, error_kw={"lw": 0.9},
            label="Realized by recursive $z$",
        )
        ax.axhline(0, color="#1a1a1a", lw=0.8)
        ax.set_xlabel(r"Total context size $|D_{\mathrm{tot}}|$")
        ax.set_ylabel(r"NLL gain over window teacher (nats)")
        ax.set_title("(b) Gain attributable to the summary")
        ax.set_xticks(totals)
        ax.legend(loc="upper left")
        fig.tight_layout()
        _save(fig, "beyond_window.png")


def chunks_figure(results):
    Ts = sorted(results.keys())
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        means = [results[T]["kl"] for T in Ts]
        ses = [results[T]["kl_se"] for T in Ts]
        ax.errorbar(
            Ts, means, yerr=[1.96 * s for s in ses], color=C_STREAM, marker="o",
            capsize=3, lw=1.8, label="LongRange-PFN, $|D| = 10$",
        )
        ax.axvspan(0.8, 3.2, color=C_STREAM, alpha=0.06, lw=0)
        ax.text(1.9, ax.get_ylim()[1], "seen in training", fontsize=8, color=C_STREAM,
                ha="center", va="top")
        ax.set_xlabel(r"Number of recursive chunks $T$ over the same $D'$")
        ax.set_ylabel("KL(teacher-full $\\Vert$ student) (nats)")
        ax.set_xticks(Ts)
        ax.legend(loc="lower right")
        fig.tight_layout()
        _save(fig, "chunks.png")


def _rolling_quantiles(losses, w):
    losses = np.asarray(losses)
    steps = np.arange(w, len(losses) + 1, w // 5)
    med, lo, hi = [], [], []
    for s in steps:
        window = losses[s - w : s]
        med.append(np.median(window))
        lo.append(np.quantile(window, 0.25))
        hi.append(np.quantile(window, 0.75))
    return steps, np.array(med), np.array(lo), np.array(hi)


def training_curves_figure(cfg):
    """Teacher NLL (rolling median + IQR) and student distillation KL
    evaluated on a FIXED, held-out validation set. Raw per-step training loss
    mixes regimes with very different scales (|D| = 5 gives KL ~ 1.2, the
    no-summary case ~ 0), which makes the optimization trend unreadable and
    the variance look like instability rather than regime mixture;
    fixed-validation curves avoid both problems by re-evaluating the SAME
    batches at every checkpoint. The student panel is split into single-shot
    (n_chunks=1) and multi-chunk (n_chunks in {2,3}) regimes: an earlier
    validation set only covered single-shot regimes, so checkpoint selection
    was blind to recursive-summarization quality; both are now tracked and
    included in model selection (train_student.VAL_REGIMES)."""
    with plt.rc_context(STYLE):
        ckpt = torch.load(cfg.student_path, map_location="cpu", weights_only=True)
        val_log = ckpt.get("val_log")
        is_tuple_keyed = bool(val_log) and isinstance(next(iter(val_log[0][1])), tuple)

        ncols = 3 if is_tuple_keyed else 2
        fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 3.4))

        # (a) teacher
        ax = axes[0]
        losses = torch.load(cfg.teacher_path, map_location="cpu", weights_only=True)["losses"]
        steps, med, lo, hi = _rolling_quantiles(losses, 100)
        ax.fill_between(steps, lo, hi, color=C_TEACHER_FULL, alpha=0.15, lw=0,
                        label="IQR across random splits")
        ax.plot(steps, med, color=C_TEACHER_FULL, lw=1.8, label="median (100-step window)")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Bar-distribution NLL (nats)")
        ax.set_title("(a) Teacher pre-training")
        ax.legend(loc="upper right", fontsize=7.5)

        regime_colors = {5: C_STUDENT, 10: C_STREAM, 20: C_TEACHER_LOCAL,
                          40: C_GP, cfg.max_context: C_ABLATION}

        if is_tuple_keyed:
            steps = [s for s, _ in val_log]

            # (b) single-shot regimes (n_chunks = 1), including null-summary
            ax = axes[1]
            single_shot = sorted({r for r in val_log[0][1] if r[1] == 1})
            for n, t in single_shot:
                ys = [v[(n, t)] for _, v in val_log]
                label = f"$|D|={n}$" if n != cfg.max_context else "no summary ($z_\emptyset$)"
                ax.plot(steps, ys, color=regime_colors.get(n, C_ABLATION), lw=1.6,
                        marker="o", ms=3, label=label)
            ax.set_yscale("log")
            ax.set_title("(b) Single-shot ($n_{\mathrm{chunks}}=1$)")
            ax.set_xlabel("Training step")
            ax.set_ylabel("KL(teacher $\Vert$ student) (nats)")
            ax.legend(loc="upper right", fontsize=7.5)

            # (c) multi-chunk regimes (n_chunks in {2,3})
            ax = axes[2]
            multi = sorted({r for r in val_log[0][1] if r[1] > 1})
            ls_by_chunks = {2: "--", 3: ":"}
            for n, t in multi:
                ys = [v[(n, t)] for _, v in val_log]
                ax.plot(steps, ys, color=regime_colors.get(n, C_ABLATION),
                        ls=ls_by_chunks.get(t, "-"), lw=1.6, marker="o", ms=3,
                        label=f"$|D|={n}$, $T={t}$")
            ax.set_yscale("log")
            ax.set_title("(c) Multi-chunk ($n_{\mathrm{chunks}} \in \{2,3\}$)")
            ax.set_xlabel("Training step")
            ax.set_ylabel("KL(teacher $\Vert$ student) (nats)")
            ax.legend(loc="upper right", fontsize=7.5)
        else:
            # legacy checkpoints: single-shot-only val_log, or raw split log
            ax = axes[1]
            if val_log:
                steps = [s for s, _ in val_log]
                regimes = sorted(val_log[0][1].keys())
                for n in regimes:
                    ys = [v[n] for _, v in val_log]
                    label = f"$|D|={n}$" if n != cfg.max_context else "no summary ($z_\emptyset$)"
                    color = regime_colors.get(n, C_ABLATION)
                    ax.plot(steps, ys, color=color, lw=1.6, marker="o", ms=3, label=label)
                ax.set_yscale("log")
                ax.set_title("(b) Student: fixed-validation KL by regime")
            else:
                splits = ckpt.get("splits")
                losses = np.asarray(ckpt["losses"])
                if splits is not None:
                    n_locals = np.array([s[0] for s in splits])
                    regimes = [
                        (n_locals <= cfg.max_local, f"$|D| \le {cfg.max_local}$", C_STUDENT),
                        ((n_locals > cfg.max_local) & (n_locals < cfg.max_context),
                         f"${cfg.max_local} < |D| < {cfg.max_context}$", C_STREAM),
                        (n_locals == cfg.max_context, "no summary ($z_\emptyset$)", C_ABLATION),
                    ]
                    for mask, label, color in regimes:
                        xs = np.arange(len(losses))[mask]
                        ys = losses[mask]
                        if len(ys) < 100:
                            continue
                        w = max(25, len(ys) // 40)
                        med = np.array(
                            [np.median(ys[max(0, i - w) : i + 1]) for i in range(len(ys))]
                        )
                        ax.plot(xs, med, color=color, lw=1.6, label=label)
                    ax.set_yscale("log")
                else:
                    steps, med, lo, hi = _rolling_quantiles(losses, 250)
                    ax.fill_between(steps, lo, hi, color=C_STUDENT, alpha=0.15, lw=0)
                    ax.plot(steps, med, color=C_STUDENT, lw=1.8, label="median (250-step window)")
                ax.set_title("(b) Student KL distillation, by split regime")
            ax.set_xlabel("Training step")
            ax.set_ylabel("KL(teacher $\Vert$ student) (nats)")
            ax.legend(loc="upper right", fontsize=7.5)

        fig.tight_layout()
        _save(fig, "training_curves.png")


@torch.no_grad()
def qualitative_figure(cfg, device, teacher, student, criterion, n_local=10, seed=7):
    torch.manual_seed(seed)
    x, y = sample_prior_batch(cfg, 1, cfg.max_context, device)
    n_far = cfg.max_context - n_local
    x_far, y_far = x[:, :n_far], y[:, :n_far]
    x_local, y_local = x[:, n_far:], y[:, n_far:]

    grid = torch.linspace(0, 1, 300, device=device)[None, :, None]
    x_local_grid = torch.cat((x_local, grid), dim=1)

    preds = {
        "Teacher, full context (reference)": (
            teacher(torch.cat((x, grid), dim=1), y), C_TEACHER_FULL),
        f"Teacher, truncated to $|D| = {n_local}$": (
            teacher(x_local_grid, y_local), C_TEACHER_LOCAL),
        "LongRange-PFN:  $z(D')\\, +\\, D$": (
            student.predict(x_far, y_far, x_local, y_local, grid), C_STUDENT),
    }

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), sharey=True)
        g = grid[0, :, 0].cpu()
        for ax, (name, (logits, color)) in zip(axes, preds.items()):
            logits = logits.cpu().float()
            mean = criterion.mean(logits)[0]
            lo = criterion.icdf(0.05, logits)[0]
            hi = criterion.icdf(0.95, logits)[0]
            ax.fill_between(g, lo, hi, color=color, alpha=0.18, lw=0, label="90% interval")
            ax.plot(g, mean, color=color, lw=1.6, label="Predictive mean")
            ax.scatter(
                x_far[0, :, 0].cpu(), y_far[0].cpu(), s=7, c="#bbbbbb", zorder=3,
                label="$D'$ (visible only via $z$)",
            )
            ax.scatter(
                x_local[0, :, 0].cpu(), y_local[0].cpu(), s=26, c="#b2182b", zorder=4,
                label="$D$ (direct context)",
            )
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("$x$")
        axes[0].set_ylabel("$y$")
        axes[0].legend(loc="lower left", fontsize=7.5)
        fig.tight_layout()
        _save(fig, "qualitative.png")


# ---------------------------------------------------------------------------


def main():
    cfg = Config()
    device = pick_device()
    torch.manual_seed(cfg.seed + 2)

    borders = torch.load(cfg.borders_path, weights_only=True)
    criterion = make_criterion(cfg, borders)  # CPU copy for icdf/mean plotting
    criterion_dev = make_criterion(cfg, borders).to(device)

    teacher = build_teacher(cfg).to(device)
    teacher.load_state_dict(
        torch.load(cfg.teacher_path, map_location=device, weights_only=True)["state_dict"]
    )
    teacher.eval()

    student = LongRangePFN(cfg).to(device)
    student.load_state_dict(
        torch.load(cfg.student_path, map_location=device, weights_only=True)["state_dict"]
    )
    student.eval()

    results = evaluate_sweep(cfg, device, teacher, student, criterion_dev)
    print(f"\n{'|D|':>4} {'method':>22} {'NLL':>8} {'+-':>7} {'KL->teacher':>12}")
    for n in sorted(results):
        for m, d in results[n].items():
            kl = f"{d['kl']:>12.4f}" if d["kl"] is not None else " " * 12
            print(f"{n:>4} {m:>22} {d['nll']:>8.4f} {d['nll_se']:>7.4f} {kl}")

    beyond = evaluate_beyond_window(cfg, device, teacher, student, criterion_dev)
    print(f"\n{'|D_total|':>10} " + " ".join(f"{m:>18}" for m in next(iter(beyond.values()))))
    for N in sorted(beyond):
        print(f"{N:>10} " + " ".join(f"{v['nll']:>18.4f}" for v in beyond[N].values()))

    chunks = evaluate_chunks(cfg, device, student, teacher, criterion_dev)
    print("\nchunks: " + " ".join(f"T={T}: {v['kl']:.4f}" for T, v in chunks.items()))

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump({"sweep": results, "beyond_window": beyond, "chunks": chunks}, f, indent=2)

    sweep_figure(results)
    certificate_figure(results)
    beyond_window_figure(beyond, cfg.max_context)
    chunks_figure(chunks)
    training_curves_figure(cfg)
    qualitative_figure(cfg, device, teacher, student, criterion)


if __name__ == "__main__":
    main()
