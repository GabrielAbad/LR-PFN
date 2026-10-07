"""Reviewer-oriented empirical diagnostics.

These figures complement the paper plots with paired distributions and
failure-mode views that make the empirical claims harder to over-interpret.

Run:
    python -m long_range_pfn.reviewer_figures
"""

from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import Config, RESULTS_DIR, pick_device
from .data import make_criterion, sample_prior_batch
from .evaluate import (
    C_ABLATION,
    C_GP,
    C_STUDENT,
    C_TEACHER_FULL,
    C_TEACHER_LOCAL,
    STYLE,
    _save,
)
from .models import LongRangePFN, build_teacher


SWEEP = [5, 10, 20, 40]


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))


@torch.no_grad()
def collect_paired_sweep(cfg, device, teacher, student, criterion, n_batches=80, batch_size=32):
    seq_len = cfg.max_context + cfg.num_test
    raw = {
        n: {
            "gain_student": [],
            "gain_null_z": [],
            "gain_full_teacher": [],
            "kl_student": [],
            "kl_local": [],
            "dist": [],
            "point_gain_student": [],
            "point_gain_full": [],
        }
        for n in SWEEP
    }

    for _ in range(n_batches):
        x, y = sample_prior_batch(cfg, batch_size, seq_len, device)
        y_test = y[:, cfg.max_context :]
        x_test = x[:, cfg.max_context :]
        t_full = teacher(x, y[:, : cfg.max_context])
        log_p = torch.log_softmax(t_full, dim=-1)

        for n_local in SWEEP:
            n_far = cfg.max_context - n_local
            x_far, y_far = x[:, :n_far], y[:, :n_far]
            x_local = x[:, n_far : cfg.max_context]
            y_local = y[:, n_far : cfg.max_context]
            x_local_test = torch.cat((x_local, x_test), dim=1)

            local = teacher(x_local_test, y_local)
            student_logits = student.predict(x_far, y_far, x_local, y_local, x_test)
            null_logits = student(
                x_local_test, y_local, student.summarizer.null_summary(x.shape[0])
            )

            nll_local = criterion(local, y_test)
            nll_student = criterion(student_logits, y_test)
            nll_null = criterion(null_logits, y_test)
            nll_full = criterion(t_full, y_test)

            gain_student = (nll_local - nll_student).mean(dim=1)
            gain_null = (nll_local - nll_null).mean(dim=1)
            gain_full = (nll_local - nll_full).mean(dim=1)

            kl_local = (log_p.exp() * (log_p - torch.log_softmax(local, dim=-1))).sum(-1)
            kl_student = (
                log_p.exp() * (log_p - torch.log_softmax(student_logits, dim=-1))
            ).sum(-1)

            dist = (x_test - x_local.transpose(1, 2)).abs().min(dim=-1).values

            raw[n_local]["gain_student"].append(gain_student.cpu())
            raw[n_local]["gain_null_z"].append(gain_null.cpu())
            raw[n_local]["gain_full_teacher"].append(gain_full.cpu())
            raw[n_local]["kl_student"].append(kl_student.mean(dim=1).cpu())
            raw[n_local]["kl_local"].append(kl_local.mean(dim=1).cpu())
            raw[n_local]["dist"].append(dist.cpu().flatten())
            raw[n_local]["point_gain_student"].append((nll_local - nll_student).cpu().flatten())
            raw[n_local]["point_gain_full"].append((nll_local - nll_full).cpu().flatten())

    out = {}
    for n_local, values in raw.items():
        out[n_local] = {k: torch.cat(v).numpy() for k, v in values.items()}
    return out


@torch.no_grad()
def collect_streaming(cfg, device, teacher, student, criterion, n_batches=50, batch_size=32):
    totals = [80, 160, 240, 320]
    raw = {N: {"delta": [], "gain": []} for N in totals}

    for _ in range(n_batches):
        for N in totals:
            x, y = sample_prior_batch(cfg, batch_size, N + cfg.num_test, device)
            n_local = 70
            n_far = N - n_local
            x_far, y_far = x[:, :n_far], y[:, :n_far]
            x_local, y_local = x[:, n_far:N], y[:, n_far:N]
            x_test, y_test = x[:, N:], y[:, N:]
            n_chunks = max(1, math.ceil(max(n_far, 1) / 70))

            student_logits = student.predict(
                x_far if n_far > 0 else None,
                y_far if n_far > 0 else None,
                x_local,
                y_local,
                x_test,
                n_chunks=n_chunks,
            )
            x_win = x[:, N - cfg.max_context : N]
            y_win = y[:, N - cfg.max_context : N]
            teacher_window = teacher(torch.cat((x_win, x_test), dim=1), y_win)

            nll_student = criterion(student_logits, y_test).mean(dim=1)
            nll_window = criterion(teacher_window, y_test).mean(dim=1)
            raw[N]["delta"].append((nll_student - nll_window).cpu())
            raw[N]["gain"].append((nll_window - nll_student).cpu())

    return {N: {k: torch.cat(v).numpy() for k, v in d.items()} for N, d in raw.items()}


def paired_gain_distribution(data):
    positions = np.arange(len(SWEEP))
    gains = [data[n]["gain_student"] for n in SWEEP]
    null_gains = [data[n]["gain_null_z"] for n in SWEEP]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
        ax = axes[0]
        parts = ax.violinplot(gains, positions=positions, showmeans=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(C_STUDENT)
            body.set_edgecolor("none")
            body.set_alpha(0.35)
        ax.boxplot(
            gains,
            positions=positions,
            widths=0.18,
            showfliers=False,
            medianprops={"color": "#1a1a1a", "lw": 1.2},
            boxprops={"color": C_STUDENT},
            whiskerprops={"color": C_STUDENT},
            capprops={"color": C_STUDENT},
        )
        ax.axhline(0, color="#1a1a1a", lw=0.8)
        ax.set_xticks(positions, [str(n) for n in SWEEP])
        ax.set_xlabel(r"Direct context size $|D|$")
        ax.set_ylabel("Paired NLL gain over truncated teacher")
        ax.set_title("(a) Distribution across fresh datasets")

        ax = axes[1]
        width = 0.34
        student_mean, student_ci, null_mean, null_ci = [], [], [], []
        win_student, win_null = [], []
        for g_s, g_n in zip(gains, null_gains):
            m, ci = _mean_ci(g_s)
            student_mean.append(m)
            student_ci.append(ci)
            m, ci = _mean_ci(g_n)
            null_mean.append(m)
            null_ci.append(ci)
            win_student.append((g_s > 0).mean())
            win_null.append((g_n > 0).mean())
        ax.bar(positions - width / 2, student_mean, width, yerr=student_ci, color=C_STUDENT,
               capsize=3, label="LongRange-PFN")
        ax.bar(positions + width / 2, null_mean, width, yerr=null_ci, color=C_ABLATION,
               capsize=3, label="null-$z$ ablation")
        for x_pos, wr in zip(positions - width / 2, win_student):
            ax.text(x_pos, 0.01, f"{100 * wr:.0f}%", ha="center", va="bottom",
                    fontsize=7.5, color="#ffffff", rotation=90)
        ax.axhline(0, color="#1a1a1a", lw=0.8)
        ax.set_xticks(positions, [str(n) for n in SWEEP])
        ax.set_xlabel(r"Direct context size $|D|$")
        ax.set_ylabel("Mean paired gain (nats)")
        ax.set_title("(b) Mean gain and win-rate labels")
        ax.legend(loc="upper right")
        fig.tight_layout()
        _save(fig, "reviewer_paired_gains.png")


def distance_heatmap(data):
    all_dist = np.concatenate([data[n]["dist"] for n in SWEEP])
    edges = np.quantile(all_dist, np.linspace(0, 1, 9))
    heat_student = np.zeros((len(SWEEP), len(edges) - 1))
    heat_full = np.zeros_like(heat_student)

    for i, n_local in enumerate(SWEEP):
        d = data[n_local]["dist"]
        g_s = data[n_local]["point_gain_student"]
        g_f = data[n_local]["point_gain_full"]
        for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (d >= lo) & (d < hi if j < len(edges) - 2 else d <= hi)
            heat_student[i, j] = g_s[mask].mean()
            heat_full[i, j] = g_f[mask].mean()

    labels = [f"{lo:.3f}-{hi:.3f}" for lo, hi in zip(edges[:-1], edges[1:])]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.8), sharey=True, constrained_layout=True)
        for ax, mat, title in [
            (axes[0], heat_student, "LongRange-PFN gain"),
            (axes[1], heat_full, "Full-context teacher gain"),
        ]:
            vmax = max(abs(mat).max(), 1e-6)
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
            ax.set_yticks(np.arange(len(SWEEP)), [str(n) for n in SWEEP])
            ax.set_xlabel("Distance to nearest point in $D$")
            ax.set_title(title)
            cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
            cbar.set_label("NLL gain (nats)")
        axes[0].set_ylabel(r"Direct context size $|D|$")
        _save(fig, "reviewer_distance_heatmap.png")


def kl_and_winrate(data):
    xs = np.arange(len(SWEEP))
    kl_local, kl_student, kl_ci = [], [], []
    winrate = []
    for n_local in SWEEP:
        d = data[n_local]
        kl_local.append(d["kl_local"].mean())
        m, ci = _mean_ci(d["kl_student"])
        kl_student.append(m)
        kl_ci.append(ci)
        winrate.append((d["gain_student"] > d["gain_null_z"]).mean())

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
        axes[0].plot(xs, kl_local, marker="o", color=C_TEACHER_LOCAL,
                     label="Teacher truncated to $D$")
        axes[0].errorbar(xs, kl_student, yerr=kl_ci, marker="o", color=C_STUDENT,
                         capsize=3, label="LongRange-PFN")
        axes[0].set_xticks(xs, [str(n) for n in SWEEP])
        axes[0].set_xlabel(r"Direct context size $|D|$")
        axes[0].set_ylabel("KL to full-context teacher")
        axes[0].set_title("(a) Distillation target fidelity")
        axes[0].legend(loc="upper right")

        axes[1].bar(xs, winrate, color=C_STUDENT, alpha=0.85)
        axes[1].axhline(0.5, color="#1a1a1a", lw=0.8, ls=":")
        axes[1].set_ylim(0, 1)
        axes[1].set_xticks(xs, [str(n) for n in SWEEP])
        axes[1].set_xlabel(r"Direct context size $|D|$")
        axes[1].set_ylabel("P(gain with $z$ > gain with null-$z$)")
        axes[1].set_title("(b) Paired ablation win-rate")
        fig.tight_layout()
        _save(fig, "reviewer_kl_ablation_winrate.png")


def streaming_failure_panel(streaming):
    totals = sorted(streaming)
    deltas = [streaming[N]["delta"] for N in totals]
    means, cis, winrates = [], [], []
    for delta in deltas:
        m, ci = _mean_ci(delta)
        means.append(m)
        cis.append(ci)
        winrates.append((delta < 0).mean())

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharex=True)
        axes[0].boxplot(
            deltas,
            positions=np.arange(len(totals)),
            widths=0.45,
            showfliers=False,
            medianprops={"color": "#1a1a1a", "lw": 1.2},
            boxprops={"color": C_STUDENT},
            whiskerprops={"color": C_STUDENT},
            capprops={"color": C_STUDENT},
        )
        axes[0].axhline(0, color="#1a1a1a", lw=0.8)
        axes[0].set_xticks(np.arange(len(totals)), [str(N) for N in totals])
        axes[0].set_xlabel(r"Total context size $|D_{\mathrm{tot}}|$")
        axes[0].set_ylabel("Student NLL - 80-window teacher NLL")
        axes[0].set_title("(a) Streaming gap distribution")

        axes[1].bar(np.arange(len(totals)), [-m for m in means], yerr=cis,
                    color=C_STUDENT, capsize=3, label="Mean gain")
        axes[1].axhline(0, color="#1a1a1a", lw=0.8)
        for x_pos, wr in enumerate(winrates):
            axes[1].text(x_pos, 0.002 if -means[x_pos] >= 0 else -0.002,
                         f"{100 * wr:.0f}%", ha="center",
                         va="bottom" if -means[x_pos] >= 0 else "top", fontsize=8)
        axes[1].set_xticks(np.arange(len(totals)), [str(N) for N in totals])
        axes[1].set_xlabel(r"Total context size $|D_{\mathrm{tot}}|$")
        axes[1].set_ylabel("NLL gain over 80-window teacher")
        axes[1].set_title("(b) Mean gain with win-rate labels")
        fig.tight_layout()
        _save(fig, "reviewer_streaming_failure.png")


def write_summary(data, streaming):
    summary = {"sweep": {}, "streaming": {}}
    for n_local in SWEEP:
        d = data[n_local]
        gain, gain_ci = _mean_ci(d["gain_student"])
        null_gain, null_ci = _mean_ci(d["gain_null_z"])
        summary["sweep"][str(n_local)] = {
            "student_gain_mean": gain,
            "student_gain_ci95": gain_ci,
            "student_gain_winrate": float((d["gain_student"] > 0).mean()),
            "null_z_gain_mean": null_gain,
            "null_z_gain_ci95": null_ci,
            "z_beats_null_winrate": float((d["gain_student"] > d["gain_null_z"]).mean()),
            "kl_local_mean": float(d["kl_local"].mean()),
            "kl_student_mean": float(d["kl_student"].mean()),
        }
    for N, d in streaming.items():
        delta, delta_ci = _mean_ci(d["delta"])
        summary["streaming"][str(N)] = {
            "student_minus_window_teacher_mean": delta,
            "student_minus_window_teacher_ci95": delta_ci,
            "student_beats_window_teacher_winrate": float((d["delta"] < 0).mean()),
        }
    with open(os.path.join(RESULTS_DIR, "reviewer_diagnostics.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    cfg = Config()
    device = torch.device(os.environ["DEVICE"]) if os.environ.get("DEVICE") else pick_device()
    torch.manual_seed(cfg.seed + 11)
    print(f"device: {device}")

    borders = torch.load(cfg.borders_path, weights_only=True)
    criterion = make_criterion(cfg, borders).to(device)

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

    paired = collect_paired_sweep(cfg, device, teacher, student, criterion)
    streaming = collect_streaming(cfg, device, teacher, student, criterion)

    paired_gain_distribution(paired)
    distance_heatmap(paired)
    kl_and_winrate(paired)
    streaming_failure_panel(streaming)
    write_summary(paired, streaming)
    print("saved reviewer diagnostics")


if __name__ == "__main__":
    main()
