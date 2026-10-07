"""Additional result figures: calibration, where-the-summary-helps, and a
linear probe of the latent summary space.

Run:  DEVICE=cpu python -m long_range_pfn.extra_figures
(uses the current checkpoints; safe to run while training occupies the GPU)
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import Config, pick_device
from .data import make_criterion, sample_prior_batch
from .evaluate import (
    _save,
    C_ABLATION,
    C_GP,
    C_STUDENT,
    C_STREAM,
    C_TEACHER_FULL,
    C_TEACHER_LOCAL,
    STYLE,
)
from .models import build_teacher, LongRangePFN


@torch.no_grad()
def calibration_figure(cfg, device, teacher, student, criterion, n_batches=40, n_local=10):
    """Empirical coverage of central predictive intervals at |D| = 10."""
    seq_len = cfg.max_context + cfg.num_test
    n_far = cfg.max_context - n_local
    levels = np.arange(0.1, 0.91, 0.1)
    methods = {
        "Teacher (full context)": (C_TEACHER_FULL, "--"),
        "Teacher (truncated to $D$)": (C_TEACHER_LOCAL, "-"),
        "LongRange-PFN": (C_STUDENT, "-"),
    }
    hits = {m: np.zeros(len(levels)) for m in methods}
    total = 0

    for _ in range(n_batches):
        x, y = sample_prior_batch(cfg, 32, seq_len, device)
        y_test = y[:, cfg.max_context :].cpu()
        x_far, y_far = x[:, :n_far], y[:, :n_far]
        x_local, y_local = x[:, n_far : cfg.max_context], y[:, n_far : cfg.max_context]
        x_test = x[:, cfg.max_context :]
        x_local_test = torch.cat((x_local, x_test), dim=1)

        logits = {
            "Teacher (full context)": teacher(x, y[:, : cfg.max_context]).cpu().float(),
            "Teacher (truncated to $D$)": teacher(x_local_test, y_local).cpu().float(),
            "LongRange-PFN": student.predict(
                x_far, y_far, x_local, y_local, x_test
            ).cpu().float(),
        }
        for m, lg in logits.items():
            for i, a in enumerate(levels):
                lo = criterion.icdf((1 - a) / 2, lg)
                hi = criterion.icdf(1 - (1 - a) / 2, lg)
                hits[m][i] += ((y_test >= lo) & (y_test <= hi)).float().sum().item()
        total += y_test.numel()

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        ax.plot([0, 1], [0, 1], color="#999999", lw=0.9, ls="-", zorder=1)
        for m, (color, ls) in methods.items():
            ax.plot(levels, hits[m] / total, marker="o", color=color, ls=ls, label=m)
        ax.set_xlabel("Nominal central-interval coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0.05, 1.0)
        ax.legend(loc="lower right")
        fig.tight_layout()
        _save(fig, "calibration.png")


@torch.no_grad()
def where_it_helps_figure(cfg, device, teacher, student, criterion, n_batches=60, n_local=10):
    """NLL improvement over truncation, binned by distance to the nearest
    direct-context point: the summary should matter most far from D."""
    seq_len = cfg.max_context + cfg.num_test
    n_far = cfg.max_context - n_local
    dists, gain_student, gain_full = [], [], []

    for _ in range(n_batches):
        x, y = sample_prior_batch(cfg, 32, seq_len, device)
        y_test = y[:, cfg.max_context :]
        x_far, y_far = x[:, :n_far], y[:, :n_far]
        x_local, y_local = x[:, n_far : cfg.max_context], y[:, n_far : cfg.max_context]
        x_test = x[:, cfg.max_context :]
        x_local_test = torch.cat((x_local, x_test), dim=1)

        nll_local = criterion(teacher(x_local_test, y_local), y_test)
        nll_full = criterion(teacher(x, y[:, : cfg.max_context]), y_test)
        nll_student = criterion(
            student.predict(x_far, y_far, x_local, y_local, x_test), y_test
        )
        # distance from each test point to the nearest direct-context point
        d = (x_test - x_local.transpose(1, 2)).abs().min(dim=-1).values  # (b, n_test)

        dists.append(d.cpu().flatten())
        gain_student.append((nll_local - nll_student).cpu().flatten())
        gain_full.append((nll_local - nll_full).cpu().flatten())

    d = torch.cat(dists).numpy()
    gs = torch.cat(gain_student).numpy()
    gf = torch.cat(gain_full).numpy()

    edges = np.quantile(d, np.linspace(0, 1, 11))
    centers, m_s, se_s, m_f, se_f = [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (d >= lo) & (d < hi if hi < edges[-1] else d <= hi)
        centers.append(d[mask].mean())
        m_s.append(gs[mask].mean())
        se_s.append(gs[mask].std() / np.sqrt(mask.sum()))
        m_f.append(gf[mask].mean())
        se_f.append(gf[mask].std() / np.sqrt(mask.sum()))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        ax.axhline(0, color="#1a1a1a", lw=0.8)
        ax.errorbar(centers, m_f, yerr=1.96 * np.array(se_f), color=C_TEACHER_FULL,
                    ls="--", marker="s", capsize=2.5, lw=1.5,
                    label="Full context (upper bound)")
        ax.errorbar(centers, m_s, yerr=1.96 * np.array(se_s), color=C_STUDENT,
                    marker="o", capsize=2.5, lw=1.8, label="LongRange-PFN ($z$)")
        ax.set_xlabel("Distance from test point to nearest point of $D$")
        ax.set_ylabel("NLL gain over truncated teacher (nats)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        _save(fig, "where_it_helps.png")


@torch.no_grad()
def latent_probe_figure(cfg, device, teacher, student, criterion, n_batches=60):
    """What does z encode? Ridge probe from vec(z(D')) to the full-context
    predictive mean on a grid, plus a 2-d PCA of the summary space."""
    grid = torch.linspace(0.05, 0.95, 9, device=device)[None, :, None]
    Z, F = [], []
    for _ in range(n_batches):
        x, y = sample_prior_batch(cfg, 32, cfg.max_context, device)
        z = student.summarize(x, y, batch_size=x.shape[0])  # summarize ALL 80 pts
        t_logits = teacher(
            torch.cat((x, grid.expand(x.shape[0], -1, -1)), dim=1), y
        ).cpu().float()
        F.append(criterion.mean(t_logits))  # (b, 9) predictive mean on grid
        Z.append(z.reshape(z.shape[0], -1).cpu())
    Z = torch.cat(Z).numpy()  # (N, k*e)
    F = torch.cat(F).numpy()  # (N, 9)

    n_train = int(0.8 * len(Z))
    Zc = Z - Z[:n_train].mean(0, keepdims=True)
    lam = 10.0
    A = Zc[:n_train].T @ Zc[:n_train] + lam * np.eye(Z.shape[1])
    W = np.linalg.solve(A, Zc[:n_train].T @ (F[:n_train] - F[:n_train].mean(0)))
    pred = Zc[n_train:] @ W + F[:n_train].mean(0)
    resid = F[n_train:] - pred
    r2 = 1 - resid.var(0) / F[n_train:].var(0)

    # PCA for panel (b)
    U, S, _ = np.linalg.svd(Zc[:2000], full_matrices=False)
    pcs = U[:, :2] * S[:2]
    color = F[:2000, 4]  # predictive mean at x = 0.5

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
        axes[0].bar(grid[0, :, 0].cpu().numpy(), r2, width=0.07, color=C_STUDENT, alpha=0.85)
        axes[0].set_xlabel("Grid location $x$")
        axes[0].set_ylabel("Probe $R^2$ (held out)")
        axes[0].set_ylim(0, 1)
        axes[0].set_title("(a) Linear probe: $z \\to$ full-context predictive mean")
        sc = axes[1].scatter(pcs[:, 0], pcs[:, 1], c=color, s=5, cmap="viridis", lw=0)
        fig.colorbar(sc, ax=axes[1], label="predictive mean at $x = 0.5$")
        axes[1].set_xlabel("PC 1")
        axes[1].set_ylabel("PC 2")
        axes[1].set_title("(b) PCA of $\\mathrm{vec}(z)$ across datasets")
        fig.tight_layout()
        _save(fig, "latent_probe.png")


def main():
    cfg = Config()
    device = torch.device(os.environ["DEVICE"]) if os.environ.get("DEVICE") else pick_device()
    torch.manual_seed(cfg.seed + 3)
    print(f"device: {device}")

    borders = torch.load(cfg.borders_path, weights_only=True)
    criterion = make_criterion(cfg, borders)  # CPU for icdf/mean

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

    crit_dev = make_criterion(cfg, borders).to(device)
    calibration_figure(cfg, device, teacher, student, criterion)
    where_it_helps_figure(cfg, device, teacher, student, crit_dev)
    latent_probe_figure(cfg, device, teacher, student, criterion)


if __name__ == "__main__":
    main()
