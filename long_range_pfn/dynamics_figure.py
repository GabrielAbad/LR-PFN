"""Training-dynamics figure: is the sudden capability jump grokking?

Uses only saved checkpoints (no model forward passes). Panel (a) shows the
fixed-validation KL per regime over training with the transition window;
panel (b) runs the grokking discriminator: matched-regime training loss vs
held-out validation KL. In the online regime every batch is a fresh prior
sample, so a persistent train-val gap (the defining grokking signature)
cannot form; the two curves should coincide, making the jump a phase
transition in capability rather than delayed generalization.

Run:  python -m long_range_pfn.dynamics_figure
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import Config
from .evaluate import _save, C_ABLATION, C_GP, C_STREAM, C_STUDENT, C_TEACHER_LOCAL, STYLE


def _load_15k_style_val_log(ck, cfg):
    """Returns (steps, {n_local: kl_series}) for single-shot regimes only,
    accepting both int-keyed (older runs) and (n_local, n_chunks)-keyed logs."""
    val_log = ck["val_log"]
    first = val_log[0][1]
    steps = np.array([s for s, _ in val_log])
    if isinstance(next(iter(first)), tuple):
        regimes = sorted({r[0] for r in first if r[1] == 1})
        series = {n: np.array([d[(n, 1)] for _, d in val_log]) for n in regimes}
    else:
        regimes = sorted(first)
        series = {n: np.array([d[n] for _, d in val_log]) for n in regimes}
    return steps, series


def main():
    cfg = Config()
    ck = torch.load(cfg.student_path, map_location="cpu", weights_only=True)
    losses = np.array(ck["losses"])
    splits = ck["splits"]
    steps, series = _load_15k_style_val_log(ck, cfg)

    # transition window from the |D|=10 curve (80%/20% of the log-drop)
    v = series[10]
    l_hi, l_lo = np.log(v[:5].mean()), np.log(v.min())
    s80 = steps[np.argmax(np.log(v) < l_hi - 0.2 * (l_hi - l_lo))]
    s20 = steps[np.argmax(np.log(v) < l_hi - 0.8 * (l_hi - l_lo))]

    n_local_arr = np.array([s[0] for s in splits])
    n_chunks_arr = np.array([s[1] for s in splits])

    regime_colors = {5: C_STUDENT, 10: C_STREAM, 20: C_TEACHER_LOCAL,
                     40: C_GP, cfg.max_context: C_ABLATION}

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))

        ax = axes[0]
        ax.axvspan(s80, s20, color="#f0e442", alpha=0.25, lw=0)
        for n, vs in series.items():
            label = f"$|D|={n}$" if n != cfg.max_context else "no summary ($z_\\emptyset$)"
            ax.plot(steps, vs, color=regime_colors.get(n, C_ABLATION),
                    lw=1.6, marker="o", ms=3, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Validation KL(teacher $\\Vert$ student) (nats)")
        ax.set_title("(a) Synchronized transition across regimes")
        ax.text(0.5 * (s80 + s20), ax.get_ylim()[1] * 0.7, "transition",
                ha="center", fontsize=8, color="#8a7d00")
        ax.legend(loc="lower left", fontsize=7.5)

        # (b) grokking discriminator at |D| = 10: matched-regime train loss
        # (causal rolling median over steps with n_local within +-3, T=1)
        ax = axes[1]
        r = 10
        mask = (np.abs(n_local_arr - r) <= 3) & (n_chunks_arr == 1)
        idx = np.arange(len(losses))
        train_pts = []
        for s in steps:
            w = mask & (idx >= s - 1500) & (idx < s + 100)
            if w.sum() >= 5:
                train_pts.append((s, np.median(losses[w])))
        tr = np.array(train_pts)
        val = np.array([series[r][i] for i, s in enumerate(steps)])
        corr = np.corrcoef(
            np.log(np.interp(steps, tr[:, 0], tr[:, 1])), np.log(val)
        )[0, 1]
        ax.plot(tr[:, 0], tr[:, 1], color=C_TEACHER_LOCAL, lw=1.8,
                label="training KL (fresh prior samples)")
        ax.plot(steps, val, color=C_STREAM, lw=1.8, marker="o", ms=3,
                label="held-out validation KL")
        ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel("KL (nats), $|D| = 10$ regime")
        ax.set_title("(b) No train--validation gap: not grokking")
        ax.text(0.03, 0.06, f"log--log correlation $= {corr:.2f}$",
                transform=ax.transAxes, fontsize=8.5)
        ax.legend(loc="upper right", fontsize=7.5)

        fig.tight_layout()
        _save(fig, "dynamics.png")
        print(f"transition window: {s80}-{s20}; corr={corr:.3f}")


if __name__ == "__main__":
    main()
