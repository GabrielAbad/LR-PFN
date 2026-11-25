"""CRUMB-style context selection on TabArena regression tasks.

Protocol mirrors CRUMB Section 5.1 on the regression subset of TabArena
(OpenML study 457): fixed context budget n = 0.1 N, methods ranked by RMSE,
average rank across (dataset, seed) pairs.

Wrappers reuse the selectors in ``context_selection.py`` around our frozen
teacher; LongRange-PFN keeps n raw tokens plus a summary of the complement.

Run:
    NUM_FEATURES=8 python -m long_range_pfn.tabarena_benchmark
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import RESULTS_DIR, config_from_env, pick_device
from .context_selection import (
    CHUNK,
    N_CLUSTERS,
    COLORS,
    DISPLAY,
    METHOD_ORDER,
    _chunks_for,
    _median_sigma,
    _standardize_train,
    _student_on_split,
    _teacher_logits,
    crumb_groups,
    greedy_mmd,
    knn_indices,
    micp_supports,
)
from .data import make_criterion
from .evaluate import STYLE, _save
from .models import LongRangePFN, build_teacher
from .tabarena_data import iter_regression_splits

PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(err * err)))


@torch.no_grad()
def _predict_mean(criterion, logits: torch.Tensor) -> np.ndarray:
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    return criterion.mean(logits).detach().cpu().numpy().reshape(-1)


@torch.no_grad()
def evaluate_split(
    teacher,
    student,
    criterion,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    budget_frac: float,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """RMSE per method on one (dataset, seed) split."""
    n_train = len(x_train)
    n = max(8, min(int(round(budget_frac * n_train)), n_train, 80))
    x_ctx = torch.as_tensor(x_train, device=device, dtype=torch.float32)
    y_ctx = torch.as_tensor(y_train, device=device, dtype=torch.float32)
    x_te = torch.as_tensor(x_test, device=device, dtype=torch.float32)
    y_te_np = np.asarray(y_test, dtype=np.float64)

    x_tr_s, x_q_s = _standardize_train(x_train, x_test)
    sigma = _median_sigma(x_tr_s)
    rmses: dict[str, float] = {}

    # Full context only when it fits the teacher window.
    if n_train <= 80:
        full = _teacher_logits(
            teacher, x_ctx.unsqueeze(0), y_ctx.unsqueeze(0), x_te.unsqueeze(0)
        )[0]
        rmses["full_context"] = _rmse(y_te_np, _predict_mean(criterion, full))

    rng = np.random.default_rng(seed + 17 * n)
    uni = rng.choice(n_train, size=n, replace=False)
    uni_t = torch.as_tensor(uni, device=device)
    uni_logits = _teacher_logits(
        teacher,
        x_ctx[uni_t].unsqueeze(0),
        y_ctx[uni_t].unsqueeze(0),
        x_te.unsqueeze(0),
    )[0]
    rmses["uniform"] = _rmse(y_te_np, _predict_mean(criterion, uni_logits))

    nn_idx = knn_indices(x_tr_s, x_q_s, n)
    nn_t = torch.as_tensor(nn_idx, device=device)
    knn_logits = _teacher_logits(
        teacher, x_ctx[nn_t], y_ctx[nn_t], x_te.unsqueeze(1)
    )[:, 0, :]
    rmses["knn"] = _rmse(y_te_np, _predict_mean(criterion, knn_logits))

    supports, centroids = micp_supports(x_tr_s, n, seed)
    micp_pred = np.empty(len(x_test), dtype=np.float64)
    for t, xq in enumerate(x_q_s):
        c = int(np.argmin(np.sum((centroids - xq) ** 2, axis=1)))
        idx = torch.as_tensor(supports[c], device=device)
        logits = _teacher_logits(
            teacher,
            x_ctx[idx].unsqueeze(0),
            y_ctx[idx].unsqueeze(0),
            x_te[t : t + 1].unsqueeze(0),
        )[0]
        micp_pred[t] = float(_predict_mean(criterion, logits)[0])
    rmses["micp"] = _rmse(y_te_np, micp_pred)

    k = min(N_CLUSTERS, max(1, len(x_test) // 5))
    crumb_pred = np.empty(len(x_test), dtype=np.float64)
    for train_idx, test_idx in crumb_groups(x_tr_s, x_q_s, n, k, seed, sigma):
        idx = torch.as_tensor(train_idx, device=device)
        te_idx = torch.as_tensor(test_idx, device=device)
        logits = _teacher_logits(
            teacher,
            x_ctx[idx].unsqueeze(0),
            y_ctx[idx].unsqueeze(0),
            x_te[te_idx].unsqueeze(0),
        )[0]
        crumb_pred[test_idx] = _predict_mean(criterion, logits)
    rmses["crumb"] = _rmse(y_te_np, crumb_pred)

    # LongRange-PFN: random raw window of size n, summarize the complement.
    local = rng.choice(n_train, size=n, replace=False)
    local_t = torch.as_tensor(local, device=device)
    n_far = n_train - n
    n_chunks = _chunks_for(n_far)
    lr_logits = _student_on_split(
        student, x_ctx, y_ctx, x_te, local_t, n_chunks=n_chunks
    )
    rmses["longrange"] = _rmse(y_te_np, _predict_mean(criterion, lr_logits))

    # LongRange + MMD window (K=1 CRUMB context as the raw window).
    mmd_local = greedy_mmd(x_tr_s, x_q_s, n, sigma)
    mmd_t = torch.as_tensor(mmd_local, device=device)
    lr_mmd_logits = _student_on_split(
        student, x_ctx, y_ctx, x_te, mmd_t, n_chunks=n_chunks
    )
    rmses["longrange_mmd"] = _rmse(y_te_np, _predict_mean(criterion, lr_mmd_logits))
    return rmses


def average_ranks(per_obs: list[dict[str, float]], methods: list[str]) -> dict[str, float]:
    """Mean rank across observations (lower RMSE -> better / lower rank)."""
    ranks = defaultdict(list)
    for obs in per_obs:
        ordered = sorted(methods, key=lambda m: obs[m])
        for r, m in enumerate(ordered, start=1):
            ranks[m].append(r)
    return {m: float(np.mean(ranks[m])) for m in methods}


def _write_table(summary: dict, path: str) -> None:
    methods = [m for m in ("full_context",) + METHOD_ORDER if m in summary["mean_rmse"]]
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Avg.\ rank $\downarrow$ & Mean RMSE $\downarrow$\\",
        r"\midrule",
    ]
    best_rank = min(summary["avg_rank"][m] for m in methods if m != "full_context")
    best_rmse = min(summary["mean_rmse"][m] for m in methods if m != "full_context")
    for m in methods:
        rank = summary["avg_rank"].get(m)
        rmse = summary["mean_rmse"][m]
        se = summary["rmse_se"][m]
        rank_s = "---" if rank is None else f"{rank:.3f}"
        if m != "full_context" and abs(rank - best_rank) < 1e-12:
            rank_s = r"\textbf{" + rank_s + "}"
        rmse_s = f"{rmse:.3f}" + r"\,{\scriptsize$\pm$" + f"{se:.3f}" + "}"
        if m != "full_context" and abs(rmse - best_rmse) < 1e-12:
            rmse_s = r"\textbf{" + rmse_s + "}"
        label = DISPLAY.get(m, m.replace("_", " "))
        if m == "full_context":
            label = "Full context"
        lines.append(f"{label} & {rank_s} & {rmse_s}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


def _plot(summary: dict, per_dataset: dict) -> None:
    plt.rcParams.update(STYLE)
    methods = [m for m in METHOD_ORDER if m in summary["avg_rank"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    ax = axes[0]
    ranks = [summary["avg_rank"][m] for m in methods]
    colors = [COLORS[m] for m in methods]
    ax.bar(np.arange(len(methods)), ranks, color=colors)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(
        ["Unif.", "kNN", "MICP", "CRUMB", "LR", "LR+MMD"][: len(methods)], fontsize=8
    )
    ax.set_ylabel("Average rank (lower better)")
    ax.set_title("TabArena regression ranks")

    ax = axes[1]
    xs = np.arange(len(methods))
    means = [summary["mean_rmse"][m] for m in methods]
    ses = [1.96 * summary["rmse_se"][m] for m in methods]
    ax.bar(xs, means, yerr=ses, color=colors, capsize=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        ["Unif.", "kNN", "MICP", "CRUMB", "LR", "LR+MMD"][: len(methods)], fontsize=8
    )
    ax.set_ylabel("Mean RMSE (standardized y)")
    ax.set_title("TabArena regression RMSE")
    fig.tight_layout()
    _save(fig, "tabarena_context_selection.png")


def main() -> None:
    cfg = config_from_env()
    device = pick_device()
    torch.manual_seed(cfg.seed)
    print(f"device={device} features={cfg.num_features}")

    if not os.path.exists(cfg.teacher_path) or not os.path.exists(cfg.student_path):
        raise FileNotFoundError(
            f"Missing checkpoints for num_features={cfg.num_features}. "
            f"Expected {cfg.teacher_path} and {cfg.student_path}. "
            "Train with NUM_FEATURES set, e.g. "
            "NUM_FEATURES=8 TEACHER_STEPS=800 STUDENT_STEPS=1200 "
            "python -m long_range_pfn.train_teacher"
        )

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

    n_seeds = int(os.environ.get("N_SEEDS", "3"))
    task_limit = os.environ.get("TASK_LIMIT")
    task_limit_i = int(task_limit) if task_limit else None
    max_train = int(os.environ.get("MAX_TRAIN", "400"))
    n_test = int(os.environ.get("N_TEST", "80"))
    budget_frac = float(os.environ.get("BUDGET_FRAC", "0.1"))

    rank_methods = list(METHOD_ORDER)
    per_obs: list[dict[str, float]] = []
    per_dataset: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for seed in range(n_seeds):
        print(f"loading TabArena regression splits (seed={seed})", flush=True)
        splits = iter_regression_splits(
            num_features=cfg.num_features,
            seed=1000 + seed,
            max_train=max_train,
            n_test=n_test,
            task_limit=task_limit_i,
        )
        for split in splits:
            print(
                f"  {split.name}: N={len(split.x_train)} T={len(split.x_test)} "
                f"d={split.x_train.shape[1]}",
                flush=True,
            )
            try:
                rmses = evaluate_split(
                    teacher,
                    student,
                    criterion,
                    split.x_train,
                    split.y_train,
                    split.x_test,
                    split.y_test,
                    budget_frac=budget_frac,
                    seed=seed,
                    device=device,
                )
            except Exception as exc:  # pragma: no cover - network / data issues
                print(f"  SKIP {split.name}: {exc}", flush=True)
                continue
            obs = {m: rmses[m] for m in rank_methods if m in rmses}
            per_obs.append(obs)
            for m, v in rmses.items():
                per_dataset[split.name][m].append(v)
            pretty = " ".join(f"{m}={rmses[m]:.3f}" for m in rank_methods if m in rmses)
            print(f"    {pretty}", flush=True)

    if not per_obs:
        raise RuntimeError("No TabArena observations were collected.")

    avg_rank = average_ranks(per_obs, rank_methods)
    mean_rmse = {}
    rmse_se = {}
    all_methods = sorted({m for obs in per_obs for m in obs} | set(avg_rank))
    # Include full_context means when present.
    for name_map in per_dataset.values():
        for m, vals in name_map.items():
            all_methods.append(m)
    all_methods = sorted(set(all_methods))
    for m in all_methods:
        vals = []
        for name_map in per_dataset.values():
            vals.extend(name_map.get(m, []))
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        mean_rmse[m] = float(arr.mean())
        rmse_se[m] = (
            float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        )

    summary = {
        "protocol": {
            "benchmark": "TabArena-v0.1 regression (OpenML study 457)",
            "budget_frac": budget_frac,
            "max_train": max_train,
            "n_test": n_test,
            "n_seeds": n_seeds,
            "num_features": cfg.num_features,
            "n_observations": len(per_obs),
            "n_clusters": N_CLUSTERS,
            "chunk": CHUNK,
            "note": (
                "Selectors wrap the frozen multi-feature teacher. "
                "LongRange-PFN attends n raw tokens plus a summary of the rest. "
                "Targets are z-scored on the training split; RMSE is on that scale."
            ),
        },
        "avg_rank": avg_rank,
        "mean_rmse": mean_rmse,
        "rmse_se": rmse_se,
        "per_dataset": {
            name: {m: [float(v) for v in vals] for m, vals in methods.items()}
            for name, methods in per_dataset.items()
        },
    }

    out_json = os.path.join(RESULTS_DIR, "tabarena_context_selection.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")
    print("average ranks:", avg_rank)
    print("mean RMSE:", {m: round(mean_rmse[m], 4) for m in rank_methods if m in mean_rmse})

    _write_table(summary, os.path.join(PAPER_DIR, "table_tabarena.tex"))
    _plot(summary, per_dataset)

    # Macros for the paper.
    macro_path = os.path.join(PAPER_DIR, "tabarena_macros.tex")
    lines = []
    for m, r in avg_rank.items():
        key = "".join(p.capitalize() for p in m.split("_"))
        lines.append(f"\\newcommand{{\\tabRank{key}}}{{{r:.3f}}}")
    for m in rank_methods:
        if m not in mean_rmse:
            continue
        key = "".join(p.capitalize() for p in m.split("_"))
        lines.append(f"\\newcommand{{\\tabRmse{key}}}{{{mean_rmse[m]:.3f}}}")
    with open(macro_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {macro_path}")


if __name__ == "__main__":
    main()
