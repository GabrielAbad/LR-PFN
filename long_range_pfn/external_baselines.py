"""External baseline benchmark for LongRange-PFN.

This is intentionally separate from ``evaluate.py``.  It compares the
checkpointed LongRange-PFN against standard non-PFN regressors and, when the
official weights are available, TabPFNRegressor.

TabPFN note: the current official package requires one-time license acceptance
and a ``TABPFN_TOKEN`` to download weights.  If the token/checkpoint is absent,
the script records the reason and still runs all other baselines.

Run:
    N_TASKS=256 python -m long_range_pfn.external_baselines
"""

from __future__ import annotations

import json
import math
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

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
EPS = 1e-8


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def _gaussian_nll(y_true, mean, std):
    var = np.maximum(np.asarray(std) ** 2, EPS)
    err = np.asarray(y_true) - np.asarray(mean)
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * err**2 / var))


def _mean_ci(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()), 0.0
    return float(values.mean()), float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))


def _append(metrics, n_local, method, rmse=None, mae=None, nll=None):
    d = metrics.setdefault(str(n_local), {}).setdefault(
        method, {"rmse": [], "mae": [], "nll": []}
    )
    if rmse is not None:
        d["rmse"].append(float(rmse))
    if mae is not None:
        d["mae"].append(float(mae))
    if nll is not None:
        d["nll"].append(float(nll))


def _kernel_for_gp():
    return (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(length_scale=0.1, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=1e-2, noise_level_bounds="fixed")
    )


def _fit_predict_sklearn(X_train, y_train, X_test, seed):
    preds = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = GaussianProcessRegressor(
            kernel=_kernel_for_gp(),
            optimizer=None,
            normalize_y=False,
            random_state=seed,
        )
        gp.fit(X_train, y_train)
        mean, std = gp.predict(X_test, return_std=True)
        preds["sklearn_gp_rbf"] = {"mean": mean, "std": std}

        krr = KernelRidge(kernel="rbf", gamma=50.0, alpha=1e-2)
        krr.fit(X_train, y_train)
        preds["kernel_ridge_rbf"] = {"mean": krr.predict(X_test)}

        knn_k = min(5, len(X_train))
        knn = KNeighborsRegressor(n_neighbors=knn_k, weights="distance")
        knn.fit(X_train, y_train)
        preds["knn_distance"] = {"mean": knn.predict(X_test)}

        rf = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        )
        rf.fit(X_train, y_train)
        preds["random_forest"] = {"mean": rf.predict(X_test)}

        poly = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=8, include_bias=False),
            Ridge(alpha=1e-2),
        )
        poly.fit(X_train, y_train)
        preds["poly8_ridge"] = {"mean": poly.predict(X_test)}

    return preds


def _maybe_tabpfn(seed):
    if os.environ.get("SKIP_TABPFN") == "1":
        return None, "SKIP_TABPFN=1"
    if not os.environ.get("TABPFN_TOKEN") and os.environ.get("TABPFN_ALLOW_INTERACTIVE") != "1":
        return None, "TABPFN_TOKEN not set; skipped to avoid interactive license prompt"
    try:
        from tabpfn import TabPFNRegressor

        return (
            TabPFNRegressor(
                n_estimators=int(os.environ.get("TABPFN_ESTIMATORS", "8")),
                device=os.environ.get("TABPFN_DEVICE", "cpu"),
                random_state=seed,
                show_progress_bar=False,
            ),
            None,
        )
    except Exception as e:  # pragma: no cover - depends on optional package
        return None, repr(e)


def _tabpfn_predict(tabpfn, X_train, y_train, X_test):
    tabpfn.fit(X_train, y_train)
    pred = tabpfn.predict(X_test, output_type="mean")
    return np.asarray(pred)


@torch.no_grad()
def run_benchmark(cfg, device, teacher, student, criterion, n_tasks=256, batch_size=32):
    metrics = {}
    failures = {}
    seq_len = cfg.max_context + cfg.num_test
    seed = cfg.seed + 29

    tabpfn, tabpfn_init_error = _maybe_tabpfn(seed)
    if tabpfn_init_error:
        failures["tabpfn"] = tabpfn_init_error

    done = 0
    while done < n_tasks:
        b = min(batch_size, n_tasks - done)
        x, y = sample_prior_batch(cfg, b, seq_len, device)
        x_test = x[:, cfg.max_context :]
        y_test = y[:, cfg.max_context :]

        teacher_full = teacher(x, y[:, : cfg.max_context])
        full_mean = criterion.mean(teacher_full).detach().cpu().numpy()
        teacher_full_nll = criterion(teacher_full, y_test).mean(dim=1).detach().cpu().numpy()

        student_cache = {}
        for n_local in SWEEP:
            n_far = cfg.max_context - n_local
            x_far, y_far = x[:, :n_far], y[:, :n_far]
            x_local = x[:, n_far : cfg.max_context]
            y_local = y[:, n_far : cfg.max_context]
            x_local_test = torch.cat((x_local, x_test), dim=1)

            teacher_local = teacher(x_local_test, y_local)
            student_logits = student.predict(x_far, y_far, x_local, y_local, x_test)
            null_logits = student(
                x_local_test, y_local, student.summarizer.null_summary(x.shape[0])
            )

            for method, logits in [
                ("teacher_full", teacher_full),
                ("teacher_local", teacher_local),
                ("longrange_pfn", student_logits),
                ("null_z_ablation", null_logits),
            ]:
                mean = criterion.mean(logits).detach().cpu().numpy()
                nll = criterion(logits, y_test).mean(dim=1).detach().cpu().numpy()
                student_cache[(n_local, method)] = (mean, nll)

            for i in range(b):
                yy = y_test[i].detach().cpu().numpy()
                for method in [
                    "teacher_full",
                    "teacher_local",
                    "longrange_pfn",
                    "null_z_ablation",
                ]:
                    mean, nll = student_cache[(n_local, method)]
                    _append(
                        metrics,
                        n_local,
                        method,
                        rmse=_rmse(yy, mean[i]),
                        mae=_mae(yy, mean[i]),
                        nll=nll[i],
                    )

                X_loc = x_local[i].detach().cpu().numpy()
                y_loc = y_local[i].detach().cpu().numpy()
                X_full = x[i, : cfg.max_context].detach().cpu().numpy()
                y_full = y[i, : cfg.max_context].detach().cpu().numpy()
                X_te = x_test[i].detach().cpu().numpy()

                for prefix, X_train, y_train in [
                    ("local", X_loc, y_loc),
                    ("full80", X_full, y_full),
                ]:
                    preds = _fit_predict_sklearn(X_train, y_train, X_te, seed + done + i)
                    for name, pred in preds.items():
                        method = f"{name}_{prefix}"
                        _append(
                            metrics,
                            n_local,
                            method,
                            rmse=_rmse(yy, pred["mean"]),
                            mae=_mae(yy, pred["mean"]),
                            nll=(
                                _gaussian_nll(yy, pred["mean"], pred["std"])
                                if "std" in pred
                                else None
                            ),
                        )

                    if tabpfn is not None and (
                        prefix == "local" or os.environ.get("TABPFN_FULL80") == "1"
                    ):
                        method = f"tabpfn_{prefix}"
                        try:
                            pred = _tabpfn_predict(tabpfn, X_train, y_train, X_te)
                            _append(
                                metrics,
                                n_local,
                                method,
                                rmse=_rmse(yy, pred),
                                mae=_mae(yy, pred),
                            )
                        except Exception as e:  # license/download/runtime failures
                            failures.setdefault(method, repr(e))
                            tabpfn = None

        done += b
        print(f"processed {done}/{n_tasks} tasks", flush=True)

    summary = {"n_tasks": n_tasks, "failures": failures, "metrics": {}}
    for n_local, methods in metrics.items():
        summary["metrics"][n_local] = {}
        for method, vals in methods.items():
            out = {}
            for metric, arr in vals.items():
                if not arr:
                    continue
                mean, ci = _mean_ci(arr)
                out[metric] = mean
                out[f"{metric}_ci95"] = ci
            summary["metrics"][n_local][method] = out

    return summary


def plot_rmse(summary):
    preferred = [
        ("teacher_full", "Teacher full", C_TEACHER_FULL, "--"),
        ("teacher_local", "Teacher local", C_TEACHER_LOCAL, "-"),
        ("longrange_pfn", "LongRange-PFN", C_STUDENT, "-"),
        ("null_z_ablation", "null-$z$", C_ABLATION, ":"),
        ("sklearn_gp_rbf_local", "GP-RBF local", C_GP, "-"),
        ("kernel_ridge_rbf_local", "KRR local", "#984ea3", "-"),
        ("knn_distance_local", "kNN local", "#4daf4a", "-"),
        ("random_forest_local", "RF local", "#a65628", "-"),
        ("tabpfn_local", "TabPFN local", "#377eb8", "-."),
    ]
    xs = SWEEP
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for key, label, color, ls in preferred:
            ys, cis = [], []
            ok = True
            for n in xs:
                d = summary["metrics"][str(n)].get(key)
                if not d or "rmse" not in d:
                    ok = False
                    break
                ys.append(d["rmse"])
                cis.append(d.get("rmse_ci95", 0.0))
            if not ok:
                continue
            ax.errorbar(xs, ys, yerr=cis, marker="o", color=color, ls=ls,
                        capsize=2.5, label=label)
        ax.set_xlabel(r"Direct context size $|D|$")
        ax.set_ylabel("Test RMSE")
        ax.set_title("External baselines on fresh GP-prior tasks")
        ax.set_xticks(xs)
        ax.legend(loc="upper right", ncol=2)
        fig.tight_layout()
        _save(fig, "external_baselines_rmse.png")


def plot_probabilistic_nll(summary):
    methods = [
        ("teacher_full", "Teacher full", C_TEACHER_FULL, "--"),
        ("teacher_local", "Teacher local", C_TEACHER_LOCAL, "-"),
        ("longrange_pfn", "LongRange-PFN", C_STUDENT, "-"),
        ("null_z_ablation", "null-$z$", C_ABLATION, ":"),
        ("sklearn_gp_rbf_local", "GP-RBF local", C_GP, "-"),
        ("sklearn_gp_rbf_full80", "GP-RBF full80", "#1b7837", "--"),
    ]
    xs = SWEEP
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        for key, label, color, ls in methods:
            ys, cis = [], []
            ok = True
            for n in xs:
                d = summary["metrics"][str(n)].get(key)
                if not d or "nll" not in d:
                    ok = False
                    break
                ys.append(d["nll"])
                cis.append(d.get("nll_ci95", 0.0))
            if not ok:
                continue
            ax.errorbar(xs, ys, yerr=cis, marker="o", color=color, ls=ls,
                        capsize=2.5, label=label)
        ax.set_xlabel(r"Direct context size $|D|$")
        ax.set_ylabel("Test NLL")
        ax.set_title("Probabilistic baselines")
        ax.set_xticks(xs)
        ax.legend(loc="upper right")
        fig.tight_layout()
        _save(fig, "external_baselines_nll.png")


def plot_winrates(summary):
    baselines = [
        "teacher_local",
        "null_z_ablation",
        "sklearn_gp_rbf_local",
        "kernel_ridge_rbf_local",
        "knn_distance_local",
        "random_forest_local",
        "tabpfn_local",
    ]
    # Win-rate needs raw per-task values; this plot uses RMSE means as a compact
    # leaderboard instead of pretending unpaired aggregate wins are paired.
    xs = np.arange(len(SWEEP))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        width = 0.8 / len([b for b in baselines if b in summary["metrics"]["5"]])
        offset = -0.4
        for baseline in baselines:
            if baseline not in summary["metrics"]["5"]:
                continue
            rel = []
            for n in SWEEP:
                ours = summary["metrics"][str(n)]["longrange_pfn"]["rmse"]
                base = summary["metrics"][str(n)][baseline]["rmse"]
                rel.append((base - ours) / max(base, EPS))
            ax.bar(xs + offset + width / 2, rel, width, label=baseline.replace("_", " "))
            offset += width
        ax.axhline(0, color="#1a1a1a", lw=0.8)
        ax.set_xticks(xs, [str(n) for n in SWEEP])
        ax.set_xlabel(r"Direct context size $|D|$")
        ax.set_ylabel("Relative RMSE reduction of LongRange-PFN")
        ax.set_title("Leaderboard-style relative gains")
        ax.legend(loc="upper right", ncol=2, fontsize=7)
        fig.tight_layout()
        _save(fig, "external_baselines_relative_rmse.png")


def main():
    cfg = Config()
    device = torch.device(os.environ["DEVICE"]) if os.environ.get("DEVICE") else pick_device()
    n_tasks = int(os.environ.get("N_TASKS", "256"))
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    torch.manual_seed(cfg.seed + 17)
    print(f"device: {device}  n_tasks: {n_tasks}")

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

    summary = run_benchmark(cfg, device, teacher, student, criterion, n_tasks, batch_size)
    out_path = os.path.join(RESULTS_DIR, "external_baselines.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_path}")
    if summary["failures"]:
        print("failures:", json.dumps(summary["failures"], indent=2)[:2000])

    plot_rmse(summary)
    plot_probabilistic_nll(summary)
    plot_winrates(summary)


if __name__ == "__main__":
    main()
