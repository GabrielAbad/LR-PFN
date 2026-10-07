"""Defensive baselines at large context (appendix).

The demanding version of the "is this task trivial / why not just use
TabPFN?" question. The tabular baselines here have NO context-window limit,
so we give them the *entire* context (up to 640 points, 8x the teacher's
window) and ask whether LongRange-PFN---restricted to a recent local window
plus an O(k) recursive summary---can rival methods that simply ingest all
the data.

Reference / bounded-state methods (see only recent points + summary):
  teacher_window   teacher on the 80 most recent points (recency baseline)
  student          LongRange-PFN: 70 recent points + recursive z over the rest

Full-context baselines (see all |Dtot| points):
  tabpfn_full      TabPFN regressor on all points (its native predictive dist.)
  gp_rbf_full      scikit-learn GP, RBF kernel, hyperparameters fit per dataset
  krr / knn / rf   kernel ridge / k-NN / random forest on all points

Oracle:
  gp_exact_full    exact GP posterior with the generating hyperparameters

NLL (nats) for probabilistic methods, RMSE for all. Point predictors report
RMSE only. Run:
  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 DEVICE=cpu \\
      python -m long_range_pfn.baselines
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor

from .config import Config, pick_device, RESULTS_DIR
from .data import make_criterion, sample_prior_batch
from .evaluate import gp_exact_nll
from .models import build_teacher, LongRangePFN

try:
    from tabpfn import TabPFNRegressor  # type: ignore

    _HAS_TABPFN = True
except Exception:  # noqa: BLE001
    _HAS_TABPFN = False

TOTALS = [80, 160, 320, 640]  # total context sizes (up to 8x the window)
LOCAL = 70                     # LongRange-PFN local window
WINDOW = 80                    # teacher recency window
CHUNK = 70                     # recursive summary chunk size
N_DATASETS = 100
TABPFN_CAP = 40                # TabPFN is slow per call on CPU


def _gauss_nll(mean, std, y):
    var = np.clip(std, 1e-6, None) ** 2
    return float(
        (0.5 * math.log(2 * math.pi) + 0.5 * np.log(var) + (y - mean) ** 2 / (2 * var)).mean()
    )


def _rmse(pred, y):
    return float(np.sqrt(((pred.ravel() - y.ravel()) ** 2).mean()))


def _fit_sklearn_gp(xc, yc, xt):
    kernel = ConstantKernel(1.0) * RBF(0.1) + WhiteKernel(1e-2)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0)
    gp.fit(xc, yc)
    mean, std = gp.predict(xt, return_std=True)
    return mean, std


def _mean_se(vals):
    a = np.asarray(vals, dtype=np.float64)
    if len(a) == 0:
        return (None, None)
    return (float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0)


def main():
    cfg = Config()
    device = torch.device(os.environ["DEVICE"]) if os.environ.get("DEVICE") else pick_device()
    torch.manual_seed(cfg.seed + 5)
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)  # avoid OpenMP double-load segfault (torch+sklearn+tabpfn)
    print(f"device: {device}  tabpfn: {_HAS_TABPFN}", flush=True)

    borders = torch.load(cfg.borders_path, weights_only=True)
    crit = make_criterion(cfg, borders).to(device)

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
    tabpfn = TabPFNRegressor(device="cpu") if _HAS_TABPFN else None

    methods = ["teacher_window", "student", "gp_exact_full",
               "tabpfn_full", "gp_rbf_full", "krr", "knn", "rf"]
    raw = {}

    for N in TOTALS:
        acc = {m: {"nll": [], "rmse": []} for m in methods}
        for i in range(N_DATASETS):
            x, y = sample_prior_batch(cfg, 1, N + cfg.num_test, device)
            xt, yt = x[:, N:], y[:, N:]
            Yt = yt[0].cpu().numpy()

            # --- bounded-state methods -----------------------------------
            xw = x[:, N - WINDOW : N]
            yw = y[:, N - WINDOW : N]
            t_logits = teacher(torch.cat((xw, xt), dim=1), yw)
            acc["teacher_window"]["nll"].append(crit(t_logits, yt).mean().item())
            acc["teacher_window"]["rmse"].append(_rmse(crit.mean(t_logits).cpu().numpy(), Yt))

            n_far = N - LOCAL
            x_far, y_far = x[:, :n_far], y[:, :n_far]
            x_loc, y_loc = x[:, n_far:N], y[:, n_far:N]
            n_chunks = max(1, math.ceil(n_far / CHUNK))
            s_logits = student.predict(x_far, y_far, x_loc, y_loc, xt, n_chunks=n_chunks)
            acc["student"]["nll"].append(crit(s_logits, yt).mean().item())
            acc["student"]["rmse"].append(_rmse(crit.mean(s_logits).cpu().numpy(), Yt))

            acc["gp_exact_full"]["nll"].append(
                gp_exact_nll(cfg, x[:, :N], y[:, :N], xt, yt)
            )

            # --- full-context baselines (all N points) -------------------
            Xc = x[0, :N].cpu().numpy()
            Yc = y[0, :N].cpu().numpy()
            Xt = xt[0].cpu().numpy()

            m, s = _fit_sklearn_gp(Xc, Yc, Xt)
            acc["gp_rbf_full"]["nll"].append(_gauss_nll(m, s, Yt))
            acc["gp_rbf_full"]["rmse"].append(_rmse(m, Yt))

            krr = KernelRidge(kernel="rbf", alpha=1e-2, gamma=1.0 / (2 * 0.1**2)).fit(Xc, Yc)
            acc["krr"]["rmse"].append(_rmse(krr.predict(Xt), Yt))

            knn = KNeighborsRegressor(n_neighbors=min(5, N)).fit(Xc, Yc)
            acc["knn"]["rmse"].append(_rmse(knn.predict(Xt), Yt))

            rf = RandomForestRegressor(n_estimators=100, n_jobs=1).fit(Xc, Yc.ravel())
            acc["rf"]["rmse"].append(_rmse(rf.predict(Xt), Yt))

            if tabpfn is not None and len(acc["tabpfn_full"]["rmse"]) < TABPFN_CAP:
                try:
                    tabpfn.fit(Xc, Yc.ravel())
                    out = tabpfn.predict(Xt, output_type="full")
                    acc["tabpfn_full"]["rmse"].append(_rmse(np.asarray(out["mean"]), Yt))
                    lg = out["logits"]
                    yt_t = torch.as_tensor(Yt.ravel(), dtype=lg.dtype)
                    acc["tabpfn_full"]["nll"].append(float(out["criterion"](lg, yt_t).mean()))
                except Exception as e:  # noqa: BLE001
                    print(f"tabpfn failed at N={N}: {e}", flush=True)

        raw[N] = {
            m: {"nll": _mean_se(d["nll"]), "rmse": _mean_se(d["rmse"]),
                "n": max(len(d["nll"]), len(d["rmse"]))}
            for m, d in acc.items()
        }
        print(f"N={N} done ({N_DATASETS} datasets)", flush=True)

    with open(os.path.join(RESULTS_DIR, "baselines.json"), "w") as f:
        json.dump(raw, f, indent=2)
    print(f"saved {os.path.join(RESULTS_DIR, 'baselines.json')}", flush=True)


if __name__ == "__main__":
    main()
