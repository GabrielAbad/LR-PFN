"""Matched-budget comparison with the context selectors used by CRUMB.

CRUMB (Heredge et al., 2026) is a training-free wrapper: cluster the test
queries, pick a small training subset by greedy MMD minimisation, and run a
frozen PFN on that subset. The same paper compares uniform subsampling,
per-query kNN, and MICP (the retrieval stage of MixturePFN). All four are
reimplemented here and wrapped around *our* frozen teacher, on the same GP
prior LongRange-PFN was distilled on.

This is the comparison that is well-defined for this model. TabArena, the
benchmark CRUMB reports, is a tabular classification suite for pretrained
TabPFN / TabICL checkpoints. LongRange-PFN is a 1-d GP regressor; scoring it
on TabArena would not measure the latent summary.

Two LongRange-PFN variants, both attending n raw tokens plus k summary tokens:

- ``longrange``: random raw window (the training distribution), summary of the
  complement. Single forward pass over the whole complement when it fits in
  one chunk, recursive chunks of 70 past that.
- ``longrange_mmd``: the raw window is the K=1 MMD-matched subset (the context
  CRUMB would feed the teacher), and the complement is summarized. Tests
  whether z still carries information after a strong selector has already
  kept the most relevant raw points.

Run:
    python -m long_range_pfn.context_selection
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
from sklearn.cluster import KMeans

from .config import RESULTS_DIR, Config, pick_device
from .data import make_criterion, sample_prior_batch
from .evaluate import STYLE, _save, kl_per_point
from .models import LongRangePFN, build_teacher

PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper")

# CRUMB uses K=20 test clusters for a few hundred queries. With T=40 that
# ratio is about 8 clusters; K=20 would collapse to near per-query kNN.
N_CLUSTERS = 8
# MICP: K' = ceil(gamma * N / n) with gamma = 1, as in CRUMB Table 1.
CHUNK = 70  # recursive summary chunk, matches the training cap

WINDOW_N = 80
WINDOW_BUDGETS = (8, 16, 40)  # 10%, 20%, 50% of the teacher window
LARGE_N = 320
LARGE_BUDGETS = (32, 64)  # 64 stays inside the student's local-window range

METHOD_ORDER = (
    "uniform",
    "knn",
    "micp",
    "crumb",
    "longrange",
    "longrange_mmd",
)
DISPLAY = {
    "teacher_full": "Teacher, full context",
    "uniform": "Uniform",
    "knn": "Per-query $k$NN",
    "micp": "MICP",
    "crumb": "CRUMB",
    "longrange": "LongRange-PFN",
    "longrange_mmd": "LongRange-PFN + MMD window",
}
COLORS = {
    "teacher_full": "#1a1a1a",
    "uniform": "#878787",
    "knn": "#e08214",
    "micp": "#1b7837",
    "crumb": "#b2182b",
    "longrange": "#2166ac",
    "longrange_mmd": "#7b3294",
}


# ---------------------------------------------------------------------------
# Selectors (numpy, one dataset)
# ---------------------------------------------------------------------------


def _standardize_train(x_train: np.ndarray, x_test: np.ndarray):
    """Z-score with training statistics, so a shifted test set stays shifted."""
    mu = x_train.mean(axis=0, keepdims=True)
    sd = np.maximum(x_train.std(axis=0, keepdims=True), 1e-6)
    return (x_train - mu) / sd, (x_test - mu) / sd


def _median_sigma(x: np.ndarray) -> float:
    if len(x) < 2:
        return 1.0
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    med = float(np.median(dist[np.triu_indices(len(x), k=1)]))
    return med if med > 1e-6 else 1.0


def _rbf(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.exp(-np.sum(diff * diff, axis=-1) / (2.0 * sigma * sigma))


def greedy_mmd(x_train: np.ndarray, x_target: np.ndarray, n: int, sigma: float) -> np.ndarray:
    """Eq. (3) of CRUMB: greedy kernel herding, one point at a time.

    Returns indices into ``x_train``, length min(n, N).
    """
    n_train = len(x_train)
    n = min(n, n_train)
    proximity = _rbf(x_train, x_target, sigma).mean(axis=1)
    chosen = np.zeros(n_train, dtype=bool)
    kernel_to_selected = np.zeros(n_train, dtype=np.float64)
    selected = np.empty(n, dtype=np.int64)
    for t in range(n):
        score = -2.0 * proximity
        if t > 0:
            score = score + (2.0 / (t + 1)) * kernel_to_selected
        score[chosen] = np.inf
        i = int(np.argmin(score))
        selected[t] = i
        chosen[i] = True
        kernel_to_selected += _rbf(x_train, x_train[i : i + 1], sigma)[:, 0]
    return selected


def knn_indices(x_train: np.ndarray, x_query: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n nearest training rows to each query. Shape (T, n)."""
    diff = x_query[:, None, :] - x_train[None, :, :]
    dist = np.sum(diff * diff, axis=-1)
    n = min(n, len(x_train))
    # argpartition is unordered within the n nearest; sort those for stability
    part = np.argpartition(dist, n - 1, axis=1)[:, :n]
    row = np.arange(len(x_query))[:, None]
    order = np.argsort(dist[row, part], axis=1)
    return part[row, order]


def _kmeans(x: np.ndarray, k: int, seed: int):
    k = int(min(k, len(x)))
    if k <= 1:
        centroids = x.mean(axis=0, keepdims=True)
        return np.zeros(len(x), dtype=np.int64), centroids
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(x)
    return labels.astype(np.int64), km.cluster_centers_


def micp_supports(x_train: np.ndarray, n: int, seed: int):
    """MICP supports: cluster the training set, n points per cluster.

    Large clusters are subsampled uniformly. Small clusters are filled out
    with the training points nearest the cluster centroid.
    """
    k_prime = int(max(1, min(len(x_train), math.ceil(len(x_train) / n))))
    labels, centroids = _kmeans(x_train, k_prime, seed)
    rng = np.random.default_rng(seed)
    supports = []
    for c in range(len(centroids)):
        members = np.flatnonzero(labels == c)
        if len(members) >= n:
            take = rng.choice(members, size=n, replace=False)
        else:
            dist = np.sum((x_train - centroids[c]) ** 2, axis=1)
            nearest = np.argsort(dist)
            take = nearest[:n]
        supports.append(np.asarray(take, dtype=np.int64))
    return supports, centroids


def crumb_groups(x_train: np.ndarray, x_test: np.ndarray, n: int, k: int, seed: int, sigma: float):
    """List of (train_idx, test_idx) pairs, one per occupied test cluster."""
    labels, _ = _kmeans(x_test, k, seed)
    groups = []
    for c in range(int(labels.max()) + 1 if len(labels) else 0):
        test_idx = np.flatnonzero(labels == c)
        if len(test_idx) == 0:
            continue
        train_idx = greedy_mmd(x_train, x_test[test_idx], n, sigma)
        groups.append((train_idx, test_idx))
    return groups


def _self_check_selectors() -> None:
    rng = np.random.default_rng(0)
    near = rng.normal(0.0, 0.05, size=(20, 1))
    far = rng.normal(5.0, 0.05, size=(20, 1))
    x_train = np.concatenate([near, far], axis=0)
    x_test = rng.normal(0.0, 0.05, size=(8, 1))
    sigma = _median_sigma(x_train)
    picked = greedy_mmd(x_train, x_test, n=5, sigma=sigma)
    assert np.all(picked < 20), "MMD herding should prefer the nearby cloud"
    nn = knn_indices(x_train, x_test, n=3)
    assert np.all(nn < 20), "kNN should return the nearby cloud"
    print("selector self-check ok")


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------


def _as_batch(idx: np.ndarray, x_rows: torch.Tensor, y_rows: torch.Tensor | None):
    """idx (n,) -> x (1, n, f), and y (1, n) if y_rows is the context vector."""
    x = x_rows[idx].unsqueeze(0)
    if y_rows is None:
        return x
    return x, y_rows[idx].unsqueeze(0)


@torch.no_grad()
def _teacher_logits(teacher, x_ctx, y_ctx, x_query):
    """x_ctx (B, n, f), y_ctx (B, n), x_query (B, t, f) -> (B, t, buckets)."""
    return teacher(torch.cat((x_ctx, x_query), dim=1), y_ctx)


def _nll(criterion, logits, y):
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if y.ndim == 1:
        y = y.unsqueeze(0)
    return float(criterion(logits, y).mean())


def _kl(reference, logits):
    if reference.ndim == 2:
        reference = reference.unsqueeze(0)
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    return float(kl_per_point(reference, logits).mean())


def _chunks_for(n_far: int) -> int:
    if n_far <= 0:
        return 1
    return int(math.ceil(n_far / CHUNK))


@torch.no_grad()
def _student_on_split(student, x_ctx, y_ctx, x_te, local_idx, n_chunks: int):
    """One dataset. ``local_idx`` indexes the raw window inside the context pool."""
    n_pool = x_ctx.shape[0]
    mask = torch.ones(n_pool, dtype=torch.bool, device=x_ctx.device)
    mask[local_idx] = False
    far_idx = mask.nonzero(as_tuple=False).flatten()
    x_local = x_ctx[local_idx].unsqueeze(0)
    y_local = y_ctx[local_idx].unsqueeze(0)
    x_test = x_te.unsqueeze(0)
    if len(far_idx) == 0:
        return student.predict(None, None, x_local, y_local, x_test, n_chunks=1)[0]
    return student.predict(
        x_ctx[far_idx].unsqueeze(0),
        y_ctx[far_idx].unsqueeze(0),
        x_local,
        y_local,
        x_test,
        n_chunks=n_chunks,
    )[0]


def _record(bucket, method, nll, kl):
    bucket[method]["nll"].append(float(nll))
    bucket[method]["kl"].append(None if kl is None else float(kl))


def _empty_methods(extra=()):
    methods = ("teacher_full",) + METHOD_ORDER + tuple(extra)
    return {m: {"nll": [], "kl": []} for m in methods}


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_dataset(
    teacher,
    student,
    criterion,
    x_ctx: torch.Tensor,
    y_ctx: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    budgets: tuple[int, ...],
    seed: int,
    reference_logits: torch.Tensor | None,
    include_teacher_full: bool,
):
    """Returns {budget: {method: {nll, kl}}} for a single dataset.

    ``x_ctx`` is (N, f). A reference logit tensor (T, buckets) enables KL to
    the full-context teacher; pass None when the pool is larger than the
    teacher's training window.
    """
    x_tr = x_ctx.detach().cpu().numpy()
    x_q = x_te.detach().cpu().numpy()
    x_tr_s, x_q_s = _standardize_train(x_tr, x_q)
    sigma = _median_sigma(x_tr_s)
    out = {n: _empty_methods() for n in budgets}

    if include_teacher_full:
        full = _teacher_logits(
            teacher, x_ctx.unsqueeze(0), y_ctx.unsqueeze(0), x_te.unsqueeze(0)
        )[0]
        ref = full
        full_nll = _nll(criterion, full, y_te)
        for n in budgets:
            _record(out[n], "teacher_full", full_nll, 0.0)
    else:
        ref = reference_logits
        for n in budgets:
            # Kept out of the budget ranking; the key is absent on purpose.
            out[n].pop("teacher_full", None)

    for n in budgets:
        # Uniform: one shared random subset, one forward pass.
        rng = np.random.default_rng(seed + 17 * n)
        uni = rng.choice(len(x_tr), size=n, replace=False)
        uni_t = torch.as_tensor(uni, device=x_ctx.device)
        logits = _teacher_logits(
            teacher,
            x_ctx[uni_t].unsqueeze(0),
            y_ctx[uni_t].unsqueeze(0),
            x_te.unsqueeze(0),
        )[0]
        _record(out[n], "uniform", _nll(criterion, logits, y_te), _kl(ref, logits) if ref is not None else None)

        # Per-query kNN. Contexts differ, so the forward is one query at a time
        # stacked into a single batch.
        nn_idx = knn_indices(x_tr_s, x_q_s, n)
        nn_t = torch.as_tensor(nn_idx, device=x_ctx.device)
        x_nn = x_ctx[nn_t]  # (T, n, f)
        y_nn = y_ctx[nn_t]
        x_one = x_te.unsqueeze(1)  # (T, 1, f)
        knn_logits = _teacher_logits(teacher, x_nn, y_nn, x_one)[:, 0, :]
        _record(
            out[n],
            "knn",
            _nll(criterion, knn_logits, y_te),
            _kl(ref, knn_logits) if ref is not None else None,
        )

        # MICP: shared support per training cluster, test points routed to it.
        supports, centroids = micp_supports(x_tr_s, n, seed + n)
        routed = np.argmin(
            np.sum((x_q_s[:, None, :] - centroids[None, :, :]) ** 2, axis=-1),
            axis=1,
        )
        micp_logits = torch.empty_like(knn_logits)
        for c, support in enumerate(supports):
            members = np.flatnonzero(routed == c)
            if len(members) == 0:
                continue
            sup = torch.as_tensor(support, device=x_ctx.device)
            mem = torch.as_tensor(members, device=x_ctx.device)
            pred = _teacher_logits(
                teacher,
                x_ctx[sup].unsqueeze(0),
                y_ctx[sup].unsqueeze(0),
                x_te[mem].unsqueeze(0),
            )[0]
            micp_logits[mem] = pred
        _record(
            out[n],
            "micp",
            _nll(criterion, micp_logits, y_te),
            _kl(ref, micp_logits) if ref is not None else None,
        )

        # CRUMB: test-side clusters, MMD-matched support, one forward per cluster.
        groups = crumb_groups(x_tr_s, x_q_s, n, N_CLUSTERS, seed + 3 * n, sigma)
        crumb_logits = torch.empty_like(knn_logits)
        for train_idx, test_idx in groups:
            tr = torch.as_tensor(train_idx, device=x_ctx.device)
            te = torch.as_tensor(test_idx, device=x_ctx.device)
            pred = _teacher_logits(
                teacher,
                x_ctx[tr].unsqueeze(0),
                y_ctx[tr].unsqueeze(0),
                x_te[te].unsqueeze(0),
            )[0]
            crumb_logits[te] = pred
        _record(
            out[n],
            "crumb",
            _nll(criterion, crumb_logits, y_te),
            _kl(ref, crumb_logits) if ref is not None else None,
        )

        # LongRange-PFN on a random window: the last n pool points, which are
        # a random subset because pool order is the sampling order (or an
        # explicit shuffle). Complement is summarized.
        local = torch.arange(len(x_tr) - n, len(x_tr), device=x_ctx.device)
        lr = _student_on_split(
            student, x_ctx, y_ctx, x_te, local, _chunks_for(len(x_tr) - n)
        )
        _record(
            out[n],
            "longrange",
            _nll(criterion, lr, y_te),
            _kl(ref, lr) if ref is not None else None,
        )

        # Same student, raw window = K=1 MMD match to the whole test set.
        mmd = greedy_mmd(x_tr_s, x_q_s, n, sigma)
        mmd_t = torch.as_tensor(mmd, device=x_ctx.device)
        lr_mmd = _student_on_split(
            student, x_ctx, y_ctx, x_te, mmd_t, _chunks_for(len(x_tr) - n)
        )
        _record(
            out[n],
            "longrange_mmd",
            _nll(criterion, lr_mmd, y_te),
            _kl(ref, lr_mmd) if ref is not None else None,
        )
    return out


def _merge(dest, src):
    for n, methods in src.items():
        bucket = dest.setdefault(str(n), {m: {"nll": [], "kl": []} for m in methods})
        for m, vals in methods.items():
            slot = bucket.setdefault(m, {"nll": [], "kl": []})
            slot["nll"].extend(vals["nll"])
            slot["kl"].extend(vals["kl"])


def _summarize(raw):
    summary = {}
    for n, methods in raw.items():
        summary[n] = {}
        for m, vals in methods.items():
            nll = np.asarray(vals["nll"], dtype=np.float64)
            entry = {
                "nll": float(nll.mean()),
                "nll_se": float(nll.std(ddof=1) / math.sqrt(len(nll))),
                "n_datasets": int(len(nll)),
            }
            kl_vals = [v for v in vals["kl"] if v is not None]
            if kl_vals:
                kl = np.asarray(kl_vals, dtype=np.float64)
                entry["kl"] = float(kl.mean())
                entry["kl_se"] = float(kl.std(ddof=1) / math.sqrt(len(kl)))
            else:
                entry["kl"] = None
                entry["kl_se"] = None
            summary[n][m] = entry
        # Paired win rate: fraction of datasets where LongRange-PFN has lower NLL.
        if "longrange" in methods:
            base = np.asarray(methods["longrange"]["nll"], dtype=np.float64)
            summary[n]["win_rate_nll"] = {}
            for m, vals in methods.items():
                if m == "longrange":
                    continue
                other = np.asarray(vals["nll"], dtype=np.float64)
                summary[n]["win_rate_nll"][m] = float(np.mean(base < other))
    return summary


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _pool_and_test(x, y, n_context):
    """x, y from the prior sampler, context is the prefix."""
    return (
        x[:n_context],
        y[:n_context],
        x[n_context:],
        y[n_context:],
    )


@torch.no_grad()
def evaluate_random_split(
    cfg, device, teacher, student, criterion, n_context, budgets, n_datasets, seed
):
    raw = {}
    seq = n_context + cfg.num_test
    done = 0
    batch = 4
    include_full = n_context <= cfg.max_context
    while done < n_datasets:
        b = min(batch, n_datasets - done)
        # Reseed per batch so a longer run extends a shorter one.
        torch.manual_seed(seed + done)
        x, y = sample_prior_batch(cfg, b, seq, device)
        for i in range(b):
            x_ctx, y_ctx, x_te, y_te = _pool_and_test(x[i], y[i], n_context)
            ref = None
            if include_full:
                ref = _teacher_logits(
                    teacher, x_ctx.unsqueeze(0), y_ctx.unsqueeze(0), x_te.unsqueeze(0)
                )[0]
            part = evaluate_dataset(
                teacher,
                student,
                criterion,
                x_ctx,
                y_ctx,
                x_te,
                y_te,
                budgets,
                seed=seed + 1000 * n_context + done + i,
                reference_logits=ref,
                include_teacher_full=include_full,
            )
            _merge(raw, part)
        done += b
        print(f"  {n_context=} {done}/{n_datasets}", flush=True)
    return _summarize(raw)


@torch.no_grad()
def evaluate_drift(
    cfg, device, teacher, student, criterion, n_datasets, budget, taus, seed, pool_size=240
):
    """Covariate drift along x, in the style of CRUMB Section 5.4.

    Points are ranked by x. The training pool is the lower half. The test
    window slides from that same half (tau=0) to the disjoint upper half
    (tau=1). Budget is fixed. The teacher is not run on the full pool: the
    pool is larger than its training window, which is the regime selectors
    are for.
    """
    raw = {str(tau): {} for tau in taus}
    seq = pool_size + cfg.num_test  # extra points so both halves can spare a test set
    # We sample a cloud and then pick train/test from it, so the sequence
    # length only has to cover the cloud. Use pool_size * 2 points.
    seq = pool_size * 2
    done = 0
    while done < n_datasets:
        torch.manual_seed(seed + done)
        x, y = sample_prior_batch(cfg, 1, seq, device)
        xs = x[0, :, 0].detach().cpu().numpy()
        order = np.argsort(xs)
        ranks = np.empty(len(order), dtype=np.float64)
        ranks[order] = (np.arange(len(order)) + 0.5) / len(order)
        lower = np.flatnonzero(ranks < 0.5)
        for tau in taus:
            lo, hi = tau / 2.0, 0.5 + tau / 2.0
            eligible = np.flatnonzero((ranks >= lo) & (ranks < hi))
            rng = np.random.default_rng(seed + done * 13 + int(tau * 100))
            test_idx = rng.choice(eligible, size=cfg.num_test, replace=False)
            train_idx = np.setdiff1d(lower, test_idx, assume_unique=False)
            if len(train_idx) > pool_size:
                train_idx = rng.choice(train_idx, size=pool_size, replace=False)
            # Shuffle so the "last n" raw window LongRange-PFN reads is random,
            # matching how the student was trained, not a spatial suffix.
            rng.shuffle(train_idx)
            tr = torch.as_tensor(train_idx, device=device)
            te = torch.as_tensor(test_idx, device=device)
            part = evaluate_dataset(
                teacher,
                student,
                criterion,
                x[0, tr],
                y[0, tr],
                x[0, te],
                y[0, te],
                (budget,),
                seed=seed + done + int(tau * 1000),
                reference_logits=None,
                include_teacher_full=False,
            )
            _merge(raw[str(tau)], part)
        done += 1
        if done % 4 == 0 or done == n_datasets:
            print(f"  drift {done}/{n_datasets}", flush=True)
    return {tau: _summarize(payload) for tau, payload in raw.items()}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _cell(mean, se, bold=False):
    body = f"{mean:.3f}\\,{{\\scriptsize$\\pm${se:.3f}}}"
    return f"\\textbf{{{body}}}" if bold else body


def _write_table(summary, budgets, path, nll_only=False):
    methods = [m for m in (("teacher_full",) + METHOD_ORDER) if m in summary[str(budgets[0])]]
    lines = [
        r"\begin{tabular}{l" + "c" * len(budgets) * (1 if nll_only else 2) + "}",
        r"\toprule",
    ]
    if nll_only:
        header = " & ".join([f"$n={n}$" for n in budgets])
        lines.append(f"Method & {header}\\\\")
        lines.append(r"\midrule")
    else:
        nll_span = " & ".join([f"$n={n}$" for n in budgets])
        kl_span = " & ".join([f"$n={n}$" for n in budgets])
        lines.append(
            f"& \\multicolumn{{{len(budgets)}}}{{c}}"
            f"{{NLL (nats)}} & \\multicolumn{{{len(budgets)}}}{{c}}{{KL to teacher-full}}\\\\"
        )
        lines.append(f"Method & {nll_span} & {kl_span}\\\\")
        lines.append(r"\midrule")
    for m in methods:
        cells = []
        for n in budgets:
            entry = summary[str(n)][m]
            # Bold the best budget-matched method. teacher_full is a ceiling,
            # not a competitor, when it is present.
            competitors = [c for c in methods if c != "teacher_full"]
            best = min(competitors, key=lambda c: summary[str(n)][c]["nll"])
            cells.append(_cell(entry["nll"], entry["nll_se"], bold=(m == best)))
        if not nll_only:
            for n in budgets:
                entry = summary[str(n)][m]
                if entry["kl"] is None:
                    cells.append("--")
                    continue
                competitors = [
                    c
                    for c in methods
                    if c != "teacher_full" and summary[str(n)][c]["kl"] is not None
                ]
                best = min(competitors, key=lambda c: summary[str(n)][c]["kl"])
                cells.append(_cell(entry["kl"], entry["kl_se"], bold=(m == best)))
        lines.append(DISPLAY[m] + " & " + " & ".join(cells) + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def _fmt_macro(name, entry, key):
    return f"\\newcommand{{\\{name}}}{{{entry[key]:.3f}}}"


def _write_macros(window, large, drift, path):
    """A few headline numbers the prose can cite. Regenerated with the table."""
    lines = [
        "% Auto-generated by long_range_pfn.context_selection. Do not edit.",
    ]
    w8 = window["8"]
    lines.append(_fmt_macro("selUniformKlEight", w8["uniform"], "kl"))
    lines.append(_fmt_macro("selKnnKlEight", w8["knn"], "kl"))
    lines.append(_fmt_macro("selMicpKlEight", w8["micp"], "kl"))
    lines.append(_fmt_macro("selCrumbKlEight", w8["crumb"], "kl"))
    lines.append(_fmt_macro("selLrKlEight", w8["longrange"], "kl"))
    lines.append(_fmt_macro("selLrMmdKlEight", w8["longrange_mmd"], "kl"))
    w16 = window["16"]
    lines.append(_fmt_macro("selLrKlSixteen", w16["longrange"], "kl"))
    lines.append(_fmt_macro("selKnnKlSixteen", w16["knn"], "kl"))
    lines.append(_fmt_macro("selCrumbKlSixteen", w16["crumb"], "kl"))
    lines.append(_fmt_macro("selUniformKlSixteen", w16["uniform"], "kl"))
    lines.append(
        f"\\newcommand{{\\selWinVsCrumbSixteen}}"
        f"{{{100 * window['16']['win_rate_nll']['crumb']:.0f}\\%}}"
    )
    lines.append(
        f"\\newcommand{{\\selWinVsKnnSixteen}}"
        f"{{{100 * window['16']['win_rate_nll']['knn']:.0f}\\%}}"
    )
    lines.append(_fmt_macro("selUniformNllEight", w8["uniform"], "nll"))
    lines.append(_fmt_macro("selKnnNllEight", w8["knn"], "nll"))
    lines.append(_fmt_macro("selCrumbNllEight", w8["crumb"], "nll"))
    lines.append(_fmt_macro("selLrNllEight", w8["longrange"], "nll"))
    lines.append(_fmt_macro("selLrMmdNllEight", w8["longrange_mmd"], "nll"))
    lines.append(_fmt_macro("selTeacherNllEight", w8["teacher_full"], "nll"))
    # Win rate of the random-window student against CRUMB at n=8, by NLL.
    lines.append(
        f"\\newcommand{{\\selWinVsCrumbEight}}"
        f"{{{100 * window['8']['win_rate_nll']['crumb']:.0f}\\%}}"
    )
    lines.append(
        f"\\newcommand{{\\selWinVsKnnEight}}"
        f"{{{100 * window['8']['win_rate_nll']['knn']:.0f}\\%}}"
    )
    # Large-pool headline at n=32 and n=64: student vs CRUMB NLL.
    for n, tag in ((32, "ThirtyTwo"), (64, "SixtyFour")):
        block = large[str(n)]
        lines.append(_fmt_macro(f"selLargeLrNll{tag}", block["longrange"], "nll"))
        lines.append(_fmt_macro(f"selLargeCrumbNll{tag}", block["crumb"], "nll"))
        lines.append(_fmt_macro(f"selLargeKnnNll{tag}", block["knn"], "nll"))
        lines.append(_fmt_macro(f"selLargeMmdNll{tag}", block["longrange_mmd"], "nll"))
        lines.append(_fmt_macro(f"selLargeUniformNll{tag}", block["uniform"], "nll"))
    # Drift: NLL at the single budget, tau=0 and tau=1.
    for tau, tag in (("0.0", "NoDrift"), ("1.0", "FullDrift")):
        block = drift[tau][list(drift[tau].keys())[0]]
        lines.append(_fmt_macro(f"selDriftCrumb{tag}", block["crumb"], "nll"))
        lines.append(_fmt_macro(f"selDriftKnn{tag}", block["knn"], "nll"))
        lines.append(_fmt_macro(f"selDriftLr{tag}", block["longrange"], "nll"))
        lines.append(_fmt_macro(f"selDriftMmd{tag}", block["longrange_mmd"], "nll"))
        lines.append(_fmt_macro(f"selDriftMicp{tag}", block["micp"], "nll"))
        lines.append(_fmt_macro(f"selDriftUniform{tag}", block["uniform"], "nll"))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def _plot(window, large, drift, n_window_datasets, n_large_datasets):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35))

    def _lines(ax, summary, budgets, ylabel):
        for method, color in COLORS.items():
            if method not in summary[str(budgets[0])]:
                continue
            ys = [summary[str(n)][method]["nll"] for n in budgets]
            se = [summary[str(n)][method]["nll_se"] for n in budgets]
            kw = dict(color=color, label=DISPLAY[method].replace("$", ""))
            if method == "teacher_full":
                ax.axhline(ys[0], color=color, ls="--", lw=1.2, label="Teacher, full context")
                continue
            ax.errorbar(budgets, ys, yerr=[1.96 * s for s in se], marker="o", **kw)
        ax.set_xlabel("Raw context budget $n$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(budgets))

    _lines(axes[0], window, WINDOW_BUDGETS, "Test NLL (nats)")
    axes[0].set_title(f"Inside the window, $N={WINDOW_N}$")
    _lines(axes[1], large, LARGE_BUDGETS, "Test NLL (nats)")
    axes[1].set_title(f"Past the window, $N={LARGE_N}$")

    ax = axes[2]
    methods = list(METHOD_ORDER)
    tau0 = drift["0.0"]
    tau1 = drift["1.0"]
    key0 = next(iter(tau0))
    key1 = next(iter(tau1))
    x = np.arange(len(methods))
    w = 0.38
    y0 = [tau0[key0][m]["nll"] for m in methods]
    e0 = [1.96 * tau0[key0][m]["nll_se"] for m in methods]
    y1 = [tau1[key1][m]["nll"] for m in methods]
    e1 = [1.96 * tau1[key1][m]["nll_se"] for m in methods]
    ax.bar(x - w / 2, y0, w, yerr=e0, color="#92c5de", label=r"$\tau=0$ (no drift)", capsize=2)
    ax.bar(x + w / 2, y1, w, yerr=e1, color="#d6604d", label=r"$\tau=1$ (disjoint $x$)", capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(["Unif.", "kNN", "MICP", "CRUMB", "LR", "LR+MMD"], fontsize=8)
    ax.set_ylabel("Test NLL (nats)")
    ax.set_title(f"Covariate drift, $n={LARGE_BUDGETS[0]}$")
    ax.legend(fontsize=7.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.36, -0.08), fontsize=8)
    fig.tight_layout()
    _save(fig, "context_selection.png")


def _print_block(title, summary):
    print(f"\n{title}")
    for n in summary:
        print(f"  n={n}")
        for m, entry in summary[n].items():
            if m == "win_rate_nll":
                continue
            kl = f"{entry['kl']:.3f}" if entry["kl"] is not None else "  --"
            print(
                f"    {m:<16} nll {entry['nll']:+.3f} ± {entry['nll_se']:.3f}   kl {kl}"
            )


def main():
    _self_check_selectors()
    cfg = Config()
    device = pick_device()
    print(f"device={device} cuda={torch.cuda.is_available()}")
    torch.manual_seed(cfg.seed)

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

    n_window = int(os.environ.get("N_WINDOW", "96"))
    n_large = int(os.environ.get("N_LARGE", "48"))
    n_drift = int(os.environ.get("N_DRIFT", "32"))

    print("in-window comparison")
    window = evaluate_random_split(
        cfg, device, teacher, student, criterion,
        WINDOW_N, WINDOW_BUDGETS, n_window, seed=100,
    )
    print("past-window comparison")
    large = evaluate_random_split(
        cfg, device, teacher, student, criterion,
        LARGE_N, LARGE_BUDGETS, n_large, seed=200,
    )
    print("drift comparison")
    drift = evaluate_drift(
        cfg, device, teacher, student, criterion,
        n_drift, budget=32, taus=(0.0, 1.0), seed=300, pool_size=160,
    )

    _print_block("window", window)
    _print_block("large", large)
    for tau, block in drift.items():
        _print_block(f"drift tau={tau}", block)

    payload = {
        "protocol": {
            "n_clusters": N_CLUSTERS,
            "micp_gamma": 1,
            "chunk": CHUNK,
            "n_window_datasets": n_window,
            "n_large_datasets": n_large,
            "n_drift_datasets": n_drift,
            "note": (
                "Selectors wrap the frozen teacher. LongRange-PFN is the "
                "distilled student. KL is to the teacher on the full pool, "
                "and is only defined when the pool fits in the teacher window."
            ),
        },
        "window": window,
        "large": large,
        "drift": drift,
    }
    out_json = os.path.join(RESULTS_DIR, "context_selection.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_json}")

    _write_table(
        window, WINDOW_BUDGETS, os.path.join(PAPER_DIR, "table_context_selection.tex")
    )
    _write_table(
        large,
        LARGE_BUDGETS,
        os.path.join(PAPER_DIR, "table_context_selection_large.tex"),
        nll_only=True,
    )
    # Drift table: methods by tau, single budget.
    fake = {}
    for tau in ("0.0", "1.0"):
        key = next(iter(drift[tau]))
        fake[tau] = drift[tau][key]
    # Hand-written: two drift intensities, one context budget.
    methods = list(METHOD_ORDER)
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Method & $\tau=0$ & $\tau=1$\\", r"\midrule"]
    for m in methods:
        cells = []
        for tau in ("0.0", "1.0"):
            entry = fake[tau][m]
            competitors = methods
            best = min(competitors, key=lambda c: fake[tau][c]["nll"])
            cells.append(_cell(entry["nll"], entry["nll_se"], bold=(m == best)))
        lines.append(DISPLAY[m] + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    drift_path = os.path.join(PAPER_DIR, "table_context_selection_drift.tex")
    with open(drift_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {drift_path}")

    _write_macros(window, large, drift, os.path.join(PAPER_DIR, "context_selection_macros.tex"))
    _plot(window, large, drift, n_window, n_large)


if __name__ == "__main__":
    main()
