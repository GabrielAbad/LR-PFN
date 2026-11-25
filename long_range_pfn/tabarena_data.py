"""TabArena regression loaders used by CRUMB-style context selection.

CRUMB (Heredge et al., 2026) evaluates context selection on the 51-dataset
TabArena suite (OpenML study 457). LongRange-PFN is a regressor, so we use
the TabArena regression tasks and map each dataset to a fixed feature width
compatible with the distilled student (pad / PCA to ``num_features``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# TabArena-v0.1 regression tasks (OpenML study 457). Prefer the smaller
# TabPFN-compatible ones first; larger sets are still supported with a
# train-size cap in the benchmark runner.
TABARENA_REGRESSION_TASKS: tuple[tuple[int, str], ...] = (
    (363698, "QSAR_fish_toxicity"),
    (363625, "concrete_compressive_strength"),
    (363675, "healthcare_insurance_expenses"),
    (363612, "airfoil_self_noise"),
    (363615, "Another-Dataset-on-used-Fiat-500"),
    (363708, "wine_quality"),
    (363686, "miami_housing"),
    (363678, "houses"),
    (363705, "superconductivity"),
    (363672, "Food_Delivery_Time"),
    (363693, "physiochemical_protein"),
    (363631, "diamonds"),
    (363697, "QSAR-TID-11"),
)


@dataclass
class TabularSplit:
    name: str
    task_id: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    y_mean: float
    y_std: float


def _to_numeric_matrix(frame) -> np.ndarray:
    """Encode categoricals as ints and cast everything to float64."""
    cols = []
    for col in frame.columns:
        series = frame[col]
        if hasattr(series, "cat") or series.dtype == object or str(series.dtype) == "string":
            encoded = LabelEncoder().fit_transform(series.astype(str).fillna("__nan__"))
            cols.append(encoded.astype(np.float64))
        else:
            vals = series.to_numpy(dtype=np.float64, copy=True)
            nan_mask = np.isnan(vals)
            if nan_mask.any():
                fill = np.nanmedian(vals) if (~nan_mask).any() else 0.0
                vals[nan_mask] = fill
            cols.append(vals)
    return np.column_stack(cols)


def fit_feature_map(
    x_train: np.ndarray, x_test: np.ndarray, num_features: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map arbitrary tabular width to ``num_features`` via PCA or zero-pad."""
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train)
    x_te = scaler.transform(x_test)
    d = x_tr.shape[1]
    if d == num_features:
        return x_tr.astype(np.float32), x_te.astype(np.float32)
    if d > num_features:
        pca = PCA(n_components=num_features, random_state=0)
        return (
            pca.fit_transform(x_tr).astype(np.float32),
            pca.transform(x_te).astype(np.float32),
        )
    pad = num_features - d
    return (
        np.pad(x_tr, ((0, 0), (0, pad))).astype(np.float32),
        np.pad(x_te, ((0, 0), (0, pad))).astype(np.float32),
    )


def standardize_targets(
    y_train: np.ndarray, y_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    if std < 1e-8:
        std = 1.0
    return (
        ((y_train - mean) / std).astype(np.float32),
        ((y_test - mean) / std).astype(np.float32),
        mean,
        std,
    )


def load_openml_task_xy(task_id: int) -> tuple[np.ndarray, np.ndarray, str]:
    task = openml.tasks.get_task(task_id, download_splits=False)
    dataset = task.get_dataset()
    target = task.target_name
    x_df, y_s, _, _ = dataset.get_data(target=target, dataset_format="dataframe")
    x = _to_numeric_matrix(x_df)
    y = np.asarray(y_s, dtype=np.float64).reshape(-1)
    return x, y, dataset.name


def make_split(
    task_id: int,
    name: str,
    num_features: int,
    seed: int,
    max_train: int,
    n_test: int,
    test_frac: float = 0.25,
) -> TabularSplit:
    """Download one TabArena regression task and build a fixed numeric split."""
    x, y, openml_name = load_openml_task_xy(task_id)
    display = name or openml_name
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_frac, random_state=seed
    )
    rng = np.random.default_rng(seed)
    if len(x_train) > max_train:
        take = rng.choice(len(x_train), size=max_train, replace=False)
        x_train, y_train = x_train[take], y_train[take]
    if len(x_test) > n_test:
        take = rng.choice(len(x_test), size=n_test, replace=False)
        x_test, y_test = x_test[take], y_test[take]

    x_train, x_test = fit_feature_map(x_train, x_test, num_features)
    y_train, y_test, y_mean, y_std = standardize_targets(y_train, y_test)
    return TabularSplit(
        name=display,
        task_id=task_id,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        y_mean=y_mean,
        y_std=y_std,
    )


def iter_regression_splits(
    num_features: int,
    seed: int = 0,
    max_train: int = 400,
    n_test: int = 80,
    task_limit: int | None = None,
) -> list[TabularSplit]:
    tasks = TABARENA_REGRESSION_TASKS
    if task_limit is not None:
        tasks = tasks[:task_limit]
    splits = []
    for i, (task_id, name) in enumerate(tasks):
        splits.append(
            make_split(
                task_id=task_id,
                name=name,
                num_features=num_features,
                seed=seed + 17 * i,
                max_train=max_train,
                n_test=n_test,
            )
        )
    return splits
