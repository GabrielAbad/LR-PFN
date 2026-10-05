"""Prior sampling and bar-distribution borders for the POC.

Datasets are drawn from the GP prior of the original PFN repo
(``pfns.priors.fast_gp``). Sampling happens on CPU (gpytorch) and the batch is
moved to the training device afterwards.
"""

from __future__ import annotations

import torch

from pfns.model.bar_distribution import (
    FullSupportBarDistribution,
    get_bucket_borders,
)
from pfns.priors import fast_gp

from .config import Config


def sample_prior_batch(
    cfg: Config,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns x (b, s, num_features) and y (b, s) sampled from the GP prior."""
    batch = fast_gp.get_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        num_features=cfg.num_features,
        device=torch.device("cpu"),
        hyperparameters=dict(cfg.gp_hyperparameters),
    )
    return batch.x.to(device), batch.y.to(device)


def make_borders(cfg: Config, num_samples: int = 100_000) -> torch.Tensor:
    """Quantile-based bucket borders estimated from prior samples of y."""
    with torch.random.fork_rng():
        torch.manual_seed(cfg.seed)
        _, y = sample_prior_batch(
            cfg,
            batch_size=max(1, num_samples // 256),
            seq_len=256,
            device=torch.device("cpu"),
        )
    return get_bucket_borders(cfg.num_buckets, ys=y.flatten())


def make_criterion(cfg: Config, borders: torch.Tensor) -> FullSupportBarDistribution:
    return FullSupportBarDistribution(borders=borders.cpu())
