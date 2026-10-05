"""Shared configuration for the Long-Range PFN proof of concept."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class Config:
    # model
    emsize: int = 128
    nhead: int = 4
    nhid: int = 512
    teacher_layers: int = 4
    summarizer_layers: int = 3
    num_summary_tokens: int = 16
    num_buckets: int = 100

    # prior (1d GP, low noise, short lengthscale so context size matters).
    # ls=0.1 keeps the teacher within ~0.1 nats of the exact posterior at
    # this model scale; much shorter lengthscales overwhelm both the teacher
    # and a k-token summary.
    num_features: int = 1
    gp_hyperparameters: dict = field(
        default_factory=lambda: {
            "noise": 1e-2,
            "outputscale": 1.0,
            "lengthscale": 0.1,
        }
    )

    # sequence layout
    max_context: int = 80  # |D_total| = |D'| + |D|
    num_test: int = 40

    # teacher training
    teacher_steps: int = 3000
    teacher_batch_size: int = 32
    teacher_lr: float = 3e-4

    # student distillation
    student_steps: int = 5000
    student_batch_size: int = 32
    student_lr: float = 3e-4
    min_local: int = 5  # smallest |D| during distillation
    max_local: int = 25  # largest |D| in the small-|D| regime
    max_local_wide: int = 75  # largest |D| in the wide regime (window - k)
    p_wide_local: float = 0.4  # prob. of sampling |D| from the wide regime
    p_no_summary: float = 0.1  # prob. of the D_total == D' case (z = null)

    warmup_steps: int = 150
    grad_clip: float = 1.0
    seed: int = 0

    teacher_path: str = os.path.join(RESULTS_DIR, "teacher.pt")
    student_path: str = os.path.join(RESULTS_DIR, "student.pt")
    borders_path: str = os.path.join(RESULTS_DIR, "borders.pt")
