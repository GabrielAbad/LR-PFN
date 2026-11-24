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

    # prior (GP; lengthscale short so context size matters).
    # ls=0.1 keeps the teacher within ~0.1 nats of the exact posterior at
    # this model scale; much shorter lengthscales overwhelm both the teacher
    # and a k-token summary. num_features=1 is the original 1-d PoC;
    # TabArena regression uses a multi-feature GP with the same one-token-
    # per-sample backbone (features_per_group = num_features).
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

    def resolve_checkpoint_paths(self) -> None:
        """Point checkpoints at feature-specific files when num_features != 1.

        Keeps the original 1-d PoC artifacts untouched while multi-feature
        TabArena runs write teacher_f{d}.pt / student_f{d}.pt / borders_f{d}.pt.
        """
        if self.num_features == 1:
            return
        tag = f"f{self.num_features}"
        self.teacher_path = os.path.join(RESULTS_DIR, f"teacher_{tag}.pt")
        self.student_path = os.path.join(RESULTS_DIR, f"student_{tag}.pt")
        self.borders_path = os.path.join(RESULTS_DIR, f"borders_{tag}.pt")


def config_from_env() -> Config:
    """Build a Config, honoring NUM_FEATURES / STEPS / LR overrides."""
    cfg = Config()
    if os.environ.get("NUM_FEATURES"):
        cfg.num_features = int(os.environ["NUM_FEATURES"])
    if os.environ.get("STEPS"):
        # Shared override used by both trainers; train_student also reads STEPS.
        cfg.teacher_steps = int(os.environ["STEPS"])
        cfg.student_steps = int(os.environ["STEPS"])
    if os.environ.get("TEACHER_STEPS"):
        cfg.teacher_steps = int(os.environ["TEACHER_STEPS"])
    if os.environ.get("STUDENT_STEPS"):
        cfg.student_steps = int(os.environ["STUDENT_STEPS"])
    if os.environ.get("LR"):
        cfg.teacher_lr = float(os.environ["LR"])
        cfg.student_lr = float(os.environ["LR"])
    # Isotropic GP lengthscales that work in 1-d are too short in higher-d:
    # typical pairwise distances grow like sqrt(d), so keep a comparable
    # correlation length for TabArena-width priors.
    if cfg.num_features > 1 and "GP_LENGTHSCALE" not in os.environ:
        # Empirically, ~1.0 keeps context informative in d~8; sqrt(d) scaling
        # of the 1-d lengthscale (0.1) is still too short for the teacher to
        # learn above the prior predictive.
        cfg.gp_hyperparameters = {
            **cfg.gp_hyperparameters,
            "lengthscale": 1.0,
        }
    if os.environ.get("GP_LENGTHSCALE"):
        cfg.gp_hyperparameters = {
            **cfg.gp_hyperparameters,
            "lengthscale": float(os.environ["GP_LENGTHSCALE"]),
        }
    cfg.resolve_checkpoint_paths()
    return cfg
