"""Trains the teacher PFN (full-context) on the GP prior with the standard
PFN objective: bar-distribution NLL at the test positions, random context
length per step so the model works for any |context| <= max_context.

Run:  python -m long_range_pfn.train_teacher
"""

from __future__ import annotations

import math
import os
import time

import torch

from .config import Config, pick_device, RESULTS_DIR
from .data import make_borders, make_criterion, sample_prior_batch
from .models import build_teacher


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    cfg = Config()
    if os.environ.get("STEPS"):
        cfg.teacher_steps = int(os.environ["STEPS"])
    if os.environ.get("LR"):
        cfg.teacher_lr = float(os.environ["LR"])
    resume = os.environ.get("RESUME") == "1"
    device = pick_device()
    torch.manual_seed(cfg.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(
        f"device: {device}  steps: {cfg.teacher_steps}  "
        f"lr: {cfg.teacher_lr}  resume: {resume}"
    )

    borders = make_borders(cfg)
    torch.save(borders, cfg.borders_path)
    criterion = make_criterion(cfg, borders).to(device)

    teacher = build_teacher(cfg).to(device)
    losses = []
    if resume:
        ckpt = torch.load(cfg.teacher_path, map_location=device, weights_only=True)
        teacher.load_state_dict(ckpt["state_dict"])
        losses = list(ckpt.get("losses", []))
        print(f"resumed teacher from {cfg.teacher_path} with {len(losses)} logged losses")
    n_params = sum(p.numel() for p in teacher.parameters())
    print(f"teacher parameters: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(teacher.parameters(), lr=cfg.teacher_lr)
    scheduler = make_scheduler(optimizer, cfg.warmup_steps, cfg.teacher_steps)

    seq_len = cfg.max_context + cfg.num_test
    t0 = time.time()
    for step in range(cfg.teacher_steps):
        sep = int(torch.randint(2, cfg.max_context + 1, (1,)).item())
        x, y = sample_prior_batch(cfg, cfg.teacher_batch_size, seq_len, device)

        logits = teacher(x, y[:, :sep])  # (b, seq_len - sep, buckets)
        loss = criterion(logits, y[:, sep:]).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        if (step + 1) % 100 == 0:
            avg = sum(losses[-100:]) / 100
            print(
                f"step {step+1:5d}/{cfg.teacher_steps}  nll {avg:.4f}  "
                f"lr {scheduler.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s",
                flush=True,
            )

    torch.save(
        {"state_dict": teacher.state_dict(), "losses": losses},
        cfg.teacher_path,
    )
    print(f"saved teacher to {cfg.teacher_path}")


if __name__ == "__main__":
    main()
