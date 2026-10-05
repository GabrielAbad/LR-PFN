"""Distills the frozen full-context teacher into the Long-Range PFN student.

Per step, a fresh prior batch with context D_total = D' | D and test points is
sampled; the split point is random. The loss is the KL divergence between the
teacher's predictive conditioned on the FULL context and the student's
predictive conditioned on (D, z(D')), averaged over test points:

    L = E_x  KL( p_teacher(y | x, D_total)  ||  q_student(y | x, D, z(D')) )

Both predictives are bar distributions over the same buckets, so the KL
reduces to the categorical KL between bucket probabilities.

Run:  python -m long_range_pfn.train_student
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .config import Config, pick_device
from .data import sample_prior_batch
from .models import build_teacher, LongRangePFN
from .train_teacher import make_scheduler


def kl_teacher_student(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """KL(p_t || q_s) per element, reduced to a scalar mean."""
    log_p = F.log_softmax(teacher_logits, dim=-1)
    log_q = F.log_softmax(student_logits, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1).mean()


#: (|D|, n_chunks) regimes covered by the fixed validation set. Single-shot
#: regimes (n_chunks=1) at |D| in {5,10,20,40} plus the null-summary case, AND
#: multi-chunk regimes at n_chunks in {2,3} for both a small and a wide |D| --
#: the earlier version of this validation set only checked n_chunks=1, so
#: checkpoint selection was blind to recursive-summarization quality and
#: silently favored single-shot fidelity at the cost of multi-chunk KL
#: (0.025 -> 0.303 nats from T=1 to T=5 on the checkpoint that criterion
#: picked). Including (10,2),(10,3),(40,2),(40,3) here closes that gap.
VAL_REGIMES = [
    (5, 1), (10, 1), (20, 1), (40, 1),
    (10, 2), (10, 3), (40, 2), (40, 3),
]


def make_fixed_validation_set(cfg: Config, device, n_batches: int = 8):
    """Fixed (seeded, held-out) validation batches keyed by (|D|, n_chunks),
    plus the null-summary case, resampled once before training and reused at
    every eval so the curve reflects optimization progress, not a moving
    target."""
    seq_len = cfg.max_context + cfg.num_test
    regimes = VAL_REGIMES + [(cfg.max_context, 1)]  # null-summary case
    with torch.random.fork_rng(devices=[] if device.type != "cuda" else [device]):
        torch.manual_seed(cfg.seed + 999)
        batches = {r: [] for r in regimes}
        for n_local, n_chunks in regimes:
            for _ in range(n_batches):
                x, y = sample_prior_batch(cfg, cfg.student_batch_size, seq_len, device)
                batches[(n_local, n_chunks)].append((x, y))
    return batches


@torch.no_grad()
def evaluate_fixed_validation(cfg, teacher, student, val_batches):
    """Mean KL(teacher_full || student) on the fixed validation batches, one
    number per (|D|, n_chunks) regime. Cheap: a handful of forward passes,
    no grad."""
    was_training = student.training
    student.eval()
    result = {}
    for (n_local, n_chunks), batches in val_batches.items():
        n_far = cfg.max_context - n_local
        kls = []
        for x, y in batches:
            teacher_logits = teacher(x, y[:, : cfg.max_context])
            x_far, y_far = (x[:, :n_far], y[:, :n_far]) if n_far > 0 else (None, None)
            x_local, y_local = x[:, n_far : cfg.max_context], y[:, n_far : cfg.max_context]
            x_test = x[:, cfg.max_context :]
            student_logits = student.predict(
                x_far, y_far, x_local, y_local, x_test, n_chunks=n_chunks
            )
            kls.append(kl_teacher_student(teacher_logits, student_logits).item())
        result[(n_local, n_chunks)] = sum(kls) / len(kls)
    if was_training:
        student.train()
    return result


def sample_split(cfg: Config) -> tuple[int, int]:
    """Returns (|D|, n_chunks) for this step; |D'| = max_context - |D|.

    n_chunks > 1 trains the recursive streaming update z_t = rep(chunk_t, z_{t-1}).
    """
    u = torch.rand(1).item()
    if u < cfg.p_no_summary:
        return cfg.max_context, 1  # D_total == D', z = null
    if u < cfg.p_no_summary + cfg.p_wide_local:
        # wide regime: |D| close to the window, z summarizes a small D'
        n_local = int(torch.randint(cfg.max_local + 1, cfg.max_local_wide + 1, (1,)).item())
    else:
        # small regime: z carries most of the information
        n_local = int(torch.randint(cfg.min_local, cfg.max_local + 1, (1,)).item())
    n_chunks = int(torch.multinomial(torch.tensor([0.4, 0.3, 0.3]), 1).item()) + 1
    return n_local, n_chunks


def main():
    import os

    cfg = Config()
    if os.environ.get("STEPS"):
        cfg.student_steps = int(os.environ["STEPS"])
    if os.environ.get("LR"):
        cfg.student_lr = float(os.environ["LR"])
    resume = os.environ.get("RESUME") == "1"
    device = pick_device()
    torch.manual_seed(cfg.seed + 1)
    print(f"device: {device}  steps: {cfg.student_steps}  lr: {cfg.student_lr}  resume: {resume}")

    ckpt = torch.load(cfg.teacher_path, map_location=device, weights_only=True)
    teacher = build_teacher(cfg).to(device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = LongRangePFN(cfg).to(device)
    if resume:
        student.load_state_dict(
            torch.load(cfg.student_path, map_location=device, weights_only=True)["state_dict"]
        )
        print(f"resumed student from {cfg.student_path}")
    else:
        student.init_from_teacher(teacher)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"student parameters: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.student_lr)
    scheduler = make_scheduler(optimizer, cfg.warmup_steps, cfg.student_steps)

    val_batches = make_fixed_validation_set(cfg, device)
    val_every = max(1, cfg.student_steps // 40)  # ~40 points along the curve
    val_log = []  # list of (step, {regime: kl})
    best_val, best_state = float("inf"), None

    seq_len = cfg.max_context + cfg.num_test
    losses, split_log = [], []
    t0 = time.time()
    for step in range(cfg.student_steps):
        n_local, n_chunks = sample_split(cfg)
        n_far = cfg.max_context - n_local

        x, y = sample_prior_batch(cfg, cfg.student_batch_size, seq_len, device)
        x_far, y_far = x[:, :n_far], y[:, :n_far]
        x_local, y_local = x[:, n_far : cfg.max_context], y[:, n_far : cfg.max_context]
        x_test = x[:, cfg.max_context :]

        with torch.no_grad():
            teacher_logits = teacher(x, y[:, : cfg.max_context])

        student_logits = student.predict(
            x_far if n_far > 0 else None,
            y_far if n_far > 0 else None,
            x_local,
            y_local,
            x_test,
            n_chunks=n_chunks,
        )
        loss = kl_teacher_student(teacher_logits, student_logits)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        split_log.append((n_local, n_chunks))
        if (step + 1) % 100 == 0:
            avg = sum(losses[-100:]) / 100
            print(
                f"step {step+1:5d}/{cfg.student_steps}  kl {avg:.4f}  "
                f"lr {scheduler.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s",
                flush=True,
            )

        if (step + 1) % val_every == 0 or step == cfg.student_steps - 1:
            val = evaluate_fixed_validation(cfg, teacher, student, val_batches)
            val_log.append((step + 1, val))
            # selection criterion covers single-shot AND multi-chunk regimes,
            # so a checkpoint can't be "best" by sacrificing recursive quality
            val_sum = sum(val[r] for r in VAL_REGIMES)
            if val_sum < best_val:
                best_val = val_sum
                best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            regimes_str = "  ".join(
                f"|D|={n}{'' if t == 1 else f'/T{t}'}:{v:.3f}" for (n, t), v in val.items()
            )
            print(f"  val @ {step+1}: {regimes_str}  (sum {val_sum:.3f})", flush=True)

    torch.save(
        {
            "state_dict": best_state if best_state is not None else student.state_dict(),
            "best_state_dict": best_state,
            "losses": losses,
            "splits": split_log,
            "val_log": val_log,
            "best_val": best_val,
        },
        cfg.student_path,
    )
    print(f"saved student to {cfg.student_path} (best val KL sum: {best_val:.4f})")


if __name__ == "__main__":
    main()
