"""Multi-feature packing keeps one token per sample."""

from __future__ import annotations

import torch

from long_range_pfn.config import Config
from long_range_pfn.models import LongRangePFN, build_teacher


def test_teacher_accepts_multi_feature_batch():
    cfg = Config(num_features=4, max_context=16, num_test=4)
    teacher = build_teacher(cfg)
    x = torch.randn(2, cfg.max_context + cfg.num_test, cfg.num_features)
    y = torch.randn(2, cfg.max_context + cfg.num_test)
    logits = teacher(x, y[:, : cfg.max_context])
    assert logits.shape == (2, cfg.num_test, cfg.num_buckets)


def test_student_predict_multi_feature():
    cfg = Config(num_features=4, max_context=16, num_test=4, num_summary_tokens=4)
    student = LongRangePFN(cfg)
    b, f = 2, cfg.num_features
    x_far = torch.randn(b, 8, f)
    y_far = torch.randn(b, 8)
    x_local = torch.randn(b, 8, f)
    y_local = torch.randn(b, 8)
    x_test = torch.randn(b, cfg.num_test, f)
    logits = student.predict(x_far, y_far, x_local, y_local, x_test, n_chunks=2)
    assert logits.shape == (b, cfg.num_test, cfg.num_buckets)
