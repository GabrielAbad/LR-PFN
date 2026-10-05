"""Teacher PFN and the Long-Range PFN student.

The teacher is the unmodified ``TableTransformer`` of the original repo,
running the classic PFN architecture (one token per sample,
``attention_between_features=False``).

The student wraps a backbone with the exact same architecture (so it can be
initialised from the teacher weights) plus a *summarizer* that compresses the
far part of the context D' into ``k`` latent tokens z. Prediction runs the
backbone on the sequence

    [ z_1 .. z_k | D | x_test ]        with single_eval_pos = k + |D|,

i.e. the summary tokens act as additional context items that every test point
attends to, exactly like real context samples would.
"""

from __future__ import annotations

import torch
from torch import nn

from pfns.model.encoders import get_linear_x_encoder, get_linear_y_encoder
from pfns.model.layer import PerFeatureLayer
from pfns.model.transformer import LayerStack, TableTransformer

from .config import Config


def build_teacher(cfg: Config) -> TableTransformer:
    return TableTransformer(
        encoder=get_linear_x_encoder(cfg.emsize, features_per_group=1),
        y_encoder=get_linear_y_encoder(cfg.emsize),
        ninp=cfg.emsize,
        nhead=cfg.nhead,
        nhid=cfg.nhid,
        nlayers=cfg.teacher_layers,
        decoder_dict={"standard": (None, cfg.num_buckets)},
        attention_between_features=False,
        features_per_group=1,
        feature_positional_embedding=None,
        batch_first=True,
        seed=cfg.seed,
    )


def _embed_tokens(
    backbone_or_summarizer,
    x: torch.Tensor,
    y: torch.Tensor | None,
    single_eval_pos: int,
) -> torch.Tensor:
    """Embeds (x, y) into one token per sample, shape (b, s, 1, e).

    Follows the same conventions as ``TableTransformer._forward`` for the
    classic architecture: positions >= single_eval_pos get a NaN y that the
    NaN-handling y-encoder maps to a "missing" embedding, and x/y embeddings
    are summed.
    """
    m = backbone_or_summarizer
    b, s, _ = x.shape

    if y is None:
        y = torch.full((b, s), torch.nan, device=x.device, dtype=x.dtype)
    elif y.ndim == 3:
        y = y.squeeze(-1)
    if y.shape[1] < s:
        pad = torch.full(
            (b, s - y.shape[1]), torch.nan, device=y.device, dtype=y.dtype
        )
        y = torch.cat((y, pad), dim=1)
    y = y.clone()
    y[:, single_eval_pos:] = torch.nan

    # encoders expect sequence-first: x as {"main": (s, b, num_features)}
    emb_x = m.encoder(
        {"main": x.transpose(0, 1)},
        single_eval_pos=single_eval_pos,
    ).transpose(0, 1)  # (b, s, e)
    emb_y = m.y_encoder(
        {"main": y.transpose(0, 1).unsqueeze(-1)},
        single_eval_pos=single_eval_pos,
    ).transpose(0, 1)  # (b, s, e)

    return (emb_x + emb_y).unsqueeze(2)  # (b, s, 1, e)


class Summarizer(nn.Module):
    """rep(D') -> z: compresses D' into ``num_summary_tokens`` latent tokens.

    D' samples are embedded with their own encoders, the k learned latent
    queries are appended as "test" positions (they attend to the D' tokens in
    every layer, while D' tokens attend to each other), and their final
    embeddings are returned as z.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = get_linear_x_encoder(cfg.emsize, features_per_group=1)
        self.y_encoder = get_linear_y_encoder(cfg.emsize)
        self.latents = nn.Parameter(torch.randn(cfg.num_summary_tokens, cfg.emsize) * 0.02)
        # z <- "#": learned null summary for the D_total == D' case
        self.null_z = nn.Parameter(torch.zeros(cfg.num_summary_tokens, cfg.emsize))
        self.layers = LayerStack(
            layer_creator=lambda: PerFeatureLayer(
                d_model=cfg.emsize,
                nhead=cfg.nhead,
                dim_feedforward=cfg.nhid,
                activation="gelu",
                zero_init=True,
                attention_between_features=False,
            ),
            num_layers=cfg.summarizer_layers,
        )

    def forward(
        self,
        x_far: torch.Tensor | None,
        y_far: torch.Tensor | None,
        prev_z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x_far (b, s', f), y_far (b, s') -> z (b, k, e).

        If ``prev_z`` is given, the summary is a *recursive update*
        z_t = rep(chunk_t, z_{t-1}): the previous summary tokens are prepended
        as extra context items, so the new summary can integrate both the new
        chunk and everything already summarized -- streaming, O(chunk) memory.
        """
        if x_far is None or x_far.shape[1] == 0:
            raise ValueError("Use null_summary() when D' is empty.")
        b, s_far, _ = x_far.shape
        tokens = _embed_tokens(self, x_far, y_far, single_eval_pos=s_far)
        if prev_z is not None:
            tokens = torch.cat((prev_z.unsqueeze(2), tokens), dim=1)
        sep = tokens.shape[1]
        queries = self.latents[None, :, None, :].expand(b, -1, 1, -1)
        seq = torch.cat((tokens, queries), dim=1)
        out = self.layers(seq, single_eval_pos=sep)
        return out[:, sep:, 0, :]  # (b, k, e)

    def null_summary(self, batch_size: int) -> torch.Tensor:
        return self.null_z[None].expand(batch_size, -1, -1)


class LongRangePFN(nn.Module):
    """q_r(y | x, D, z) = g(x, D, z): a PFN conditioned on a latent summary."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_teacher(cfg)
        self.summarizer = Summarizer(cfg)

    def init_from_teacher(self, teacher: TableTransformer) -> None:
        self.backbone.load_state_dict(teacher.state_dict())
        # start the summarizer's embedding of (x, y) pairs from the teacher too
        self.summarizer.encoder.load_state_dict(teacher.encoder.state_dict())
        self.summarizer.y_encoder.load_state_dict(teacher.y_encoder.state_dict())

    def summarize(
        self,
        x_far: torch.Tensor | None,
        y_far: torch.Tensor | None,
        batch_size: int,
        n_chunks: int = 1,
    ) -> torch.Tensor:
        """z = rep(D'). With ``n_chunks > 1``, D' is consumed in a stream of
        chunks with the recursive update z_t = rep(chunk_t, z_{t-1}), starting
        from the learned null summary."""
        if x_far is None or x_far.shape[1] == 0:
            return self.summarizer.null_summary(batch_size)
        z = self.summarizer.null_summary(batch_size)
        for x_c, y_c in zip(
            torch.chunk(x_far, n_chunks, dim=1), torch.chunk(y_far, n_chunks, dim=1)
        ):
            if x_c.shape[1] == 0:
                continue
            z = self.summarizer(x_c, y_c, prev_z=z)
        return z

    def forward(
        self,
        x_local: torch.Tensor,  # (b, |D| + n_test, f): local context then test
        y_local: torch.Tensor,  # (b, |D|)
        z: torch.Tensor,  # (b, k, e)
    ) -> torch.Tensor:
        """Returns bar-distribution logits for the test part, (b, n_test, buckets)."""
        n_local = y_local.shape[1]
        tokens = _embed_tokens(self.backbone, x_local, y_local, single_eval_pos=n_local)
        seq = torch.cat((z.unsqueeze(2), tokens), dim=1)
        sep = z.shape[1] + n_local
        out = self.backbone.transformer_layers(seq, single_eval_pos=sep)
        test_out = out[:, sep:, 0, :]
        return self.backbone.decoder_dict["standard"](test_out)

    def predict(
        self,
        x_far: torch.Tensor | None,
        y_far: torch.Tensor | None,
        x_local: torch.Tensor,
        y_local: torch.Tensor,
        x_test: torch.Tensor,
        n_chunks: int = 1,
    ) -> torch.Tensor:
        z = self.summarize(x_far, y_far, batch_size=x_local.shape[0], n_chunks=n_chunks)
        return self(torch.cat((x_local, x_test), dim=1), y_local, z)
