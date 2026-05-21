"""ModalityGatedFusion architecture for Exp.15."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityGatedFusion(nn.Module):
    """
    Three-branch architecture with learnable per-video gating over modalities.

    Modalities:
      M_d: detector score (scalar input)
      M_e: emotion descriptors (emotion_dim-dim input, ~49 features from data)
      M_q: quality features (quality_dim-dim input)

    Each branch projects its input to a shared embedding dim.
    A gating head computes per-video softmax weights over the three modalities.
    Final prediction is a gated mixture of the three per-branch scalar logits.
    """

    def __init__(
        self,
        emotion_dim: int = 49,
        quality_dim: int = 4,
        embed_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.emotion_dim = emotion_dim
        self.quality_dim = quality_dim
        self.embed_dim = embed_dim

        # Per-modality embedders
        self.det_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.emo_embed = nn.Sequential(
            nn.Linear(emotion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )
        self.qual_embed = nn.Sequential(
            nn.Linear(quality_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Per-modality logit heads
        self.det_head = nn.Linear(embed_dim, 1)
        self.emo_head = nn.Linear(embed_dim, 1)
        self.qual_head = nn.Linear(embed_dim, 1)

        # Gating head — takes concatenated embeddings, produces 3 softmax weights
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

    def forward(self, det: torch.Tensor, emo: torch.Tensor, qual: torch.Tensor) -> dict:
        """
        Args:
            det:  (B, 1)           detector score
            emo:  (B, emotion_dim) emotion features
            qual: (B, quality_dim) quality features

        Returns dict with keys:
            logit         (B,)   final gated logit
            gate_weights  (B, 3) softmax weights [det, emo, qual]
            branch_logits (B, 3) individual branch logits [det, emo, qual]
        """
        h_d = self.det_embed(det)    # (B, embed_dim)
        h_e = self.emo_embed(emo)    # (B, embed_dim)
        h_q = self.qual_embed(qual)  # (B, embed_dim)

        z_d = self.det_head(h_d).squeeze(-1)   # (B,)
        z_e = self.emo_head(h_e).squeeze(-1)   # (B,)
        z_q = self.qual_head(h_q).squeeze(-1)  # (B,)

        h_concat = torch.cat([h_d, h_e, h_q], dim=-1)   # (B, embed_dim * 3)
        gate_logits = self.gate(h_concat)                # (B, 3)
        gate_weights = F.softmax(gate_logits, dim=-1)    # (B, 3)

        z_stacked = torch.stack([z_d, z_e, z_q], dim=-1)   # (B, 3)
        z_final = (gate_weights * z_stacked).sum(dim=-1)    # (B,)

        return {
            "logit": z_final,
            "gate_weights": gate_weights,
            "branch_logits": z_stacked,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
