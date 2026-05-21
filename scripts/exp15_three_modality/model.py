"""
ThreeModalityGated — three-branch gated fusion network.

Modalities:
  M_q: quality    (static technical signals)
  M_s: emotion static  (aggregated semantic content)
  M_t: emotion temporal (dynamics over time)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThreeModalityGated(nn.Module):
    def __init__(
        self,
        quality_dim: int,
        emo_static_dim: int,
        emo_temporal_dim: int,
        embed_dim: int = 16,
        gate_hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.q_embed = nn.Sequential(
            nn.Linear(quality_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, embed_dim),
            nn.ReLU(),
        )
        self.s_embed = nn.Sequential(
            nn.Linear(emo_static_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )
        self.t_embed = nn.Sequential(
            nn.Linear(emo_temporal_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, embed_dim),
            nn.ReLU(),
        )

        self.q_head = nn.Linear(embed_dim, 1)
        self.s_head = nn.Linear(embed_dim, 1)
        self.t_head = nn.Linear(embed_dim, 1)

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 3),
        )

    def forward(self, x_q, x_s, x_t):
        h_q = self.q_embed(x_q)
        h_s = self.s_embed(x_s)
        h_t = self.t_embed(x_t)

        z_q = self.q_head(h_q).squeeze(-1)
        z_s = self.s_head(h_s).squeeze(-1)
        z_t = self.t_head(h_t).squeeze(-1)

        gate_logits = self.gate(torch.cat([h_q, h_s, h_t], dim=-1))
        gate_weights = F.softmax(gate_logits, dim=-1)

        z_stacked = torch.stack([z_q, z_s, z_t], dim=-1)
        z_final = (gate_weights * z_stacked).sum(dim=-1)

        return {
            "logit": z_final,
            "gate_weights": gate_weights,       # (B, 3)  [q, s, t]
            "branch_logits": z_stacked,          # (B, 3)
        }


# ── Two-branch variants for ablation ──────────────────────────────────────────

class TwoModalityGated(nn.Module):
    """Generic 2-branch version used in ablation (one modality removed)."""

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        embed_dim: int = 16,
        gate_hidden: int = 32,
        dropout: float = 0.2,
        hidden_a: int = 32,
        hidden_b: int = 32,
    ):
        super().__init__()

        self.a_embed = nn.Sequential(
            nn.Linear(dim_a, hidden_a),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_a, embed_dim),
            nn.ReLU(),
        )
        self.b_embed = nn.Sequential(
            nn.Linear(dim_b, hidden_b),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_b, embed_dim),
            nn.ReLU(),
        )

        self.a_head = nn.Linear(embed_dim, 1)
        self.b_head = nn.Linear(embed_dim, 1)

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 2),
        )

    def forward(self, x_a, x_b):
        h_a = self.a_embed(x_a)
        h_b = self.b_embed(x_b)

        z_a = self.a_head(h_a).squeeze(-1)
        z_b = self.b_head(h_b).squeeze(-1)

        gate_logits = self.gate(torch.cat([h_a, h_b], dim=-1))
        gate_weights = F.softmax(gate_logits, dim=-1)

        z_stacked = torch.stack([z_a, z_b], dim=-1)
        z_final = (gate_weights * z_stacked).sum(dim=-1)

        return {
            "logit": z_final,
            "gate_weights": gate_weights,   # (B, 2)
            "branch_logits": z_stacked,     # (B, 2)
        }


def build_ablation_model(config_name: str, quality_dim: int, emo_static_dim: int,
                          emo_temporal_dim: int, embed_dim: int = 16,
                          gate_hidden: int = 32, dropout: float = 0.2):
    """Factory for ablation variants."""
    if config_name == "full":
        return ThreeModalityGated(
            quality_dim, emo_static_dim, emo_temporal_dim,
            embed_dim=embed_dim, gate_hidden=gate_hidden, dropout=dropout,
        )
    elif config_name == "no_quality":
        # static + temporal
        return TwoModalityGated(
            emo_static_dim, emo_temporal_dim,
            embed_dim=embed_dim, gate_hidden=gate_hidden, dropout=dropout,
            hidden_a=64, hidden_b=32,
        )
    elif config_name == "no_emotion_static":
        # quality + temporal
        return TwoModalityGated(
            quality_dim, emo_temporal_dim,
            embed_dim=embed_dim, gate_hidden=gate_hidden, dropout=dropout,
            hidden_a=32, hidden_b=32,
        )
    elif config_name == "no_emotion_temporal":
        # quality + static
        return TwoModalityGated(
            quality_dim, emo_static_dim,
            embed_dim=embed_dim, gate_hidden=gate_hidden, dropout=dropout,
            hidden_a=32, hidden_b=64,
        )
    else:
        raise ValueError(f"Unknown ablation config: {config_name}")
