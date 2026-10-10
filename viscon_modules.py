"""Standalone VisCon components for composed video retrieval.

Copyright (c) 2023, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
Inherited source notice retained; see THIRD_PARTY_NOTICES.md and licenses/.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ReferenceAwareQueryRefinement(nn.Module):
    """Refine composed query tokens with gated reference-token evidence."""

    def __init__(
        self,
        query_dim,
        reference_dim,
        num_heads=8,
        ffn_dim=1536,
        dropout=0.1,
    ):
        super().__init__()
        if query_dim % num_heads != 0:
            raise ValueError(
                f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(reference_dim, query_dim)
        self.v_proj = nn.Linear(reference_dim, query_dim)
        self.attention_dropout = dropout

        self.gate_proj = nn.Linear(3 * query_dim, query_dim)
        self.reference_ffn = nn.Sequential(
            nn.Linear(query_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, query_dim),
        )
        self.output_norm = nn.LayerNorm(query_dim)

        nn.init.zeros_(self.reference_ffn[-1].weight)
        nn.init.zeros_(self.reference_ffn[-1].bias)

    def _split_heads(self, tokens):
        batch_size, num_tokens, _ = tokens.shape
        return tokens.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, composed_tokens, reference_tokens):
        query = self._split_heads(self.q_proj(composed_tokens))
        key = self._split_heads(self.k_proj(reference_tokens))
        value = self._split_heads(self.v_proj(reference_tokens))

        attended_reference = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attended_reference = attended_reference.transpose(1, 2).contiguous()
        attended_reference = attended_reference.view_as(composed_tokens)

        gate_input = torch.cat(
            [
                composed_tokens,
                attended_reference,
                composed_tokens * attended_reference,
            ],
            dim=-1,
        )
        reference_gate = torch.sigmoid(self.gate_proj(gate_input))
        reference_residual = self.reference_ffn(attended_reference)
        return self.output_norm(
            composed_tokens + reference_gate * reference_residual
        )


class SimilarityConditionedVisualRegularization(nn.Module):
    """Training-only visual-preservation regularization with sample weights."""

    def __init__(
        self,
        high_threshold=0.92,
        medium_threshold=0.8,
        high_weight=2.0,
        medium_weight=1.2,
        low_weight=0.0,
        eps=1e-6,
    ):
        super().__init__()
        if high_threshold <= medium_threshold:
            raise ValueError("high_threshold must be greater than medium_threshold")
        if not high_weight > medium_weight > 0:
            raise ValueError(
                "Expected high_weight > medium_weight > 0"
            )
        if low_weight < 0 or low_weight >= medium_weight:
            raise ValueError("Expected 0 <= low_weight < medium_weight")

        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.high_weight = high_weight
        self.medium_weight = medium_weight
        self.low_weight = low_weight
        self.eps = eps

    def forward(self, composed_features, reference_features, target_features):
        composed_features = F.normalize(composed_features, dim=-1)
        reference_features = F.normalize(reference_features.detach(), dim=-1)
        target_features = F.normalize(target_features.detach(), dim=-1)

        pair_similarity = (reference_features * target_features).sum(dim=-1)
        high = torch.full_like(pair_similarity, self.high_weight)
        medium = torch.full_like(pair_similarity, self.medium_weight)
        low = torch.full_like(pair_similarity, self.low_weight)
        sample_weight = torch.where(
            pair_similarity > self.high_threshold,
            high,
            torch.where(pair_similarity > self.medium_threshold, medium, low),
        )

        preservation_loss = 1.0 - (
            composed_features * reference_features
        ).sum(dim=-1)
        loss = (sample_weight * preservation_loss).sum() / (
            sample_weight.sum() + self.eps
        )
        stats = {
            "pair_similarity": pair_similarity.detach(),
            "sample_weight": sample_weight.detach(),
            "preservation_loss": preservation_loss.detach(),
        }
        return loss, stats
