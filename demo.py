"""CPU-only demonstration with synthetic inputs; no retrieval result is produced."""

import torch

from viscon_modules import (
    ReferenceAwareQueryRefinement,
    SimilarityConditionedVisualRegularization,
)


def main():
    torch.manual_seed(1234)
    refinement = ReferenceAwareQueryRefinement(
        query_dim=8, reference_dim=12, num_heads=2, ffn_dim=16, dropout=0.0
    )
    composed_tokens = torch.randn(2, 3, 8, requires_grad=True)
    reference_tokens = torch.randn(2, 5, 12)
    refined_tokens = refinement(composed_tokens, reference_tokens)

    # Synthetic vectors isolate the regularizer from any backbone or retrieval model.
    query_features = torch.tensor([[0.8, 0.6], [0.8, 0.6]], requires_grad=True)
    reference_features = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    target_features = torch.tensor([[1.0, 0.0], [0.85, 0.5267827]])
    regularizer = SimilarityConditionedVisualRegularization()
    loss, stats = regularizer(query_features, reference_features, target_features)
    (refined_tokens.square().mean() + loss).backward()

    print("PyTorch:", torch.__version__)
    print("Refined token shape:", tuple(refined_tokens.shape))
    print("Synthetic sample weights:", stats["sample_weight"].tolist())
    print("Synthetic regularization loss:", round(loss.item(), 6))
    print("Backward pass: OK")
    print("Demo only: no videos, trained model, or benchmark metrics.")


if __name__ == "__main__":
    main()
