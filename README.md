# VisCon — Submission-stage Code Preview

This repository provides a **partial code release** for VisCon, containing two
standalone components for composed video retrieval:

- **RAQR**: reference-aware query refinement using reference-token attention and
  gated residual injection.
- **SCVR**: similarity-conditioned visual-preservation regularization during
  training.

Alongside this code preview, we will provide the VGIR-revised training
annotations as a separate CSV file. The complete code and trained model weights
will be publicly released upon manuscript acceptance.

This preview does not include the full retrieval model, backbone integration,
training/evaluation pipelines, offline instruction rewriting pipeline, or model
checkpoints. It is not a complete benchmark reproduction package.

## VGIR-revised training annotations

The VGIR-revised training CSV contains reference video IDs, target video IDs,
and the final edit instructions used for training. Instructions selected for
revision are replaced with their VGIR-generated versions; the remaining
instructions are retained. The CSV will be provided separately from the source
code archive.

The training loader uses `pth1` for the reference video ID, `pth2` for the target
video ID, and `edit` for the final instruction. Use the revised CSV for training
and keep the official validation/test annotations unchanged. Raw video files
are not included and should be obtained through the original dataset sources.

## Quick start

Requirements: Python 3.10+ and PyTorch 2.0+.

```bash
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -s tests -v
```

The demo and tests run on CPU with synthetic tensors. They do not require data,
pretrained weights, API credentials, or model downloads. Install the appropriate
PyTorch build for your operating system if it is not already installed.

## Component interfaces

```python
from viscon_modules import (
    ReferenceAwareQueryRefinement,
    SimilarityConditionedVisualRegularization,
)

refinement = ReferenceAwareQueryRefinement(query_dim=768, reference_dim=1408)
# composed_tokens: [batch, query_tokens, query_dim]
# reference_tokens: [batch, reference_tokens, reference_dim]
refined_tokens = refinement(composed_tokens, reference_tokens)

regularizer = SimilarityConditionedVisualRegularization()
# All feature tensors have shape [batch, embedding_dim].
# This regularizer is used during training, not query-time inference.
loss, diagnostics = regularizer(query_features, reference_features, target_features)
```

The snippet documents interfaces; `demo.py` supplies concrete synthetic inputs.
RAQR consumes already composed query tokens and reference visual tokens; it does
not accept a target video or target caption. SCVR uses reference-target features
for training supervision. Backbone encoding, query composition, and feature
pooling/projection are outside this preview.

The demo reports tensor shapes and a synthetic loss only, not retrieval results.

## Release plan

The submission-stage release consists of the standalone RAQR/SCVR components
and the separately provided VGIR-revised training CSV. Upon manuscript
acceptance, we will publicly release the complete code and trained model
weights, including the integrated retrieval model, training/evaluation scripts,
experiment configurations, and usage instructions. Dataset sharing remains
subject to the original dataset's redistribution terms.

## License and acknowledgements

The existing project license and inherited source notices have been retained.
See `LICENSE` and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Please acknowledge the corresponding VisCon manuscript when using these
components in research.
