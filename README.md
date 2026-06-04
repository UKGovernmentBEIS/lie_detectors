# lie_detectors

Load the pre-trained lie-detection probes published at [`ai-safety-institute/lie-detection`](https://huggingface.co/collections/ai-safety-institute/lie-detection).

## Install

```bash
uv add lie-detectors        # or: pip install lie-detectors
```

## Usage

```python
import torch
from lie_detectors import get_probe

# Downloads the repo's default checkpoint (probe.pt = the best sweep result).
probe = get_probe("ai-safety-institute/dyl-qwen-qwen3.5-122b-a10b-fp8")

# Which model layer the probe reads activations from.
# Note e.g. the Unrelated Questions probe is instead trained on logprobs
print(probe.layer)  # e.g. 28

# `activations` are residual-stream activations of shape (..., d_model) for the layer the
# probe was trained on.
scores = probe(activations)        # higher score => more likely deceptive
flagged = scores > probe.threshold # threshold is calibrated to ~1% FPR
```

Pick a specific checkpoint from the hyperparameter sweep with `filename=` (see each repo's
`sweep.json` for the full list and their metrics):

```python
probe = get_probe(
    "ai-safety-institute/dyl-qwen-qwen3.5-122b-a10b-fp8",
    filename="l_40_ar_mlp_wd_0_001_lr_0_0001_ep_100.pt",
)
```
