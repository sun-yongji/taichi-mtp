# TaiChi-MTP: Multi-Token Prediction Engine

Hexagram-inspired parallel inference — each yao position maps to a
prediction head, coupled through hexagonal (C6) topology.

## Overview

Standard autoregressive models predict **one** token at a time.
TaiChi-MTP predicts **up to 6 tokens** in parallel, using a hexagonal
coupling structure derived from the C6 symmetry group.

Each of the 6 prediction heads corresponds to one yao of the hexagram
(初→上), with coupling weights determined by cyclic neighbor adjacency
on the hexagon:

```
    初爻 ── 二爻
    /           \
  上爻           三爻
    \           /
    五爻 ── 四爻
```

## Dynamic Depth

The engine automatically adjusts prediction depth based on S-field
coupling strength:

| Coupling | Regime | Depth | Use case |
|----------|--------|-------|----------|
| c < 0.25 | Shallow | 1-2 | High-variance input |
| 0.25 ≤ c < 0.55 | Moderate | 3-4 | Normal input |
| c ≥ 0.55 | Deep | 5-6 | Stable input |

## Quick Start

```python
import numpy as np
from taichi_mtp import MTPEngine, create_mtp_engine

# Create engine
engine = create_mtp_engine(hidden_dim=512, output_dim=256, preset="balanced")

# Forward pass with (seq_len, hidden_dim) input
hidden = np.random.randn(128, 512)
result = engine(hidden)

print(f"Depth: {result.depth_used}, Mode: {result.mode.value}")
print(f"Head couplings: {result.head_couplings.round(3)}")
```

## License

Apache 2.0 — see LICENSE file.

## Part of TaiChi-Matrix

M2 module of the TaiChi-Matrix: Eastern-Numerology-Driven MoE Training
& Quantization Open-Source Toolkit for CCF OSS 2026.
