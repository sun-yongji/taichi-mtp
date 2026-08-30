"""
TaiChi-MTP: Multi-Token Prediction Engine

C6-symmetry-inspired parallel inference: six group positions map to six
prediction heads, coupled through hexagonal (C6) topology.

Key components:
- MTPEngine: Core engine with dynamic depth scheduling
- MTPHead: Single prediction head with hexagonal sibling coupling
- DepthScheduler: Adaptive depth selection via S-field coupling
"""

from .predictor import (
    MTPEngine,
    MTPHead,
    DepthScheduler,
    MTPResult,
    DepthMode,
    compute_head_coupling,
    decide_depth_mode,
    depth_for_mode,
    create_mtp_engine,
    MAX_DEPTH,
    HEX_ANGLE,
    HEX_COUPLING_MATRIX,
    GOLD_RATIO_COMP,
)

__all__ = [
    "MTPEngine",
    "MTPHead",
    "DepthScheduler",
    "MTPResult",
    "DepthMode",
    "compute_head_coupling",
    "decide_depth_mode",
    "depth_for_mode",
    "create_mtp_engine",
    "MAX_DEPTH",
    "HEX_ANGLE",
    "HEX_COUPLING_MATRIX",
    "GOLD_RATIO_COMP",
]

__version__ = "0.1.0"
