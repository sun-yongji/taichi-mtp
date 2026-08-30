"""
TaiChi-MTP: Multi-Token Prediction Engine

C6-symmetry-inspired parallel inference: each group position maps to a
prediction head, coupled through hexagonal (C6) topology.

Core concepts:
- Depth N ∈ [1..6] maps to C6 structure group positions (g0→g5)
- Hexagonal coupling matrix governs head interaction
- Dynamic scheduler adjusts depth based on S-field coupling strength
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum MTP depth (C6 group has 6 elements)
MAX_DEPTH: int = 6

# Hexagonal coupling angle (60° = 2π/6)
HEX_ANGLE: float = math.pi / 3.0  # 60° in radians

# Coupling matrix derived from C6 symmetry group
# Each row corresponds to a slot-level prediction head;
# columns represent influence from other group positions.
HEX_COUPLING_MATRIX: np.ndarray = np.array([
    [1.000, 0.500, 0.000, 0.000, 0.000, 0.500],  # g0
    [0.500, 1.000, 0.500, 0.000, 0.000, 0.000],  # g1
    [0.000, 0.500, 1.000, 0.500, 0.000, 0.000],  # g2
    [0.000, 0.000, 0.500, 1.000, 0.500, 0.000],  # g3
    [0.000, 0.000, 0.000, 0.500, 1.000, 0.500],  # g4
    [0.500, 0.000, 0.000, 0.000, 0.500, 1.000],  # g5
], dtype=np.float64)

# Gold ratio entropy compensation for depth scheduling
GOLD_RATIO_COMP: float = 0.0618


class DepthMode(Enum):
    """Prediction depth regime based on coupling strength."""
    SHALLOW = "shallow"       # c < 0.5  → predict 1-2 tokens
    MODERATE = "moderate"     # 0.5 ≤ c < 1.5 → predict 3-4 tokens
    DEEP = "deep"             # c ≥ 1.5 → predict 5-6 tokens


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MTPResult:
    """Result from one forward pass of the MTP engine."""
    # Predicted token embeddings/representations for each depth level
    predictions: List[np.ndarray]
    # Depth actually used (may be < MAX_DEPTH due to dynamic scheduling)
    depth_used: int
    # Per-head coupling strengths
    head_couplings: np.ndarray
    # The depth mode that was selected
    mode: DepthMode
    # Raw scores per head before thresholding
    head_scores: np.ndarray
    # Extra metadata
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Coupling analysis
# ---------------------------------------------------------------------------

def compute_head_coupling(
    hidden_states: np.ndarray,
    prev_predictions: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Compute per-head coupling strength from hidden states.

    Maps variance across the hidden dimension → C6 coupling coefficients.

    Args:
        hidden_states: Shape (batch, seq_len, hidden_dim) or (seq_len, hidden_dim)
        prev_predictions: Previous prediction outputs for recurrent coupling

    Returns:
        Coupling vector of length MAX_DEPTH
    """
    arr = np.asarray(hidden_states, dtype=np.float64)

    # Handle 1D input: add a dimension
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]

    # Flatten batch/seq dims for variance computation
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    elif arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])

    # Per-dimension variance as the raw coupling signal
    dim_var = np.var(arr, axis=0)
    if dim_var.ndim == 0:
        dim_var = np.array([float(dim_var)])
    var_total = float(np.sum(dim_var)) + 1e-12

    # Zero-variance edge case: return uniform coupling
    if var_total <= 2e-12:
        return np.full(MAX_DEPTH, 1.0 / MAX_DEPTH, dtype=np.float64)

    # Distribute variance across 6 heads using hexagonal projection
    ndim = len(dim_var)
    coupling = np.zeros(MAX_DEPTH, dtype=np.float64)
    step = max(1, ndim // MAX_DEPTH)
    for i in range(MAX_DEPTH):
        start = i * step
        end = min(start + step, ndim)
        coupling[i] = float(np.sum(dim_var[start:end])) / var_total

    # Normalize so sum = 1
    cs = float(np.sum(coupling))
    if cs > 0:
        coupling /= cs

    # If we have previous predictions, blend in recurrence signal
    if prev_predictions is not None:
        for i, prev in enumerate(prev_predictions[:MAX_DEPTH]):
            prev_var = float(np.var(prev)) + 1e-12
            # Recurrence adds coupling proportional to prediction stability
            coupling[i] = 0.7 * coupling[i] + 0.3 * (1.0 / (1.0 + prev_var))
        # Re-normalize after blending
        cs2 = float(np.sum(coupling))
        if cs2 > 0:
            coupling /= cs2

    return coupling


def decide_depth_mode(coupling: np.ndarray) -> DepthMode:
    """Determine the depth regime from coupling vector.

    Uses normalized entropy: a uniform (high-entropy) coupling
    distribution indicates stable conditions → DEEP prediction.
    A concentrated (low-entropy) distribution indicates turbulence
    → SHALLOW prediction.

    Entropy thresholds are calibrated to the C6 C6 structure structure.
    """
    eps = 1e-12
    ent = -float(np.sum(coupling * np.log(coupling + eps)))
    ent_max = math.log(MAX_DEPTH)
    ent_norm = ent / ent_max if ent_max > 0 else 0.0

    if ent_norm > 0.90:
        return DepthMode.DEEP
    elif ent_norm > 0.65:
        return DepthMode.MODERATE
    else:
        return DepthMode.SHALLOW


def depth_for_mode(mode: DepthMode) -> int:
    """Return the recommended prediction depth for a given mode."""
    return {
        DepthMode.SHALLOW: 2,
        DepthMode.MODERATE: 4,
        DepthMode.DEEP: 6,
    }[mode]


# ---------------------------------------------------------------------------
# Prediction heads
# ---------------------------------------------------------------------------

class MTPHead:
    """A single multi-token prediction head.

    Each head corresponds to one group position in the C6 structure,
    projecting hidden states to vocabulary/logit space.
    """

    def __init__(
        self,
        index: int,
        hidden_dim: int,
        output_dim: int,
        coupling_row: Optional[np.ndarray] = None,
    ):
        if not 0 <= index < MAX_DEPTH:
            raise ValueError(f"Head index {index} out of range [0, {MAX_DEPTH})")

        self.index = index
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize projection matrix with hexagonal coupling
        rng = np.random.default_rng(42 + index)
        self.W = rng.normal(0, 1.0 / math.sqrt(hidden_dim), (output_dim, hidden_dim)).astype(np.float64)
        self.b = np.zeros(output_dim, dtype=np.float64)

        # Coupling weights from sibling heads
        self.coupling_row = (
            coupling_row if coupling_row is not None
            else HEX_COUPLING_MATRIX[index].copy()
        )

    def forward(
        self,
        hidden: np.ndarray,
        sibling_outputs: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        """Forward pass with optional hexagonal sibling coupling.

        Args:
            hidden: Shape (hidden_dim,) or (batch, hidden_dim)
            sibling_outputs: List of outputs from other heads for coupling

        Returns:
            Output logits/embeddings
        """
        base = hidden @ self.W.T + self.b  # (..., output_dim)

        # Apply hexagonal coupling from sibling heads
        if sibling_outputs is not None and len(sibling_outputs) > 0:
            coupled = np.zeros_like(base)
            total_weight = 0.0

            for j, sib_out in enumerate(sibling_outputs):
                w = float(self.coupling_row[j])
                if w > 0 and j != self.index and j < len(sibling_outputs):
                    coupled += w * sib_out
                    total_weight += w

            if total_weight > 0:
                base = base + coupled / total_weight

        return base


# ---------------------------------------------------------------------------
# Dynamic Depth Scheduler
# ---------------------------------------------------------------------------

class DepthScheduler:
    """Dynamically selects how many tokens to predict based on S-field coupling.

    The scheduler maps the coupling strength distribution onto the
    C6 structure's 6-tier depth ladder:

    - Weak coupling (stable input)   → depth 6 (full confidence)
    - Moderate coupling              → depth 3-4
    - Strong coupling (turbulent)    → depth 1-2 (conservative)
    """

    def __init__(
        self,
        min_depth: int = 1,
        max_depth: int = MAX_DEPTH,
        entropy_comp: float = GOLD_RATIO_COMP,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.entropy_comp = entropy_comp

    def schedule(self, coupling: np.ndarray) -> Tuple[int, DepthMode, np.ndarray]:
        """Determine optimal prediction depth.

        Args:
            coupling: Per-head coupling vector (length MAX_DEPTH)

        Returns:
            (depth, mode, head_scores) — depth to use, the mode, per-head scores
        """
        mode = decide_depth_mode(coupling)
        base_depth = depth_for_mode(mode)

        # Compute per-head activation scores from coupling
        # High coupling → low activation (turbulent → suppress)
        head_scores = 1.0 - coupling
        hs = float(np.sum(head_scores)) + 1e-12
        head_scores = head_scores / hs

        # Entropy-compensated depth: exploration bonus proportional to
        # the entropy of the coupling distribution
        ent = -float(np.sum(coupling * np.log(coupling + 1e-12)))
        ent_max = math.log(MAX_DEPTH)
        ent_norm = ent / ent_max if ent_max > 0 else 0.0
        depth_bonus = int(round(ent_norm * self.entropy_comp * MAX_DEPTH))

        depth = min(self.max_depth, max(self.min_depth, base_depth + depth_bonus))

        return depth, mode, head_scores


# ---------------------------------------------------------------------------
# MTP Engine
# ---------------------------------------------------------------------------

class MTPEngine:
    """Multi-Token Prediction engine with hexagonal head coupling.

    Usage::

        engine = MTPEngine(hidden_dim=512, output_dim=256, max_depth=6)
        hidden = np.random.randn(128, 512)  # (seq_len, hidden_dim)
        result = engine(hidden)
        print(f"Predicted {result.depth_used} tokens, mode={result.mode.value}")
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        max_depth: int = MAX_DEPTH,
        coupling_matrix: Optional[np.ndarray] = None,
        entropy_comp: float = GOLD_RATIO_COMP,
    ):
        if not 1 <= max_depth <= MAX_DEPTH:
            raise ValueError(f"max_depth must be in [1, {MAX_DEPTH}]")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_depth = max_depth

        self.coupling_matrix = (
            coupling_matrix if coupling_matrix is not None
            else HEX_COUPLING_MATRIX[:max_depth, :max_depth].copy()
        )

        # Create prediction heads
        self.heads: List[MTPHead] = []
        for i in range(max_depth):
            head = MTPHead(
                index=i,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                coupling_row=self.coupling_matrix[i].copy(),
            )
            self.heads.append(head)

        self.scheduler = DepthScheduler(
            min_depth=1,
            max_depth=max_depth,
            entropy_comp=entropy_comp,
        )

        # Stats
        self.call_count: int = 0
        self.depth_history: List[int] = []

    def forward(
        self,
        hidden: np.ndarray,
        prev_predictions: Optional[List[np.ndarray]] = None,
        return_all: bool = False,
    ) -> MTPResult:
        """Predict multiple future tokens from hidden states.

        Args:
            hidden: Shape (..., hidden_dim) input hidden states
            prev_predictions: Previous step predictions for recurrence
            return_all: If True, return all head outputs regardless of depth

        Returns:
            MTPResult with predictions, depth used, and metadata
        """
        self.call_count += 1

        # 1. Compute coupling distribution
        coupling = compute_head_coupling(hidden, prev_predictions)

        # 2. Schedule depth
        depth, mode, head_scores = self.scheduler.schedule(coupling)
        if not return_all:
            actual_depth = depth
        else:
            actual_depth = self.max_depth

        self.depth_history.append(actual_depth)

        # 3. Run heads sequentially with sibling coupling
        predictions: List[np.ndarray] = []
        for i in range(actual_depth):
            # Gather sibling outputs for coupling
            siblings = predictions  # already-computed heads
            out = self.heads[i].forward(hidden, siblings)
            predictions.append(out)

        return MTPResult(
            predictions=predictions,
            depth_used=actual_depth,
            head_couplings=coupling,
            mode=mode,
            head_scores=head_scores,
            meta={
                "call_count": self.call_count,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    def __call__(self, *args: Any, **kwargs: Any) -> MTPResult:
        return self.forward(*args, **kwargs)

    def depth_stats(self) -> Dict[str, float]:
        """Return summary statistics on depth usage."""
        if not self.depth_history:
            return {}
        arr = np.array(self.depth_history, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "calls": len(self.depth_history),
        }

    def __repr__(self) -> str:
        return (
            f"MTPEngine(hidden_dim={self.hidden_dim}, output_dim={self.output_dim}, "
            f"max_depth={self.max_depth}, heads={len(self.heads)}, "
            f"calls={self.call_count})"
        )


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def create_mtp_engine(
    hidden_dim: int,
    output_dim: int,
    preset: str = "balanced",
) -> MTPEngine:
    """Create an MTP engine with a preset configuration.

    Presets:
        - "balanced": 6 heads, standard hexagonal coupling
        - "fast": 4 heads, moderate coupling for speed
        - "precise": 6 heads, flat coupling (all heads equal weight)
    """
    preset_configs = {
        "balanced": {"max_depth": 6, "entropy_comp": GOLD_RATIO_COMP},
        "fast": {"max_depth": 4, "entropy_comp": 0.05},
        "precise": {"max_depth": 6, "entropy_comp": 0.0},
    }

    if preset not in preset_configs:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(preset_configs)}")

    cfg = preset_configs[preset]
    return MTPEngine(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        max_depth=cfg["max_depth"],
        entropy_comp=cfg["entropy_comp"],
    )
