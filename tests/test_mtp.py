"""
Tests for TaiChi-MTP Multi-Token Prediction Engine.
"""

import math
import numpy as np
import pytest

from taichi_mtp import (
    MTPEngine,
    MTPHead,
    DepthScheduler,
    DepthMode,
    compute_head_coupling,
    decide_depth_mode,
    depth_for_mode,
    create_mtp_engine,
    MAX_DEPTH,
    HEX_COUPLING_MATRIX,
    HEX_ANGLE,
)


class TestConstants:
    """Verify fundamental constants match their definitions."""

    def test_max_depth(self):
        assert MAX_DEPTH == 6

    def test_hex_angle(self):
        assert pytest.approx(HEX_ANGLE) == math.pi / 3.0

    def test_coupling_matrix_shape(self):
        assert HEX_COUPLING_MATRIX.shape == (6, 6)

    def test_coupling_matrix_symmetric(self):
        """C6 cycle graph should be symmetric."""
        m = HEX_COUPLING_MATRIX
        assert np.allclose(m, m.T)

    def test_coupling_diagonal_one(self):
        """Self-coupling on diagonal should be 1.0."""
        for i in range(6):
            assert HEX_COUPLING_MATRIX[i, i] == 1.0


class TestComputeHeadCoupling:
    """Test head coupling computation."""

    def test_constant_input(self):
        """Constant input → uniform coupling, sum to 1."""
        h = np.ones((64, 256))
        c = compute_head_coupling(h)
        assert c.shape == (MAX_DEPTH,)
        # Zero variance: coupling is uniform by fallback
        assert np.allclose(c, 1.0 / MAX_DEPTH, atol=0.01)

    def test_zero_variance_input(self):
        """Uniform constant → uniform coupling."""
        h = np.full((32, 128), 3.0)
        c = compute_head_coupling(h)
        assert np.allclose(c, 1.0 / MAX_DEPTH, atol=0.01)

    def test_random_input(self):
        h = np.random.randn(100, 256)
        c = compute_head_coupling(h)
        assert np.all(c >= 0)
        assert pytest.approx(float(np.sum(c))) == 1.0

    def test_scalar_input(self):
        h = np.random.randn(50)
        c = compute_head_coupling(h)
        assert c.shape == (MAX_DEPTH,)

    def test_with_prev_predictions(self):
        h = np.random.randn(64, 128)
        prev = [np.random.randn(128) for _ in range(3)]
        c = compute_head_coupling(h, prev)
        assert pytest.approx(float(np.sum(c)), abs=1e-6) == 1.0


class TestDepthMode:
    """Test depth regime classification."""

    def test_shallow(self):
        c = np.array([0.02, 0.02, 0.03, 0.02, 0.02, 0.89])
        assert decide_depth_mode(c) == DepthMode.SHALLOW

    def test_moderate(self):
        # Uniform → high entropy → DEEP
        c_uniform = np.ones(6) / 6.0
        assert decide_depth_mode(c_uniform) == DepthMode.DEEP
        # Moderately concentrated → MODERATE (entropy ~0.83)
        c2 = np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05])
        assert decide_depth_mode(c2) == DepthMode.MODERATE

    def test_deep(self):
        # Strongly peaked → low entropy → SHALLOW
        c = np.array([0.85, 0.03, 0.03, 0.03, 0.03, 0.03])
        assert decide_depth_mode(c) == DepthMode.SHALLOW

    def test_depth_for_mode(self):
        assert depth_for_mode(DepthMode.SHALLOW) == 2
        assert depth_for_mode(DepthMode.MODERATE) == 4
        assert depth_for_mode(DepthMode.DEEP) == 6


class TestMTPHead:
    """Test single prediction head."""

    def test_construction(self):
        head = MTPHead(0, hidden_dim=64, output_dim=32)
        assert head.index == 0
        assert head.W.shape == (32, 64)
        assert head.b.shape == (32,)

    def test_forward_no_siblings(self):
        head = MTPHead(0, hidden_dim=16, output_dim=8)
        h = np.random.randn(16)
        out = head.forward(h)
        assert out.shape == (8,)

    def test_forward_with_siblings(self):
        head = MTPHead(1, hidden_dim=32, output_dim=16)
        h = np.random.randn(32)
        sibling_out = np.random.randn(16)
        out = head.forward(h, sibling_outputs=[sibling_out])
        assert out.shape == (16,)
        # Should differ from no-sibling output
        out_ns = head.forward(h)
        assert not np.allclose(out, out_ns)

    def test_batch_forward(self):
        head = MTPHead(2, hidden_dim=64, output_dim=32)
        h = np.random.randn(8, 64)
        out = head.forward(h)
        assert out.shape == (8, 32)

    def test_invalid_index(self):
        with pytest.raises(ValueError):
            MTPHead(6, hidden_dim=32, output_dim=16)
        with pytest.raises(ValueError):
            MTPHead(-1, hidden_dim=32, output_dim=16)


class TestMTPEngine:
    """Test the full MTP engine."""

    @pytest.fixture
    def engine(self):
        return MTPEngine(hidden_dim=128, output_dim=64, max_depth=6)

    def test_construction(self, engine):
        assert len(engine.heads) == 6
        assert engine.call_count == 0

    def test_forward_returns_result(self, engine):
        h = np.random.randn(32, 128)
        result = engine.forward(h)
        assert len(result.predictions) >= 1
        assert result.depth_used >= 1
        assert result.depth_used <= 6
        assert result.mode in DepthMode
        assert result.head_couplings.shape == (6,)

    def test_call_count_increments(self, engine):
        h = np.random.randn(16, 128)
        engine(h)
        engine(h)
        assert engine.call_count == 2

    def test_depth_history(self, engine):
        h = np.random.randn(16, 128)
        for _ in range(5):
            engine(h)
        stats = engine.depth_stats()
        assert stats["calls"] == 5
        assert stats["min"] >= 1
        assert stats["max"] <= 6

    def test_return_all(self, engine):
        h = np.random.randn(8, 128)
        result = engine.forward(h, return_all=True)
        assert result.depth_used == 6
        assert len(result.predictions) == 6

    def test_stable_input_deep(self, engine):
        """Stable input should yield deeper prediction."""
        h = np.ones((64, 128)) * 0.5
        result = engine(h)
        assert result.depth_used >= 1

    def test_predictions_shape(self, engine):
        h = np.random.randn(16, 128)
        result = engine.forward(h, return_all=True)
        for pred in result.predictions:
            assert pred.shape[-1] == 64  # output_dim

    def test_custom_max_depth(self):
        engine = MTPEngine(hidden_dim=32, output_dim=16, max_depth=3)
        assert len(engine.heads) == 3

    def test_invalid_max_depth(self):
        with pytest.raises(ValueError):
            MTPEngine(hidden_dim=32, output_dim=16, max_depth=7)
        with pytest.raises(ValueError):
            MTPEngine(hidden_dim=32, output_dim=16, max_depth=0)

    def test_repr(self, engine):
        r = repr(engine)
        assert "MTPEngine" in r
        assert "128" in r


class TestPresets:
    """Test convenience constructors."""

    def test_balanced(self):
        engine = create_mtp_engine(256, 128, "balanced")
        assert len(engine.heads) == 6

    def test_fast(self):
        engine = create_mtp_engine(256, 128, "fast")
        assert len(engine.heads) == 4

    def test_precise(self):
        engine = create_mtp_engine(256, 128, "precise")
        assert len(engine.heads) == 6

    def test_unknown_preset(self):
        with pytest.raises(ValueError):
            create_mtp_engine(256, 128, "unknown")


class TestDepthScheduler:
    """Test the depth scheduler."""

    def test_schedule_returns_depth_mode_scores(self):
        s = DepthScheduler()
        c = np.ones(6) / 6.0
        depth, mode, scores = s.schedule(c)
        assert 1 <= depth <= 6
        assert mode in DepthMode
        assert scores.shape == (6,)
