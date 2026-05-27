"""
tests/test_hooks.py
====================
Unit tests for the MEMOSCOPE hook engine and metric math.

Run with:  pytest tests/ -v
"""

import math
import pytest
import torch
import torch.nn as nn

from memoscope.core.hooks import (
    MemoryInspector,
    StepSnapshot,
    _cosine_drift,
    _attention_entropy,
    _token_retention,
    _l2_norm,
    _collapse_score,
)
from memoscope.core.mock_models import (
    MockTransformer,
    MockMamba,
    MockRNN,
    get_mock_model,
    synthetic_token_stream,
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric math tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCosineDistrift:
    def test_identical_vectors(self):
        v = torch.randn(128)
        assert _cosine_drift(v, v) == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert _cosine_drift(a, b) == pytest.approx(1.0, abs=1e-5)

    def test_opposite_vectors(self):
        v = torch.tensor([1.0, 0.5])
        assert _cosine_drift(v, -v) == pytest.approx(2.0, abs=1e-5)

    def test_shape_mismatch_handled(self):
        a = torch.randn(256)
        b = torch.randn(512)
        # Should not raise — truncates to shorter
        drift = _cosine_drift(a, b)
        assert 0.0 <= drift <= 2.0

    def test_zero_vector_safe(self):
        a = torch.zeros(64)
        b = torch.randn(64)
        drift = _cosine_drift(a, b)
        assert math.isfinite(drift)


class TestAttentionEntropy:
    def test_uniform_attention_max_entropy(self):
        seq = 16
        attn = torch.ones(1, seq, seq) / seq
        H = _attention_entropy(attn)
        assert H == pytest.approx(1.0, abs=0.05)

    def test_delta_attention_low_entropy(self):
        seq = 16
        attn = torch.zeros(1, seq, seq)
        attn[0, :, 0] = 1.0   # all mass on first token
        H = _attention_entropy(attn)
        assert H < 0.2

    def test_multi_head_input(self):
        attn = torch.rand(4, 8, 8)   # 4 heads, seq=8
        attn = attn / attn.sum(dim=-1, keepdim=True)
        H = _attention_entropy(attn)
        assert 0.0 <= H <= 1.0

    def test_1d_input_handled(self):
        attn = torch.rand(16)
        H = _attention_entropy(attn)
        assert 0.0 <= H <= 1.0


class TestTokenRetention:
    def test_output_length(self):
        attn = torch.rand(1, 8, 8)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        ret = _token_retention(attn, max_tokens=32)
        assert len(ret) == 32

    def test_values_bounded(self):
        attn = torch.rand(2, 16, 16)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        ret = _token_retention(attn, max_tokens=32)
        assert all(0.0 <= v <= 1.0 for v in ret)


class TestL2Norm:
    def test_known_norm(self):
        v = torch.tensor([3.0, 4.0])
        assert _l2_norm(v) == pytest.approx(5.0, abs=1e-5)

    def test_multidim(self):
        v = torch.randn(4, 16, 32)
        norm = _l2_norm(v)
        assert norm > 0


class TestCollapseScore:
    def test_healthy_state(self):
        score = _collapse_score([0.7, 0.8, 0.75], [5.0, 5.1, 4.9])
        assert score < 0.4

    def test_collapse_state(self):
        score = _collapse_score([0.05, 0.03, 0.04], [1.0, 100.0, 0.5])
        assert score > 0.4

    def test_empty_inputs(self):
        score = _collapse_score([], [])
        assert score == pytest.approx(0.0)

    def test_bounded(self):
        for _ in range(20):
            H = [float(torch.rand(1)) for _ in range(4)]
            N = [float(torch.rand(1) * 100) for _ in range(4)]
            score = _collapse_score(H, N)
            assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Mock model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMockTransformer:
    @pytest.fixture
    def model(self):
        m = MockTransformer(vocab_size=64, d_model=32, num_heads=2, num_layers=2, max_seq=16)
        m.eval()
        return m

    def test_forward_shape(self, model):
        ids = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (1, 8, 64)

    def test_eval_mode(self, model):
        assert not model.training


class TestMockMamba:
    @pytest.fixture
    def model(self):
        return MockMamba(vocab_size=64, d_model=32, d_state=8, num_layers=2)

    def test_forward_shape(self, model):
        ids = torch.randint(0, 64, (1, 6))
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (1, 6, 64)


class TestMockRNN:
    @pytest.fixture
    def model(self):
        return MockRNN(vocab_size=64, d_model=32, hidden_size=32, num_layers=2)

    def test_forward_shape(self, model):
        ids = torch.randint(0, 64, (1, 5))
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (1, 5, 64)


class TestGetMockModel:
    @pytest.mark.parametrize("mtype", ["transformer", "mamba", "rnn", "ssm", "lstm", "gpt"])
    def test_all_types(self, mtype):
        model = get_mock_model(mtype)
        assert isinstance(model, nn.Module)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            get_mock_model("banana")


class TestSyntheticStream:
    def test_yields_correct_count(self):
        stream = synthetic_token_stream(vocab_size=64, seq_len=10)
        batches = list(stream)
        assert len(batches) == 10

    def test_growing_sequence(self):
        stream = synthetic_token_stream(vocab_size=64, seq_len=5)
        batches = list(stream)
        for i, b in enumerate(batches):
            assert b.shape == (1, i + 1)


# ─────────────────────────────────────────────────────────────────────────────
# MemoryInspector integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryInspector:
    @pytest.fixture
    def inspector(self):
        model = MockTransformer(vocab_size=64, d_model=32, num_heads=2, num_layers=2, max_seq=16)
        return MemoryInspector(model)

    def test_hooks_attached(self, inspector):
        assert len(inspector._hook_handles) > 0

    def test_step_returns_snapshot(self, inspector):
        ids = torch.randint(0, 64, (1, 4))
        with torch.no_grad():
            _ = inspector.model(ids)
        snap = inspector.step()
        assert isinstance(snap, StepSnapshot)
        assert snap.step == 0

    def test_step_increments(self, inspector):
        ids = torch.randint(0, 64, (1, 4))
        with torch.no_grad():
            for _ in range(5):
                _ = inspector.model(ids)
                inspector.step()
        assert inspector._step == 5

    def test_drift_after_multiple_steps(self, inspector):
        for t in range(3):
            ids = torch.randint(0, 64, (1, t + 1))
            with torch.no_grad():
                _ = inspector.model(ids)
            snap = inspector.step()
        # By step 2, drift should be computable
        assert len(snap.layer_drift) > 0
        assert all(math.isfinite(d) for d in snap.layer_drift)

    def test_reset_clears_state(self, inspector):
        ids = torch.randint(0, 64, (1, 4))
        with torch.no_grad():
            _ = inspector.model(ids)
        inspector.step()
        inspector.reset()
        assert inspector._step == 0
        assert len(inspector._history) == 0

    def test_detach_removes_hooks(self, inspector):
        inspector.detach()
        assert len(inspector._hook_handles) == 0

    def test_snapshot_to_dict(self, inspector):
        ids = torch.randint(0, 64, (1, 4))
        with torch.no_grad():
            _ = inspector.model(ids)
        snap = inspector.step()
        d = snap.to_dict()
        assert "step" in d
        assert "mean_drift" in d
        assert "collapse_score" in d
        assert "token_retention" in d
        assert isinstance(d["layer_norms"], list)

    def test_history_accumulates(self, inspector):
        for _ in range(10):
            ids = torch.randint(0, 64, (1, 4))
            with torch.no_grad():
                _ = inspector.model(ids)
            inspector.step()
        assert len(inspector.history()) == 10

    @pytest.mark.parametrize("mtype", ["transformer", "mamba", "rnn"])
    def test_all_model_types(self, mtype):
        model = get_mock_model(mtype)
        insp = MemoryInspector(model, model_type=mtype)
        vocab = getattr(model.embed, "num_embeddings", 64)
        ids = torch.randint(0, vocab, (1, 6))
        with torch.no_grad():
            _ = model(ids)
        snap = insp.step()
        assert snap.model_type == mtype
        insp.detach()
