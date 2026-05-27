"""
memoscope/core/hooks.py
=======================
PyTorch Forward Hook Engine for MEMOSCOPE.

Captures hidden states, attention matrices, and SSM recurrent states
during inference WITHOUT modifying the model's computational graph.

Metrics computed per step
--------------------------
1. Hidden State Drift   D(t) = 1 - cos(h_t, h_{t-1})
2. Layer-wise Norm      ||h_t||_2  per layer
3. Token Decay          Exponential decay of early-token attention mass
4. Context Entropy      H(attention) = -sum(p * log(p+eps))
5. Collapse Indicator   Sharp entropy spike / state saturation signal
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepSnapshot:
    """All memory metrics captured at a single inference step."""

    step: int                              # global token position
    timestamp: float                       # wall-clock seconds since epoch

    # Per-layer hidden state L2 norms  [num_layers]
    layer_norms: List[float] = field(default_factory=list)

    # Per-layer cosine drift from previous step  [num_layers]
    layer_drift: List[float] = field(default_factory=list)

    # Flattened token-retention heatmap  [num_layers x seq_len] (row-major)
    token_retention: List[List[float]] = field(default_factory=list)

    # Shannon entropy of attention distribution per layer  [num_layers]
    attention_entropy: List[float] = field(default_factory=list)

    # Single scalar: context collapse score  [0, 1]
    collapse_score: float = 0.0

    # Average drift across all layers (headline metric)
    mean_drift: float = 0.0

    # Model type tag ("transformer" | "mamba" | "rnn")
    model_type: str = "transformer"

    # Current sequence length
    seq_len: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict for WebSocket broadcast."""
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "layer_norms": self.layer_norms,
            "layer_drift": self.layer_drift,
            "token_retention": self.token_retention,
            "attention_entropy": self.attention_entropy,
            "collapse_score": round(self.collapse_score, 4),
            "mean_drift": round(self.mean_drift, 4),
            "model_type": self.model_type,
            "seq_len": self.seq_len,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Metric Utilities
# ─────────────────────────────────────────────────────────────────────────────

_EPS = 1e-8


def _cosine_drift(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute cosine-based drift between two state vectors.

    D(t) = 1 - (h_t · h_{t-1}) / (||h_t|| * ||h_{t-1}||)

    Returns a value in [0, 2].  0 = identical, 1 = orthogonal, 2 = opposite.
    """
    a_flat = a.detach().float().reshape(-1)
    b_flat = b.detach().float().reshape(-1)

    # Guard: truncate / pad to the shorter length to handle shape mismatches
    min_len = min(a_flat.shape[0], b_flat.shape[0])
    a_flat = a_flat[:min_len]
    b_flat = b_flat[:min_len]

    dot = torch.dot(a_flat, b_flat)
    norm_a = torch.norm(a_flat).clamp(min=_EPS)
    norm_b = torch.norm(b_flat).clamp(min=_EPS)
    cosine_sim = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    return float(1.0 - cosine_sim)


def _attention_entropy(attn: torch.Tensor) -> float:
    """
    Shannon entropy of an attention weight matrix.

    H = -sum_i p_i * log(p_i + eps)

    attn: [heads, seq, seq] or [seq, seq]
    Returns entropy normalised to [0, 1] by log(seq_len).
    """
    # Flatten to a 2-D probability distribution over the last axis
    p = attn.detach().float()
    if p.dim() == 3:
        p = p.mean(dim=0)          # average over heads -> [seq, seq]
    if p.dim() == 1:
        p = p.unsqueeze(0)

    # Ensure proper probability distribution
    p = p / (p.sum(dim=-1, keepdim=True).clamp(min=_EPS))
    H = -(p * torch.log(p + _EPS)).sum(dim=-1).mean()   # scalar

    # Normalise by maximum possible entropy log(seq_len)
    seq_len = p.shape[-1]
    H_max = math.log(seq_len + 1) + _EPS
    return float((H / H_max).clamp(0.0, 1.0))


def _token_retention(attn: torch.Tensor, max_tokens: int = 32) -> List[float]:
    """
    Build a token-retention row: how much attention mass each early token
    still receives as sequence grows.

    Returns a list of `max_tokens` floats in [0, 1].
    """
    p = attn.detach().float()
    if p.dim() == 3:
        p = p.mean(dim=0)          # [seq, seq]
    if p.dim() == 1:
        p = p.unsqueeze(0)

    # Column sums = total attention received per token position
    col_sums = p.sum(dim=0)
    col_sums = col_sums / (col_sums.sum().clamp(min=_EPS))

    # Pad / truncate to max_tokens
    seq = col_sums.shape[0]
    retention = col_sums[:max_tokens].tolist()
    retention += [0.0] * max(0, max_tokens - seq)
    return retention


def _l2_norm(tensor: torch.Tensor) -> float:
    """L2 norm of an arbitrary tensor, cast to scalar float."""
    return float(tensor.detach().float().norm(p=2))


def _collapse_score(entropies: List[float], norms: List[float]) -> float:
    """
    Context collapse indicator.

    A score near 1 means the model's hidden states have become
    saturated / degenerate.  Triggered by:
      - Very low entropy (sharp peaky attention → attending to nothing new)
      - Abnormally high or low norm growth

    Score = 0.5 * (1 - mean_entropy) + 0.5 * norm_anomaly
    """
    if not entropies:
        return 0.0
    mean_H = float(np.mean(entropies))
    norm_arr = np.array(norms, dtype=np.float32) if norms else np.array([1.0])
    # Coefficient of variation: high CV → norms diverging → potential collapse
    norm_cv = float(norm_arr.std() / (norm_arr.mean() + _EPS))
    norm_anomaly = min(norm_cv / 2.0, 1.0)      # cap at 1
    score = 0.5 * (1.0 - mean_H) + 0.5 * norm_anomaly
    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Hook Engine
# ─────────────────────────────────────────────────────────────────────────────

class MemoryInspector:
    """
    Lightweight, non-invasive memory inspector for PyTorch models.

    Usage
    -----
    >>> inspector = MemoryInspector(model)
    >>> output = model(input_ids)
    >>> snapshot = inspector.latest_snapshot()

    Hooks are automatically removed on ``inspector.detach()``.
    """

    # Maximum number of snapshots kept in-memory for historical queries
    MAX_HISTORY = 512

    def __init__(self, model: nn.Module, model_type: str = "auto"):
        self.model = model
        self.model_type = model_type if model_type != "auto" else self._infer_type(model)

        # Rolling buffer: previous hidden states keyed by module name
        self._prev_states: Dict[str, torch.Tensor] = {}

        # Captured buffers filled by hooks during a forward pass
        self._hidden_states: Dict[str, torch.Tensor] = {}
        self._attention_maps: Dict[str, torch.Tensor] = {}

        # Step counter and history
        self._step = 0
        self._history: deque[StepSnapshot] = deque(maxlen=self.MAX_HISTORY)
        self._lock = threading.Lock()

        # Registered hook handles (for cleanup)
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []

        self._attach_hooks()

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _infer_type(model: nn.Module) -> str:
        """Heuristically detect model family from class hierarchy."""
        name = type(model).__name__.lower()
        mro_names = " ".join(c.__name__.lower() for c in type(model).__mro__)
        if "mamba" in name or "ssm" in name or "s4" in name:
            return "mamba"
        if "rnn" in mro_names or "lstm" in mro_names or "gru" in mro_names:
            return "rnn"
        return "transformer"

    def _attach_hooks(self):
        """
        Walk the module tree and attach forward hooks to:
          - Linear / LayerNorm / Attention layers  → hidden state capture
          - Any module named with 'attn' / 'attention' → attention map capture
          - SSM-specific: 'ssm', 'selective_scan', 'mamba' → recurrent state
        """
        for name, module in self.model.named_modules():
            # Skip the root module itself
            if name == "":
                continue

            n_lower = name.lower()

            # Attention map capture (for transformers)
            if any(k in n_lower for k in ("attn", "attention", "self_attn")):
                handle = module.register_forward_hook(
                    self._make_attention_hook(name)
                )
                self._hook_handles.append(handle)

            # Hidden state capture for key layer types
            elif isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
                handle = module.register_forward_hook(
                    self._make_hidden_hook(name)
                )
                self._hook_handles.append(handle)

            # SSM / recurrent state capture
            elif any(k in n_lower for k in ("ssm", "mamba", "selective", "scan", "rnn", "lstm", "gru")):
                handle = module.register_forward_hook(
                    self._make_hidden_hook(name)
                )
                self._hook_handles.append(handle)

    def _make_hidden_hook(self, name: str) -> Callable:
        """Factory: returns a forward hook closure capturing `name`."""

        def hook(module, inputs, output):
            # output can be a tensor or a tuple/list of tensors
            if isinstance(output, torch.Tensor) and output.is_floating_point():
                with self._lock:
                    self._hidden_states[name] = output.detach()
            elif isinstance(output, (tuple, list)):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor) and o.is_floating_point():
                        with self._lock:
                            self._hidden_states[f"{name}[{i}]"] = o.detach()
                        break  # only capture the first valid tensor

        return hook

    def _make_attention_hook(self, name: str) -> Callable:
        """Factory: returns a forward hook that captures attention weights."""

        def hook(module, inputs, output):
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                # Standard HF / custom pattern: (context, attn_weights, ...)
                maybe_attn = output[1]
                if isinstance(maybe_attn, torch.Tensor) and maybe_attn.is_floating_point():
                    with self._lock:
                        self._attention_maps[name] = maybe_attn.detach()
            elif isinstance(output, torch.Tensor) and output.dim() >= 2:
                # Some implementations return only the attention matrix
                with self._lock:
                    self._attention_maps[name] = output.detach()

        return hook

    # ── Public API ─────────────────────────────────────────────────────────

    def step(self) -> StepSnapshot:
        """
        Call once per forward pass AFTER the model has run.
        Computes all memory metrics and returns a StepSnapshot.
        """
        with self._lock:
            hidden_copy = dict(self._hidden_states)
            attn_copy = dict(self._attention_maps)
            self._hidden_states.clear()
            self._attention_maps.clear()

        layer_norms: List[float] = []
        layer_drift: List[float] = []
        layer_entropy: List[float] = []
        layer_retention: List[List[float]] = []

        # ── Hidden-state metrics ──────────────────────────────────────────
        for layer_name, h in hidden_copy.items():
            norm = _l2_norm(h)
            layer_norms.append(norm)

            if layer_name in self._prev_states:
                drift = _cosine_drift(h, self._prev_states[layer_name])
            else:
                drift = 0.0
            layer_drift.append(drift)

            self._prev_states[layer_name] = h

        # ── Attention metrics ─────────────────────────────────────────────
        for layer_name, attn in attn_copy.items():
            try:
                H = _attention_entropy(attn)
                ret = _token_retention(attn)
            except Exception:          # never crash during inference
                H = 0.5
                ret = [0.0] * 32
            layer_entropy.append(H)
            layer_retention.append(ret)

        # If no explicit attention maps (RNNs / SSMs), derive from hidden norms
        if not layer_entropy and layer_norms:
            norm_arr = np.array(layer_norms)
            norm_arr = norm_arr / (norm_arr.sum() + _EPS)
            H_approx = float(-np.sum(norm_arr * np.log(norm_arr + _EPS)) /
                             (math.log(len(norm_arr) + 1) + _EPS))
            layer_entropy = [float(np.clip(H_approx, 0.0, 1.0))] * max(1, len(layer_norms))
            # Synthetic token-retention from norm distribution
            retention_row = norm_arr[:32].tolist()
            retention_row += [0.0] * max(0, 32 - len(retention_row))
            layer_retention = [retention_row]

        mean_drift = float(np.mean(layer_drift)) if layer_drift else 0.0
        collapse = _collapse_score(layer_entropy, layer_norms)

        snapshot = StepSnapshot(
            step=self._step,
            timestamp=time.time(),
            layer_norms=layer_norms,
            layer_drift=layer_drift,
            token_retention=layer_retention,
            attention_entropy=layer_entropy,
            collapse_score=collapse,
            mean_drift=mean_drift,
            model_type=self.model_type,
            seq_len=self._step + 1,
        )

        self._history.append(snapshot)
        self._step += 1
        return snapshot

    def latest_snapshot(self) -> Optional[StepSnapshot]:
        """Return the most recently computed snapshot, or None."""
        return self._history[-1] if self._history else None

    def history(self) -> List[StepSnapshot]:
        """Return the full snapshot history as a list (oldest first)."""
        return list(self._history)

    def reset(self):
        """Clear all state — useful between separate inference runs."""
        with self._lock:
            self._prev_states.clear()
            self._hidden_states.clear()
            self._attention_maps.clear()
        self._history.clear()
        self._step = 0

    def detach(self):
        """Remove all hooks from the model."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def __del__(self):
        try:
            self.detach()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"MemoryInspector("
            f"model={type(self.model).__name__}, "
            f"type={self.model_type}, "
            f"hooks={len(self._hook_handles)}, "
            f"steps={self._step})"
        )
