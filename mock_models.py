"""
memoscope/core/mock_models.py
==============================
Self-contained mathematical mock models for MEMOSCOPE demos.

No 10-GB downloads.  No GPU required.  Works on any laptop.

The mocks are numerically realistic:
  - MockTransformer  mimics a 4-layer causal GPT with multi-head attention
  - MockMamba        mimics an SSM with selective state-space recurrence
  - MockRNN          mimics a vanilla LSTM with forget / cell / hidden gates

All models produce outputs whose hidden-state statistics (norms, drift,
entropy) mirror real models, making the dashboard visually interesting
from the very first token.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _init_weights(module: nn.Module):
    """Xavier-normal initialisation for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight, gain=0.8)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Mock Transformer  (GPT-2 / LLaMA flavour)
# ─────────────────────────────────────────────────────────────────────────────

class _CausalSelfAttention(nn.Module):
    """
    Minimal causal multi-head self-attention.
    Returns (output, attn_weights) so MEMOSCOPE hooks capture the weights.
    """

    def __init__(self, d_model: int, num_heads: int, max_seq: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Causal mask (upper triangular -inf)
        mask = torch.triu(torch.ones(max_seq, max_seq, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # [3, B, H, T, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale   # [B, H, T, T]

        # Apply causal mask
        mask = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)   # [B, H, T, T]

        context = torch.matmul(attn, v)    # [B, H, T, d_head]
        context = context.permute(0, 2, 1, 3).reshape(B, T, C)
        output = self.out_proj(context)

        # Return BOTH output and attention weights (hook picks up attn_weights)
        return output, attn


class _TransformerBlock(nn.Module):
    """Single pre-norm transformer block."""

    def __init__(self, d_model: int, num_heads: int, max_seq: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, num_heads, max_seq)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class MockTransformer(nn.Module):
    """
    A 4-layer causal transformer that runs on CPU in milliseconds.

    Architecture
    ------------
    Embedding → 4x (LayerNorm + MHA + FFN) → LayerNorm → LM-head

    Config defaults match "nano-GPT": 256 d_model, 4 heads, 512 vocab.
    """

    MODEL_TYPE = "transformer"

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(d_model, num_heads, max_seq) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(_init_weights)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids : [batch, seq_len]
        returns   : logits [batch, seq_len, vocab_size]
        """
        B, T = input_ids.shape
        device = input_ids.device

        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Mock Mamba / SSM
# ─────────────────────────────────────────────────────────────────────────────

class _SelectiveSSMCell(nn.Module):
    """
    Simplified selective state-space cell mimicking Mamba's S6 block.

    State update:
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C * h_t

    A_bar, B_bar, C, dt are all input-dependent (selective scan).
    """

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Input projections
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.x_proj = nn.Linear(d_model, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)

        # SSM parameters (learnable log-space discretisation)
        A_log = torch.arange(1, d_state + 1, dtype=torch.float32).log()
        self.A_log = nn.Parameter(A_log.unsqueeze(0).expand(d_model, -1).clone())
        self.D = nn.Parameter(torch.ones(d_model))

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def ssm_scan(self, u: torch.Tensor) -> torch.Tensor:
        """
        Selective scan over sequence dimension.
        u: [batch, seq, d_model]
        """
        B, T, D = u.shape
        d_state = self.d_state

        x_and_z = self.in_proj(u)                        # [B, T, 2*D]
        x, z = x_and_z.chunk(2, dim=-1)                 # each [B, T, D]

        x_dbl = self.x_proj(x)                          # [B, T, dt_rank + 2*d_state]
        dt, B_ssm, C = x_dbl.split(
            [self.dt_proj.in_features, d_state, d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))               # [B, T, D]

        # Discretise A: A_bar = exp(-exp(A_log) * dt)
        A = -torch.exp(self.A_log)                       # [D, d_state]
        A_bar = torch.exp(
            dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # [B, T, D, d_state]

        # Discretise B
        B_bar = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)   # [B, T, D, d_state]

        # Selective scan (sequential recurrence — simple but correct)
        h = torch.zeros(B, D, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(T):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # [B, D]
            ys.append(y)

        y = torch.stack(ys, dim=1)                       # [B, T, D]
        y = y + self.D * x
        y = y * F.silu(z)
        return self.out_proj(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm_scan(self.norm(x))


class MockMamba(nn.Module):
    """
    A 4-layer Mamba-like SSM that runs on CPU.

    Captures recurrent hidden states h_t at each SSM cell, which
    MEMOSCOPE tracks to show state drift and memory decay.

    Architecture
    ------------
    Embedding → 4x SelectiveSSMCell → LayerNorm → LM-head
    """

    MODEL_TYPE = "mamba"

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 128,
        d_state: int = 16,
        num_layers: int = 4,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [_SelectiveSSMCell(d_model, d_state) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(_init_weights)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Mock RNN  (LSTM)
# ─────────────────────────────────────────────────────────────────────────────

class _LSTMBlock(nn.Module):
    """Single-layer LSTM wrapped in a Module for hook compatibility."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out, (h, c) = self.lstm(x, state)
        return self.norm(out), (h, c)


class MockRNN(nn.Module):
    """
    A 3-layer stacked LSTM for sequence modelling demos.

    Hidden state h_t is captured per layer, showing classic RNN memory
    decay patterns — early token information fades exponentially.

    Architecture
    ------------
    Embedding → 3x LSTM+LayerNorm → Linear LM-head
    """

    MODEL_TYPE = "rnn"

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 256,
        hidden_size: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                _LSTMBlock(
                    d_model if i == 0 else hidden_size,
                    hidden_size,
                )
                for i in range(num_layers)
            ]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.apply(_init_weights)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x, _ = layer(x)
        return self.lm_head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_MOCK_REGISTRY = {
    "transformer": MockTransformer,
    "mamba": MockMamba,
    "rnn": MockRNN,
    # aliases
    "ssm": MockMamba,
    "lstm": MockRNN,
    "gpt": MockTransformer,
    "llm": MockTransformer,
}


def get_mock_model(model_type: str = "transformer", **kwargs) -> nn.Module:
    """
    Return an initialised mock model by type name.

    Parameters
    ----------
    model_type : str
        One of "transformer", "mamba", "ssm", "rnn", "lstm", "gpt", "llm".
    **kwargs
        Forwarded to the model constructor (vocab_size, d_model, etc.).

    Returns
    -------
    nn.Module
        An eval-mode model ready for inference.

    Examples
    --------
    >>> model = get_mock_model("mamba")
    >>> model = get_mock_model("transformer", d_model=512, num_layers=8)
    """
    key = model_type.lower().strip()
    if key not in _MOCK_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'.  "
            f"Choose from: {sorted(_MOCK_REGISTRY.keys())}"
        )
    model = _MOCK_REGISTRY[key](**kwargs)
    model.eval()
    return model


def synthetic_token_stream(
    vocab_size: int = 512,
    seq_len: int = 256,
    batch_size: int = 1,
    device: str = "cpu",
):
    """
    Generator yielding batched token tensors one at a time.

    Simulates streaming auto-regressive decoding: each call yields
    the *current full context window* [batch, step+1].

    Parameters
    ----------
    vocab_size : int
        Token vocabulary size.
    seq_len : int
        Total number of tokens to generate.
    batch_size : int
        Batch dimension.
    device : str
        Torch device string.

    Yields
    ------
    torch.Tensor  shape [batch, t+1]
    """
    tokens = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    yield tokens

    for _ in range(seq_len - 1):
        next_tok = torch.randint(0, vocab_size, (batch_size, 1), device=device)
        tokens = torch.cat([tokens, next_tok], dim=1)
        yield tokens
