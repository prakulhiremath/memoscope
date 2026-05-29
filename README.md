# 🔬 MEMOSCOPE — Live Model Memory Inspector

> **Real-time visual debugging for Transformers, SSMs (Mamba), RWKV, and RNNs.**
> Watch your model's memory live — drift, decay, collapse, all in one dashboard.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/fastapi-0.110%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Medium Deep Dive](https://img.shields.io/badge/medium-deep--dive-green?style=flat-square)](https://medium.com/@prakulhiremath/the-ghost-in-the-context-window-introducing-memoscope-5011be9a01c9)
[![Stars](https://img.shields.io/github/stars/prakulhiremath/memoscope?style=flat-square)](https://github.com/prakulhiremath/memoscope/stargazers)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20440915-blue?style=flat-square)](https://doi.org/10.5281/zenodo.20440915)


```
  __  __ _____ __  __  ___  ____   ____ ___  ____  _____
 |  \/  | ____|  \/  |/ _ \/ ___| / ___/ _ \|  _ \| ____|
 | |\/| |  _| | |\/| | | | \___ \| |  | | | | |_) |  _|
 | |  | | |___| |  | | |_| |___) | |__| |_| |  __/| |___
 |_|  |_|_____|_|  |_|\___/|____/ \____\___/|_|   |_____|

 Live Model Memory Inspector  ·  v0.1.0
```

---

## What is MEMOSCOPE?

When your LLM starts forgetting earlier tokens, produces incoherent long outputs, or silently collapses its context representation — **you won't see it in loss curves**.

MEMOSCOPE gives you a live window into the model's memory:

| What you see | What it means |
|---|---|
| 🌊 **Hidden State Drift** | How much the internal representation changed after each new token |
| 🧠 **Token Retention Heatmap** | Which early tokens the model is still "attending to" |
| 📉 **Memory Decay Curve** | Exponential fall-off of early-token attention mass |
| ⚡ **Context Collapse Score** | When entropy spikes — the model has run out of useful context |

---

## Quickstart (zero configuration)

```bash
# Install
pip install memoscope

# Run the demo — opens browser automatically, no GPU needed
memoscope

# Try different architectures
memoscope --model mamba       # State Space Model
memoscope --model rnn         # LSTM
```

Or from source:

```bash
git clone https://github.com/your-org/memoscope
cd memoscope
pip install -e .
python app.py
```

**One-liner API:**

```python
from memoscope import inspect_memory
inspect_memory()   # that's it — browser opens, data flows
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        YOUR MODEL                            │
│                                                              │
│  token_1 ──▶ [Layer 0] ──▶ [Layer 1] ──▶ ... ──▶ logits    │
│                   │              │                           │
│               hook_0         hook_1    ← zero-overhead       │
│                   │              │       forward hooks       │
└───────────────────┼──────────────┼───────────────────────────┘
                    │              │
                    ▼              ▼
          ┌─────────────────────────────┐
          │      MemoryInspector        │
          │                             │
          │  • cosine drift  D(t)       │
          │  • layer norms   ‖h_t‖₂    │
          │  • attention entropy  H     │
          │  • token retention          │
          │  • collapse score           │
          └──────────────┬──────────────┘
                         │  asyncio.Queue
          ┌──────────────▼──────────────┐
          │   FastAPI + WebSocket       │
          │   server (uvicorn)          │
          │                             │
          │   GET  /          → SPA     │
          │   GET  /history   → replay  │
          │   WS   /ws        → live    │
          └──────────────┬──────────────┘
                         │  JSON frames
          ┌──────────────▼──────────────┐
          │   Browser Dashboard         │
          │                             │
          │   Chart.js   live charts    │
          │   CSS grid   heatmap        │
          │   TailwindCSS  dark UI      │
          └─────────────────────────────┘
```

---

## Metrics — The Math

### 1. Hidden State Drift

How much does the model's internal representation change after each new token?

$$D(t) = 1 - \frac{h_t \cdot h_{t-1}}{\|h_t\| \cdot \|h_{t-1}\|}$$

- `D(t) = 0` → state identical to previous step (stagnation)
- `D(t) = 1` → fully orthogonal (healthy exploration)
- `D(t) = 2` → state has reversed (instability / collapse)

### 2. Attention Entropy

How "spread out" is attention across tokens?

$$H(t) = -\sum_{i} p_i \log(p_i + \varepsilon), \quad \text{normalised by} \log(T)$$

Low entropy = model attending to only a few tokens = memory narrowing.

### 3. Context Collapse Score

A composite indicator of state saturation:

$$\text{collapse}(t) = 0.5 \times (1 - \bar{H}) + 0.5 \times \text{CV}(\|h\|)$$

where CV is the coefficient of variation of layer norms.  
Score > 0.55 → warning. Score > 0.78 → critical.

### 4. Token Retention

Column sums of the attention matrix, normalised:

$$r_j = \frac{\sum_i A_{ij}}{\sum_{i,j} A_{ij}}$$

Plotted as a bar chart: early positions (j=0,1,2…) should retain mass in healthy models. Flat curves near zero = the model has forgotten everything.

---

## API Reference

### `inspect_memory()`

```python
from memoscope import inspect_memory

inspector = inspect_memory(
    model=None,           # nn.Module or None (uses mock)
    data_stream=None,     # Iterable[Tensor] or None (uses synthetic)
    host="127.0.0.1",
    port=8765,
    open_browser=True,
    mock_model_type="transformer",  # "transformer" | "mamba" | "rnn"
    stream_delay=0.15,    # seconds between steps
)
```

### `MemoryInspector`

```python
from memoscope import MemoryInspector
import torch.nn as nn

model = MyModel()
inspector = MemoryInspector(model)

# Run inference manually
output = model(input_ids)
snapshot = inspector.step()   # compute metrics for this step

print(snapshot.mean_drift)
print(snapshot.collapse_score)
print(snapshot.layer_norms)

# Detach hooks when done
inspector.detach()
```

### `StepSnapshot` fields

| Field | Type | Description |
|---|---|---|
| `step` | `int` | Global token position |
| `layer_norms` | `List[float]` | L2 norm per layer |
| `layer_drift` | `List[float]` | Cosine drift per layer |
| `token_retention` | `List[List[float]]` | Attention retention per layer |
| `attention_entropy` | `List[float]` | Shannon entropy per layer |
| `collapse_score` | `float` | Context collapse indicator [0,1] |
| `mean_drift` | `float` | Average drift across layers |
| `model_type` | `str` | "transformer" / "mamba" / "rnn" |
| `seq_len` | `int` | Current context window length |

### Mock Models

```python
from memoscope import get_mock_model, MockTransformer, MockMamba, MockRNN

# Factory
model = get_mock_model("mamba")

# Direct instantiation with custom config
model = MockTransformer(
    vocab_size=512,
    d_model=256,
    num_heads=4,
    num_layers=4,
    max_seq=128,
)

model = MockMamba(
    vocab_size=512,
    d_model=128,
    d_state=16,
    num_layers=4,
)

model = MockRNN(
    vocab_size=512,
    d_model=256,
    hidden_size=256,
    num_layers=3,
)
```

### Synthetic Token Stream

```python
from memoscope.core.mock_models import synthetic_token_stream

stream = synthetic_token_stream(
    vocab_size=512,
    seq_len=1024,
    batch_size=1,
    device="cpu",
)

for token_batch in stream:   # shape: [1, t+1]
    output = model(token_batch)
```

---

## Attach to a Real HuggingFace Model

```python
from transformers import GPT2LMHeadModel
from memoscope import MemoryInspector, inspect_memory

# Load any HF model
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# Option A: full dashboard
inspect_memory(model, mock_model_type="transformer")

# Option B: programmatic access only
inspector = MemoryInspector(model, model_type="transformer")

import torch
input_ids = torch.randint(0, 50257, (1, 32))
with torch.no_grad():
    _ = model(input_ids, output_attentions=True)

snapshot = inspector.step()
print(f"Drift: {snapshot.mean_drift:.4f}")
print(f"Collapse risk: {snapshot.collapse_score:.4f}")
```

---

## CLI Reference

```
Usage: memoscope [OPTIONS]

Options:
  --model TEXT       transformer | mamba | rnn | ssm | lstm  [default: transformer]
  --host TEXT        Server host                              [default: 127.0.0.1]
  --port INTEGER     Server port                              [default: 8765]
  --delay FLOAT      Seconds between steps                    [default: 0.15]
  --seq-len INTEGER  Tokens to stream                         [default: 512]
  --no-browser       Skip auto-opening browser
  --help             Show this message and exit.
```

---

## File Structure

```
memoscope/
├── app.py                          ← Demo launcher (run this)
├── pyproject.toml                  ← Package metadata & deps
├── README.md                       ← You are here
│
└── memoscope/
    ├── __init__.py                 ← Public API: inspect_memory()
    ├── __main__.py                 ← python -m memoscope
    ├── cli.py                      ← CLI entry point
    │
    ├── core/
    │   ├── hooks.py                ← PyTorch hook engine + metric math
    │   └── mock_models.py          ← Transformer / Mamba / RNN mocks
    │
    └── server/
        ├── app.py                  ← FastAPI + WebSocket server
        └── templates/
            └── index.html          ← SPA dashboard (Chart.js + Tailwind)
```

---

## Dashboard Panels

```
┌────────────────────────────────────────────────────────────────┐
│  MEMOSCOPE  [TRANSFORMER]              step: 247   drift: 0.183│
├──────────────┬──────────────┬──────────────────────────────────┤
│ Context Len  │ Mem Entropy  │   Collapse Score                 │
│    247       │   0.612      │   ████░░░░░░ 0.321               │
├──────────────┴──────────────┴──────────────────────────────────┤
│  Hidden State Drift  D(t) = 1 - cos(h_t · h_{t-1})            │
│  ~~~~~~~~~~~~~~~~~~~~~~/\/\~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~│
│  _____________________/    \___________________________________│
├──────────────────┬─────────────────┬───────────────────────────┤
│ Attention Entropy│  Layer Norms    │  Memory Decay             │
│ per layer (live) │  L00 ████  4.21 │  ▇▅▃▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  │
│                  │  L01 ███   3.89 │  token position →         │
│                  │  L02 ██    2.14 │                           │
│                  │  L03 █     1.02 │                           │
├──────────────────┴─────────────────┴───────────────────────────┤
│  Token Retention Heatmap (layer × token position)              │
│  ░░▒▒▓▓████▓▒░░░░░░░░░░░░░░░░░░░░░  ← step 230               │
│  ░░░░▒▒▓▓██▓▒░░░░░░░░░░░░░░░░░░░░░  ← step 231               │
├──────────────────────────────────────────────────────────────  ┤
│  [SYS] MEMOSCOPE v0.1.0 booting…                               │
│  [NET] WebSocket connected                                      │
│  [INFO] Step 200 — drift 0.183 — entropy 0.612                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Interpreting the Signals

### Hidden State Drift — What to look for

| Pattern | Interpretation |
|---|---|
| Stable low drift (~0.05–0.15) | Model in steady auto-regressive rhythm |
| Periodic spikes | Semantic boundaries (sentence ends, topic shifts) |
| Sustained high drift (>0.8) | Unstable generation / degenerate outputs |
| Drift → 0 plateau | State has frozen — model ignoring new input |

### Context Collapse — When to worry

| Score | Status | Meaning |
|---|---|---|
| 0.00–0.54 | ✅ Healthy | Normal operation |
| 0.55–0.77 | ⚠️ Warning | Context narrowing — watch entropy |
| 0.78–1.00 | 🔴 Critical | Likely generating garbage |

### Memory Decay — Architecture differences

```
Transformer (causal):       RNN/LSTM:              Mamba/SSM:
▇▅▄▃▂▁▁▁▁▁▁▁▁▁▁▁▁          ▇▄▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁      ▇▆▅▅▄▄▃▃▂▂▁▁▁▁▁▁▁
Soft decay via               Hard exponential        Selective retention —
  softmax dilution            forgetting             learned decay rate
```

---

## Performance Notes

- **No GPU required.** All mock models run on CPU in <5ms per step.
- **Zero model modification.** Hooks are read-only; they never affect gradients or outputs.
- **Memory overhead.** ~20MB for the inspector + rolling buffer of 512 snapshots.
- **Real models.** Hook overhead on GPT-2 (124M) is under 0.3ms per step.
- **Thread safety.** All state transitions are protected by `threading.Lock`.

---

## Roadmap

- [ ] HuggingFace model auto-detection (parse `config.json`)
- [ ] RWKV-specific WKV state inspector
- [ ] Export snapshots to Parquet / W&B
- [ ] Gradient-weighted attention maps (GRAD-CAM style)
- [ ] Multi-model side-by-side comparison view
- [ ] Alerting webhooks (Slack / Discord) on collapse events
- [ ] Plugin system for custom metrics

---

## Contributing

```bash
git clone https://github.com/your-org/memoscope
cd memoscope
pip install -e ".[dev]"
ruff check .
pytest tests/
```

Pull requests welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Citation

```bibtex
@software{memoscope2024,
  title  = {MEMOSCOPE: Live Model Memory Inspector},
  year   = {2024},
  url    = {https://github.com/your-org/memoscope},
}
```

---

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
Built for the curious minds who want to see <em>inside</em> the model, not just its outputs.
</p>
