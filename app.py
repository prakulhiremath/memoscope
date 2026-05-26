#!/usr/bin/env python3
"""
app.py — MEMOSCOPE Demo Launcher
==================================
The quickest way to see MEMOSCOPE in action:

    python app.py                   # Transformer demo
    python app.py --model mamba     # Mamba/SSM demo
    python app.py --model rnn       # LSTM/RNN demo

This script:
  1. Instantiates a mock model (no downloads required)
  2. Attaches MemoryInspector hooks
  3. Launches the FastAPI + WebSocket server in a background thread
  4. Streams synthetic tokens through the model continuously
  5. Opens the live dashboard in your default browser

Architecture
------------

   ┌─────────────────────────────────────────────────────┐
   │  Thread A: uvicorn server (FastAPI + WebSockets)     │
   │    GET /          → SPA dashboard HTML               │
   │    GET /history   → cached snapshot replay           │
   │    WS  /ws        → live metric stream               │
   └─────────────────────────────┬───────────────────────┘
                                 │ asyncio.Queue
   ┌─────────────────────────────▼───────────────────────┐
   │  Thread B: inference loop                            │
   │    for token in data_stream:                         │
   │        model(token)                                  │
   │        snapshot = inspector.step()   ← hooks fire   │
   │        queue.put(snapshot.to_dict())                 │
   └─────────────────────────────────────────────────────┘
"""

import signal
import sys
import time

# ── Parse CLI args (mirrors memoscope/cli.py for standalone use) ──────────
import argparse

parser = argparse.ArgumentParser(
    prog="app.py",
    description="MEMOSCOPE Demo — Live Model Memory Inspector",
)
parser.add_argument(
    "--model",
    choices=["transformer", "mamba", "rnn", "ssm", "lstm"],
    default="transformer",
    help="Which mock model to demo (default: transformer)",
)
parser.add_argument("--host",  default="127.0.0.1")
parser.add_argument("--port",  type=int, default=8765)
parser.add_argument("--delay", type=float, default=0.12,
                    help="Seconds between inference steps")
parser.add_argument("--no-browser", action="store_true",
                    help="Skip auto-opening the browser")
parser.add_argument("--seq-len", type=int, default=1024,
                    help="Total tokens to stream (default: 1024)")

args = parser.parse_args()

# ── Banner ────────────────────────────────────────────────────────────────
BANNER = r"""
  __  __ _____ __  __  ___  ____   ____ ___  ____  _____
 |  \/  | ____|  \/  |/ _ \/ ___| / ___/ _ \|  _ \| ____|
 | |\/| |  _| | |\/| | | | \___ \| |  | | | | |_) |  _|
 | |  | | |___| |  | | |_| |___) | |__| |_| |  __/| |___
 |_|  |_|_____|_|  |_|\___/|____/ \____\___/|_|   |_____|

 Live Model Memory Inspector  ·  v0.1.0
 ─────────────────────────────────────────────────────────
"""

print(BANNER)
print(f"  model  : {args.model.upper()}")
print(f"  server : http://{args.host}:{args.port}")
print(f"  tokens : {args.seq_len}")
print(f"  delay  : {args.delay}s / step")
print()
print("  Starting inference stream…  Press Ctrl+C to quit.\n")

# ── Launch ────────────────────────────────────────────────────────────────
def _sigint(sig, frame):
    print("\n  MEMOSCOPE stopped cleanly.\n")
    sys.exit(0)

signal.signal(signal.SIGINT, _sigint)

from memoscope import inspect_memory
from memoscope.core.mock_models import synthetic_token_stream, get_mock_model

# Build the mock model
model = get_mock_model(args.model)

# Build a finite token stream of the requested length
vocab_size = getattr(model.embed, "num_embeddings", 512)
data_stream = synthetic_token_stream(
    vocab_size=vocab_size,
    seq_len=args.seq_len,
    batch_size=1,
    device="cpu",
)

# Launch everything — this call is non-blocking (returns MemoryInspector)
inspector = inspect_memory(
    model=model,
    data_stream=data_stream,
    host=args.host,
    port=args.port,
    open_browser=not args.no_browser,
    mock_model_type=args.model,
    stream_delay=args.delay,
)

# Keep the main thread alive so daemon threads continue
while True:
    time.sleep(1)
