"""
memoscope/server/app.py
=======================
FastAPI + WebSocket server that streams real-time memory metrics to the
MEMOSCOPE dashboard.

Architecture
------------
  ┌──────────────────────────────────────┐
  │  inference thread                    │
  │   model(token) → inspector.step()   │
  │       → StepSnapshot                 │
  │           → broadcast queue         │
  └────────────────┬─────────────────────┘
                   │  asyncio.Queue (JSON)
  ┌────────────────▼─────────────────────┐
  │  WebSocket endpoint /ws              │
  │   → all connected browser clients   │
  └──────────────────────────────────────┘

HTTP endpoints
--------------
  GET /           →  SPA index.html
  GET /history    →  JSON array of last N snapshots (for page reload)
  WS  /ws         →  live metric stream
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Set

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from memoscope.core.hooks import MemoryInspector
from memoscope.core.mock_models import synthetic_token_stream

log = logging.getLogger("memoscope.server")

# ─────────────────────────────────────────────────────────────────────────────
# Connection manager (fan-out to all connected dashboards)
# ─────────────────────────────────────────────────────────────────────────────

class _ConnectionManager:
    """Thread-safe WebSocket connection pool with JSON broadcast."""

    def __init__(self):
        self.active: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self.active.add(ws)
        log.info(f"[ws] client connected  ({len(self.active)} total)")

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self.active.discard(ws)
        log.info(f"[ws] client disconnected ({len(self.active)} remaining)")

    async def broadcast(self, data: dict):
        """Send `data` as JSON to every active WebSocket."""
        if not self.active:
            return
        payload = json.dumps(data)
        dead = set()
        async with self._lock:
            targets = set(self.active)
        for ws in targets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                self.active -= dead


# ─────────────────────────────────────────────────────────────────────────────
# Shared state (module-level singletons, reset on run_server call)
# ─────────────────────────────────────────────────────────────────────────────

manager = _ConnectionManager()
_broadcast_queue: asyncio.Queue = None          # set up inside async context
_snapshot_cache: list = []
_MAX_CACHE = 256

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MEMOSCOPE",
    description="Live Model Memory Inspector",
    version="0.1.0",
)

TEMPLATES_DIR = Path(__file__).parent / "templates"


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the single-page dashboard."""
    index_path = TEMPLATES_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/history", response_class=JSONResponse)
async def get_history():
    """Return cached snapshot history for clients that connect mid-run."""
    return JSONResponse(content={"snapshots": _snapshot_cache[-128:]})


@app.get("/health")
async def health():
    return {"status": "ok", "snapshots_broadcast": len(_snapshot_cache)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Immediately send the history so new clients see existing data
        if _snapshot_cache:
            await websocket.send_text(
                json.dumps({"type": "history", "snapshots": _snapshot_cache[-64:]})
            )
        # Keep connection alive — messages come from broadcast queue
        while True:
            await asyncio.sleep(30)   # heartbeat (client sends nothing)
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket)


# ─────────────────────────────────────────────────────────────────────────────
# Background queue consumer (runs inside asyncio event loop)
# ─────────────────────────────────────────────────────────────────────────────

async def _queue_consumer():
    """Drain the broadcast queue and fan-out to all WebSocket clients."""
    global _broadcast_queue
    _broadcast_queue = asyncio.Queue(maxsize=1024)

    while True:
        snapshot_dict = await _broadcast_queue.get()
        snapshot_dict["type"] = "snapshot"
        _snapshot_cache.append(snapshot_dict)
        if len(_snapshot_cache) > _MAX_CACHE:
            _snapshot_cache.pop(0)
        await manager.broadcast(snapshot_dict)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(_queue_consumer())
    log.info("MEMOSCOPE server started.  Queue consumer active.")


# ─────────────────────────────────────────────────────────────────────────────
# Inference thread (runs in background, pumps metrics into the queue)
# ─────────────────────────────────────────────────────────────────────────────

def _inference_loop(
    inspector: MemoryInspector,
    data_stream,
    stream_delay: float,
    loop: asyncio.AbstractEventLoop,
):
    """
    Runs model inference step-by-step and pushes StepSnapshots to the
    asyncio broadcast queue.

    Designed to run in a daemon thread alongside the uvicorn event loop.
    """
    log.info(f"[inference] starting loop  model={type(inspector.model).__name__}")

    # Give the server a moment to fully bind before we start pushing data
    time.sleep(1.5)

    model = inspector.model

    with torch.no_grad():
        for token_batch in data_stream:
            t0 = time.perf_counter()

            try:
                _ = model(token_batch)
                snapshot = inspector.step()
                snapshot_dict = snapshot.to_dict()

                # Thread-safe enqueue into the asyncio loop
                asyncio.run_coroutine_threadsafe(
                    _broadcast_queue.put(snapshot_dict), loop
                )
            except Exception as exc:
                log.warning(f"[inference] step error: {exc}")

            elapsed = time.perf_counter() - t0
            wait = max(0.0, stream_delay - elapsed)
            time.sleep(wait)

    log.info("[inference] stream finished.")


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_server(
    inspector: MemoryInspector,
    data_stream=None,
    host: str = "127.0.0.1",
    port: int = 8765,
    stream_delay: float = 0.15,
):
    """
    Launch the MEMOSCOPE HTTP + WebSocket server (blocking).

    Called from a daemon thread so the caller can return immediately.

    Parameters
    ----------
    inspector : MemoryInspector
        Attached to the model whose metrics we want to stream.
    data_stream : Iterable | None
        Token generator.  Defaults to ``synthetic_token_stream()``.
    host, port : str, int
        Server binding.
    stream_delay : float
        Seconds between steps (controls dashboard animation speed).
    """
    if data_stream is None:
        model = inspector.model
        vocab_size = getattr(model, "embed", None)
        if vocab_size is not None:
            vocab_size = getattr(model.embed, "num_embeddings", 512)
        else:
            vocab_size = 512
        data_stream = synthetic_token_stream(
            vocab_size=vocab_size,
            seq_len=512,
            batch_size=1,
        )

    # We need to wire the inference thread to the event loop *after* uvicorn
    # starts.  We do this by patching a startup hook.

    _inference_started = threading.Event()

    @app.on_event("startup")
    async def _start_inference():
        loop = asyncio.get_running_loop()
        t = threading.Thread(
            target=_inference_loop,
            args=(inspector, data_stream, stream_delay, loop),
            daemon=True,
        )
        t.start()
        _inference_started.set()
        log.info("[inference] thread launched from startup event.")

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    server.run()
