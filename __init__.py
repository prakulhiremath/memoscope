"""
╔╦╗╔═╗╔╦╗╔═╗╔═╗╔═╗╔═╗╔═╗╔═╗
║║║║╣ ║║║║ ║╚═╗║  ║ ║╠═╝║╣
╩ ╩╚═╝╩ ╩╚═╝╚═╝╚═╝╚═╝╩  ╚═╝

MEMOSCOPE — Live Model Memory Inspector
Real-time visual debugging for Transformers, SSMs, RWKV, and RNNs.

Public API
----------
>>> from memoscope import inspect_memory
>>> inspect_memory(model, data_stream)
"""

from memoscope.core.hooks import MemoryInspector
from memoscope.core.mock_models import (
    MockTransformer,
    MockMamba,
    MockRNN,
    get_mock_model,
)

__version__ = "0.1.0"
__all__ = [
    "inspect_memory",
    "MemoryInspector",
    "MockTransformer",
    "MockMamba",
    "MockRNN",
    "get_mock_model",
]


def inspect_memory(
    model=None,
    data_stream=None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    mock_model_type: str = "transformer",
    stream_delay: float = 0.15,
):
    """
    Launch MEMOSCOPE: attach memory hooks to *model*, stream *data_stream*
    through it, and open the live dashboard in your browser.

    Parameters
    ----------
    model : nn.Module | None
        A PyTorch model.  If None, a built-in mock model is used.
    data_stream : Iterable | None
        An iterable of token tensors.  If None, synthetic tokens are generated.
    host : str
        WebSocket / HTTP server host (default 127.0.0.1).
    port : int
        Port for the dashboard server (default 8765).
    open_browser : bool
        Automatically open the dashboard URL (default True).
    mock_model_type : str
        "transformer" | "mamba" | "rnn" -- which mock to use when model=None.
    stream_delay : float
        Seconds between inference steps (controls animation speed).

    Returns
    -------
    MemoryInspector
        The active inspector instance (useful for programmatic access).

    Examples
    --------
    >>> # Zero-config demo -- runs entirely on CPU with mock models
    >>> from memoscope import inspect_memory
    >>> inspect_memory()

    >>> # Attach to your own model
    >>> import torch.nn as nn
    >>> my_model = MyTransformer()
    >>> inspect_memory(my_model, data_stream=my_token_generator())
    """
    from memoscope.core.hooks import MemoryInspector
    from memoscope.core.mock_models import get_mock_model
    from memoscope.server.app import run_server
    import threading, webbrowser, time

    if model is None:
        model = get_mock_model(mock_model_type)

    inspector = MemoryInspector(model)

    server_thread = threading.Thread(
        target=run_server,
        args=(inspector, data_stream),
        kwargs={"host": host, "port": port, "stream_delay": stream_delay},
        daemon=True,
    )
    server_thread.start()

    # Give the server a moment to bind
    time.sleep(0.8)

    url = f"http://{host}:{port}"
    if open_browser:
        webbrowser.open(url)
        print(f"\n  MEMOSCOPE dashboard -> {url}\n   Press Ctrl+C to stop.\n")

    return inspector
