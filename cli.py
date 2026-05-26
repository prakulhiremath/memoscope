"""
memoscope/cli.py
================
Command-line interface for MEMOSCOPE.

Usage
-----
  memoscope                          # default transformer demo
  memoscope --model mamba            # SSM demo
  memoscope --model rnn              # LSTM demo
  memoscope --port 9000              # custom port
  memoscope --no-browser             # headless mode
  memoscope --delay 0.05             # faster streaming
"""

import argparse
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        prog="memoscope",
        description="MEMOSCOPE — Live Model Memory Inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  memoscope                    Run the default Transformer demo
  memoscope --model mamba      Run the Mamba/SSM demo
  memoscope --model rnn        Run the LSTM/RNN demo
  memoscope --port 9000        Use port 9000
  memoscope --no-browser       Don't auto-open browser
  memoscope --delay 0.05       Stream at ~20 steps/second
        """,
    )
    parser.add_argument(
        "--model",
        choices=["transformer", "mamba", "rnn", "ssm", "lstm", "gpt"],
        default="transformer",
        help="Mock model architecture to demonstrate (default: transformer)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Server port (default: 8765)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help="Seconds between inference steps (default: 0.15)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Number of tokens to stream (default: 512)",
    )

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════╗
║  M E M O S C O P E  —  Live Model Memory Inspector  ║
║  v0.1.0                                              ║
╚══════════════════════════════════════════════════════╝

  Model type : {args.model.upper()}
  Dashboard  : http://{args.host}:{args.port}
  Seq length : {args.seq_len} tokens
  Step delay : {args.delay}s

  Press Ctrl+C to stop.
""")

    from memoscope import inspect_memory

    def _sigint(sig, frame):
        print("\n\n  MEMOSCOPE stopped.\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    inspector = inspect_memory(
        model=None,
        data_stream=None,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        mock_model_type=args.model,
        stream_delay=args.delay,
    )

    # Block main thread so daemon threads keep running
    while True:
        time.sleep(1)
