"""
vektor.cli
----------
Command-line entry point: `vektor serve`.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="vektor")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Start the Vektor API server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--data-dir", default="./vektor_data")
    serve_parser.add_argument("--log-level", default="info")
    serve_parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        from vektor.api.app import create_app

        app = create_app(data_dir=args.data_dir)
        uvicorn.run(
            app, host=args.host, port=args.port,
            log_level=args.log_level, reload=args.reload,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()