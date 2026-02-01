"""Unified command line entrypoint for diffusionrl."""

from __future__ import annotations

import argparse
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="diffusionrl")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train", help="Run training loop")
    subparsers.add_parser("version", help="Print package version")

    args, remaining = parser.parse_known_args(argv)

    if args.command == "version":
        from diffusionrl import __version__

        print(__version__)
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    from diffusionrl.train import main as train_main

    train_main(remaining)
    return 0
