#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility shim for legacy run_eval.py. "
            "Use scripts/run_eval_unprocessed.py or scripts/run_eval_processed.py directly."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["unprocessed", "processed"],
        default=None,
        help=(
            "Optional explicit mode for delegation. If omitted, mode is inferred from "
            "--target_transform/--model_set flags and defaults to processed."
        ),
    )
    return parser.parse_known_args(sys.argv[1:] if argv is None else argv)


def _infer_mode(args: argparse.Namespace, passthrough: list[str]) -> str:
    if args.mode:
        return str(args.mode)

    tokens = list(passthrough)
    for idx, tok in enumerate(tokens):
        if tok == "--target_transform" and idx + 1 < len(tokens):
            if str(tokens[idx + 1]).strip().lower() == "log_level":
                return "unprocessed"
        if tok.startswith("--target_transform="):
            value = tok.split("=", 1)[1].strip().lower()
            if value == "log_level":
                return "unprocessed"

        if tok == "--model_set" and idx + 1 < len(tokens):
            if str(tokens[idx + 1]).strip().lower() == "ll":
                return "unprocessed"
        if tok.startswith("--model_set="):
            value = tok.split("=", 1)[1].strip().lower()
            if value == "ll":
                return "unprocessed"

    return "processed"


def main() -> int:
    args, passthrough = parse_args()
    mode = _infer_mode(args=args, passthrough=passthrough)

    target_script = "run_eval_unprocessed.py" if mode == "unprocessed" else "run_eval_processed.py"
    script_path = (ROOT / "scripts" / target_script).resolve()

    print(
        "[DEPRECATION] scripts/run_eval.py is deprecated. "
        f"Delegating to scripts/{target_script}.",
        file=sys.stderr,
    )

    cmd = [sys.executable, str(script_path), *passthrough]
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
