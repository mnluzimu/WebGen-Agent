import os
import argparse
from pathlib import Path

import sys
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from agent import WebGenAgent


def build_parser() -> argparse.ArgumentParser:
    """Create and return an ArgumentParser for WebGenAgent CLI."""
    parser = argparse.ArgumentParser(
        description="Launch a WebGenAgent experiment."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model name or path."
    )
    parser.add_argument(
        "--vlm_model", 
        required=True,
        help="VLM Model name or path."
    )
    parser.add_argument(
        "--instruction",
        required=True,
        help="Natural‑language instruction to execute."
    )
    parser.add_argument(
        "--workspace-dir",
        required=True,
        type=Path,
        help="Directory where the agent creates / modifies files."
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        type=Path,
        help="Directory where run logs will be written."
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        metavar="N",
        help="Maximum reasoning / action iterations (default: 20)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true"
    )
    parser.add_argument(
        "--error-limit",
        type=int,
        default=5,
        metavar="N",
        help="Maximum reasoning / action iterations (default: 20)."
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    agent = WebGenAgent(
        model=args.model,
        vlm_model=args.vlm_model,
        workspace_dir=str(args.workspace_dir),
        log_dir=str(args.log_dir),
        instruction=args.instruction,
        max_iter=args.max_iter,
        overwrite=args.overwrite,
        error_limit=args.error_limit,
    )

    agent.run()


if __name__ == "__main__":
    main()
