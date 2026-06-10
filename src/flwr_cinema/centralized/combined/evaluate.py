#!/usr/bin/env python3
"""Evaluate centralized model on test set with per-client breakdown.

Thin wrapper around the unified evaluation orchestrator.

Usage:
  python -m flwr_cinema.centralized.combined.evaluate --checkpoint-dir <dir>
  python -m flwr_cinema.centralized.combined.evaluate --checkpoint-dir <dir> --checkpoint-epoch 100
"""

from __future__ import annotations

from pathlib import Path

from flwr_cinema.scripts.eval_utils import (
    build_shared_argument_parser,
    run_unified_evaluation,
)


def main():
    parser = build_shared_argument_parser()
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=None,
        help="Specific epoch to evaluate (overrides best selection)",
    )
    args = parser.parse_args()

    run_unified_evaluation(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_path=args.checkpoint_path,
        checkpoint_epoch=args.checkpoint_epoch,
        output_dir=args.output_dir,
        output_file_name=args.output_file_name,
        mc_dropout=args.mc_dropout,
        mc_dropout_passes=args.mc_dropout_passes,
        is_federated=False,
    )


if __name__ == "__main__":
    main()
