#!/usr/bin/env python3
"""
Script to remove all global_model_best_steps_*.pt checkpoint files.

These files accumulate when using the old checkpoint saving logic.
After the fix, only round-based checkpoints should be kept.

Usage:
    python scripts/cleanup_best_checkpoints.py [--root-dir ROOT_DIR] [--dry-run]
    
Examples:
    # Clean outputs directory
    python scripts/cleanup_best_checkpoints.py --root-dir ./outputs
    
    # Dry run (preview files to be deleted)
    python scripts/cleanup_best_checkpoints.py --root-dir ./outputs --dry-run
    
    # Clean specific experiment directory
    python scripts/cleanup_best_checkpoints.py --root-dir ./outputs/cinema_classification/federated/acdc_mnms_combined
"""

import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def cleanup_best_checkpoints(root_dir: Path, dry_run: bool = False) -> None:
    """Find and remove all global_model_best_steps_*.pt files.
    
    Args:
        root_dir: Root directory to search from.
        dry_run: If True, only print files that would be deleted without deleting them.
    """
    root_dir = Path(root_dir).resolve()
    
    if not root_dir.exists():
        logger.error(f"Directory does not exist: {root_dir}")
        return
    
    logger.info(f"Searching for global_model_best_steps_*.pt in {root_dir}")
    
    # Find all matching checkpoint files
    checkpoint_files = list(root_dir.glob("**/global_model_best_steps_*.pt"))
    
    if not checkpoint_files:
        logger.info("No matching checkpoint files found.")
        return
    
    logger.info(f"Found {len(checkpoint_files)} checkpoint(s) to remove:")
    
    total_size = 0
    for ckpt_file in sorted(checkpoint_files):
        file_size = ckpt_file.stat().st_size / (1024 ** 2)  # Convert to MB
        total_size += ckpt_file.stat().st_size
        logger.info(f"  - {ckpt_file.relative_to(root_dir)} ({file_size:.1f} MB)")
    
    logger.info(f"Total size: {total_size / (1024 ** 2):.1f} MB")
    
    if dry_run:
        logger.info("[DRY RUN] No files were deleted. Run without --dry-run to actually remove files.")
        return
    
    # Delete files
    deleted_count = 0
    for ckpt_file in checkpoint_files:
        try:
            ckpt_file.unlink()
            logger.info(f"Deleted: {ckpt_file.relative_to(root_dir)}")
            deleted_count += 1
        except Exception as e:
            logger.error(f"Failed to delete {ckpt_file}: {e}")
    
    logger.info(f"Successfully deleted {deleted_count}/{len(checkpoint_files)} checkpoint(s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root-dir",
        type=str,
        default="./outputs",
        help="Root directory to search from (default: ./outputs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview files to be deleted without actually removing them",
    )
    
    args = parser.parse_args()
    
    cleanup_best_checkpoints(root_dir=args.root_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
