#!/usr/bin/env python3
"""
Prepare lightweight version of extracted Celeb-DF-v3 dataset.

This script creates extracted-light with:
- 300 files per manipulation family subfolder (randomly sampled)
- Youtube-real copied as-is (no reduction)
- TalkingFace removed entirely
- Same directory structure as source

Usage:
    python prepare_light_dataset.py \
        --source /home/n-salikhova/datasets/extracted/Celeb-DF-v3 \
        --dest /home/n-salikhova/datasets/extracted-light \
        --files-per-type 300 \
        --seed 42
"""

import argparse
import shutil
import random
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LightDatasetBuilder:
    """Build lightweight dataset by sampling files per manipulation family."""

    # Skip these folders entirely
    SKIP_FOLDERS = {"TalkingFace", "talking-face"}
    
    # Copy these folders as-is (no sampling)
    NO_SAMPLE_FOLDERS = {"Youtube-real", "youtube-real"}

    def __init__(
        self,
        source_root: Path,
        dest_root: Path,
        files_per_type: int = 300,
        seed: int = 42,
        dry_run: bool = False,
    ):
        self.source_root = Path(source_root).resolve()
        self.dest_root = Path(dest_root).resolve()
        self.files_per_type = files_per_type
        self.seed = seed
        self.dry_run = dry_run
        self.rng = random.Random(seed)

        # Validation
        if not self.source_root.exists():
            raise FileNotFoundError(f"Source root not found: {self.source_root}")
        if not self.source_root.is_dir():
            raise NotADirectoryError(f"Source root is not a directory: {self.source_root}")

    def _is_skip_folder(self, folder_name: str) -> bool:
        """Check if folder should be skipped entirely."""
        return any(skip.lower() == folder_name.lower() for skip in self.SKIP_FOLDERS)

    def _is_no_sample_folder(self, folder_name: str) -> bool:
        """Check if folder should be copied as-is without sampling."""
        return any(skip.lower() == folder_name.lower() for skip in self.NO_SAMPLE_FOLDERS)

    def _get_video_files(self, folder: Path) -> List[Path]:
        """Get all video files in a folder."""
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        return [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in video_exts
        ]

    def _sample_files(self, folder: Path, n: int) -> List[Path]:
        """Randomly sample n files from folder."""
        all_files = self._get_video_files(folder)
        if len(all_files) <= n:
            logger.info(f"  Folder has {len(all_files)} files (≤ {n}), keeping all")
            return all_files
        sampled = self.rng.sample(all_files, n)
        logger.info(f"  Sampled {len(sampled)} / {len(all_files)} files")
        return sampled

    def _copy_files(self, src_files: List[Path], dest_folder: Path) -> int:
        """Copy files to destination folder."""
        dest_folder.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src_file in src_files:
            dest_file = dest_folder / src_file.name
            if not self.dry_run:
                shutil.copy2(src_file, dest_file)
            copied += 1
        return copied

    def _process_real_folder(self, real_folder: Path) -> Dict[str, int]:
        """Process real-videos folder (usually Youtube-real + Celeb-real)."""
        stats = {"folders_processed": 0, "files_copied": 0, "folders_skipped": 0}
        
        for subfolder in sorted(real_folder.iterdir()):
            if not subfolder.is_dir():
                continue

            folder_name = subfolder.name
            logger.info(f"  → {folder_name}")

            if self._is_skip_folder(folder_name):
                logger.info(f"    ✗ Skipping (in SKIP_FOLDERS)")
                stats["folders_skipped"] += 1
                continue

            if self._is_no_sample_folder(folder_name):
                logger.info(f"    ✓ Copying as-is (Youtube-real)")
                all_files = self._get_video_files(subfolder)
                dest_subfolder = self.dest_root / "real" / folder_name
                copied = self._copy_files(all_files, dest_subfolder)
                stats["files_copied"] += copied
                stats["folders_processed"] += 1
            else:
                logger.info(f"    • Sampling {self.files_per_type} files")
                sampled = self._sample_files(subfolder, self.files_per_type)
                dest_subfolder = self.dest_root / "real" / folder_name
                copied = self._copy_files(sampled, dest_subfolder)
                stats["files_copied"] += copied
                stats["folders_processed"] += 1

        return stats

    def _process_fake_folder(self, fake_folder: Path) -> Dict[str, int]:
        """Process fake (Celeb-synthesis) folder with nested structure."""
        stats = {"families_processed": 0, "types_processed": 0, "files_copied": 0, "folders_skipped": 0}
        
        for family_folder in sorted(fake_folder.iterdir()):
            if not family_folder.is_dir():
                continue

            family_name = family_folder.name
            logger.info(f"  → {family_name}")

            if self._is_skip_folder(family_name):
                logger.info(f"    ✗ Skipping (in SKIP_FOLDERS)")
                stats["folders_skipped"] += 1
                continue

            stats["families_processed"] += 1

            # Process manipulation types under each family
            for type_folder in sorted(family_folder.iterdir()):
                if not type_folder.is_dir():
                    continue

                type_name = type_folder.name
                logger.info(f"    • {type_name} - Sampling {self.files_per_type} files")

                sampled = self._sample_files(type_folder, self.files_per_type)
                dest_type_folder = self.dest_root / "fake" / family_name / type_name
                copied = self._copy_files(sampled, dest_type_folder)
                stats["files_copied"] += copied
                stats["types_processed"] += 1

        return stats

    def build(self) -> Dict[str, int]:
        """Build the light dataset."""
        logger.info("=" * 70)
        logger.info(f"SOURCE: {self.source_root}")
        logger.info(f"DEST:   {self.dest_root}")
        logger.info(f"FILES PER TYPE: {self.files_per_type}")
        logger.info(f"SEED: {self.seed}")
        logger.info(f"DRY RUN: {self.dry_run}")
        logger.info("=" * 70)

        if not self.dry_run:
            self.dest_root.mkdir(parents=True, exist_ok=True)

        total_stats = {
            "real_folders_processed": 0,
            "fake_families_processed": 0,
            "fake_types_processed": 0,
            "files_copied": 0,
            "folders_skipped": 0,
        }

        # Process real folder if it exists
        real_folder = self.source_root / "real"
        if real_folder.exists():
            logger.info("\n[PROCESSING REAL VIDEOS]")
            stats = self._process_real_folder(real_folder)
            total_stats["real_folders_processed"] = stats["folders_processed"]
            total_stats["folders_skipped"] += stats["folders_skipped"]
            total_stats["files_copied"] += stats["files_copied"]

        # Process fake folder if it exists
        fake_folder = self.source_root / "fake"
        if fake_folder.exists():
            logger.info("\n[PROCESSING FAKE VIDEOS]")
            stats = self._process_fake_folder(fake_folder)
            total_stats["fake_families_processed"] = stats["families_processed"]
            total_stats["fake_types_processed"] = stats["types_processed"]
            total_stats["folders_skipped"] += stats["folders_skipped"]
            total_stats["files_copied"] += stats["files_copied"]

        logger.info("\n" + "=" * 70)
        logger.info("BUILD COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Real folders processed:      {total_stats['real_folders_processed']}")
        logger.info(f"Fake families processed:     {total_stats['fake_families_processed']}")
        logger.info(f"Fake types processed:        {total_stats['fake_types_processed']}")
        logger.info(f"Total files copied:          {total_stats['files_copied']:,}")
        logger.info(f"Folders skipped:             {total_stats['folders_skipped']}")
        logger.info("=" * 70)

        return total_stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare lightweight version of extracted Celeb-DF-v3 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview what will be done)
  python prepare_light_dataset.py \\
    --source /home/user/datasets/extracted/Celeb-DF-v3 \\
    --dest /home/user/datasets/extracted-light \\
    --dry-run

  # Actual run with 300 files per type
  python prepare_light_dataset.py \\
    --source /home/user/datasets/extracted/Celeb-DF-v3 \\
    --dest /home/user/datasets/extracted-light \\
    --files-per-type 300 \\
    --seed 42
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source extracted folder (e.g., /home/user/datasets/extracted/Celeb-DF-v3)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="Path where extracted-light will be created",
    )
    parser.add_argument(
        "--files-per-type",
        type=int,
        default=300,
        help="Number of files to sample per manipulation type (default: 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what will be done without copying files",
    )

    args = parser.parse_args()

    try:
        builder = LightDatasetBuilder(
            source_root=args.source,
            dest_root=args.dest,
            files_per_type=args.files_per_type,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        builder.build()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
