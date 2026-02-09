#!/usr/bin/env python3
"""
Data Preparation Pipeline for Classification Dataset

Usage:
    python pipeline.py                          # 기본 cfg/config.yaml 사용
    python pipeline.py --config my.yaml         # 커스텀 config 사용
    python pipeline.py split.enabled=false      # 설정 오버라이드

Pipeline:
    1. Download images from AWS S3
    2. EDA (Exploratory Data Analysis)
    3. Text extraction (OCR) from images
    4. Train/Val/Test split
"""

import argparse
import logging
from pathlib import Path

from tasks import (
    ImageDownloader,
    ClassificationEDA,
    TextExtractor,
    DatasetSplitter,
    load_config,
    setup_logging,
)

logger = logging.getLogger(__name__)


def run_pipeline(cfg, script_dir: Path):
    """Run the full data preparation pipeline."""
    logger.info(f"Class: {cfg.target.class_name} ({cfg.target.class_name_kr})")
    logger.info(f"CSV: {cfg.target.csv_file}")

    if cfg.download.enabled:
        ImageDownloader(cfg, script_dir).run()
    else:
        logger.info("Step 1 skipped (download.enabled=false)")

    if cfg.eda.enabled:
        ClassificationEDA(cfg, script_dir).run()
    else:
        logger.info("Step 2 skipped (eda.enabled=false)")

    if cfg.text_extraction.get('enabled', False):
        TextExtractor(cfg, script_dir).run()
    else:
        logger.info("Step 3 skipped (text_extraction.enabled=false)")

    if cfg.split.get('enabled', True):
        DatasetSplitter(cfg, script_dir).run()
    else:
        logger.info("Step 4 skipped (split.enabled=false)")

    logger.info("Pipeline complete!")


def main():
    parser = argparse.ArgumentParser(description='Data Preparation Pipeline')
    parser.add_argument('--config', default='cfg/config.yaml', help='Config file path')
    parser.add_argument('overrides', nargs='*', help='Config overrides (key=value)')
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides or None)
    setup_logging(cfg.logging.level)
    script_dir = Path(__file__).parent.resolve()
    run_pipeline(cfg, script_dir)


if __name__ == '__main__':
    main()
