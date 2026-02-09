"""Task 1: Image Downloader - Download images from AWS S3"""

import os
import io
import time
import logging

import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
from omegaconf import DictConfig

from .utils import resolve_path, get_filename_from_url, build_local_filename

logger = logging.getLogger(__name__)


class ImageDownloader:
    def __init__(self, cfg: DictConfig, script_dir: Path):
        self.cfg = cfg
        self.csv_path = script_dir / cfg.target.csv_file
        self.output_dir = resolve_path(cfg.paths.image_dir, script_dir)
        self.max_retries = cfg.download.max_retries
        self.max_workers = cfg.download.max_workers
        self.timeout = cfg.download.timeout
        self.output_format = cfg.download.format.lower()
        self.jpg_quality = cfg.download.jpg_quality

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = pd.read_csv(self.csv_path)

        self.success_count = 0
        self.fail_count = 0
        self.skip_count = 0
        self.failed_downloads = []

    def _download_single(self, row: dict) -> dict:
        """Download a single image"""
        creative_id = row['creative_id']
        image_url = row['image_url']
        label = row.get('label', 'unknown')

        filename = build_local_filename(image_url, label, self.output_format, creative_id)
        output_path = self.output_dir / filename

        if output_path.exists():
            return {'creative_id': creative_id, 'filename': filename,
                    'status': 'skipped', 'message': 'File already exists',
                    'path': str(output_path)}

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(image_url, timeout=self.timeout, stream=True)
                response.raise_for_status()

                with Image.open(io.BytesIO(response.content)) as img:
                    if img.mode != 'RGB':
                        if img.mode == 'RGBA':
                            bg = Image.new('RGB', img.size, (255, 255, 255))
                            bg.paste(img, mask=img.split()[3])
                            img = bg
                        else:
                            img = img.convert('RGB')

                    original = get_filename_from_url(image_url) or f"{creative_id}.jpg"
                    original_ext = os.path.splitext(original)[1].lower()

                    if self.output_format == 'png':
                        img.save(output_path, 'PNG')
                    elif self.output_format == 'original' and original_ext == '.png':
                        img.save(output_path, 'PNG')
                    else:
                        img.save(output_path, 'JPEG', quality=self.jpg_quality)

                return {'creative_id': creative_id, 'filename': filename,
                        'status': 'success', 'message': f'Downloaded on attempt {attempt}',
                        'path': str(output_path), 'attempts': attempt}
            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))
                else:
                    return {'creative_id': creative_id, 'filename': filename,
                            'status': 'failed',
                            'message': f'Failed after {self.max_retries} attempts: {e}',
                            'path': None, 'attempts': attempt}

    def run(self) -> pd.DataFrame:
        """Download all images"""
        logger.info("=" * 60)
        logger.info("Task 1: Image Download")
        logger.info(f"Total: {len(self.df)} | Dir: {self.output_dir} | "
                     f"Format: {self.output_format.upper()} | Workers: {self.max_workers}")
        logger.info("=" * 60)

        results = []
        rows = self.df.to_dict('records')

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._download_single, r): r for r in rows}
            for future in tqdm(as_completed(futures), total=len(rows), desc="Downloading"):
                result = future.result()
                results.append(result)
                if result['status'] == 'success':
                    self.success_count += 1
                elif result['status'] == 'failed':
                    self.fail_count += 1
                    self.failed_downloads.append(result)
                else:
                    self.skip_count += 1

        logger.info(f"Success: {self.success_count} | Skipped: {self.skip_count} | Failed: {self.fail_count}")
        if self.failed_downloads:
            for item in self.failed_downloads[:5]:
                logger.warning(f"  FAIL {item['creative_id']}: {item['message']}")

        results_df = pd.DataFrame(results)
        results_path = self.output_dir / 'download_results.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to: {results_path}")
        return results_df
