"""Task 3: Text Extractor - Extract text from images using OCR."""

import os
import logging

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
from omegaconf import DictConfig

from .utils import resolve_path, build_local_filename, clean_text

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extract text from images using Google Vision API or GLM-OCR."""

    def __init__(self, cfg: DictConfig, script_dir: Path):
        self.cfg = cfg
        self.csv_path = script_dir / cfg.target.csv_file
        self.image_dir = resolve_path(cfg.paths.image_dir, script_dir)
        self.output_dir = resolve_path(cfg.paths.output_dir, script_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        te = cfg.text_extraction
        self.engine = te.engine
        self.save_bbox = te.save_bbox_images
        self.min_height = te.min_height_pixels
        self.min_confidence = te.min_confidence
        self.output_format = cfg.download.format.lower() if hasattr(cfg.download, 'format') else 'jpg'

        self.df = pd.read_csv(self.csv_path)
        self._build_path_map()

        self.client = None
        self.model = None
        self.processor = None

    def _build_path_map(self):
        """Build creative_id -> image_path mapping."""
        self.path_map = {}
        # Prefer download_results.csv (accurate downloaded paths)
        results_path = self.image_dir / 'download_results.csv'
        if results_path.exists():
            try:
                rdf = pd.read_csv(results_path)
                if 'path' in rdf.columns and 'creative_id' in rdf.columns:
                    for _, r in rdf[rdf['path'].notna()].iterrows():
                        self.path_map[r['creative_id']] = r['path']
            except Exception:
                pass
        # Fallback: construct expected paths
        for _, row in self.df.iterrows():
            cid = row['creative_id']
            if cid not in self.path_map:
                fname = build_local_filename(
                    row['image_url'], row.get('label', 'unknown'),
                    self.output_format, cid)
                p = self.image_dir / fname
                if p.exists():
                    self.path_map[cid] = str(p)

    def _init_engine(self):
        """Lazy-initialize the OCR engine."""
        if self.engine == 'glm':
            try:
                from transformers import AutoProcessor, AutoModelForImageTextToText
            except ImportError:
                raise ImportError("GLM-OCR requires 'transformers' package: pip install transformers")
            model_path = self.cfg.text_extraction.glm_model_path
            logger.info(f"Loading GLM-OCR model: {model_path}")
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype="auto", device_map="auto")
        else:
            try:
                from google.cloud import vision
            except ImportError:
                raise ImportError("Vision API requires 'google-cloud-vision' package: pip install google-cloud-vision")
            logger.info("Initializing Google Vision API client")
            self.client = vision.ImageAnnotatorClient()

    def _extract_vision(self, image_path: str) -> str:
        """Extract text using Google Vision API."""
        from google.cloud import vision
        try:
            with open(image_path, 'rb') as f:
                content = f.read()
            response = self.client.document_text_detection(image=vision.Image(content=content))
            if response.error.message:
                return ""
            annotation = response.full_text_annotation
            if not annotation:
                return ""

            words = []
            for page in annotation.pages:
                for block in page.blocks:
                    if block.block_type != vision.Block.BlockType.TEXT:
                        continue
                    if block.confidence < self.min_confidence:
                        continue
                    for para in block.paragraphs:
                        for word in para.words:
                            verts = word.bounding_box.vertices
                            y_coords = [v.y for v in verts]
                            if max(y_coords) - min(y_coords) < self.min_height:
                                continue
                            wt = "".join(s.text for s in word.symbols)
                            if wt:
                                words.append(wt)

            if self.save_bbox and response.text_annotations:
                self._draw_bboxes(image_path, response.text_annotations)

            return clean_text(" ".join(words))
        except Exception as e:
            logger.warning(f"Vision API error ({image_path}): {e}")
            return ""

    def _extract_glm(self, image_path: str) -> str:
        """Extract text using GLM-OCR model."""
        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": "Text Recognition:"},
            ]}]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(self.model.device)
            inputs.pop("token_type_ids", None)
            ids = self.model.generate(**inputs, max_new_tokens=8192)
            text = self.processor.decode(
                ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return clean_text(text.strip())
        except Exception as e:
            logger.warning(f"GLM-OCR error ({image_path}): {e}")
            return ""

    def _draw_bboxes(self, image_path: str, text_annotations):
        """Draw bounding boxes on image and save."""
        try:
            bbox_dir = self.output_dir / 'ocr_bbox'
            bbox_dir.mkdir(parents=True, exist_ok=True)
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            for text in text_annotations[1:]:
                points = [(v.x, v.y) for v in text.bounding_poly.vertices]
                draw.polygon(points, outline='red')
            img.save(bbox_dir / os.path.basename(image_path))
        except Exception as e:
            logger.warning(f"Bbox drawing error: {e}")

    def run(self):
        """Run text extraction on all images."""
        logger.info("=" * 60)
        logger.info(f"Task 3: Text Extraction (engine: {self.engine})")
        logger.info(f"Images mapped: {len(self.path_map)}/{len(self.df)}")
        logger.info("=" * 60)

        self._init_engine()
        extract_fn = self._extract_glm if self.engine == 'glm' else self._extract_vision

        text_results = []
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting text"):
            image_path = self.path_map.get(row['creative_id'])
            if image_path and os.path.exists(image_path):
                text_results.append(extract_fn(image_path))
            else:
                text_results.append("")

        self.df['text_info'] = text_results

        csv_stem = Path(self.cfg.target.csv_file).stem
        suffix = self.cfg.text_extraction.output_suffix
        output_path = self.output_dir / f'{csv_stem}{suffix}.csv'
        self.df.to_csv(output_path, index=False)

        extracted = sum(1 for t in text_results if t)
        logger.info(f"Extracted text from {extracted}/{len(self.df)} images")
        logger.info(f"Output saved to: {output_path}")
        logger.info("Text extraction completed!")
