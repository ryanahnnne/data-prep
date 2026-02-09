"""Shared utilities for the data preparation pipeline."""

import os
import re
import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Optional

import matplotlib.pyplot as plt
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


def load_config(config_path: str = None, overrides: List[str] = None) -> DictConfig:
    """Load configuration from YAML file with optional CLI overrides."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'cfg' / 'config.yaml'
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve path: absolute stays absolute, relative is joined to base_dir."""
    p = Path(path_str)
    return p if p.is_absolute() else base_dir / p


def get_filename_from_url(url: str) -> Optional[str]:
    """Extract filename from URL path."""
    filename = os.path.basename(urlparse(url).path)
    return filename if filename and '.' in filename else None


def build_local_filename(url: str, label, fmt: str = 'jpg', creative_id=None) -> str:
    """Build local filename: {base}_{label}.{ext}."""
    original = get_filename_from_url(url) or f"{creative_id or 'unknown'}.jpg"
    base = os.path.splitext(original)[0]
    if fmt == 'original':
        ext = os.path.splitext(original)[1]
    elif fmt == 'png':
        ext = '.png'
    else:
        ext = '.jpg'
    return f"{base}_{label}{ext}"


def md_table(headers: List[str], rows: List[list]) -> str:
    """Generate a markdown table."""
    lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def save_fig(fig, path: Path, dpi: int = 150):
    """Save and close a matplotlib figure."""
    plt.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# --- Text cleaning utilities ---
_RE_WHITESPACE = re.compile(r'\s+')
_RE_EMAIL = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
_RE_URL = re.compile(r'(?:https?://|www\.)\S+')
_RE_PHONE = re.compile(r'[\d\-.\(\)+]{7,}')
_RE_DATE_YMD4 = re.compile(r'\b\d{4}[./\-]\d{1,2}[./\-]\d{1,2}\b')
_RE_DATE_YMD2 = re.compile(r'\b\d{2}[./\-]\d{1,2}[./\-]\d{1,2}\b')
_RE_DATE_MD = re.compile(r'\b\d{1,2}[./\-]\d{1,2}\b')
_RE_PERCENT_SPACE = re.compile(r'(\d)\s+%')
_RE_PERCENT_ALONE = re.compile(r'(?<!\d)%')
_RE_SPECIAL_CHARS = re.compile(r'[!+\-&@#$^*|\\<>{}\[\]~`]')
_RE_JAMO_ONLY = re.compile(r'^[ㄱ-ㅎㅏ-ㅣ]+$')
_RE_NON_WORD_ONLY = re.compile(r'^[\s\W]+$')
_RE_NON_WORD_EDGES = re.compile(r'^[\s\W]+|[\s\W]+$')


def clean_text(text: str) -> str:
    """Clean OCR text: remove noise, special chars, emails, URLs, dates, etc."""
    if not text or not text.strip():
        return ""
    text = _RE_WHITESPACE.sub(' ', text.replace('\n', ' ').replace('\r', ' ')).strip()
    for pat in [_RE_EMAIL, _RE_URL, _RE_PHONE, _RE_DATE_YMD4, _RE_DATE_YMD2, _RE_DATE_MD]:
        text = pat.sub('', text)
    text = _RE_PERCENT_SPACE.sub(r'\1%', text)
    text = _RE_PERCENT_ALONE.sub('', text)
    text = _RE_SPECIAL_CHARS.sub('', text)
    text = ' '.join(w for w in text.split() if not _RE_JAMO_ONLY.match(w))
    text = _RE_WHITESPACE.sub(' ', text).strip()
    if _RE_NON_WORD_ONLY.match(text):
        return ""
    text = _RE_NON_WORD_EDGES.sub('', text)
    return text if len(text) >= 2 else ""
