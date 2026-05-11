"""MVP 2.0 PDF parser via Marker (VikParuchuri/marker)."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def parse_pdf(pdf_path: str | Path) -> str:
    """Parse a PDF into markdown (with $...$ formulas) using Marker.

    Marker downloads ~3GB of model weights on first call (cached under ~/.cache/huggingface).
    Pre-warm by running once on a login node before sbatch.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError as e:
        raise ImportError(
            "marker-pdf is not installed. Install MVP 2.0 deps: "
            "pip install -e .[mvp2]  (or PAPER2MANIM_MVP=2 source scripts/setup_env.sh)"
        ) from e

    log.info("[marker] parsing %s (will load models if first run)", pdf_path)
    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(pdf_path))
    text, _meta, _images = text_from_rendered(rendered)
    log.info("[marker] produced %d chars of markdown", len(text))
    return text
