"""FALCON end-to-end analysis and report rendering."""

from .analyze import analyze_pair
from .report import render_markdown

__all__ = ["analyze_pair", "render_markdown"]
