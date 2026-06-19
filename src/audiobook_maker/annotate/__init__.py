"""Speaker attribution: identify characters and narrator per line via LLM."""

from .annotator import annotate_book, ScriptEntry, AnnotationConfig, save_script, load_script
from .reviewer import review_script

__all__ = ["annotate_book", "ScriptEntry", "AnnotationConfig", "save_script", "load_script", "review_script"]
