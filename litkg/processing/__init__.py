"""Processing modules for LitKG SDK."""

try:
    from .pdf_processor import PDFProcessor
    __all__ = ["PDFProcessor"]
except ImportError:
    __all__ = []