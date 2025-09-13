"""
LitKG SDK: LLM-Powered Literature Knowledge Graph Construction

A Python SDK for converting PDF literature into interactive knowledge graphs
using Large Language Models with human-in-the-loop validation.

Simple 4-line API:
    import litkg
    session = litkg.create_session(llm="gpt-4")
    kg = session.upload_pdf("paper.pdf")
    kg.collaborate_interactively()
    kg.export("knowledge_graph.neo4j")
"""

__version__ = "0.1.0"
__author__ = "LitKG Team"
__email__ = "contact@litkg.ai"

from .core.session import Session, create_session
from .core.knowledge_graph import KnowledgeGraph
from .core.config import Config

# Simple API exports
__all__ = [
    "create_session",
    "Session",
    "KnowledgeGraph",
    "Config",
    "__version__"
]

# Version check
import sys
if sys.version_info < (3, 8):
    raise RuntimeError("LitKG SDK requires Python 3.8 or higher")

# Optional imports with graceful fallbacks
try:
    from .providers.neo4j_builder import Neo4jBuilder
    __all__.append("Neo4jBuilder")
except ImportError:
    pass

try:
    from .human_loop.jupyter_widgets import JupyterInterface
    __all__.append("JupyterInterface")
except ImportError:
    pass

try:
    from .temporal.graphiti import TemporalKG
    __all__.append("TemporalKG")
except ImportError:
    pass