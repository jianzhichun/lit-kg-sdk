"""Human-in-the-loop modules for LitKG SDK."""

try:
    from .langgraph_workflow import HumanLoopWorkflow
    __all__ = ["HumanLoopWorkflow"]
except ImportError:
    __all__ = []

try:
    from .jupyter_widgets import JupyterInterface
    __all__.append("JupyterInterface")
except ImportError:
    pass