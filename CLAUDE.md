# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup Development Environment
```bash
# Clone and setup
git clone https://github.com/jianzhichun/lit-kg-sdk
cd lit-kg-sdk
pip install -e ".[dev]"
pre-commit install
```

### Testing and Quality
```bash
# Run tests
pytest tests/

# Code formatting and linting
black .
isort .
flake8
mypy litkg/

# Install with all features for development
pip install -e ".[all]"
```

### Package Installation Options
```bash
# Basic installation
pip install lit-kg-sdk

# With specific features
pip install lit-kg-sdk[pdf]        # PDF processing
pip install lit-kg-sdk[jupyter]    # Jupyter widgets
pip install lit-kg-sdk[temporal]   # Temporal tracking
pip install lit-kg-sdk[community]  # Community detection
pip install lit-kg-sdk[local-llm]  # Local LLM support
pip install lit-kg-sdk[all]        # All features
```

## Architecture Overview

### Core Components

**Session Management (`litkg/core/session.py`)**
- Central orchestration point for all operations
- Lazy-loaded components for optimal performance
- Configuration management and LLM provider selection
- Main entry point: `litkg.create_session(llm="gpt-4")`

**Knowledge Graph (`litkg/core/knowledge_graph.py`)**
- Entity and Relation data structures with confidence scores
- NetworkX-based graph representation
- Export capabilities (Neo4j, GraphML, JSON, HTML)
- Metadata tracking and processing history

**Configuration (`litkg/core/config.py`)**
- Centralized configuration with validation
- Environment variable integration
- Domain-specific settings and thresholds

### Module Structure

```
litkg/
├── core/                    # Core session, KG, and config management
├── providers/               # LLM providers and Neo4j integration
│   ├── llm_providers.py    # Multi-LLM support (OpenAI, Claude, Gemini, Ollama)
│   └── neo4j_builder.py    # Neo4j database integration
├── processing/              # PDF processing and text extraction
├── human_loop/              # Interactive validation components
│   ├── jupyter_widgets.py  # Rich Jupyter interface widgets
│   └── langgraph_workflow.py # Human-in-the-loop workflows
├── temporal/                # Time-aware knowledge graphs
└── communities/             # Community detection algorithms
```

### Key Design Patterns

**Lazy Loading**: Components are only loaded when needed, with graceful fallbacks for missing dependencies
**Optional Dependencies**: Features gracefully degrade when optional packages aren't installed
**Provider Pattern**: Pluggable LLM providers with consistent interface
**Human-in-the-Loop**: Interactive validation through Jupyter widgets and LangGraph workflows

## API Design

### Simple 4-Line API
```python
import litkg
session = litkg.create_session(llm="gpt-4")
kg = session.upload_pdf("paper.pdf")
kg.collaborate_interactively()
kg.export("knowledge_graph.neo4j")
```

### Advanced Configuration
```python
session = litkg.create_session(
    llm="claude-3.5-sonnet",
    confidence_threshold=0.8,
    neo4j_uri="bolt://localhost:7687",
    enable_communities=True,
    temporal_tracking=True,
    domain="biomedical"
)
```

## Dependencies and Integration

### LLM Providers
- **OpenAI**: GPT-4, GPT-3.5-turbo via `openai` package
- **Anthropic**: Claude models via `anthropic` package
- **Local Models**: Ollama integration via `litellm`
- **Provider Selection**: Automatic based on model string in `create_session()`

### Graph Storage
- **Neo4j**: Primary graph database integration
- **NetworkX**: In-memory graph operations and analysis
- **Export Formats**: Neo4j, GraphML, JSON, interactive HTML

### Human-in-the-Loop
- **Jupyter Widgets**: Rich interactive validation interface
- **LangGraph**: Workflow orchestration for human validation loops
- **Progressive Enhancement**: 12% precision improvement through human feedback

### Optional Features
- **PDF Processing**: LLMSherpa, PyMuPDF, pdfplumber for text extraction
- **Temporal Graphs**: Graphiti integration for time-aware knowledge graphs
- **Community Detection**: Python-louvain, scikit-learn for graph clustering
- **Visualization**: Plotly, Dash for interactive graph visualization

## Environment Configuration

### Required for Cloud LLMs
```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

### Optional for Neo4j
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## Common Development Patterns

### Adding New LLM Providers
1. Extend `providers/llm_providers.py` with new provider class
2. Implement consistent interface following existing patterns
3. Add provider detection logic in `get_llm_provider()`
4. Update optional dependencies in `pyproject.toml`

### Extending Entity/Relation Types
1. Modify `Entity` and `Relation` dataclasses in `core/knowledge_graph.py`
2. Update validation logic in configuration
3. Extend export methods for new properties

### Adding Export Formats
1. Implement export method in `KnowledgeGraph` class
2. Follow existing pattern with format detection
3. Add format-specific dependencies to optional groups