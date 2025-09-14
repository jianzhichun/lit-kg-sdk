# LitKG SDK Examples

This directory contains example scripts and Jupyter notebooks demonstrating how to use the LitKG SDK.

## 🚀 Quick Setup

### 1. Environment Configuration

Copy the environment template and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required for cloud LLMs
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-claude-key-here

# Optional for local Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your-neo4j-password
```

### 2. Install Dependencies

```bash
pip install lit-kg-sdk[all]
```

## 📚 Examples

### Python Scripts

- **`basic_usage.py`** - Simple 4-line API demonstration
- **`advanced_features.py`** - Advanced configuration and batch processing
- **`demo_mode.py`** - Demo mode with sample data

### Jupyter Notebooks

- **`jupyter_notebook_example.ipynb`** - Interactive notebook with widgets

## 🔧 Usage Examples

### Basic PDF Processing

```python
import litkg

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Create session
session = litkg.create_session(llm="gpt-4")

# Process PDF
kg = session.upload_pdf("research_paper.pdf")

# Interactive validation
kg.collaborate_interactively()

# Export results
kg.export("knowledge_graph.neo4j")
```

### Local LLM Usage (No API Keys Required)

```python
import litkg

# Use local Ollama model
session = litkg.create_session(
    llm="ollama/llama3.1",
    local_processing=True
)

kg = session.upload_pdf("paper.pdf")
kg.visualize()
```

## 🔑 API Key Sources

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic (Claude)**: https://console.anthropic.com/
- **Google AI**: https://makersuite.google.com/app/apikey

## 🛠️ Local Setup (No API Keys)

If you prefer to use local models:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.1

# Use in LitKG
session = litkg.create_session(llm="ollama/llama3.1")
```

## 🔒 Security Note

- Never commit `.env` files to version control
- Keep your API keys secure and rotate them regularly
- Use environment variables in production deployments