# Contributing to LitKG SDK

Thank you for your interest in contributing to LitKG SDK! This guide will help you get started with contributing to this open-source project.

## 🎯 How to Contribute

There are many ways to contribute to LitKG SDK:

- **🐛 Report bugs** - Help us identify and fix issues
- **💡 Suggest features** - Propose new functionality
- **📝 Improve documentation** - Help make our docs better
- **🔧 Submit code** - Fix bugs or implement features
- **🧪 Write tests** - Improve test coverage
- **📚 Create examples** - Help others learn to use the SDK

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic familiarity with knowledge graphs and LLMs

### Setting Up Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/lit-kg-sdk.git
   cd lit-kg-sdk
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev,all]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Verify installation**
   ```bash
   python -c "import litkg; print('✅ LitKG SDK installed successfully')"
   pytest tests/ -v
   ```

## 🏗️ Project Structure

```
lit-kg-sdk/
├── litkg/                   # Main package
│   ├── core/               # Core functionality
│   ├── providers/          # LLM and database providers
│   ├── human_loop/         # Human-in-the-loop components
│   ├── temporal/           # Temporal knowledge graphs
│   └── processing/         # PDF and text processing
├── examples/               # Usage examples
├── tests/                  # Test suite
├── docs/                   # Documentation
└── pyproject.toml          # Project configuration
```

## 🔧 Development Workflow

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and checks**
   ```bash
   # Run tests
   pytest tests/ -v

   # Run linting
   flake8 litkg/ tests/

   # Run type checking
   mypy litkg/

   # Format code
   black litkg/ tests/
   isort litkg/ tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add awesome new feature"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Coding Standards

#### Python Style Guide

- **PEP 8** compliance (enforced by flake8)
- **Type hints** for all public functions
- **Docstrings** for all public classes and methods
- **Maximum line length**: 88 characters (Black default)

#### Example Code Style

```python
"""Module docstring explaining the purpose."""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ExampleClass:
    """Class docstring explaining the purpose and usage."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the class.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def process_data(self,
                    input_data: List[str],
                    confidence_threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """Process input data and return results.

        Args:
            input_data: List of text strings to process
            confidence_threshold: Minimum confidence for results

        Returns:
            Dictionary containing processed results, or None if failed

        Raises:
            ValueError: If input_data is empty
        """
        if not input_data:
            raise ValueError("Input data cannot be empty")

        logger.info(f"Processing {len(input_data)} items")

        # Implementation here
        return {"status": "success", "count": len(input_data)}
```

#### Documentation Style

- Use **Google-style docstrings**
- Include **type hints** in function signatures
- Provide **examples** in docstrings when helpful
- Use **meaningful variable names**

### Testing Guidelines

#### Test Structure

```python
"""Test module for ExampleClass."""

import pytest
from unittest.mock import Mock, patch

from litkg.core.example import ExampleClass


class TestExampleClass:
    """Test cases for ExampleClass."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {"key": "value"}
        self.example = ExampleClass(self.config)

    def test_process_data_success(self):
        """Test successful data processing."""
        input_data = ["text1", "text2"]
        result = self.example.process_data(input_data)

        assert result is not None
        assert result["status"] == "success"
        assert result["count"] == 2

    def test_process_data_empty_input(self):
        """Test error handling for empty input."""
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            self.example.process_data([])

    @patch('litkg.core.example.external_api_call')
    def test_process_data_with_mock(self, mock_api):
        """Test with mocked external dependencies."""
        mock_api.return_value = {"response": "success"}

        result = self.example.process_data(["test"])

        assert result is not None
        mock_api.assert_called_once()
```

#### Test Categories

- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Performance tests**: Test speed and memory usage

#### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_core.py

# Run with coverage
pytest tests/ --cov=litkg

# Run only unit tests
pytest tests/unit/

# Run with verbose output
pytest tests/ -v -s
```

## 📝 Documentation

### Types of Documentation

1. **API Documentation** - Docstrings in code
2. **User Guide** - How to use the SDK
3. **Developer Guide** - How to contribute
4. **Examples** - Real-world usage scenarios

### Writing Documentation

- Use **clear, concise language**
- Include **code examples** where helpful
- Keep **up-to-date** with code changes
- Test all **code examples** to ensure they work

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear title** describing the issue
2. **Step-by-step reproduction** instructions
3. **Expected vs actual behavior**
4. **Environment details** (Python version, OS, dependencies)
5. **Error messages** and stack traces
6. **Minimal code example** that reproduces the issue

### Bug Report Template

```markdown
## Bug Description
Brief description of the bug.

## Steps to Reproduce
1. First step
2. Second step
3. Third step

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- Python version: 3.9.7
- LitKG SDK version: 0.1.0
- Operating System: macOS 12.6
- LLM Provider: OpenAI GPT-4

## Additional Context
Any other context about the problem.
```

## 💡 Feature Requests

When suggesting new features:

1. **Describe the problem** the feature would solve
2. **Explain the proposed solution**
3. **Consider alternatives** you've evaluated
4. **Provide use cases** and examples
5. **Discuss implementation** if you have ideas

### Feature Request Template

```markdown
## Problem Statement
What problem does this feature solve?

## Proposed Solution
Describe your proposed solution.

## Example Usage
```python
# Show how the feature would be used
session = litkg.create_session()
result = session.new_feature()
```

## Alternatives Considered
What other approaches did you consider?

## Additional Context
Any other context or motivation.
```

## 🔄 Pull Request Process

### Before Submitting

- [ ] **Tests pass** locally
- [ ] **Code is formatted** with Black and isort
- [ ] **Type checking passes** with mypy
- [ ] **Documentation updated** if needed
- [ ] **CHANGELOG.md updated** if applicable

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added and passing
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Discussion** and iteration if needed
4. **Approval** and merge

## 🏷️ Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(core): add temporal knowledge graph support

fix(pdf): handle corrupted PDF files gracefully

docs(api): update session creation examples

test(human-loop): add validation workflow tests
```

## 🎯 Areas for Contribution

### High Priority

- **LLM Provider Integration** - Add support for new LLM providers
- **PDF Processing** - Improve text extraction and structure preservation
- **Visualization** - Enhanced graph visualization capabilities
- **Performance** - Optimize processing speed and memory usage
- **Documentation** - Improve user guides and API documentation

### Medium Priority

- **Export Formats** - Add new knowledge graph export formats
- **Validation UI** - Enhance human-in-the-loop interfaces
- **Community Detection** - Improve clustering algorithms
- **Error Handling** - Better error messages and recovery

### Beginner Friendly

- **Examples** - Create more usage examples
- **Tests** - Increase test coverage
- **Documentation** - Fix typos and improve clarity
- **Bug Fixes** - Address issues labeled "good first issue"

## 📞 Getting Help

### Communication Channels

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - Questions and general discussion
- **Documentation** - Comprehensive guides and API reference

### Asking for Help

When asking for help:

1. **Search existing issues** first
2. **Provide context** about what you're trying to do
3. **Include relevant code** snippets
4. **Be specific** about the problem
5. **Be patient** - maintainers are volunteers

## 📄 License

By contributing to LitKG SDK, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be:

- **Listed** in CONTRIBUTORS.md
- **Mentioned** in release notes for significant contributions
- **Invited** to join the core team for outstanding contributions

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Please read and follow it.

---

**Thank you for contributing to LitKG SDK! 🎉**

Your contributions help make literature knowledge graph construction accessible to researchers worldwide.