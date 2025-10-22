# Development Guide

## Environment Setup

1. **Create Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Variables**
```bash
export OPENAI_API_KEY="your-key-here"
```

## Project Structure

```
src/
  orchestrator/    # Core orchestration logic
    agents.py      # Agent definitions
    bandit.py      # Thompson sampling
    cli.py         # Command-line interface
    contracts.py   # Data models
    crew.py        # CrewAI integration
    redaction.py   # PHI handling
    tasks.py       # Agent task definitions
    terminology.py # Code system mappings
tests/
  conftest.py     # Test configuration
  test_bandit.py  # Unit tests
  test_redaction.py
docs/             # Documentation
```

## Development Workflow

### 1. Running Tests
```bash
pytest tests/
pytest tests/test_bandit.py -v  # Specific test
```

### 2. Code Style
- Follow PEP 8
- Use type hints
- Document functions and classes

### 3. Git Workflow
```bash
git checkout -b feature/new-feature
git commit -m "feat: add new feature"
git push origin feature/new-feature
```

## CLI Usage

### Basic Usage
```bash
python -m src.orchestrator.cli \
  --text "Patient narrative..." \
  --manager-llm gpt-4
```

### Output Options
```bash
# JSON output
python -m src.orchestrator.cli \
  --text "..." \
  --output-format json \
  --save-results results.json

# Markdown report
python -m src.orchestrator.cli \
  --text "..." \
  --output-format markdown \
  --save-results report.md
```

### Debug Mode
```bash
python -m src.orchestrator.cli \
  --text "..." \
  --verbose
```

## Adding New Features

### 1. New Agent
1. Define agent in `agents.py`
2. Add tasks in `tasks.py`
3. Update `crew.py` integration
4. Add tests

### 2. New Model Integration
1. Update `bandit.py`
2. Add model config
3. Update tests

### 3. New Output Format
1. Modify `cli.py`
2. Add formatter
3. Update documentation

## Debugging

### Logging
- Use `--verbose` flag
- Check `logging` output
- Monitor agent interactions

### Common Issues
1. **Missing API Key**
   - Set OPENAI_API_KEY
   - Check environment

2. **Model Errors**
   - Verify model availability
   - Check input format

3. **Output Issues**
   - Validate FHIR format
   - Check entity normalization