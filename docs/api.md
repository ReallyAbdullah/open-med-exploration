# API Reference

## Orchestrator CLI

### Command Line Interface

```bash
python -m src.orchestrator.cli [OPTIONS]
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| --text | str | Required | Patient narrative text |
| --manager-llm | str | "gpt-4o" | LLM model for manager agent |
| --verbose | flag | False | Enable verbose logging |
| --output-format | str | None | Output format (json/markdown) |
| --save-results | str | None | Path to save results |

### Output Formats

#### JSON Structure
```json
{
  "intake": {
    "chief_complaint": str,
    "history": str,
    "medications": str,
    "allergies": str
  },
  "entities": [
    {
      "text": str,
      "type": str,
      "code": str,
      "system": str,
      "confidence": float
    }
  ],
  "plan": {
    "recommendations": str,
    "safety_considerations": str,
    "follow_up": str
  },
  "summary": str,
  "fhir_bundle": object
}
```

#### Markdown Report Sections
1. Clinical Intake
2. Extracted Entities
3. Treatment Plan
4. FHIR Bundle Preview

## Core Modules

### Agents (`agents.py`)

```python
class ManagerAgent:
    """Coordinates workflow and agent interactions."""
    
class IntakeSpecialist:
    """Handles initial patient data processing."""
    
class NERSpecialist:
    """Performs entity extraction and classification."""
```

### Tasks (`tasks.py`)

```python
class IntakeTask:
    """Initial patient data structuring."""
    
class EntityExtractionTask:
    """Named entity recognition and processing."""
    
class PlanningTask:
    """Treatment protocol generation."""
```

### Terminology (`terminology.py`)

```python
def normalize_entities(entities: List[Entity]) -> List[Entity]:
    """Map entities to standard terminologies."""
    
def fhir_bundle(entities: List[Entity], summary: str) -> FHIRBundle:
    """Generate FHIR bundle from processed data."""
```

## Integration Examples

### Basic Usage
```python
from src.orchestrator.crew import run_hierarchical

# Process text
result = run_hierarchical(
    text="Patient narrative...",
    manager_llm="gpt-4"
)

# Access results
print(result.intake)
print(result.entities)
print(result.plan)
```

### Custom Agent Configuration
```python
from src.orchestrator.agents import ManagerAgent
from src.orchestrator.tasks import CustomTask

# Create custom agent
agent = ManagerAgent(
    llm="gpt-4",
    tasks=[CustomTask()]
)

# Run workflow
result = agent.execute()
```

### FHIR Integration
```python
from src.orchestrator.terminology import fhir_bundle

# Create FHIR bundle
bundle = fhir_bundle(
    entities=extracted_entities,
    summary=summary_text
)

# Export
print(bundle.model_dump_json(indent=2))
```