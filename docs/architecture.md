# Architecture Overview

## System Architecture

### Component Overview

```
Clinical Text → Safe Harbor → Model Router → NER Pipeline → Terminology → FHIR
     ↓             ↓             ↓              ↓              ↓           ↓
   Agents      Redaction    OpenMed API    Entity Merge    Normalization  Bundle
```

### Core Components

1. **Orchestrator (`src/orchestrator/`)**
   - `cli.py`: Command-line interface
   - `crew.py`: CrewAI integration
   - `agents.py`: Specialized agent definitions
   - `tasks.py`: Task specifications
   - `contracts.py`: Data models

2. **Data Processing**
   - `redaction.py`: PHI handling
   - `terminology.py`: Code system mappings
   - `bandit.py`: Thompson sampling for model selection

### Agent Roles

#### Manager Agent
- Coordinates workflow
- Assigns tasks to specialist agents
- Ensures coherent output

#### Specialist Agents
- **Intake Specialist**: Structured data extraction
- **NER Specialist**: Entity recognition and classification
- **Protocol Planner**: Treatment planning
- **Safety Reviewer**: Contraindication checking
- **Clinical Writer**: Report generation

## Data Flow

1. **Input Processing**
   - Text sanitization
   - PHI redaction
   - Initial structuring

2. **Entity Processing**
   - NER with OpenMed models
   - Entity deduplication
   - Confidence scoring

3. **Knowledge Integration**
   - SNOMED CT mapping
   - RxNorm normalization
   - Relationship extraction

4. **Output Generation**
   - FHIR bundle construction
   - Report formatting
   - Safety validation

## Integration Points

### OpenMed API
- Model selection
- Entity extraction
- Confidence scoring

### CrewAI
- Agent orchestration
- Task management
- Workflow optimization

### FHIR
- Resource mapping
- Bundle construction
- Terminology binding

## Performance Considerations

- Parallel processing where possible
- Caching of terminology mappings
- Efficient text chunking
- Response streaming