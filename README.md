# OpenMed Exploration

This repository explores the use of **OpenMed** clinical language models for healthcare
applications.  It contains scripts to inspect available models, analyze patient
descriptions, and build simple prototypes such as a multi‑agent inference pipeline
and a disease–drug knowledge graph.

## Getting started

1. Install dependencies (requires Python 3.8+).  At minimum you will need
   the [`openmed`](https://pypi.org/project/openmed/) package and its
   dependencies.  You can install it with:

   ```bash
   pip install openmed
    ```

2. Clone this repository and run the example scripts described below.  Many
   scripts will automatically detect your available hardware and run on CPU,
   GPU (CUDA) or Apple MPS when available.

## Scripts

* **`list_models.py`** – Lists all available OpenMed models and
  categories.  This script is useful for discovering model names and
  understanding what kinds of entities each model can extract.

* **`model_suggestions.py`** – Demonstrates the `get_model_suggestions` API,
  which inspects a piece of text and suggests which OpenMed models are most
  relevant.  Use this to decide which specialist models to run on your data.

* **`analyze_patients.py`** – Samples records from the public
  **Open‑Patients** dataset and runs both the disease and pharmaceutical
  detection models.  It prints entity type distributions, average confidence
  scores and common disease–drug co‑occurrences.  You can adapt this script
  to analyze other datasets such as `patient_notes.csv`.

* **`multi_agent_pipeline.py`** – Implements a simple multi‑agent
  pipeline that automatically selects and runs relevant OpenMed models on an
  arbitrary text input.  It uses the `get_model_suggestions` function to
  determine which models to call, executes each suggested model in turn and
  prints a summary of the detected entities with their confidence scores.  See
  the script’s built‑in example or provide your own text at the command line.
* **`src/orchestrator/cli.py`** – Entry point for the IBS digital‑therapeutic
  orchestrator.  It can run in a fully local fallback mode or, when CrewAI and
  an LLM provider are configured, execute the hierarchical multi‑agent flow
  described in the build plan.

* **`knowledge_graph_demo.py`** – Builds a disease–drug association
  graph from de‑identified patient notes.  The script reads a CSV file of
  patient histories, uses the disease and pharmaceutical detection models to
  extract entities, records co‑occurrences of diseases and medications within
  the same note and prints a summary of the most frequent associations.  The
  updated exporter now adds weighted network analytics, a clearer static map
  and an optional interactive HTML dashboard geared toward drug repurposing,
  pharmacovigilance and treatment‑pathway exploration.

* **`OpenMed_Research.md`** – A research note summarising the OpenMed
  model suite, architecture options, healthcare use cases and integration
  patterns.  It also lists potential SaaS applications such as clinical
  NLP‑as‑a‑service, knowledge‑graph builders and pharmacovigilance systems.

## Running the multi‑agent pipeline

You can run the multi‑agent pipeline on any piece of clinical text.  For example:

```bash
python3 multi_agent_pipeline.py "Patient diagnosed with acute lymphoblastic leukemia and started on imatinib."
```

The script will automatically determine which OpenMed models are appropriate
for the text, run them and print the detected entities grouped by model.

## IBS digital‑therapeutic orchestrator

The orchestrator wraps HIPAA Safe Harbor redaction, OpenMed model routing,
terminology normalization and FHIR bundling into a single command‑line tool.
It can execute purely with the repository’s local fallback logic, or—if you
install the optional dependencies—it will run the full CrewAI hierarchical
workflow.

1. (Optional) Create a virtual environment and install the extended
   dependencies:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install crewai openmed pydantic==2.* rich python-dotenv
   ```

2. Run the orchestrator against a patient narrative.  The command below uses
   the module path so it works from the repository root:

   ```bash
   python -m src.orchestrator.cli --text "IBS symptoms with abdominal pain and bloating; tried peppermint oil; on omeprazole; anxiety present."
   ```

   By default this uses the local fallback pipeline, which still produces
   structured intake data, merged entities, a protocol plan, a Markdown
   summary and a minimal FHIR bundle preview.  To enable CrewAI’s hierarchical
   manager, install the optional dependencies above and supply a supported
   `--manager-llm` or configure your provider credentials (for example via
   `OPENAI_API_KEY`).

### End-to-end CrewAI example

To run the full hierarchical workflow with live OpenMed models and CrewAI
agents, follow the steps below.  This assumes you have network access and
credentials for your chosen LLM provider (OpenAI is used in the example).

1. Install the extended dependencies inside a virtual environment:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install crewai openmed pydantic==2.* rich python-dotenv
   ```

2. Export the credentials required by CrewAI and OpenMed.  At minimum you need
   an API key for your LLM provider.  If you want to call authenticated OpenMed
   endpoints, export those variables too.

   ```bash
   export OPENAI_API_KEY="sk-..."
   # export OPENMED_API_KEY="..."           # only if your deployment requires it
   ```

3. Invoke the orchestrator with a realistic patient narrative and specify the
   CrewAI manager model you want to use (for example, `gpt-4o`).

   ```bash
   python -m src.orchestrator.cli \
     --manager-llm gpt-4o \
     --text "Patient reports 6 months of abdominal pain, diarrhea, and bloating. Currently taking omeprazole and peppermint oil capsules. Sleep disruption and anxiety noted."
   ```

4. Inspect the outputs:

   * The CLI prints a structured JSON-like dictionary containing the intake
     assessment, merged entities (with SNOMED/RxNorm mappings when available),
     and the selected therapeutic protocol with safety flags.
   * A Markdown summary is emitted, suitable for pasting into a clinical note.
   * A FHIR bundle preview is printed so you can verify downstream EHR payloads.

   When CrewAI is active, you will also see verbose logs that trace the
   hierarchical manager’s decisions, the agents assigned to each task, and any
   guardrail retries.  This makes it easy to validate that the multi-agent flow
   is functioning end to end.

## Building a disease–drug knowledge graph

To generate a simple knowledge graph from de‑identified patient notes, run:

```bash
python3 knowledge_graph_demo.py --file patient_notes.csv --sample-size 50 \
    --min-weight 2 --output disease_drug_network.png \
    --html-output disease_drug_network.html
```

This reads a random subset of notes from `patient_notes.csv`, extracts
co‑occurring disease and drug entities and prints the most common
associations.  The static PNG export now emphasises association strength and
node prominence, while the interactive HTML view (requires `pip install pyvis`)
reveals weighted degrees, top partners and the programme’s long‑term discovery
goals.  You can adjust the sample size, thresholds or output paths to suit your
dataset.

## License

This project is provided for educational and research purposes.  Consult the
OpenMed documentation and your local regulations before using these models on
real patient data.
