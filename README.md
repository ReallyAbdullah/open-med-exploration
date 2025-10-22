# Open Med Exploration

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

* **`knowledge_graph_demo.py`** – Builds a disease–drug association
  graph from de‑identified patient notes.  The script reads a CSV file of
  patient histories, uses the disease and pharmaceutical detection models to
  extract entities, records co‑occurrences of diseases and medications within
  the same note and prints a summary of the most frequent associations.  This
  provides a starting point for constructing a medical knowledge graph.

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

## Building a disease–drug knowledge graph

To generate a simple knowledge graph from de‑identified patient notes, run:

```bash
python3 knowledge_graph_demo.py --file patient_notes.csv --sample-size 50
```

This reads a random subset of notes from `patient_notes.csv`, extracts
co‑occurring disease and drug entities and prints the most common
associations.  You can adjust the sample size or point the script at a
different CSV file as needed.

## License

This project is provided for educational and research purposes.  Consult the
OpenMed documentation and your local regulations before using these models on
real patient data.
