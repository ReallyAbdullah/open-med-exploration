"""
knowledge_graph_demo.py
=======================

This script demonstrates how to extract simple disease–drug association
statistics from a CSV file of de‑identified patient notes using the
OpenMed models.  It is a companion to `analyze_patients.py` but focuses
on building a rudimentary knowledge graph by counting co‑occurrences of
disease and pharmaceutical entities within the same patient note.

Each note is analyzed with both the disease detection and the
pharmaceutical detection models.  For each disease mention, the
script records which drugs appear in the same note.  At the end of
processing it prints a table of the most frequently co‑occurring
disease–drug pairs.

Usage::

    python3 knowledge_graph_demo.py --file patient_notes.csv --sample-size 50

Requirements:
    - pandas
    - openmed (installed via pip)
"""

import argparse
import csv
import logging
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import torch

# Suppress noisy logs from downstream libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

try:
    from openmed import analyze_text
except ImportError as e:
    raise ImportError(
        "OpenMed must be installed to run this script. Install it via `pip install openmed`."
    ) from e


def get_device() -> torch.device:
    """Return the best available torch device (MPS, CUDA or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda")
    return torch.device("cpu")


def analyze_note(
    text: str,
    device: torch.device,
) -> Tuple[List, List]:
    """
    Analyze a single note using OpenMed's disease and pharmaceutical models.

    Returns a tuple (disease_entities, pharma_entities), where each is a list
    of entity objects.  Entities include attributes `text`, `label`, `start`,
    `end` and `confidence`.
    """
    # Disease model
    disease_result = analyze_text(
        text,
        model_name="disease_detection_superclinical",
        group_entities=True,
        include_confidence=True,
        device=device,
    )
    # Pharmaceutical model
    pharma_result = analyze_text(
        text,
        model_name="pharma_detection_superclinical",
        group_entities=True,
        include_confidence=True,
        device=device,
    )
    return list(disease_result.entities), list(pharma_result.entities)


def build_association_graph(
    records: pd.DataFrame,
    *,
    sample_size: int,
    device: torch.device,
) -> Dict[str, Counter]:
    """
    Build disease–drug association counts from a subset of records.

    Parameters
    ----------
    records:
        A DataFrame containing at least a `pn_history` (text) column or a
        `description` column.
    sample_size:
        Number of records to sample for analysis.  A smaller subset can
        significantly reduce runtime when working with large datasets.
    device:
        Torch device on which to perform model inference.

    Returns
    -------
    Dict[str, Counter]
        A mapping from disease text to a Counter of drug texts and the
        number of times the pair appeared together.
    """
    # Sample the dataset
    df = records.sample(n=min(sample_size, len(records)), random_state=42)

    associations: Dict[str, Counter] = defaultdict(Counter)

    for idx, row in df.iterrows():
        text = None
        # Determine which column contains the note text
        for col in ["pn_history", "description", "text"]:
            if col in row and isinstance(row[col], str):
                text = row[col]
                break
        if not text:
            continue

        disease_entities, pharma_entities = analyze_note(text, device=device)

        # Record associations: for each disease mention, record all drugs in the note
        for d_entity in disease_entities:
            disease_name = d_entity.text
            for p_entity in pharma_entities:
                drug_name = p_entity.text
                associations[disease_name][drug_name] += 1

    return associations


def print_top_associations(associations: Dict[str, Counter], top_n: int = 5) -> None:
    """Print the top N drugs associated with each disease."""
    for disease, counter in associations.items():
        print(f"\nDisease: {disease}")
        for drug, count in counter.most_common(top_n):
            print(f"  - {drug}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple disease–drug association graph from patient notes.")
    parser.add_argument(
        "--file",
        required=True,
        help="Path to a CSV file containing de‑identified patient notes. Must have a `pn_history` or `description` column.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of notes to sample for analysis.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top associated drugs to display per disease.",
    )
    args = parser.parse_args()

    # Load the CSV.  We rely on pandas to handle quoting and delimiters.
    df = pd.read_csv(args.file)
    device = get_device()
    associations = build_association_graph(df, sample_size=args.sample_size, device=device)
    print_top_associations(associations, top_n=args.top_n)


if __name__ == "__main__":
    main()