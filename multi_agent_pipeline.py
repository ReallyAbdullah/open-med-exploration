"""
multi_agent_pipeline.py
=======================

This script implements a simple **multi‑agent pipeline** for running
OpenMed models on arbitrary clinical text.  Rather than choosing a
single model manually, it first invokes `get_model_suggestions` to
determine which specialist models are relevant to the text, then runs
each suggested model in turn.  The results are aggregated and
presented to the user with entity labels and confidence scores.

Example usage::

    python3 multi_agent_pipeline.py "Patient diagnosed with acute lymphoblastic leukemia and started on imatinib."

If no text is provided on the command line, a default example is used.

Note: this script requires the `openmed` package to be installed.  It
will automatically select an available compute device (MPS, CUDA or
CPU) for inference.  Confidence scores are included in the output to
assist with downstream decision‑making.
"""

import logging
import warnings
from typing import Dict, List, Tuple

import torch

# Suppress noisy logs from downstream libraries before importing OpenMed
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

try:
    from openmed import get_model_suggestions, analyze_text
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


def run_multi_agent(
    text: str,
    *,
    group_entities: bool = True,
    include_confidence: bool = True,
    max_models: int = 5,
    device: torch.device = None,
) -> Dict[str, object]:
    """
    Run a multi‑agent inference pipeline on a piece of clinical text.

    Parameters
    ----------
    text:
        The clinical narrative to analyze.
    group_entities:
        If True, the output will group contiguous tokens belonging to the same
        entity.  See `openmed.analyze_text` for details.
    include_confidence:
        If True, confidence scores will be included in each entity.
    max_models:
        Maximum number of suggested models to run.  Some texts may trigger
        more model suggestions than are practical to execute; you can cap
        execution with this argument.
    device:
        Optional torch device on which to perform inference.  If None, the
        function will call `get_device()` to select one automatically.

    Returns
    -------
    Dict[str, object]
        A mapping from model display name to the result object returned by
        `openmed.analyze_text`.  Each result has an `.entities` attribute
        containing the extracted entities.
    """
    if not text or not text.strip():
        raise ValueError("Input text must be a non‑empty string.")

    # Determine the compute device if none is provided
    if device is None:
        device = get_device()

    # Obtain model suggestions.  Each suggestion is a tuple
    # (priority_key, model_info, reason).  We sort by the key to
    # prioritize more relevant models.
    suggestions = get_model_suggestions(text)
    suggestions = sorted(suggestions, key=lambda x: x[0])

    results: Dict[str, object] = {}
    for priority, model_info, reason in suggestions[:max_models]:
        # model_info provides metadata about the model.  Attempt to
        # resolve its internal model name; fall back to its display name.
        model_name = getattr(model_info, "model_name", None) or getattr(model_info, "name", None)
        display_name = getattr(model_info, "display_name", model_name)

        # Some model suggestions may not include an actual model name; skip
        # those entries.
        if not model_name:
            continue

        # Execute the model on the text
        try:
            result = analyze_text(
                text,
                model_name=model_name,
                group_entities=group_entities,
                include_confidence=include_confidence,
                device=device,
            )
            results[display_name] = result
        except Exception as ex:
            # Log the error and continue to the next model
            logging.error(f"Error running model {model_name}: {ex}")
            continue

    return results


def print_summary(results: Dict[str, object]) -> None:
    """Pretty‑print the entities extracted by each model."""
    if not results:
        print("No models were executed or no entities were found.")
        return

    for model_name, result in results.items():
        print(f"\n=== {model_name} ===")
        for entity in result.entities:
            try:
                conf = f" ({entity.confidence:.2f})" if hasattr(entity, "confidence") else ""
                print(f"{entity.label}: {entity.text}{conf}")
            except Exception:
                # In case the entity object does not follow the expected structure
                print(repr(entity))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a multi‑agent OpenMed pipeline on clinical text.")
    parser.add_argument(
        "text",
        nargs="*",
        help="The clinical text to analyze. If empty, a default example will be used.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=5,
        help="Maximum number of suggested models to run.",
    )
    parser.add_argument(
        "--no-grouping",
        action="store_true",
        help="Disable grouping of tokens into entities.",
    )
    parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="Disable inclusion of confidence scores.",
    )
    args = parser.parse_args()

    input_text = " ".join(args.text) if args.text else (
        "Patient presents with chest pain and shortness of breath. "
        "Past medical history includes type 2 diabetes mellitus and hypertension. "
        "Current medications include metformin 1000mg BID and lisinopril 10mg daily."
    )

    results = run_multi_agent(
        input_text,
        group_entities=not args.no_grouping,
        include_confidence=not args.no_confidence,
        max_models=args.max_models,
    )
    print_summary(results)


if __name__ == "__main__":
    main()