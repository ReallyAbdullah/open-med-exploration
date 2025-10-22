import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Literal

from rich import print as rprint

from .crew import run_hierarchical
from .redaction import safe_harbor_redact
from .terminology import fhir_bundle, normalize_entities


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def save_results(
    results: Dict[str, Any],
    output_format: Literal["json", "markdown"],
    output_file: str
) -> None:
    """Save results to a file in the specified format."""
    logger = logging.getLogger(__name__)
    output_path = Path(output_file)
    
    logger.info(f"Saving results to {output_path} in {output_format} format")
    
    if output_format == "json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    elif output_format == "markdown":
        with open(output_path, "w") as f:
            f.write("# Patient Analysis Report\n\n")
            
            # Write intake section
            f.write("## Clinical Intake\n")
            for key, value in results["intake"].items():
                f.write(f"### {key.replace('_', ' ').title()}\n{value}\n\n")
            
            # Write entities section
            f.write("## Extracted Entities\n")
            for entity in results["entities"]:
                f.write(f"- {entity['text']} ({entity['type']})\n")
            f.write("\n")
            
            # Write plan section
            f.write("## Treatment Plan\n")
            for key, value in results["plan"].items():
                f.write(f"### {key.replace('_', ' ').title()}\n{value}\n\n")
            
            # Write FHIR preview
            f.write("## FHIR Bundle Preview\n```json\n")
            f.write(results["fhir_bundle"])
            f.write("\n```\n")
    
    logger.info(f"Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Patient narrative.")
    parser.add_argument("--manager-llm", default="gpt-4o", help="LLM model to use as manager (default: gpt-4o)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--output-format",
        choices=["json", "markdown"],
        help="Format for saving results (json or markdown)"
    )
    parser.add_argument(
        "--save-results",
        help="Path to save the results file"
    )
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    
    logger.info("Starting patient narrative processing")
    logger.debug(f"Input text length: {len(args.text)} characters")
    
    logger.info("Performing safe harbor redaction")
    redacted = safe_harbor_redact(args.text)
    logger.debug(f"Redacted text length: {len(redacted)} characters")
    
    logger.info(f"Running hierarchical workflow with manager LLM: {args.manager_llm}")
    intake, entities, plan, summary_md = run_hierarchical(
        redacted, manager_llm=args.manager_llm
    )
    
    logger.info("Normalizing extracted entities")
    entities = normalize_entities(entities)
    logger.debug(f"Normalized {len(entities)} entities")
    
    logger.info("Generating FHIR bundle")
    bundle = fhir_bundle(entities, summary_md)

    # Prepare results dictionary
    results = {
        "intake": intake.model_dump(),
        "entities": [entity.model_dump() for entity in entities],
        "plan": plan.model_dump(),
        "summary": summary_md,
        "fhir_bundle": bundle.model_dump_json(indent=2)
    }
    
    # Always display results in terminal
    rprint(results)
    print("\n=== SUMMARY (Markdown) ===\n", summary_md)
    print("\n=== FHIR BUNDLE (preview) ===\n", bundle.model_dump_json(indent=2))
    
    # Save results if requested
    if args.output_format and args.save_results:
        save_results(results, args.output_format, args.save_results)


if __name__ == "__main__":
    main()
