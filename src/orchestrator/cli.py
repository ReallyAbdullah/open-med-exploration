import argparse

from rich import print as rprint

from .crew import run_hierarchical
from .redaction import safe_harbor_redact
from .terminology import fhir_bundle, normalize_entities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Patient narrative.")
    parser.add_argument("--manager-llm", default="gpt-4o")
    args = parser.parse_args()

    redacted = safe_harbor_redact(args.text)
    intake, entities, plan, summary_md = run_hierarchical(
        redacted, manager_llm=args.manager_llm
    )
    entities = normalize_entities(entities)
    bundle = fhir_bundle(entities, summary_md)

    rprint(
        {
            "intake": intake.model_dump(),
            "entities": [entity.model_dump() for entity in entities],
            "plan": plan.model_dump(),
        }
    )
    print("\n=== SUMMARY (Markdown) ===\n", summary_md)
    print("\n=== FHIR BUNDLE (preview) ===\n", bundle.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
