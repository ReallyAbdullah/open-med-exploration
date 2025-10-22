from typing import Dict, List

from .contracts import ExtractionEntity, FHIRBundle

SNOMED_MAP = {"ibs": "10743008"}
RXNORM_MAP = {"omeprazole": "763668"}


def normalize_entities(ents: List[ExtractionEntity]) -> List[ExtractionEntity]:
    for ent in ents:
        if ent.type in ("disease", "condition"):
            ent.snomed = SNOMED_MAP.get(ent.text.lower().strip())
        if ent.type == "drug":
            ent.rxnorm = RXNORM_MAP.get(ent.text.lower().strip())
    return ents


def fhir_bundle(ents: List[ExtractionEntity], summary_markdown: str) -> FHIRBundle:
    entries: List[Dict] = []
    for ent in ents:
        if ent.type in ("disease", "condition"):
            entries.append(
                {
                    "resourceType": "Condition",
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": ent.snomed or "unknown",
                                "display": ent.text,
                            }
                        ]
                    },
                }
            )
        elif ent.type == "drug":
            entries.append(
                {
                    "resourceType": "MedicationStatement",
                    "medicationCodeableConcept": {
                        "coding": [
                            {
                                "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                                "code": ent.rxnorm or "unknown",
                                "display": ent.text,
                            }
                        ]
                    },
                }
            )
    entries.append(
        {
            "resourceType": "DocumentReference",
            "content": [
                {
                    "attachment": {
                        "contentType": "text/markdown",
                        "data": summary_markdown,
                    }
                }
            ],
        }
    )
    return FHIRBundle(entries=entries)
