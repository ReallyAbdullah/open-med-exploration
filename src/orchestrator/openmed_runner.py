"""Utilities for routing OpenMed models with resilient fallbacks."""

from typing import Dict, Iterable, List, Tuple

from .contracts import ExtractionEntity


def _import_openmed() -> Tuple:
    """Attempt to import OpenMed helpers, returning graceful fallbacks."""

    try:
        from openmed import analyze_text, get_model_suggestions

        return analyze_text, get_model_suggestions
    except Exception:
        return None, None


default_analyze_text, default_model_suggestions = _import_openmed()

DEFAULT_TRIAGE = ["disease_detection_tiny"]
ESCALATE_ON = ["ibs", "bloating", "abdominal", "gi", "gut"]

# Speciality models surfaced in the Open Health AI overview. Keywords map common
# clinical or research cues to a higher-value model.
SPECIALTY_MODEL_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "disease_detection_superclinical": ("flare", "flare-up", "severe", "hospital"),
    "pharma_detection_superclinical": ("medication", "tablet", "capsule", "dose", "prescribed"),
    "chemical_detection_supermedical": (
        "chemical",
        "compound",
        "solvent",
        "ppm",
        "formulation",
        "additive",
    ),
    "genomic_detection_superclinical": (
        "genetic",
        "genome",
        "genomic",
        "mutation",
        "variant",
        "snv",
        "polymorphism",
    ),
    "dna_detection_superclinical": ("dna", "rna", "sequencing", "amplicon"),
    "oncology_detection_ultra": (
        "tumor",
        "cancer",
        "metastatic",
        "oncology",
        "chemotherapy",
    ),
}


def _extend_with_specialty_models(text: str, models: List[str]) -> List[str]:
    """Extend a model list with speciality options when keywords match."""

    lowered = text.lower()
    for model_name, keywords in SPECIALTY_MODEL_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            models.append(model_name)
    # Preserve order but drop duplicates.
    return list(dict.fromkeys(models))


def _normalize_suggestions(raw: Iterable) -> List[str]:
    """Normalise suggestion payloads to a flat list of model names."""

    suggestions: List[str] = []
    for item in raw:
        if isinstance(item, str):
            suggestions.append(item)
        elif isinstance(item, (list, tuple)) and item:
            # ``get_model_suggestions`` can return (model_key, metadata, rationale).
            suggestions.append(str(item[0]))
    return suggestions


def suggest_models(text: str) -> List[str]:
    if default_model_suggestions:
        try:
            suggestions = _normalize_suggestions(default_model_suggestions(text) or [])
            if suggestions:
                return _extend_with_specialty_models(text, suggestions)
        except Exception:
            pass
    models = DEFAULT_TRIAGE.copy()
    lowered = text.lower()
    if any(keyword in lowered for keyword in ESCALATE_ON):
        models.append("disease_detection_superclinical")
    if "omeprazole" in lowered or "peppermint oil" in lowered:
        models.append("pharma_detection_superclinical")
    return _extend_with_specialty_models(text, models)


def run_models(text: str, models: List[str]) -> Dict[str, List[ExtractionEntity]]:
    results: Dict[str, List[ExtractionEntity]] = {}
    if default_analyze_text:
        for model in models:
            try:
                output = default_analyze_text(text, models=[model]).get(model, [])
            except Exception:
                output = []
            results[model] = [
                ExtractionEntity(
                    text=item.get("entity", ""),
                    type=item.get("type", "unknown"),
                    confidence=item.get("confidence", 0.0),
                )
                for item in output
            ]
        return results
    synthetic = {
        "disease_detection_tiny": [
            ExtractionEntity(text="irritable bowel syndrome", type="disease", confidence=0.86)
        ],
        "disease_detection_superclinical": [
            ExtractionEntity(text="IBS", type="disease", confidence=0.92)
        ],
        "pharma_detection_superclinical": [
            ExtractionEntity(text="omeprazole", type="drug", confidence=0.88),
            ExtractionEntity(text="peppermint oil", type="supplement", confidence=0.73),
        ],
        "chemical_detection_supermedical": [
            ExtractionEntity(text="menthol", type="chemical", confidence=0.81),
            ExtractionEntity(text="linalool", type="chemical", confidence=0.77),
        ],
        "genomic_detection_superclinical": [
            ExtractionEntity(text="HLA-DQ2", type="gene", confidence=0.79)
        ],
        "dna_detection_superclinical": [
            ExtractionEntity(text="16S rRNA", type="dna", confidence=0.74)
        ],
        "oncology_detection_ultra": [
            ExtractionEntity(text="colorectal cancer", type="disease", confidence=0.90)
        ],
    }
    for model in models:
        results[model] = synthetic.get(model, [])
    return results
