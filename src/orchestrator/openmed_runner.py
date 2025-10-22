from typing import Dict, List, Tuple

from .contracts import ExtractionEntity


def _import_openmed() -> Tuple:
    try:
        from openmed import analyze_text, get_model_suggestions

        return analyze_text, get_model_suggestions
    except Exception:
        return None, None


default_analyze_text, default_model_suggestions = _import_openmed()

DEFAULT_TRIAGE = ["disease_detection_tiny"]
ESCALATE_ON = ["ibs", "bloating", "abdominal", "gi", "gut"]


def suggest_models(text: str) -> List[str]:
    if default_model_suggestions:
        try:
            models = default_model_suggestions(text)
            if models:
                return models
        except Exception:
            pass
    models = DEFAULT_TRIAGE.copy()
    lowered = text.lower()
    if any(keyword in lowered for keyword in ESCALATE_ON):
        models.append("disease_detection_superclinical")
    if "omeprazole" in lowered or "peppermint oil" in lowered:
        models.append("pharma_detection_superclinical")
    return list(dict.fromkeys(models))


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
    }
    for model in models:
        results[model] = synthetic.get(model, [])
    return results
