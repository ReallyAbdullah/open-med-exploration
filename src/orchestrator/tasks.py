import json
from typing import Optional, Tuple

try:
    from crewai import Task
except Exception:  # pragma: no cover - fallback when CrewAI missing
    Task = object  # type: ignore[misc,assignment]


def guardrail_has_condition(output) -> bool:
    try:
        data = json.loads(output) if isinstance(output, str) else output
    except Exception:
        return False
    entities = data.get("entities", []) if isinstance(data, dict) else []
    return any(ent.get("type", "").lower() in {"disease", "condition"} for ent in entities)


def make_tasks(
    intake_agent,
    triage_agent,
    specialist_agent,
    planner_agent,
    safety_agent,
    summary_agent,
) -> Tuple:
    if not all(
        [
            intake_agent,
            triage_agent,
            specialist_agent,
            planner_agent,
            safety_agent,
            summary_agent,
        ]
    ):
        return (None,) * 6

    intake_task = Task(
        description=(
            "Extract symptoms[], onset_weeks, current_meds[], context_notes from the patient narrative. "
            "Return JSON with those keys."
        ),
        expected_output="JSON with keys: symptoms, onset_weeks, current_meds, context_notes.",
        agent=intake_agent,
        output_json=dict,
        markdown=False,
    )

    triage_task = Task(
        description=(
            "Given the intake JSON, select OpenMed models: start with disease_detection_tiny; "
            "if GI cues or uncertainty, add disease_detection_superclinical; if meds present add pharma_detection_superclinical. "
            "Return JSON {'models': [...]}"
        ),
        expected_output="JSON with key 'models'.",
        agent=triage_agent,
        output_json=dict,
        context=[intake_task],
        markdown=False,
    )

    specialist_task = Task(
        description=(
            "Run selected OpenMed models on the ORIGINAL narrative and merge entities "
            "as JSON: {'entities':[{'text':..., 'type':..., 'confidence':...}, ...]} (dedupe by text, keep max confidence)."
        ),
        expected_output="JSON with key 'entities'.",
        agent=specialist_agent,
        output_json=dict,
        context=[triage_task],
        markdown=False,
        guardrail=guardrail_has_condition,
        guardrail_max_retries=2,
    )

    planner_task = Task(
        description=(
            "Choose one variant among ['calm_breathing','gut_directed_relax','progressive_muscle_relax'] "
            "with rationale and 3-5 steps. Return JSON {'variant','rationale','steps':[]}"
        ),
        expected_output="JSON with keys variant, rationale, steps.",
        agent=planner_agent,
        output_json=dict,
        context=[specialist_task],
        markdown=False,
    )

    safety_task = Task(
        description=(
            "Review plan & entities for contraindications (e.g., reflux with menthol-heavy supplements). "
            "Add 'safety_flags':[]; return full plan JSON."
        ),
        expected_output="JSON with keys variant, rationale, steps, safety_flags.",
        agent=safety_agent,
        output_json=dict,
        context=[planner_task, specialist_task],
        markdown=False,
    )

    summary_task = Task(
        description=(
            "Produce Markdown note with: 1) Key entities (diseases/drugs) 2) Chosen variant + rationale + steps "
            "3) Safety flags 4) Next-session data to collect."
        ),
        expected_output="Polished Markdown suitable for a clinical note.",
        agent=summary_agent,
        context=[safety_task],
        markdown=True,
    )

    return (
        intake_task,
        triage_task,
        specialist_task,
        planner_task,
        safety_task,
        summary_task,
    )
