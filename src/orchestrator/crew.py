from typing import List, Tuple

from .agents import (
    make_intake_agent,
    make_planner_agent,
    make_safety_agent,
    make_specialist_agent,
    make_summary_agent,
    make_triage_agent,
)
from .bandit import ThompsonBandit
from .contracts import ExtractionEntity, PatientIntake, ProtocolPlan
from .openmed_runner import run_models, suggest_models
from .tasks import make_tasks

try:
    from crewai import Crew, Process

    CREW_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when CrewAI missing
    CREW_AVAILABLE = False


def _merge_entities(raw_results: List[ExtractionEntity]) -> List[ExtractionEntity]:
    merged = {}
    for entity in raw_results:
        key = entity.text.lower()
        if key not in merged or entity.confidence > merged[key].confidence:
            merged[key] = entity
    return list(merged.values())


def _fallback_pipeline(narrative: str):
    intake = PatientIntake(
        symptoms=["bloating", "abdominal pain"] if "bloat" in narrative.lower() else ["abdominal discomfort"],
        onset_weeks=12,
        current_meds=["omeprazole"] if "omeprazole" in narrative.lower() else [],
        context_notes="anxiety present" if "anxiety" in narrative.lower() else "",
    )
    models = suggest_models(narrative)
    model_outputs = run_models(narrative, models)
    merged_entities = _merge_entities([entity for entities in model_outputs.values() for entity in entities])
    if not any(entity.type in {"disease", "condition"} for entity in merged_entities):
        merged_entities.append(ExtractionEntity(text="IBS", type="disease", confidence=0.6))
    bandit = ThompsonBandit(["calm_breathing", "gut_directed_relax", "progressive_muscle_relax"])
    variant = bandit.choose()
    plan = ProtocolPlan(
        variant=variant,
        rationale="Adaptive selection for GI + anxiety co-management.",
        steps=["5m diaphragmatic breathing", "10m gut-directed relaxation", "3m reflection"],
        safety_flags=[],
    )
    if any(med.lower() == "omeprazole" for med in intake.current_meds):
        if any(entity.text.lower() == "peppermint oil" for entity in merged_entities):
            plan.safety_flags.append(
                "Monitor reflux; avoid menthol-heavy supplements if reflux worsens."
            )
    summary_md = "# Summary\n"
    summary_md += "## Entities\n" + "\n".join(
        f"- **{entity.text}** ({entity.type}, conf={entity.confidence:.2f})" for entity in merged_entities
    )
    summary_md += "\n## Variant\n"
    summary_md += f"- `{plan.variant}`\n- {plan.rationale}\n- Steps:\n"
    summary_md += "".join(f"  - {step}\n" for step in plan.steps)
    summary_md += "## Safety\n"
    summary_md += "\n".join(f"- {flag}" for flag in plan.safety_flags) if plan.safety_flags else "- None\n"
    return intake, merged_entities, plan, summary_md


def run_hierarchical(narrative: str, manager_llm: str = "gpt-4o"):
    if not CREW_AVAILABLE:
        return _fallback_pipeline(narrative)

    intake_agent = make_intake_agent()
    triage_agent = make_triage_agent()
    specialist_agent = make_specialist_agent()
    planner_agent = make_planner_agent()
    safety_agent = make_safety_agent()
    summary_agent = make_summary_agent()
    tasks = make_tasks(
        intake_agent,
        triage_agent,
        specialist_agent,
        planner_agent,
        safety_agent,
        summary_agent,
    )
    if tasks[0] is None:
        return _fallback_pipeline(narrative)

    crew = Crew(
        agents=[
            intake_agent,
            triage_agent,
            specialist_agent,
            planner_agent,
            safety_agent,
            summary_agent,
        ],
        tasks=list(tasks),
        process=Process.hierarchical,
        manager_llm=manager_llm,
        verbose=True,
    )

    markdown_summary = crew.kickoff(inputs={"narrative": narrative})
    try:
        intake_json = tasks[0].output.json_dict or {}
        entities_json = tasks[2].output.json_dict.get("entities", [])
        plan_json = tasks[4].output.json_dict or {}
    except Exception:
        return _fallback_pipeline(narrative)

    intake = PatientIntake(
        symptoms=intake_json.get("symptoms", []),
        onset_weeks=intake_json.get("onset_weeks"),
        current_meds=intake_json.get("current_meds", []),
        context_notes=intake_json.get("context_notes", ""),
    )
    entities = [
        ExtractionEntity(
            text=entity.get("text", ""),
            type=entity.get("type", "unknown"),
            confidence=entity.get("confidence", 0.0),
        )
        for entity in entities_json
    ]
    plan = ProtocolPlan(
        variant=plan_json.get("variant", "calm_breathing"),
        rationale=plan_json.get("rationale", "N/A"),
        steps=plan_json.get("steps", ["5m breathing", "10m relaxation"]),
        safety_flags=plan_json.get("safety_flags", []),
    )
    return intake, entities, plan, str(markdown_summary)
