from typing import Optional

try:
    from crewai import Agent
except Exception:  # pragma: no cover - fallback for environments without CrewAI
    Agent = object  # type: ignore[misc,assignment]


def make_intake_agent() -> Optional[Agent]:
    try:
        return Agent(
            role="Clinical Intake Specialist",
            goal="Extract structured symptoms & meds and keep a supportive tone.",
            backstory="Careful, empathetic, detail-oriented.",
            allow_code_execution=False,
            verbose=True,
        )
    except Exception:
        return None


def make_triage_agent() -> Optional[Agent]:
    try:
        return Agent(
            role="Triage Model Router",
            goal="Prefer tiny models; escalate on uncertainty or GI cues.",
            backstory="Optimizes latency then quality.",
            allow_code_execution=True,
            verbose=True,
        )
    except Exception:
        return None


def make_specialist_agent() -> Optional[Agent]:
    try:
        return Agent(
            role="Clinical Extraction Specialist",
            goal="Run OpenMed models and merge entities with confidences.",
            backstory="Experienced in biomedical NER.",
            allow_code_execution=True,
            verbose=True,
        )
    except Exception:
        return None


def make_planner_agent() -> Optional[Agent]:
    try:
        return Agent(
            role="Therapeutic Planner",
            goal="Select protocol variant (15 min) with rationale and steps.",
            backstory="Knows fixed-time psychophysiological protocols.",
            allow_code_execution=False,
            verbose=True,
        )
    except Exception:
        return None


def make_safety_agent() -> Optional[Agent]:
    try:
        return Agent(
            role="Safety & Guardrails Reviewer",
            goal="Check contraindications and add safety flags.",
            backstory="Flags reflux interactions, risky advice, or prescriptive language.",
            allow_code_execution=False,
            verbose=True,
        )
    except Exception:
        return None


def make_summary_agent() -> Optional[Agent]:
    try:
        return Agent(
            role="Clinical Communication Writer",
            goal="Write concise clinician summary (Markdown).",
            backstory="Crisp, clinically accurate, patient-aware tone.",
            allow_code_execution=False,
            verbose=True,
        )
    except Exception:
        return None
