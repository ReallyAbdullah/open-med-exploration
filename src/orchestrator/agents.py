from functools import lru_cache
import os
from typing import Optional

try:
    from crewai import Agent, LLM
except Exception:  # pragma: no cover - fallback for environments without CrewAI
    Agent = object  # type: ignore[misc,assignment]
    LLM = None  # type: ignore[assignment]


@lru_cache(maxsize=1)
def _agent_llm():
    """Return a shared LLM instance for orchestrator agents."""

    if LLM is None:
        return None

    model_name = os.getenv("ORCHESTRATOR_AGENT_LLM", "gpt-4o-mini")
    try:
        return LLM(model=model_name)
    except Exception:
        return None


def make_intake_agent() -> Optional[Agent]:
    try:
        llm = _agent_llm()
        if llm is None:
            raise RuntimeError("Agent LLM unavailable")
        return Agent(
            role="Clinical Intake Specialist",
            goal="Extract structured symptoms & meds and keep a supportive tone.",
            backstory="Careful, empathetic, detail-oriented.",
            allow_code_execution=False,
            verbose=True,
            llm=llm,
        )
    except Exception:
        return None


def make_triage_agent() -> Optional[Agent]:
    try:
        llm = _agent_llm()
        if llm is None:
            raise RuntimeError("Agent LLM unavailable")
        return Agent(
            role="Triage Model Router",
            goal="Prefer tiny models; escalate on uncertainty or GI cues.",
            backstory="Optimizes latency then quality.",
            allow_code_execution=True,
            verbose=True,
            llm=llm,
        )
    except Exception:
        return None


def make_specialist_agent() -> Optional[Agent]:
    try:
        llm = _agent_llm()
        if llm is None:
            raise RuntimeError("Agent LLM unavailable")
        return Agent(
            role="Clinical Extraction Specialist",
            goal="Run OpenMed models and merge entities with confidences.",
            backstory="Experienced in biomedical NER.",
            allow_code_execution=True,
            verbose=True,
            llm=llm,
        )
    except Exception:
        return None


def make_planner_agent() -> Optional[Agent]:
    try:
        llm = _agent_llm()
        if llm is None:
            raise RuntimeError("Agent LLM unavailable")
        return Agent(
            role="Therapeutic Planner",
            goal="Select protocol variant (15 min) with rationale and steps.",
            backstory="Knows fixed-time psychophysiological protocols.",
            allow_code_execution=False,
            verbose=True,
            llm=llm,
        )
    except Exception:
        return None


def make_safety_agent() -> Optional[Agent]:
    try:
        llm = _agent_llm()
        if llm is None:
            raise RuntimeError("Agent LLM unavailable")
        return Agent(
            role="Safety & Guardrails Reviewer",
            goal="Check contraindications and add safety flags.",
            backstory="Flags reflux interactions, risky advice, or prescriptive language.",
            allow_code_execution=False,
            verbose=True,
            llm=llm,
        )
    except Exception:
        return None


def make_summary_agent() -> Optional[Agent]:
    try:
        llm = _agent_llm()
        if llm is None:
            raise RuntimeError("Agent LLM unavailable")
        return Agent(
            role="Clinical Communication Writer",
            goal="Write concise clinician summary (Markdown).",
            backstory="Crisp, clinically accurate, patient-aware tone.",
            allow_code_execution=False,
            verbose=True,
            llm=llm,
        )
    except Exception:
        return None
