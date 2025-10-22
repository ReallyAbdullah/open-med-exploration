from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PatientIntake(BaseModel):
    symptoms: List[str] = Field(default_factory=list)
    onset_weeks: Optional[int] = None
    current_meds: List[str] = Field(default_factory=list)
    context_notes: str = ""


class ExtractionEntity(BaseModel):
    text: str
    type: str  # "disease" | "drug" | "condition" | "supplement" ...
    confidence: float
    cui: Optional[str] = None
    snomed: Optional[str] = None
    rxnorm: Optional[str] = None


class NERResult(BaseModel):
    model_name: str
    entities: List[ExtractionEntity] = Field(default_factory=list)


class ProtocolPlan(BaseModel):
    variant: str  # e.g., "gut_directed_relax"
    rationale: str
    steps: List[str]
    safety_flags: List[str] = Field(default_factory=list)


class FHIRBundle(BaseModel):
    """Minimal FHIR outputs for downstream EHR interop."""

    entries: List[Dict] = Field(default_factory=list)
