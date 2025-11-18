from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field


class PatientCase(BaseModel):
    case_id: str = Field(..., description="Unique identifier for the case")
    age: int
    sex: Literal["Male", "Female", "Other", "Unknown"]
    symptoms: List[str] = Field(default_factory=list)
    duration: Optional[str] = None
    notes: Optional[str] = None


class SpecialistAssessment(BaseModel):
    provisional_diagnosis: str
    differential: List[str] = Field(default_factory=list)
    rationale: Optional[str] = None
    icd10: Optional[str] = None


class WorkerEvidence(BaseModel):
    source: Literal["webmd", "mayoclinic", "other"] = "other"
    summary: str = ""
    citations: List[str] = Field(default_factory=list)
    quotes: List[str] = Field(default_factory=list)


class JudgeRequest(BaseModel):
    case: PatientCase
    specialist: SpecialistAssessment
    evidence: List[WorkerEvidence] = Field(default_factory=list)


class JudgeVerdict(BaseModel):
    verdict: Literal["agree", "disagree", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    critique: str
    key_points: List[str] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    alignment_scores: Dict[str, float] = Field(default_factory=dict)
    citations: List[str] = Field(default_factory=list)


class JudgeResponse(BaseModel):
    request_id: str
    verdict: JudgeVerdict
    model: str = "dummy-judge"
    raw: Optional[dict] = None