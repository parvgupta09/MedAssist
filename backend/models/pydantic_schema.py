from pydantic import BaseModel, Field
from typing import List, Optional

# -----------------------------
# Symptom Models
# -----------------------------
class SymptomDetail(BaseModel):
    name: str
    severity: Optional[str] = Field(None, description="mild/moderate/severe")
    frequency: Optional[str] = Field(None, description="common/less common/rare")
    onset_timing: Optional[str] = None
    duration: Optional[str] = None
    warning_sign: bool = False

class Symptoms(BaseModel):
    primary: List[SymptomDetail]
    secondary: List[SymptomDetail]
    distinguishing: List[SymptomDetail]
    asymptomatic_possible: bool = False
    notes: Optional[str] = None

# -----------------------------
# Population & Risk
# -----------------------------
class AffectedPopulation(BaseModel):
    age_groups: List[str]
    genders: List[str]
    high_risk_groups: List[str]

class RiskFactor(BaseModel):
    factor: str
    description: Optional[str] = None
    increases_severity: bool = False

# -----------------------------
# Transmission & Cause
# -----------------------------
class Transmission(BaseModel):
    modes: List[str]
    human_to_human: bool
    contagious: bool
    incubation_period: Optional[str] = None
    seasonality: Optional[str] = None

class Cause(BaseModel):
    pathogen_type: str
    pathogen_name: Optional[str] = None
    description: Optional[str] = None

# -----------------------------
# Diagnosis
# -----------------------------
class DiagnosisClues(BaseModel):
    key_differentiators: List[str]
    commonly_confused_with: List[str]
    red_flags: List[str]
    typical_tests: List[str]

# -----------------------------
# Complications & Prognosis
# -----------------------------
class Complication(BaseModel):
    name: str
    life_threatening: bool = False

class Prognosis(BaseModel):
    typical_recovery_time: Optional[str] = None
    mortality_risk: Optional[str] = None
    long_term_effects: Optional[List[str]] = None
    recurrence_possible: bool = False

# -----------------------------
# Prevention
# -----------------------------
class PreventionMethod(BaseModel):
    method: str
    description: Optional[str] = None

# -----------------------------
# Follow-up Question System
# -----------------------------
class FollowUpQuestion(BaseModel):
    question: str
    purpose: str = Field(..., description="What this question helps differentiate")
    related_symptoms: Optional[List[str]] = None
    expected_answer_type: str = Field(..., description="yes/no, multiple choice, free text, number")
    expected_positive_answer: Optional[str] = Field(None, description="Answer that increases probability of this disease")

class QuestionFlow(BaseModel):
    initial_questions: List[FollowUpQuestion]
    refinement_questions: List[FollowUpQuestion]
    red_flag_screening_questions: List[FollowUpQuestion]

# -----------------------------
# Main Disease Schema
# -----------------------------
class DiseaseSchema(BaseModel):
    id: str
    name: str
    category: str
    body_system: str = Field(..., description="respiratory/gastrointestinal/neurological/skin/musculoskeletal/infectious/urinary/metabolic/ENT")
    icd_code: Optional[str] = None
    overview: str
    when_to_see_doctor: str = Field(..., description="low/medium/high/emergency")
    typical_duration: str

    causes: List[Cause]
    transmission: Transmission
    symptoms: Symptoms
    affected_population: AffectedPopulation
    risk_factors: List[RiskFactor]
    diagnosis_clues: DiagnosisClues
    complications: List[Complication]
    prognosis: Prognosis
    prevention: List[PreventionMethod]
    question_flow: QuestionFlow
    differential_diagnosis: List[str]