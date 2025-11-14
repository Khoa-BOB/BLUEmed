from pydantic import BaseModel

class MedRequest(BaseModel):
    patient_text: str   # free text: meds, doses, context

class MedResponse(BaseModel):
    decision: str
    expert1_analysis: str
    expert2_analysis: str
