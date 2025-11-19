import os
import uuid
from typing import List

import requests

from models import DebateRequest, DebateResponse, DebateArgument, JudgeDecision


class DummyJudgeClient:
    """
    A simple, deterministic debate simulator that:
    - Simulates Expert A (Mayo Clinic) and Expert B (WebMD) providing 2 rounds of arguments
    - Simulates a judge evaluating the debate and making a final decision
    """

    def run(self, req: DebateRequest) -> DebateResponse:
        """Simulate a 2-round debate between experts and a judge decision."""
        
        # Simulate Expert A (Mayo Clinic) arguments
        expertA_round1 = DebateArgument(
            round=1,
            content=(
                "Based on Mayo Clinic guidelines, I analyze this medical note.\n\n"
                "The key clinical findings include the patient's symptoms and history. "
                "According to Mayo Clinic protocols, these findings suggest specific diagnostic considerations. "
                "The treatment approach mentioned aligns with standard medical practice.\n\n"
                "Based on my analysis, this note is CORRECT because it follows established clinical guidelines "
                "and the diagnostic and therapeutic decisions are appropriate for the presented case."
            )
        )
        
        expertA_round2 = DebateArgument(
            round=2,
            content=(
                "In response to Expert B's counter-argument:\n\n"
                "I maintain my position. Mayo Clinic literature supports this diagnosis and management plan. "
                "While Expert B raises valid points about alternative considerations, the clinical presentation "
                "and patient history strongly favor the current assessment. The treatment protocol follows "
                "evidence-based guidelines.\n\n"
                "Based on my analysis, this note is CORRECT because the clinical reasoning is sound and "
                "supported by Mayo Clinic evidence-based practices."
            )
        )
        
        # Simulate Expert B (WebMD) arguments
        expertB_round1 = DebateArgument(
            round=1,
            content=(
                "According to WebMD medical resources, I examine this clinical case.\n\n"
                "The symptoms and patient history require careful evaluation. WebMD guidelines indicate "
                "that similar presentations can have multiple differential diagnoses. The management plan "
                "described appears reasonable given the clinical context.\n\n"
                "Based on my analysis, this note is CORRECT because the diagnostic approach and treatment "
                "are consistent with standard patient care guidelines from WebMD resources."
            )
        )
        
        expertB_round2 = DebateArgument(
            round=2,
            content=(
                "Addressing Expert A's arguments:\n\n"
                "I agree with Expert A's assessment. The clinical findings and management approach are "
                "appropriate. WebMD resources corroborate the diagnostic reasoning. While there could be "
                "alternative considerations, the evidence supports the conclusions in this medical note.\n\n"
                "Based on my analysis, this note is CORRECT because it demonstrates sound clinical judgment "
                "and follows recognized patient care standards."
            )
        )
        
        # Simulate Judge decision
        judge_decision = JudgeDecision(
            final_answer="CORRECT",
            confidence_score=8,
            winner="Tie",
            reasoning=(
                "Both Expert A (Mayo Clinic) and Expert B (WebMD) provided well-reasoned arguments "
                "supporting the medical note's accuracy.\n\n"
                "Expert A demonstrated strong clinical reasoning with specific references to Mayo Clinic "
                "protocols. Expert B corroborated this assessment with WebMD guidelines.\n\n"
                "In Round 2, both experts maintained consistent positions and addressed each other's points "
                "professionally. Neither expert identified significant errors in the medical note.\n\n"
                "The quality of arguments was comparable, with both experts citing relevant medical guidelines "
                "and providing logical reasoning. This is a clear case where both experts agree on the "
                "correctness of the medical note."
            )
        )
        
        return DebateResponse(
            request_id=str(uuid.uuid4()),
            medical_note=req.medical_note,
            expertA_arguments=[expertA_round1, expertA_round2],
            expertB_arguments=[expertB_round1, expertB_round2],
            expertA_retrieved_docs=["Mayo Clinic: Clinical Guidelines (simulated)", "Mayo Clinic: Diagnostic Standards (simulated)"],
            expertB_retrieved_docs=["WebMD: Patient Care Guidelines (simulated)", "WebMD: Medical Reference (simulated)"],
            judge_decision=judge_decision,
            model_info={"expert": "dummy-expert", "judge": "dummy-judge"}
        )


class APIJudgeClient:
    """
    HTTP API client for the real debate system backend.

    Expected backend contract:
    POST {base_url}
    Headers:
      Authorization: Bearer {api_key}
      Content-Type: application/json
    Body: DebateRequest (as JSON)
    Response: DebateResponse (as JSON)
    """

    def __init__(self, base_url: str, api_key: str | None = None, timeout: int = 120):
        self.base_url = base_url
        self.api_key = api_key or os.getenv("JUDGE_API_KEY", "")
        self.timeout = timeout

    def run(self, req: DebateRequest) -> DebateResponse:
        if not self.base_url:
            raise ValueError("API base_url is required in HTTP API mode.")
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        resp = requests.post(
            self.base_url, 
            headers=headers, 
            data=req.model_dump_json(), 
            timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Parse backend response into DebateResponse
        return DebateResponse.model_validate(data)