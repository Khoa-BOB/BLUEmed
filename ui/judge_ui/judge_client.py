import os
import re
import uuid
from typing import Dict, List, Tuple

import requests

from models import JudgeRequest, JudgeResponse, JudgeVerdict


class DummyJudgeClient:
    """
    A simple, deterministic 'judge' that:
    - Tokenizes the specialist's provisional diagnosis
    - Compares with each evidence summary
    - Computes a naive alignment score per source
    - Produces agree/uncertain/disagree + confidence
    """

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) > 2]

    def _alignment(self, diag_tokens: List[str], evidence_text: str) -> float:
        if not evidence_text.strip():
            return 0.0
        ev_tokens = set(self._tokenize(evidence_text))
        if not ev_tokens:
            return 0.0
        overlap = sum(1 for t in set(diag_tokens) if t in ev_tokens)
        denom = max(1, len(set(diag_tokens)))
        return min(1.0, overlap / denom)

    def run(self, req: JudgeRequest) -> JudgeResponse:
        diag_tokens = self._tokenize(req.specialist.provisional_diagnosis)
        alignment_scores: Dict[str, float] = {}

        for ev in req.evidence:
            score = self._alignment(diag_tokens, ev.summary)
            alignment_scores[ev.source] = round(score, 3)

        # If no evidence provided, base confidence low and uncertain
        if not alignment_scores:
            verdict = "uncertain"
            confidence = 0.3
            critique = "No worker evidence provided; unable to corroborate the specialist's diagnosis."
            key_points = ["Provide worker summaries from WebMD and Mayo Clinic to improve adjudication."]
            citations: List[str] = []
        else:
            avg_alignment = sum(alignment_scores.values()) / max(1, len(alignment_scores))
            if avg_alignment >= 0.5:
                verdict = "agree"
                confidence = min(0.9, 0.6 + avg_alignment * 0.4)
            elif avg_alignment >= 0.25:
                verdict = "uncertain"
                confidence = 0.45 + (avg_alignment - 0.25) * 0.5  # 0.45..0.55
            else:
                verdict = "disagree"
                confidence = 0.35 - avg_alignment * 0.2  # 0.35..0.3

            critique = (
                f"Average alignment between the specialist's diagnosis and worker evidence is {avg_alignment:.2f}. "
                f"Verdict: {verdict}."
            )
            key_points = [
                f"Specialist diagnosis tokens: {', '.join(set(diag_tokens)) or '(none)'}",
                "Higher token overlap with evidence increases agreement in this dummy model.",
            ]
            citations = [c for ev in req.evidence for c in ev.citations][:10]

        jv = JudgeVerdict(
            verdict=verdict, confidence=round(confidence, 2), critique=critique, key_points=key_points,
            missing_info=[], suggestions=["Consider including labs/imaging and structured findings."],
            alignment_scores=alignment_scores, citations=citations
        )
        return JudgeResponse(request_id=str(uuid.uuid4()), verdict=jv, model="dummy-judge", raw=None)


class APIJudgeClient:
    """
    HTTP API client skeleton. Implement once your judge backend is available.

    Expected backend contract (suggested):
    POST {base_url}
    Headers:
      Authorization: Bearer {api_key}
      Content-Type: application/json
    Body: JudgeRequest (as JSON)
    Response: JudgeResponse (as JSON)
    """

    def __init__(self, base_url: str, api_key: str | None = None, timeout: int = 60):
        self.base_url = base_url
        self.api_key = api_key or os.getenv("JUDGE_API_KEY", "")
        self.timeout = timeout

    def run(self, req: JudgeRequest) -> JudgeResponse:
        if not self.base_url:
            raise ValueError("API base_url is required in HTTP API mode.")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(self.base_url, headers=headers, data=req.model_dump_json(), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # If your backend returns the same schema, this will parse cleanly:
        return JudgeResponse.model_validate(data)