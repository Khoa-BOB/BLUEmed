from typing import List, Dict, Any


def parse_list_from_textarea(text: str, sep: str = ",") -> List[str]:
    if not text:
        return []
    raw = [t.strip() for t in text.split(sep)]
    return [t for t in raw if t]


def _lines(items: List[str]) -> str:
    return "".join([f"- {it}\n" for it in items]) if items else "- (none)\n"


def generate_markdown_report(req: Dict[str, Any], resp: Dict[str, Any]) -> str:
    case = req["case"]
    spec = req["specialist"]
    verdict = resp["verdict"]

    md = []
    md.append(f"# BLUEmed Judge Report — Case {case['case_id']}\n")
    md.append("## Verdict\n")
    md.append(f"- Decision: {verdict['verdict']}\n")
    md.append(f"- Confidence: {verdict['confidence']:.2f}\n\n")

    md.append("## Patient Case\n")
    md.append(f"- Age: {case['age']}\n")
    md.append(f"- Sex: {case['sex']}\n")
    md.append(f"- Duration: {case.get('duration') or '(unspecified)'}\n")
    md.append("### Symptoms\n")
    md.append(_lines(case.get("symptoms", [])))
    md.append("\n")
    md.append("### Clinical Notes\n")
    md.append(f"{case.get('notes') or '(none)'}\n\n")

    md.append("## Specialist Assessment\n")
    md.append(f"- Provisional diagnosis: {spec['provisional_diagnosis']}\n")
    md.append(f"- ICD-10: {spec.get('icd10') or '(none)'}\n")
    md.append("### Differential diagnoses\n")
    md.append(_lines(spec.get("differential", [])))
    md.append("\n")
    md.append("### Rationale\n")
    md.append(f"{spec.get('rationale') or '(none)'}\n\n")

    md.append("## Judge Rationale\n")
    md.append(f"{verdict.get('critique') or ''}\n\n")
    if verdict.get("key_points"):
        md.append("### Key Points\n")
        md.append(_lines(verdict["key_points"]))
        md.append("\n")
    if verdict.get("alignment_scores"):
        md.append("### Alignment Scores\n")
        for src, score in verdict["alignment_scores"].items():
            md.append(f"- {src}: {score:.2f}\n")
        md.append("\n")
    if verdict.get("citations"):
        md.append("### Citations\n")
        for c in verdict["citations"]:
            md.append(f"- {c}\n")
        md.append("\n")

    return "".join(md)