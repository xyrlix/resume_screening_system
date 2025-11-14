import re
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None

def _rule_extract(text: str):
    ents = []
    # education
    edu_patterns = [r"\b(Bachelor|Master|PhD|MSc|BSc|MBA)\b"]
    for pat in edu_patterns:
        for m in re.finditer(pat, text, flags=re.I):
            ents.append({"type": "学历", "text": m.group(0)})
    # company
    comp_patterns = [r"\b[A-Z][A-Za-z0-9&\- ]+(?:Inc|LLC|Ltd|Co|Company|Corporation)\b"]
    for pat in comp_patterns:
        for m in re.finditer(pat, text):
            ents.append({"type": "公司名称", "text": m.group(0)})
    # position
    pos_patterns = [r"\b(Engineer|Developer|Manager|Analyst|Scientist|Architect|Consultant)\b"]
    for pat in pos_patterns:
        for m in re.finditer(pat, text, flags=re.I):
            ents.append({"type": "职位", "text": m.group(0)})
    return ents

def extract_en_entities(text: str):
    if _nlp is not None:
        doc = _nlp(text or "")
        ents = []
        for e in doc.ents:
            if e.label_ in ("ORG", "WORK_OF_ART", "FAC"):
                ents.append({"type": "公司名称", "text": e.text})
            elif e.label_ in ("PERSON",):
                ents.append({"type": "姓名", "text": e.text})
            elif e.label_ in ("GPE", "LOC"):
                ents.append({"type": "地区", "text": e.text})
            elif e.label_ in ("DATE",):
                ents.append({"type": "日期", "text": e.text})
        ents.extend(_rule_extract(text or ""))
        return ents
    return _rule_extract(text or "")