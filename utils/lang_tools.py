import re

def detect_language(text: str) -> str:
    t = text or ""
    if not t.strip():
        return "unknown"
    zh = len(re.findall(r"[\u4e00-\u9fa5]", t))
    en = sum(c.isalpha() for c in t)
    tot = max(1, len(t))
    zh_ratio = zh / tot
    en_ratio = en / tot
    if zh_ratio >= 0.2 and zh >= 20:
        return "zh"
    if en_ratio >= 0.3 and en >= 50:
        return "en"
    return "unknown"