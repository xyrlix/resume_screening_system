from typing import Dict
import os
import json
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
_sent_model_en = None
_sent_model_multi = None
def _get_model(text_a: str, text_b: str):
    global _sent_model_en, _sent_model_multi
    if SentenceTransformer is None:
        return None
    ta = (text_a or "").lower()
    tb = (text_b or "").lower()
    en_ratio = sum(c.isalpha() for c in ta+tb) / max(1, len(ta+tb))
    if en_ratio > 0.5:
        if _sent_model_en is None:
            _sent_model_en = SentenceTransformer("all-MiniLM-L6-v2")
        return _sent_model_en
    if _sent_model_multi is None:
        _sent_model_multi = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _sent_model_multi

def base_similarity(a: str, b: str) -> float:
    m = _get_model(a, b)
    if m is None:
        rb = (b or "").lower()
        toks = []
        import re
        toks += re.findall(r"[a-zA-Z0-9]+", rb)
        toks = list({t for t in toks if len(t) >= 2})
        if not toks:
            return 0.0
        ra = (a or "").lower()
        hits = sum(1 for t in toks if t in ra)
        return float(hits) / float(len(toks))
    va = m.encode([a])[0]
    vb = m.encode([b])[0]
    num = float((va * vb).sum())
    den = float((va**2).sum()) ** 0.5 * float((vb**2).sum()) ** 0.5
    if den == 0:
        return 0.0
    s = num / den
    return float(max(0.0, min(1.0, s)))

def format_score(text: str) -> float:
    n_digits = sum(c.isdigit() for c in text or "")
    n_lines = max(1, (text or "").count("\n") + 1)
    density = n_digits / n_lines
    return float(max(0.0, min(1.0, density / 5.0)))

_W_PROF = 0.0
_W_METRIC = 0.0
def _load_weights():
    global _W_PROF, _W_METRIC
    try:
        cfg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "matching.json")
        if os.path.isfile(cfg):
            with open(cfg, "r", encoding="utf-8") as f:
                obj = json.load(f)
            w = obj.get("weights", {})
            _W_PROF = float(w.get("proficiency", 0.0) or 0.0)
            _W_METRIC = float(w.get("metric", 0.0) or 0.0)
    except Exception:
        _W_PROF = 0.0
        _W_METRIC = 0.0
_load_weights()

def composite_score(base: float, skill: float, implicit: float, fmt: float, prof: float | None = None, metric: float | None = None) -> float:
    s = base * 0.6 + skill * 0.2 + implicit * 0.15 + fmt * 0.05
    if prof is not None:
        s += _W_PROF * float(prof)
    if metric is not None:
        s += _W_METRIC * float(metric)
    return round(max(0.0, min(1.0, s)), 4)