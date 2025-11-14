from typing import List, Dict, Any
import importlib.util
import os

def _load_mod(relpath: str):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", relpath))
    spec = importlib.util.spec_from_file_location(relpath.replace('/', '_'), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_llm = _load_mod("modules/llm_utils.py")

def _completeness(profile: Dict[str, Any]) -> float:
    keys = ["skills", "degree", "years", "position"]
    hit = 0
    for k in keys:
        v = profile.get(k)
        if isinstance(v, list) and len(v) > 0:
            hit += 1
        elif isinstance(v, str) and v:
            hit += 1
        elif isinstance(v, int) and v > 0:
            hit += 1
    return float(hit) / float(len(keys))

def dynamic_threshold(job_desc: str, results: List[Dict[str, Any]]) -> float:
    logger = get_logger("decision")
    if not results:
        logger.info("empty_results")
        return 0.7
    comp = sum(_completeness(r.get("resume_profile", {})) for r in results) / float(len(results))
    # 语言差异化：若英文占比高，提高分数线
    en_count = 0
    zh_count = 0
    for r in results:
        text = r.get("resume_profile", {}).get("skills", [])
        joined = ",".join(text)
        if any(c.isalpha() for c in joined):
            en_count += 1
        if any("\u4e00" <= ch <= "\u9fa5" for ch in joined):
            zh_count += 1
    if en_count > zh_count:
        base_thr = 0.82
    else:
        base_thr = 0.8
    if comp >= 0.9:
        return base_thr
    return base_thr - 0.1

def recommend(job_desc: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    thr = dynamic_threshold(job_desc, results)
    picks = []
    for r in results:
        if float(r.get("score", 0)) >= thr:
            prof = r.get("resume_profile", {})
            text = "技能:" + ",".join(prof.get("skills", [])) + " 职位:" + str(prof.get("position", ""))
            if hasattr(_llm, "generate_interview_questions_ctx"):
                qs = _llm.generate_interview_questions_ctx(job_desc, text)
            else:
                qs = _llm.generate_interview_questions(text)
            reason = f"base={r.get('base')} skill={r.get('skill')} implicit={r.get('implicit')} format={r.get('format')}"
            picks.append({"id": r.get("id"), "score": r.get("score"), "interview_questions": qs, "reason": reason, "details": {"base": r.get("base"), "skill": r.get("skill"), "implicit": r.get("implicit"), "format": r.get("format")}})
    logger = get_logger("decision")
    logger.info(f"threshold={thr} picks={len(picks)}")
    return {"threshold": thr, "recommended": picks}
from utils.logger import get_logger