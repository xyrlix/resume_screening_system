from typing import List, Dict, Any
import os
try:
    from chromadb import PersistentClient
except Exception:
    PersistentClient = None
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
import sys
_root = os.path.dirname(os.path.dirname(__file__))
if _root not in sys.path:
    sys.path.append(_root)
import importlib.util
import time
from utils.logger import get_logger
from utils.lang_tools import detect_language
import importlib.util

_memory_store: List[Dict[str, Any]] = []
if PersistentClient is not None:
    _client = PersistentClient(path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db"))
    try:
        _collection = _client.get_collection("resume_collection")
    except Exception:
        _collection = _client.create_collection("resume_collection")
else:
    _collection = None
_sent = SentenceTransformer("all-MiniLM-L6-v2") if SentenceTransformer is not None else None
_emb_cache: Dict[str, tuple] = {}

def _load_matcher():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    path = os.path.abspath(os.path.join(base_dir, "04_matcher_model.py"))
    spec = importlib.util.spec_from_file_location("matcher_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_matcher = _load_matcher()

def _load_mod(relpath: str):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", relpath))
    spec = importlib.util.spec_from_file_location(relpath.replace('/', '_'), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_rule = _load_mod("modules/rule_engine.py")
_llm = _load_mod("modules/llm_utils.py")
_score = _load_mod("modules/scoring.py")

def vector_index_upsert(items: List[Dict[str, Any]]):
    logger = get_logger("vector")
    if _collection is None or _sent is None:
        for it in items or []:
            rid = str(it.get("id"))
            txt = str(it.get("text", ""))
            if not rid or not txt:
                continue
            _memory_store.append({"id": rid, "text": txt})
        logger.info(f"memory_upsert count={len(items or [])}")
        return {"count": len(items or [])}
    ids = []
    docs = []
    embs = []
    for it in items or []:
        rid = str(it.get("id"))
        txt = str(it.get("text", ""))
        if not rid or not txt:
            continue
        ids.append(rid)
        docs.append(txt)
        embs.append(_sent.encode([txt])[0])
    if not ids:
        return {"count": 0}
    try:
        _collection.upsert(ids=ids, documents=docs, embeddings=embs)
    except Exception:
        _collection.add(ids=ids, documents=docs, embeddings=embs)
    logger.info(f"upsert count={len(ids)}")
    return {"count": len(ids)}

def vector_search(job_desc: str, top_k: int = 50) -> List[Dict[str, Any]]:
    if _collection is None or _sent is None:
        scored = []
        for it in _memory_store:
            s = _score.base_similarity(it.get("text", ""), job_desc or "")
            scored.append((s, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [{"id": it.get("id"), "text": it.get("text")} for s, it in scored[:top_k]]
        return out
    q = _encode_cached(job_desc)
    res = _collection.query(query_embeddings=[q], n_results=max(1, top_k))
    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    for i, d in zip(ids, docs):
        out.append({"id": i, "text": d})
    return out

def three_stage_filter(job_desc: str, custom_rules: List[Dict[str, Any]] = None, top_k: int = 50) -> List[Dict[str, Any]]:
    logger = get_logger("funnel")
    candidates = vector_search(job_desc, top_k=top_k)
    implicit = _llm.infer_implicit_demands(job_desc)
    results = []
    for c in candidates:
        t = c.get("text", "")
        rprof = _matcher.profile_from_text(t)
        jprof = _matcher.job_profile_from_text(job_desc)
        base_sim = _score.base_similarity(t, job_desc)
        lang = detect_language(t)
        thr = 0.5 if _sent is not None else 0.2
        if _sent is not None:
            if lang == "en":
                thr = 0.55
            elif lang == "zh":
                thr = 0.5
            else:
                thr = 0.45
        if base_sim < thr:
            continue
        ok, reason = _rule.apply_custom_rules(rprof, custom_rules or [])
        if not ok:
            continue
        skill_ratio = 0.0
        if len(jprof.get("skills", [])) > 0:
            skill_ratio = len(set(jprof.get("skills", [])) & set(rprof.get("skills", []))) / float(len(jprof.get("skills", [])))
        implicit_score = _llm.implicit_match_score(t, implicit)
        fmt_score = _score.format_score(t)
        final = _score.composite_score(base_sim, skill_ratio, implicit_score, fmt_score)
        results.append({"id": c.get("id"), "score": final, "base": base_sim, "skill": round(skill_ratio, 4), "implicit": round(implicit_score, 4), "format": round(fmt_score, 4), "resume_profile": rprof, "job_profile": jprof})
    results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
    logger.info(f"job_desc_len={len(job_desc)} candidates={len(candidates)} results={len(results)}")
    return results
def _encode_cached(text: str):
    if _sent is None:
        return None
    now = time.time()
    ent = _emb_cache.get(text)
    if ent and now - ent[1] < 3600:
        return ent[0]
    emb = _sent.encode([text])[0]
    _emb_cache[text] = (emb, now)
    return emb