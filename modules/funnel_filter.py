from typing import List, Dict, Any
import os
try:
    from chromadb import PersistentClient
    try:
        from chromadb.config import Settings
    except Exception:
        Settings = None
except Exception:
    PersistentClient = None
    Settings = None
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
_job_memory_store: List[Dict[str, Any]] = []
if PersistentClient is not None:
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
    if Settings is not None:
        try:
            _client = PersistentClient(path=db_path, settings=Settings(allow_anonymous_telemetry=False))
        except Exception:
            _client = PersistentClient(path=db_path)
    else:
        _client = PersistentClient(path=db_path)
    try:
        _collection = _client.get_collection("resume_collection")
    except Exception:
        _collection = _client.create_collection("resume_collection")
    try:
        _job_collection = _client.get_collection("job_collection")
    except Exception:
        _job_collection = _client.create_collection("job_collection")
else:
    _collection = None
    _job_collection = None
_sent = SentenceTransformer("all-MiniLM-L6-v2") if SentenceTransformer is not None else None
_emb_cache: Dict[str, tuple] = {}
_vsearch_cache: Dict[str, tuple] = {}
_funnel_cache: Dict[str, tuple] = {}

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
    for it in items or []:
        rid = str(it.get("id"))
        txt = str(it.get("text", ""))
        if not rid or not txt:
            continue
        ids.append(rid)
        docs.append(txt)
    if not ids:
        return {"count": 0}
    chunk = 128
    total = 0
    for i in range(0, len(ids), chunk):
        c_ids = ids[i:i+chunk]
        c_docs = docs[i:i+chunk]
        c_embs = _sent.encode(c_docs)
        try:
            _collection.upsert(ids=c_ids, documents=c_docs, embeddings=list(c_embs))
        except Exception:
            _collection.add(ids=c_ids, documents=c_docs, embeddings=list(c_embs))
        total += len(c_ids)
    logger.info(f"upsert count={total}")
    return {"count": total}

def job_index_upsert(items: List[Dict[str, Any]]):
    logger = get_logger("job_index")
    if _job_collection is None or _sent is None:
        for it in items or []:
            jid = str(it.get("id"))
            txt = str(it.get("text", ""))
            if not jid or not txt:
                continue
            _job_memory_store.append({"id": jid, "text": txt})
        logger.info(f"job_memory_upsert count={len(items or [])}")
        return {"count": len(items or [])}
    ids = []
    docs = []
    for it in items or []:
        jid = str(it.get("id"))
        txt = str(it.get("text", ""))
        if not jid or not txt:
            continue
        ids.append(jid)
        docs.append(txt)
    if not ids:
        return {"count": 0}
    chunk = 128
    total = 0
    for i in range(0, len(ids), chunk):
        c_ids = ids[i:i+chunk]
        c_docs = docs[i:i+chunk]
        c_embs = _sent.encode(c_docs)
        try:
            _job_collection.upsert(ids=c_ids, documents=c_docs, embeddings=list(c_embs))
        except Exception:
            _job_collection.add(ids=c_ids, documents=c_docs, embeddings=list(c_embs))
        total += len(c_ids)
    logger.info(f"job_upsert count={total}")
    return {"count": total}

def vector_search(job_desc: str, top_k: int = 50) -> List[Dict[str, Any]]:
    if _collection is None or _sent is None:
        scored = []
        for it in _memory_store:
            s = _score.base_similarity(it.get("text", ""), job_desc or "")
            scored.append((s, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [{"id": it.get("id"), "text": it.get("text")} for s, it in scored[:top_k]]
        return out
    now = time.time()
    key = f"{job_desc}::{top_k}"
    ent = _vsearch_cache.get(key)
    if ent and now - ent[1] < 30:
        return ent[0]
    q = _encode_cached(job_desc)
    res = _collection.query(query_embeddings=[q], n_results=max(1, top_k))
    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    for i, d in zip(ids, docs):
        out.append({"id": i, "text": d})
    _vsearch_cache[key] = (out, now)
    return out

def job_vector_search_by_text(query_text: str, top_k: int = 20) -> List[Dict[str, Any]]:
    if _job_collection is None or _sent is None:
        scored = []
        for it in _job_memory_store:
            s = _score.base_similarity(it.get("text", ""), query_text or "")
            scored.append((s, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = [{"id": it.get("id"), "text": it.get("text") } for s, it in scored[:top_k]]
        return out
    now = time.time()
    key = f"JOB::{top_k}::{hash(query_text)}"
    ent = _vsearch_cache.get(key)
    if ent and now - ent[1] < 30:
        return ent[0]
    q = _encode_cached(query_text)
    res = _job_collection.query(query_embeddings=[q], n_results=max(1, top_k))
    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    for i, d in zip(ids, docs):
        out.append({"id": i, "text": d})
    _vsearch_cache[key] = (out, now)
    return out

def three_stage_filter(job_desc: str, custom_rules: List[Dict[str, Any]] = None, top_k: int = 50) -> List[Dict[str, Any]]:
    logger = get_logger("funnel")
    key = f"{job_desc}::{top_k}::{str(custom_rules or [])}"
    now = time.time()
    ent = _funnel_cache.get(key)
    if ent and now - ent[1] < 60:
        logger.info(f"cache_hit candidates={len(ent[0])}")
        return ent[0]
    candidates = vector_search(job_desc, top_k=top_k)
    implicit = _llm.infer_implicit_demands(job_desc)
    results = []
    jprof = _matcher.job_profile_from_text(job_desc)
    for c in candidates:
        t = c.get("text", "")
        rprof = _matcher.profile_from_text(t)
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
        prof_score = 0.0
        metric_score = 0.0
        kw = set(rprof.get("keywords", []))
        methods = {"transformer","transformers","rag","ranking","bert","gpt"}
        if any(k in methods for k in kw):
            prof_score = 0.2
        if any(ch.isdigit() for ch in t):
            metric_score = min(1.0, 0.2 + 0.2 * sum(ch.isdigit() for ch in t) / 20.0)
        final = _score.composite_score(base_sim, skill_ratio, implicit_score, fmt_score, prof_score, metric_score)
        # v2 解释：根据关键词分类领域/方法，简单打分（不影响综合分）
        kw = set(rprof.get("keywords", []))
        domains = {"金融","电商","医疗","制造","物联网"}
        methods = {"transformer","transformers","rag","ranking","bert","gpt"}
        matched_domains = sorted(list(kw & domains))
        matched_methods = sorted([k for k in kw if k in methods])
        # 证书与语言命中（以岗位画像为基准）
        matched_certs = sorted(list(set(jprof.get("certs", [])) & set(rprof.get("certs", []))))
        matched_langs = sorted(list(set(jprof.get("languages", [])) & set(rprof.get("languages", []))))
        prof_list = list(rprof.get("proficiency", []) or [])
        prof_score = 0.0
        metric_score = 0.0
        # 以格式分与关键词命中简单近似（示例占位）
        if matched_methods:
            prof_score = min(1.0, 0.2 + 0.2 * len(matched_methods))
        if any(x.isdigit() for x in t):
            metric_score = min(1.0, 0.2 + 0.2 * sum(ch.isdigit() for ch in t) / 20.0)
        results.append({"id": c.get("id"), "score": final, "base": base_sim, "skill": round(skill_ratio, 4), "implicit": round(implicit_score, 4), "format": round(fmt_score, 4), "resume_profile": rprof, "job_profile": jprof})
    results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
    logger.info(f"job_desc_len={len(job_desc)} candidates={len(candidates)} results={len(results)}")
    _funnel_cache[key] = (results, now)
    return results
def funnel_explain(job_desc: str, custom_rules: List[Dict[str, Any]] = None, top_k: int = 50) -> Dict[str, Any]:
    logger = get_logger("funnel")
    candidates = vector_search(job_desc, top_k=top_k)
    jprof = _matcher.job_profile_from_text(job_desc)
    items = []
    passed = []
    for c in candidates:
        t = c.get("text", "")
        rprof = _matcher.profile_from_text(t)
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
            items.append({
                "id": c.get("id"),
                "rule_ok": False,
                "rule_reason": "similarity_below_threshold",
                "base": round(base_sim, 4),
                "skill": 0.0,
                "implicit": 0.0,
                "format": 0.0,
                "score": 0.0,
                "resume_profile": rprof,
                "job_profile": jprof
            })
            continue
        ok, reason = _rule.apply_custom_rules(rprof, custom_rules or [])
        if not ok:
            items.append({
                "id": c.get("id"),
                "rule_ok": False,
                "rule_reason": reason,
                "base": round(base_sim, 4),
                "skill": 0.0,
                "implicit": 0.0,
                "format": 0.0,
                "score": 0.0,
                "resume_profile": rprof,
                "job_profile": jprof
            })
            continue
        skill_ratio = 0.0
        if len(jprof.get("skills", [])) > 0:
            skill_ratio = len(set(jprof.get("skills", [])) & set(rprof.get("skills", []))) / float(len(jprof.get("skills", [])))
        implicit_score = _llm.implicit_match_score(t, _llm.infer_implicit_demands(job_desc))
        fmt_score = _score.format_score(t)
        kw = set(rprof.get("keywords", []))
        domains = {"金融","电商","医疗","制造","物联网"}
        methods = {"transformer","transformers","rag","ranking","bert","gpt"}
        matched_domains = sorted(list(kw & domains))
        matched_methods = sorted([k for k in kw if k in methods])
        matched_certs = sorted(list(set(jprof.get("certs", [])) & set(rprof.get("certs", []))))
        matched_langs = sorted(list(set(jprof.get("languages", [])) & set(rprof.get("languages", []))))
        prof_list = list(rprof.get("proficiency", []) or [])
        prof_score = 0.0
        metric_score = 0.0
        if matched_methods:
            prof_score = min(1.0, 0.2 + 0.2 * len(matched_methods))
        if any(x.isdigit() for x in t):
            metric_score = min(1.0, 0.2 + 0.2 * sum(ch.isdigit() for ch in t) / 20.0)
        final = _score.composite_score(base_sim, skill_ratio, implicit_score, fmt_score, prof_score, metric_score)
        expl = []
        if matched_methods:
            expl.append("方法:" + "/".join(matched_methods))
        if matched_domains:
            expl.append("领域:" + "/".join(matched_domains))
        if matched_certs:
            expl.append("证书:" + "/".join(matched_certs))
        if matched_langs:
            expl.append("语言:" + "/".join(matched_langs))
        if prof_list:
            expl.append("熟练度:" + "/".join(prof_list))
        row = {
            "id": c.get("id"),
            "rule_ok": True,
            "rule_reason": "ok",
            "base": round(base_sim, 4),
            "skill": round(skill_ratio, 4),
            "implicit": round(implicit_score, 4),
            "format": round(fmt_score, 4),
            "score": final,
            "resume_profile": rprof,
            "job_profile": jprof,
            "matched_domains": matched_domains,
            "matched_methods": matched_methods,
            "matched_certs": matched_certs,
            "matched_lang_levels": matched_langs,
            "proficiency_list": prof_list,
            "proficiency_score": round(prof_score, 4),
            "metric_score": round(metric_score, 4),
            "explain": ";".join(expl)
        }
        items.append(row)
        passed.append(row)
    passed_sorted = sorted(passed, key=lambda x: x.get("score", 0.0), reverse=True)
    logger.info(f"funnel_explain recall={len(candidates)} passed={len(passed_sorted)}")
    return {
        "recall_count": len(candidates),
        "rule_passed_count": len(passed_sorted),
        "items": items,
        "passed": passed_sorted
    }
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