from typing import List, Dict, Any
import os
# 设置HuggingFace镜像源
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

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
import numpy as np

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
    
    # 修复ChromaDB集合配置问题
    try:
        _collection = _client.get_collection("resume_collection")
    except Exception:
        try:
            _collection = _client.create_collection(
                name="resume_collection",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"创建resume_collection时出错: {e}")
            _collection = None
            
    try:
        _job_collection = _client.get_collection("job_collection")
    except Exception:
        try:
            _job_collection = _client.create_collection(
                name="job_collection", 
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"创建job_collection时出错: {e}")
            _job_collection = None
else:
    _collection = None
    _job_collection = None

# 使用BGE-M3模型替代SentenceTransformer模型
_sent = None
_use_bge_m3 = False  # 默认不使用BGE-M3模型

# 尝试加载BGE-M3模型
def _load_embedding_model():
    global _sent, _use_bge_m3
    if SentenceTransformer is None:
        return None
        
    try:
        # 导入FlagEmbedding用于BGE-M3模型（如果已安装）
        from FlagEmbedding import BGEM3FlagModel
        _sent = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        _use_bge_m3 = True  # 标记使用BGE-M3模型
        print("成功加载BGE-M3模型")
    except ImportError:
        print("FlagEmbedding未安装，使用SentenceTransformer模型")
        try:
            _sent = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            print(f"加载SentenceTransformer模型时出错: {e}")
            _sent = None
    except Exception as e:
        print(f"加载BGE-M3模型时出错: {e}")
        try:
            _sent = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            print(f"加载SentenceTransformer模型时出错: {e}")
            _sent = None

_load_embedding_model()
        
_emb_cache: Dict[str, tuple] = {}
_vsearch_cache: Dict[str, tuple] = {}
_funnel_cache: Dict[str, tuple] = {}

def _load_matcher():
    base_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    path = os.path.abspath(os.path.join(base_dir, "04_matcher_model.py"))
    spec = importlib.util.spec_from_file_location("matcher_module", path)
    if spec is not None:
        mod = importlib.util.module_from_spec(spec)
        if spec.loader is not None:
            spec.loader.exec_module(mod)
        return mod
    return None

_matcher = _load_matcher()

def _load_mod(relpath: str):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", relpath))
    spec = importlib.util.spec_from_file_location(relpath.replace('/', '_'), path)
    mod = None
    if spec is not None:
        mod = importlib.util.module_from_spec(spec)
        if spec.loader is not None:
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
        # 转换为numpy数组列表以确保兼容性
        c_embs_list = [np.array(emb).tolist() for emb in c_embs]
        try:
            _collection.upsert(ids=c_ids, documents=c_docs, embeddings=c_embs_list)
        except Exception:
            _collection.add(ids=c_ids, documents=c_docs, embeddings=c_embs_list)
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
        # 转换为numpy数组列表以确保兼容性
        c_embs_list = [np.array(emb).tolist() for emb in c_embs]
        try:
            _job_collection.upsert(ids=c_ids, documents=c_docs, embeddings=c_embs_list)
        except Exception:
            _job_collection.add(ids=c_ids, documents=c_docs, embeddings=c_embs_list)
        total += len(c_ids)
    logger.info(f"job_upsert count={total}")
    return {"count": total}

def vector_search(job_desc: str, top_k: int = 50) -> List[Dict[str, Any]]:
    if _collection is None or _sent is None:
        scored = []
        if _score is not None:
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
    if q is None:
        return []
    try:
        # 确保q是正确的类型
        if hasattr(q, 'tolist'):
            res = _collection.query(query_embeddings=[q.tolist()], n_results=max(1, top_k))  # type: ignore
        else:
            res = _collection.query(query_embeddings=[q], n_results=max(1, top_k))  # type: ignore
    except Exception:
        return []
    out = []
    try:
        # 安全地访问结果
        ids_result = res.get("ids", [])
        docs_result = res.get("documents", [])
        if ids_result and docs_result:
            ids = ids_result[0]
            docs = docs_result[0]
            for i, d in zip(ids, docs):
                out.append({"id": i, "text": d})
    except Exception:
        pass
    _vsearch_cache[key] = (out, now)
    return out

def job_vector_search_by_text(query_text: str, top_k: int = 20) -> List[Dict[str, Any]]:
    if _job_collection is None or _sent is None:
        scored = []
        if _score is not None:
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
    if q is not None:
        res = _job_collection.query(query_embeddings=[q.tolist()], n_results=max(1, top_k))
        out = []
        if res and "ids" in res and "documents" in res:
            ids = res["ids"][0] if res["ids"] else []
            docs = res["documents"][0] if res["documents"] else []
            for i, d in zip(ids, docs):
                out.append({"id": i, "text": d})
        _vsearch_cache[key] = (out, now)
        return out
    return []

def _match_hot_industry_or_position(text: str) -> bool:
    """
    检查文本是否匹配任何行业或岗位（现在支持全行业全岗位）
    """
    # 现在支持所有行业和岗位，总是返回True
    return True

def three_stage_filter_v2(job_desc: str, custom_rules: List[Dict[str, Any]] | None = None, top_k: int = 50) -> List[Dict[str, Any]]:
    logger = get_logger("funnel")
    key = f"{job_desc}::{top_k}::{str(custom_rules or [])}"
    now = time.time()
    ent = _funnel_cache.get(key)
    if ent and now - ent[1] < 60:
        logger.info(f"cache_hit candidates={len(ent[0])}")
        return ent[0]
    
    # 检查岗位描述是否匹配热门行业或岗位
    if not _match_hot_industry_or_position(job_desc):
        logger.info("Job description does not match hot industries or positions")
        return []
        
    candidates = vector_search(job_desc, top_k=top_k)
    implicit = _llm.infer_implicit_demands(job_desc) if _llm else []
    results = []
    jprof = _matcher.job_profile_from_text(job_desc) if _matcher else {}
    for c in candidates:
        t = c.get("text", "")
        rprof = _matcher.profile_from_text(t) if _matcher else {}
        base_sim = _score.base_similarity(t, job_desc) if _score else 0.0
        lang = detect_language(t)
        
        # 只处理中文简历
        if lang != "zh":
            continue
            
        thr = 0.5 if _sent is not None else 0.2
        if base_sim < thr:
            continue
        ok, reason = _rule.apply_custom_rules(rprof, custom_rules or []) if _rule else (True, "ok")
        if not ok:
            continue
        skill_ratio = 0.0
        if len(jprof.get("skills", [])) > 0:
            skill_ratio = len(set(jprof.get("skills", [])) & set(rprof.get("skills", []))) / float(len(jprof.get("skills", [])))
        implicit_score = _llm.implicit_match_score(t, implicit) if _llm else 0.0
        fmt_score = _score.format_score(t) if _score else 0.0
        prof_score = 0.0
        metric_score = 0.0
        kw = set(rprof.get("keywords", []))
        methods = {"transformer","transformers","rag","ranking","bert","gpt"}
        if any(k in methods for k in kw):
            prof_score = 0.2
        if any(ch.isdigit() for ch in t):
            metric_score = min(1.0, 0.2 + 0.2 * sum(ch.isdigit() for ch in t) / 20.0)
        final = _score.composite_score(base_sim, skill_ratio, implicit_score, fmt_score, prof_score, metric_score) if _score else 0.0
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

def funnel_explain(job_desc: str, custom_rules: List[Dict[str, Any]] | None = None, top_k: int = 50) -> Dict[str, Any]:
    logger = get_logger("funnel")
    candidates = vector_search(job_desc, top_k=top_k)
    jprof = _matcher.job_profile_from_text(job_desc) if _matcher else {}
    items = []
    passed = []
    for c in candidates:
        t = c.get("text", "")
        rprof = _matcher.profile_from_text(t) if _matcher else {}
        base_sim = _score.base_similarity(t, job_desc) if _score else 0.0
        lang = detect_language(t)
        
        # 只处理中文简历
        if lang != "zh":
            items.append({
                "id": c.get("id"),
                "rule_ok": False,
                "rule_reason": "not_chinese_resume",
                "base": round(base_sim, 4),
                "skill": 0.0,
                "implicit": 0.0,
                "format": 0.0,
                "score": 0.0,
                "resume_profile": rprof,
                "job_profile": jprof
            })
            continue
            
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
        ok, reason = _rule.apply_custom_rules(rprof, custom_rules or []) if _rule else (True, "ok")
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
        implicit_score = _llm.implicit_match_score(t, _llm.infer_implicit_demands(job_desc) if _llm else []) if _llm else 0.0
        fmt_score = _score.format_score(t) if _score else 0.0
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
        final = _score.composite_score(base_sim, skill_ratio, implicit_score, fmt_score, prof_score, metric_score) if _score else 0.0
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

def generate_evaluation_report(resume_text: str, job_text: str) -> Dict[str, Any]:
    """
    生成评估报告
    包括匹配度分析、优势分析、风险分析等
    """
    report: Dict[str, Any] = {
        "generated_at": time.time(),
        "resume_summary": {},
        "job_requirements": {},
        "match_analysis": {},
        "strengths": [],
        "risks": [],
        "recommendations": []
    }
    
    # 提取简历摘要
    if _matcher:
        rprof = _matcher.profile_from_text(resume_text)
        report["resume_summary"] = rprof
    
    # 提取岗位要求
    if _matcher:
        jprof = _matcher.job_profile_from_text(job_text)
        report["job_requirements"] = jprof
    
    # 匹配分析
    base_sim = _score.base_similarity(resume_text, job_text) if _score else 0.0
    
    skill_match = 0.0
    if (_matcher and report["job_requirements"].get("skills") and 
        report["resume_summary"].get("skills")):
        matched_skills = set(report["job_requirements"]["skills"]) & set(report["resume_summary"]["skills"])
        skill_match = len(matched_skills) / len(report["job_requirements"]["skills"]) if report["job_requirements"]["skills"] else 0.0
    
    report["match_analysis"] = {
        "base_similarity": base_sim,
        "skill_match_ratio": skill_match,
        "overall_score": 0.7 * base_sim + 0.3 * skill_match
    }
    
    # 优势分析
    if skill_match > 0.7:
        report["strengths"].append("技能匹配度高")
    if base_sim > 0.6:
        report["strengths"].append("整体匹配度较高")
    if report["resume_summary"].get("years", 0) > 5:
        report["strengths"].append("工作经验丰富")
    if len(report["resume_summary"].get("skills", [])) > 10:
        report["strengths"].append("技能覆盖面广")
    
    # 风险分析
    if skill_match < 0.3:
        report["risks"].append("关键技能缺失")
    if base_sim < 0.4:
        report["risks"].append("整体匹配度较低")
    if report["resume_summary"].get("years", 0) < 1:
        report["risks"].append("工作经验较少")
    if not report["resume_summary"].get("degree"):
        report["risks"].append("学历信息不明确")
    
    # 建议
    if skill_match < 0.5:
        report["recommendations"].append("建议加强相关技能学习")
    if base_sim < 0.5:
        report["recommendations"].append("建议优化简历内容，更贴近岗位要求")
    if not report["resume_summary"].get("degree"):
        report["recommendations"].append("建议补充学历信息")
    if report["resume_summary"].get("years", 0) < 2:
        report["recommendations"].append("建议积累更多项目经验")
    
    # 添加更详细的分析
    if _llm:
        # 使用LLM生成更详细的评估报告
        try:
            llm_report = _llm.generate_evaluation_report(job_text, resume_text)
            if isinstance(llm_report, dict) and "report" in llm_report:
                report["llm_analysis"] = llm_report["report"]
        except Exception:
            # 如果LLM调用失败，使用简单的分析
            report["llm_analysis"] = "未能生成详细的LLM分析报告"
    
    return report

def _encode_cached(text: str):
    if _sent is None:
        return None
    now = time.time()
    ent = _emb_cache.get(text)
    if ent and now - ent[1] < 3600:
        return ent[0]
    # 根据使用的模型类型进行不同的编码操作
    if _use_bge_m3:
        # 使用BGE-M3模型编码
        emb_result = _sent.encode(text)
        emb = np.array(emb_result['dense_vecs'])
    else:
        # 使用SentenceTransformer模型编码
        emb = _sent.encode([text])[0]
    _emb_cache[text] = (emb, now)
    return emb
