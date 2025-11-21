from fastapi import FastAPI, Depends, Request, UploadFile, File, Form
from fastapi import HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer
from starlette.responses import RedirectResponse
try:
    import jwt
    from jwt.algorithms import RSAAlgorithm
except Exception:
    jwt = None
from pydantic import BaseModel
from typing import Dict, Any, List
import os
import importlib.util
import socket
import sys
BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from utils.logger import get_logger

# 动态加载同目录下的脚本：04_predict.py 与 04_matcher_model.py

def _load_func(module_filename: str, func_name: str):
    path = os.path.join(BASE_DIR, module_filename)
    spec = importlib.util.spec_from_file_location(module_filename.replace('.py',''), path)
    mod = importlib.util.module_from_spec(spec)
    if spec and spec.loader:
        spec.loader.exec_module(mod)
    return getattr(mod, func_name)

def _load_func_project(module_relpath: str, func_name: str):
    path = os.path.join(PROJECT_ROOT, module_relpath)
    spec = importlib.util.spec_from_file_location(module_relpath.replace('.py','').replace('/','_'), path)
    mod = importlib.util.module_from_spec(spec)
    if spec and spec.loader:
        spec.loader.exec_module(mod)
    return getattr(mod, func_name)

ner_predict = None

def _ensure_ner():
    global ner_predict
    if ner_predict is None:
        try:
            ner_predict = _load_func('04_predict.py', 'predict')
        except Exception:
            def _fallback(text: str):
                return []
            ner_predict = _fallback
quick_match = _load_func('04_matcher_model.py', 'quick_match')
vector_index_upsert = _load_func_project('modules/funnel_filter.py', 'vector_index_upsert')
three_stage_filter = _load_func_project('modules/funnel_filter.py', 'three_stage_filter')
funnel_explain = _load_func_project('modules/funnel_filter.py', 'funnel_explain')
decision_recommend = _load_func_project('modules/decision_support.py', 'recommend')
job_index_upsert = _load_func_project('modules/funnel_filter.py', 'job_index_upsert')
job_vector_search_by_text = _load_func_project('modules/funnel_filter.py', 'job_vector_search_by_text')
get_industry_templates = _load_func_project('modules/llm_utils.py', 'get_industry_templates')
update_industry_templates = _load_func_project('modules/llm_utils.py', 'update_industry_templates')
online_ingest = _load_func_project('modules/online_ingest.py', 'ingest_urls')
fetch_bosszhipin = _load_func_project('modules/adapters/bosszhipin.py', 'fetch_bosszhipin')
fetch_zhilian = _load_func_project('modules/adapters/zhilian.py', 'fetch_zhilian')
fetch_liepin = _load_func_project('modules/adapters/liepin.py', 'fetch_liepin')
fetch_linkedin = _load_func_project('modules/adapters/linkedin.py', 'fetch_linkedin')
set_llm_enabled = _load_func_project('modules/llm_utils.py', 'set_llm_enabled')
generate_structured_jd = _load_func_project('modules/llm_utils.py', 'generate_structured_jd')
optimize_resume = _load_func_project('modules/llm_utils.py', 'optimize_resume')
generate_interview_questions_ctx = _load_func_project('modules/llm_utils.py', 'generate_interview_questions_ctx')
base_similarity = _load_func_project('modules/scoring.py', 'base_similarity')
generate_evaluation_report = _load_func_project('modules/llm_utils.py', 'generate_evaluation_report')
llm_generate_questions = _load_func_project('modules/llm_providers.py', 'generate_questions')
llm_generate_analysis = _load_func_project('modules/llm_providers.py', 'generate_analysis')


app = FastAPI(title="Resume Screening API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def _parse_token_map() -> dict:
    m = {}
    env = os.getenv("RBAC_TOKEN_MAP", "")
    for seg in env.split(","):
        if not seg.strip():
            continue
        if ":" in seg:
            role, token = seg.split(":", 1)
            m[token.strip()] = role.strip()
    for role, envk in {
        "admin": "AUTH_TOKEN_ADMIN",
        "hr": "AUTH_TOKEN_HR",
        "interviewer": "AUTH_TOKEN_INTERVIEWER",
        "candidate": "AUTH_TOKEN_CANDIDATE",
    }.items():
        tok = os.getenv(envk)
        if tok:
            m[tok.strip()] = role
    return m

TOKEN_ROLE = _parse_token_map()

JWT_SECRET = os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY") or ""
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_PUBLIC_KEY = os.getenv("JWT_PUBLIC_KEY") or ""
JWT_JWKS_URL = os.getenv("JWT_JWKS_URL") or ""
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

_JWKS_CACHE: Dict[str, Any] = {}

def _jwt_decode(token: str) -> Dict[str, Any] | None:
    if jwt is None:
        return None
    try:
        hdr = jwt.get_unverified_header(token)
    except Exception:
        hdr = {}
    kid = hdr.get("kid")
    if JWT_JWKS_URL:
        try:
            import requests
            if not _JWKS_CACHE.get("jwks"):
                r = requests.get(JWT_JWKS_URL, timeout=5)
                _JWKS_CACHE["jwks"] = r.json()
            keys = _JWKS_CACHE.get("jwks", {}).get("keys", [])
            key = None
            for k in keys:
                if not kid or k.get("kid") == kid:
                    key = k
                    break
            if key is not None:
                pub = RSAAlgorithm.from_jwk(key)
                return jwt.decode(token, pub, algorithms=[hdr.get("alg") or "RS256"])
        except Exception:
            pass
    if JWT_PUBLIC_KEY:
        try:
            return jwt.decode(token, JWT_PUBLIC_KEY, algorithms=[hdr.get("alg") or "RS256"])
        except Exception:
            pass
    if JWT_SECRET:
        try:
            return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        except Exception:
            pass
    return None

def _jwt_encode(payload: Dict[str, Any]) -> str:
    if jwt is None:
        raise RuntimeError("PyJWT not available")
    if JWT_SECRET:
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    raise RuntimeError("JWT secret not configured")

def _role_from_request(req: Request) -> str | None:
    h = req.headers.get("Authorization") or ""
    if h.lower().startswith("bearer "):
        tok = h[7:].strip()
        claims = _jwt_decode(tok)
        if claims and isinstance(claims.get("roles"), list):
            roles = claims.get("roles")
            return str(roles[0]) if roles else None
        return TOKEN_ROLE.get(tok)
    k = req.headers.get("X-API-Key")
    if k:
        return TOKEN_ROLE.get(k.strip())
    return None

def require_roles(roles: list[str]):
    def _dep(req: Request):
        r = _role_from_request(req)
        if r is None or r not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        return {"role": r}
    return _dep

def _parse_role_permissions() -> dict[str, set[str]]:
    default = {
        "admin": {"config.read", "config.write", "templates.read", "ingest.write", "index.write", "decision.read", "match.read", "filter.read", "uploads.write"},
        "hr": {"config.read", "templates.read", "decision.read", "match.read", "filter.read", "uploads.write", "index.write"},
        "interviewer": {"config.read", "templates.read", "decision.read"},
        "candidate": {"uploads.write"}
    }
    env = os.getenv("ROLE_PERMS_MAP", "")
    if not env.strip():
        return default
    out = {}
    for seg in env.split(","):
        if not seg.strip() or ":" not in seg:
            continue
        role, perms = seg.split(":", 1)
        out[role.strip()] = set([p.strip() for p in perms.split("|") if p.strip()])
    for r, s in default.items():
        if r not in out:
            out[r] = s
    return out

ROLE_PERMS = _parse_role_permissions()

def require_perms(perms: list[str]):
    def _dep(req: Request):
        h = req.headers.get("Authorization") or ""
        if h.lower().startswith("bearer "):
            tok = h[7:].strip()
            claims = _jwt_decode(tok)
            if claims:
                cperms = set(claims.get("permissions", []) or [])
                for p in perms:
                    if p not in cperms:
                        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
                return {"role": (claims.get("roles") or [None])[0]}
        r = _role_from_request(req)
        if r is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        have = ROLE_PERMS.get(r, set())
        for p in perms:
            if p not in have:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
        return {"role": r}
    return _dep


class PredictRequest(BaseModel):
    text: str


class MatchRequest(BaseModel):
    resume_text: str
    job_text: str

class VectorIndexItem(BaseModel):
    id: str
    text: str

class VectorIndexRequest(BaseModel):
    items: List[VectorIndexItem]

class FilterRequest(BaseModel):
    job_desc: str
    top_k: int = 50
    custom_rules: List[Dict[str, Any]] | None = None

class DecisionRequest(BaseModel):
    job_desc: str
    top_k: int = 50
    custom_rules: List[Dict[str, Any]] | None = None
    page: int = 1
    page_size: int = 20

class DecisionBatchRequest(BaseModel):
    job_descs: List[str]
    top_k: int = 50
    custom_rules: List[Dict[str, Any]] | None = None

class IndustryTemplatesUpdate(BaseModel):
    templates: Dict[str, str]

class OnlineIngestRequest(BaseModel):
    urls: List[str]


class BossZhipinRequest(BaseModel):
    urls: List[str]
    cookie: str

class ZhiLianRequest(BaseModel):
    urls: List[str]
    cookie: str

class LLMEnabled(BaseModel):
    enabled: bool

class LiePinRequest(BaseModel):
    urls: List[str]
    cookie: str

class LinkedInRequest(BaseModel):
    urls: List[str]
    cookie: str

class TokenRequest(BaseModel):
    username: str
    password: str

class JDGenerateRequest(BaseModel):
    text: str

class ResumeOptimizeRequest(BaseModel):
    text: str

class InterviewQuestionsRequest(BaseModel):
    job_desc: str | None = None
    resume_text: str
    provider: str | None = None
    model: str | None = None
    top_n: int = 3

class LLMSettings(BaseModel):
    provider: str
    api_url: str | None = None
    api_key: str | None = None
    model: str | None = None

class RecommendJobsRequest(BaseModel):
    resume_text: str
    jobs_dir: str | None = None
    top_k: int = 5
    industry: str | None = None
    region: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    data_source: str | None = None
    sqlite_path: str | None = None
    fusion_weight: float | None = 0.7

def _users_from_env_or_file() -> Dict[str, Dict[str, Any]]:
    db = {}
    cfg = os.getenv("USER_MAP", "")
    for seg in cfg.split(","):
        if not seg.strip() or seg.count(":") < 3:
            continue
        u, pw, role, perms = seg.split(":", 3)
        db[u.strip()] = {"password": pw.strip(), "roles": [role.strip()], "permissions": [p.strip() for p in perms.split("|") if p.strip()]}
    path = os.getenv("USER_DB_PATH") or ""
    if path and os.path.isfile(path):
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            for it in obj.get("users", []) or []:
                u = str(it.get("username"))
                if u:
                    db[u] = {
                        "password": str(it.get("password", "")),
                        "roles": list(it.get("roles", []) or []),
                        "permissions": list(it.get("permissions", []) or [])
                    }
        except Exception:
            pass
    return db


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


ADMIN_UI_ENABLED = False

@app.get("/")
def root():
    if ADMIN_UI_ENABLED:
        return RedirectResponse(url="/ui")
    return RedirectResponse(url="/docs")


@app.post("/predict")
def predict_api(req: PredictRequest) -> Dict[str, Any]:
    _ensure_ner()
    ents = ner_predict(req.text)
    get_logger("api").info(f"predict len={len(req.text)} ents={len(ents)}")
    return {"entities": ents}


if os.getenv("ALLOW_PUBLIC_MATCH", "0") == "1":
    @app.post("/match")
    def match_api(req: MatchRequest) -> Dict[str, Any]:
        res = quick_match(req.resume_text, req.job_text)
        get_logger("api").info(f"match score={res.get('score')}")
        return res
else:
    @app.post("/match", dependencies=[Depends(require_perms(["match.read"]))])
    def match_api(req: MatchRequest) -> Dict[str, Any]:
        res = quick_match(req.resume_text, req.job_text)
        get_logger("api").info(f"match score={res.get('score')}")
        return res


if os.getenv("ALLOW_PUBLIC_INDEX", "0") == "1":
    @app.post("/vector_index")
    def vector_index_api(req: VectorIndexRequest) -> Dict[str, Any]:
        count = vector_index_upsert([{"id": it.id, "text": it.text} for it in req.items])
        get_logger("api").info(f"vector_index count={count.get('count')}")
        return count
else:
    @app.post("/vector_index", dependencies=[Depends(require_perms(["index.write"]))])
    def vector_index_api(req: VectorIndexRequest) -> Dict[str, Any]:
        count = vector_index_upsert([{"id": it.id, "text": it.text} for it in req.items])
        get_logger("api").info(f"vector_index count={count.get('count')}")
        return count

if os.getenv("ALLOW_PUBLIC_FILTER", "0") == "1":
    @app.post("/filter")
    def filter_api(req: FilterRequest) -> Dict[str, Any]:
        res = three_stage_filter(req.job_desc, req.custom_rules or [], req.top_k)
        get_logger("api").info(f"filter results={len(res)}")
        return {"results": res}
else:
    @app.post("/filter", dependencies=[Depends(require_perms(["filter.read"]))])
    def filter_api(req: FilterRequest) -> Dict[str, Any]:
        res = three_stage_filter(req.job_desc, req.custom_rules or [], req.top_k)
        get_logger("api").info(f"filter results={len(res)}")
        return {"results": res}

if os.getenv("ALLOW_PUBLIC_FILTER", "0") == "1":
    @app.post("/funnel_explain")
    def funnel_explain_api(req: FilterRequest) -> Dict[str, Any]:
        obj = funnel_explain(req.job_desc, req.custom_rules or [], req.top_k)
        return obj
else:
    @app.post("/funnel_explain", dependencies=[Depends(require_perms(["filter.read"]))])
    def funnel_explain_api(req: FilterRequest) -> Dict[str, Any]:
        obj = funnel_explain(req.job_desc, req.custom_rules or [], req.top_k)
        return obj

def _decision_impl(req: DecisionRequest) -> Dict[str, Any]:
    res = three_stage_filter(req.job_desc, req.custom_rules or [], req.top_k)
    rec = decision_recommend(req.job_desc, res)
    total = len(res)
    p = max(1, int(req.page))
    ps = max(1, int(req.page_size))
    start = (p - 1) * ps
    end = start + ps
    page_res = res[start:end]
    get_logger("api").info(f"decision total={len(res)} picks={len(rec.get('recommended', []))}")
    return {"results": res, "page_results": page_res, "total": total, "page": p, "page_size": ps, "decision": rec}

if os.getenv("ALLOW_PUBLIC_DECISION", "0") == "1":
    @app.post("/decision")
    def decision_api(req: DecisionRequest) -> Dict[str, Any]:
        return _decision_impl(req)
else:
    @app.post("/decision", dependencies=[Depends(require_perms(["decision.read"]))])
    def decision_api(req: DecisionRequest) -> Dict[str, Any]:
        return _decision_impl(req)

def _decision_batch_impl(req: DecisionBatchRequest) -> Dict[str, Any]:
    out = []
    for jd in req.job_descs:
        res = three_stage_filter(jd, req.custom_rules or [], req.top_k)
        rec = decision_recommend(jd, res)
        get_logger("api").info(f"decision_batch jd_len={len(jd)} total={len(res)} picks={len(rec.get('recommended', []))}")
        out.append({"job_desc": jd, "results": res, "decision": rec})
    return {"items": out}

if os.getenv("ALLOW_PUBLIC_DECISION", "0") == "1":
    @app.post("/decision_batch")
    def decision_batch_api(req: DecisionBatchRequest) -> Dict[str, Any]:
        return _decision_batch_impl(req)
else:
    @app.post("/decision_batch", dependencies=[Depends(require_perms(["decision.read"]))])
    def decision_batch_api(req: DecisionBatchRequest) -> Dict[str, Any]:
        return _decision_batch_impl(req)

@app.get("/config/industry_templates", dependencies=[Depends(require_perms(["templates.read"]))])
def get_industry_templates_api() -> Dict[str, Any]:
    return {"templates": get_industry_templates()}

@app.put("/config/industry_templates", dependencies=[Depends(require_perms(["config.write"]))])
def update_industry_templates_api(req: IndustryTemplatesUpdate) -> Dict[str, Any]:
    ok = update_industry_templates(req.templates)
    return {"ok": ok}

@app.post("/ingest_online", dependencies=[Depends(require_perms(["ingest.write"]))])
def ingest_online_api(req: OnlineIngestRequest) -> Dict[str, Any]:
    items = online_ingest(req.urls)
    return {"count": len(items), "items": items}

@app.post("/ingest_bosszhipin", dependencies=[Depends(require_perms(["ingest.write"]))])
def ingest_bosszhipin_api(req: BossZhipinRequest) -> Dict[str, Any]:
    items = fetch_bosszhipin(req.urls, req.cookie)
    return {"count": len(items), "items": items}

@app.post("/ingest_zhilian", dependencies=[Depends(require_perms(["ingest.write"]))])
def ingest_zhilian_api(req: ZhiLianRequest) -> Dict[str, Any]:
    items = fetch_zhilian(req.urls, req.cookie)
    return {"count": len(items), "items": items}

@app.post("/ingest_liepin", dependencies=[Depends(require_perms(["ingest.write"]))])
def ingest_liepin_api(req: LiePinRequest) -> Dict[str, Any]:
    items = fetch_liepin(req.urls, req.cookie)
    return {"count": len(items), "items": items}

@app.post("/ingest_linkedin", dependencies=[Depends(require_perms(["ingest.write"]))])
def ingest_linkedin_api(req: LinkedInRequest) -> Dict[str, Any]:
    items = fetch_linkedin(req.urls, req.cookie)
    return {"count": len(items), "items": items}

if os.getenv("ALLOW_PUBLIC_CONFIG", "0") == "1":
    @app.post("/config/llm_enabled")
    def set_llm_enabled_api(req: LLMEnabled) -> Dict[str, Any]:
        set_llm_enabled(bool(req.enabled))
        return {"ok": True, "enabled": bool(req.enabled)}
else:
    @app.post("/config/llm_enabled", dependencies=[Depends(require_perms(["config.write"]))])
    def set_llm_enabled_api(req: LLMEnabled) -> Dict[str, Any]:
        set_llm_enabled(bool(req.enabled))
        return {"ok": True, "enabled": bool(req.enabled)}

@app.post("/jd_generate")
def jd_generate_api(req: JDGenerateRequest) -> Dict[str, Any]:
    obj = generate_structured_jd(req.text)
    return {"jd": obj}

@app.post("/resume_optimize")
def resume_optimize_api(req: ResumeOptimizeRequest) -> Dict[str, Any]:
    obj = optimize_resume(req.text)
    return obj

@app.post("/interview_questions")
def interview_questions_api(req: InterviewQuestionsRequest) -> Dict[str, Any]:
    qs = llm_generate_questions(req.job_desc or "", req.resume_text, req.provider, req.model, int(req.top_n or 3))
    return {"questions": qs}

@app.post("/evaluation_report")
def evaluation_report_api(req: MatchRequest) -> Dict[str, Any]:
    txt = llm_generate_analysis(req.job_text, req.resume_text)
    return {"report": txt}

if os.getenv("ALLOW_PUBLIC_CONFIG", "0") == "1":
    @app.post("/config/llm_settings")
    def set_llm_settings(req: LLMSettings) -> Dict[str, Any]:
        import json, os
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
        path = os.path.join(base_dir, "llm.json")
        os.makedirs(base_dir, exist_ok=True)
        data = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        sect = data.get(req.provider.lower(), {})
        if req.api_url is not None:
            sect["api_url"] = req.api_url
        if req.api_key is not None:
            sect["api_key"] = req.api_key
        if req.model is not None:
            sect["model"] = req.model
        data[req.provider.lower()] = sect
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
        return {"ok": True, "provider": req.provider.lower(), "saved": sect}
else:
    @app.post("/config/llm_settings", dependencies=[Depends(require_perms(["config.write"]))])
    def set_llm_settings(req: LLMSettings) -> Dict[str, Any]:
        import json, os
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
        path = os.path.join(base_dir, "llm.json")
        os.makedirs(base_dir, exist_ok=True)
        data = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        sect = data.get(req.provider.lower(), {})
        if req.api_url is not None:
            sect["api_url"] = req.api_url
        if req.api_key is not None:
            sect["api_key"] = req.api_key
        if req.model is not None:
            sect["model"] = req.model
        data[req.provider.lower()] = sect
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
        return {"ok": True, "provider": req.provider.lower(), "saved": sect}

def _recommend_jobs_impl(req: RecommendJobsRequest) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    if (req.data_source or "dir").lower() == "sqlite":
        import sqlite3
        sp = req.sqlite_path or os.getenv("RECOMMEND_JOBS_SQLITE", "")
        if not sp or not os.path.isfile(sp):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        conn = sqlite3.connect(sp)
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, text FROM jobs")
            for jid, txt in cur.fetchall():
                items.append({"id": str(jid), "text": str(txt or "")})
        finally:
            conn.close()
    else:
        jobs_dir = req.jobs_dir or os.getenv("RECOMMEND_JOBS_DIR", os.path.join("data", "raw_jobs"))
        if not os.path.isdir(jobs_dir):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        for name in os.listdir(jobs_dir):
            low = name.lower()
            if not low.endswith((".txt", ".md")):
                continue
            p = os.path.join(jobs_dir, name)
            if not os.path.isfile(p):
                continue
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    jt = f.read()
            except Exception:
                jt = ""
            if not jt:
                continue
            items.append({"id": name, "text": jt})
    if items:
        job_index_upsert(items)
    candidates = job_vector_search_by_text(req.resume_text, top_k=max(1, int(req.top_k) * 3))
    ind = (req.industry or "").strip().lower()
    reg = (req.region or "").strip().lower()
    sal_min = float(req.salary_min) if req.salary_min is not None else None
    sal_max = float(req.salary_max) if req.salary_max is not None else None
    def _salary_hint(t: str) -> float | None:
        import re
        m = re.findall(r"(\d+[\.]?\d*)\s*[kK]?", t)
        if not m:
            return None
        try:
            v = float(m[0])
            if "k" in t.lower():
                v = v * 1000.0
            return v
        except Exception:
            return None
    rows = []
    for c in candidates:
        txt = c.get("text", "")
        low = txt.lower()
        if ind and ind not in low:
            continue
        if reg and reg not in low:
            continue
        s_hint = _salary_hint(txt)
        if sal_min is not None and (s_hint is None or s_hint < sal_min):
            continue
        if sal_max is not None and (s_hint is None or s_hint > sal_max):
            continue
        res = quick_match(req.resume_text, txt)
        score = float(res.get("score", 0.0))
        sim = float(base_similarity(req.resume_text, txt))
        alpha = float(req.fusion_weight or 0.7)
        score = alpha * score + (1.0 - alpha) * sim
        rows.append({"id": c.get("id"), "text": txt, "score": score, "details": res.get("details", {})})
    rows = sorted(rows, key=lambda x: x.get("score", 0.0), reverse=True)
    top = rows[: max(1, int(req.top_k))]
    return {"items": [{"id": it.get("id"), "score": it.get("score"), "file": it.get("id")} for it in top]}

if os.getenv("ALLOW_PUBLIC_RECOMMEND", "0") == "1":
    @app.post("/recommend_jobs")
    def recommend_jobs_api(req: RecommendJobsRequest) -> Dict[str, Any]:
        return _recommend_jobs_impl(req)
else:
    @app.post("/recommend_jobs", dependencies=[Depends(require_perms(["match.read"]))])
    def recommend_jobs_api(req: RecommendJobsRequest) -> Dict[str, Any]:
        return _recommend_jobs_impl(req)

TRAINING_RESUMES_JSON = os.getenv("TRAINING_RESUMES_JSON", os.path.join("data", "processed", "resumes_for_annotation.json"))
UPLOADS_DIR = os.getenv("UPLOADS_DIR", os.path.join("data", "uploads"))


@app.post("/uploads", dependencies=[Depends(require_perms(["uploads.write"]))])
async def uploads_api(file: UploadFile | None = File(None), text: str | None = Form(None), filename: str | None = Form(None)) -> Dict[str, Any]:
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    import time
    ts = int(time.time() * 1000)
    if file is not None:
        name = filename or file.filename or f"upload_{ts}"
        safe = name.replace("..", "_")
        path = os.path.join(UPLOADS_DIR, safe)
        data = await file.read()
        with open(path, "wb") as f:
            f.write(data)
        return {"ok": True, "path": path, "size": len(data)}
    if text is not None and text.strip():
        name = filename or f"text_{ts}.txt"
        safe = name.replace("..", "_")
        path = os.path.join(UPLOADS_DIR, safe)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return {"ok": True, "path": path, "size": len(text.encode("utf-8"))}
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

parse_word = None
parse_pdf = None
try:
    parse_word = _load_func('02_data_preprocess.py', 'parse_word')
except Exception:
    parse_word = None
try:
    parse_pdf = _load_func('02_data_preprocess.py', 'parse_pdf')
except Exception:
    parse_pdf = None

@app.post("/vector_index_warmup", dependencies=[Depends(require_perms(["index.write"]))])
def vector_index_warmup() -> Dict[str, Any]:
    if not os.path.isfile(TRAINING_RESUMES_JSON):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    import json
    with open(TRAINING_RESUMES_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = [{"id": str(i), "text": d.get("text", "")} for i, d in enumerate(data)]
    count = vector_index_upsert(items)
    return {"count": count.get("count", 0)}

class UploadsIndexRequest(BaseModel):
    hours: int | None = None
    name_pattern: str | None = None

@app.post("/uploads_index", dependencies=[Depends(require_perms(["index.write"]))])
def uploads_index(req: UploadsIndexRequest | None = None) -> Dict[str, Any]:
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    items = []
    import re, time
    now = time.time()
    hours = int(req.hours) if req and req.hours else None
    pattern = str(req.name_pattern) if req and req.name_pattern else None
    reg = re.compile(pattern) if pattern else None
    for name in os.listdir(UPLOADS_DIR):
        p = os.path.join(UPLOADS_DIR, name)
        if not os.path.isfile(p):
            continue
        if reg and not reg.search(name):
            continue
        if hours is not None:
            try:
                mtime = os.path.getmtime(p)
                if (now - mtime) > (hours * 3600):
                    continue
            except Exception:
                pass
        low = name.lower()
        txt = ""
        try:
            if low.endswith(('.txt', '.md')):
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    txt = f.read()
            elif low.endswith('.docx') and parse_word:
                txt = parse_word(p)
            elif low.endswith('.pdf') and parse_pdf:
                txt = parse_pdf(p)
        except Exception:
            txt = ""
        if txt:
            items.append({"id": name, "text": txt})
    if not items:
        return {"count": 0}
    count = vector_index_upsert(items)
    return {"count": count.get("count", 0)}

@app.post("/auth/token")
def issue_token(req: TokenRequest) -> Dict[str, Any]:
    users = _users_from_env_or_file()
    if not users:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED)
    u = users.get(req.username)
    if not u or u.get("password") != req.password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    payload = {"sub": req.username, "roles": u.get("roles", []), "permissions": u.get("permissions", [])}
    tok = _jwt_encode(payload)
    return {"access_token": tok, "token_type": "bearer", "roles": u.get("roles", []), "permissions": u.get("permissions", [])}

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    # 优先使用环境变量端口，否则默认从 8000 开始尝试回退
    env_port = os.getenv("API_PORT")

    def is_port_free(p: int) -> bool:
        # 通过尝试绑定 0.0.0.0:port 判断端口是否可用（更可靠）
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return True
            except OSError:
                return False

    def choose_port(h: str) -> int:
        if env_port:
            try:
                p = int(env_port)
                return p
            except Exception:
                pass
        # 依次尝试 8000-8005，找到第一个可用端口
        for p in range(8000, 8006):
            if is_port_free(p):
                return p
        # 若都不可用，最终仍返回 8000（交由 uvicorn 报错）
        return 8000

    port = choose_port(host)
    print(f"Starting API on {host}:{port} (env API_PORT={env_port})")
    static_dir = os.path.join(PROJECT_ROOT, "static", "admin")
    disable_admin_ui = os.getenv("DISABLE_ADMIN_UI", "0") == "1"
    if not disable_admin_ui and os.path.isdir(static_dir):
        app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")
        ADMIN_UI_ENABLED = True
        print(f"Management UI: http://127.0.0.1:{port}/ui")
    uvicorn.run(app, host=host, port=port)
