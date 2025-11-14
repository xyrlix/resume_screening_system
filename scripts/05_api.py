from fastapi import FastAPI
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
quick_match = _load_func('04_matcher_model.py', 'quick_match')
vector_index_upsert = _load_func_project('modules/funnel_filter.py', 'vector_index_upsert')
three_stage_filter = _load_func_project('modules/funnel_filter.py', 'three_stage_filter')
decision_recommend = _load_func_project('modules/decision_support.py', 'recommend')
get_industry_templates = _load_func_project('modules/llm_utils.py', 'get_industry_templates')
update_industry_templates = _load_func_project('modules/llm_utils.py', 'update_industry_templates')
online_ingest = _load_func_project('modules/online_ingest.py', 'ingest_urls')
extract_en_entities = _load_func_project('modules/en_entities.py', 'extract_en_entities')
fetch_bosszhipin = _load_func_project('modules/adapters/bosszhipin.py', 'fetch_bosszhipin')


app = FastAPI(title="Resume Screening API", version="0.1.0")


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

class PredictEnRequest(BaseModel):
    text: str

class BossZhipinRequest(BaseModel):
    urls: List[str]
    cookie: str


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "Resume Screening API is running",
        "docs": "/docs",
        "endpoints": ["GET /health", "POST /predict", "POST /match"],
    }


@app.post("/predict")
def predict_api(req: PredictRequest) -> Dict[str, Any]:
    _ensure_ner()
    ents = ner_predict(req.text)
    get_logger("api").info(f"predict len={len(req.text)} ents={len(ents)}")
    return {"entities": ents}


@app.post("/match")
def match_api(req: MatchRequest) -> Dict[str, Any]:
    res = quick_match(req.resume_text, req.job_text)
    get_logger("api").info(f"match score={res.get('score')}")
    return res

@app.post("/predict_en")
def predict_en_api(req: PredictEnRequest) -> Dict[str, Any]:
    ents = extract_en_entities(req.text)
    return {"entities": ents}

@app.post("/vector_index")
def vector_index_api(req: VectorIndexRequest) -> Dict[str, Any]:
    count = vector_index_upsert([{"id": it.id, "text": it.text} for it in req.items])
    get_logger("api").info(f"vector_index count={count.get('count')}")
    return count

@app.post("/filter")
def filter_api(req: FilterRequest) -> Dict[str, Any]:
    res = three_stage_filter(req.job_desc, req.custom_rules or [], req.top_k)
    get_logger("api").info(f"filter results={len(res)}")
    return {"results": res}

@app.post("/decision")
def decision_api(req: DecisionRequest) -> Dict[str, Any]:
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

@app.post("/decision_batch")
def decision_batch_api(req: DecisionBatchRequest) -> Dict[str, Any]:
    out = []
    for jd in req.job_descs:
        res = three_stage_filter(jd, req.custom_rules or [], req.top_k)
        rec = decision_recommend(jd, res)
        get_logger("api").info(f"decision_batch jd_len={len(jd)} total={len(res)} picks={len(rec.get('recommended', []))}")
        out.append({"job_desc": jd, "results": res, "decision": rec})
    return {"items": out}

@app.get("/config/industry_templates")
def get_industry_templates_api() -> Dict[str, Any]:
    return {"templates": get_industry_templates()}

@app.put("/config/industry_templates")
def update_industry_templates_api(req: IndustryTemplatesUpdate) -> Dict[str, Any]:
    ok = update_industry_templates(req.templates)
    return {"ok": ok}

@app.post("/ingest_online")
def ingest_online_api(req: OnlineIngestRequest) -> Dict[str, Any]:
    items = online_ingest(req.urls)
    return {"count": len(items), "items": items}

@app.post("/ingest_bosszhipin")
def ingest_bosszhipin_api(req: BossZhipinRequest) -> Dict[str, Any]:
    items = fetch_bosszhipin(req.urls, req.cookie)
    return {"count": len(items), "items": items}


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
    uvicorn.run(app, host=host, port=port)
def _ensure_ner():
    global ner_predict
    if ner_predict is None:
        try:
            ner_predict = _load_func('04_predict.py', 'predict')
        except Exception:
            def _fallback(text: str):
                return []
            ner_predict = _fallback