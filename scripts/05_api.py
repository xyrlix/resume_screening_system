from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import os
import importlib.util
import socket

# 动态加载同目录下的脚本：04_predict.py 与 04_matcher_model.py
BASE_DIR = os.path.dirname(__file__)

def _load_func(module_filename: str, func_name: str):
    path = os.path.join(BASE_DIR, module_filename)
    spec = importlib.util.spec_from_file_location(module_filename.replace('.py',''), path)
    mod = importlib.util.module_from_spec(spec)
    if spec and spec.loader:
        spec.loader.exec_module(mod)
    return getattr(mod, func_name)

ner_predict = _load_func('04_predict.py', 'predict')
quick_match = _load_func('04_matcher_model.py', 'quick_match')


app = FastAPI(title="Resume Screening API", version="0.1.0")


class PredictRequest(BaseModel):
    text: str


class MatchRequest(BaseModel):
    resume_text: str
    job_text: str


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
    ents = ner_predict(req.text)
    return {"entities": ents}


@app.post("/match")
def match_api(req: MatchRequest) -> Dict[str, Any]:
    res = quick_match(req.resume_text, req.job_text)
    return res


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