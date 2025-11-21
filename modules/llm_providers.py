import os
import json
import time
import pathlib

def _local_questions(job_desc: str, resume_text: str, top_n: int = 3) -> list[str]:
    qs = []
    base = resume_text or ""
    if "python" in base.lower():
        qs.append("请结合项目经历说明你最熟悉的Python生态组件，并举例性能优化案例")
    if "java" in base.lower():
        qs.append("请描述你在Java服务中的线程池与连接池调优经验")
    if "docker" in base.lower() or "k8s" in base.lower():
        qs.append("请说明一次容器化上线的完整流程，并阐述出现故障的定位方法")
    qs.append("请结合该岗位要求，说明你最能胜任的两项能力及其验证方式")
    return qs[:max(1, int(top_n))]

def _local_analysis(job_desc: str, resume_text: str) -> str:
    s = []
    if job_desc:
        s.append("岗位要求已提取，匹配度以技能、学历、年限、职位四维为主")
    base = resume_text or ""
    if any(k in base.lower() for k in ["python","java","go","docker","k8s"]):
        s.append("优势：核心技术词命中，具备相关项目经历")
    s.append("风险：请补充量化成果与关键指标以提升说服力")
    return "；".join(s)

def _qwen_call(prompt: str, api_key: str, model: str) -> str:
    try:
        import requests
        url = os.getenv("QWEN_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {"model": model or os.getenv("QWEN_MODEL", "qwen2.5-instruct"), "messages": [{"role": "user", "content": prompt}]}
        r = requests.post(url, headers=headers, data=json.dumps(data), timeout=12)
        if r.status_code == 200:
            obj = r.json()
            c = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
            return c or ""
        return ""
    except Exception:
        return ""

def _read_llm_config() -> dict:
    try:
        cfg_path = pathlib.Path(__file__).resolve().parent.parent / "config" / "llm.json"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _openai_compat_call(prompt: str, api_url: str, api_key: str, model: str) -> str:
    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        r = requests.post(api_url, headers=headers, data=json.dumps(data), timeout=12)
        if r.status_code == 200:
            obj = r.json()
            c = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
            return c or ""
        return ""
    except Exception:
        return ""

def _gemini_call(prompt: str, api_key: str, model: str) -> str:
    try:
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        r = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data), timeout=12)
        if r.status_code == 200:
            obj = r.json()
            candidates = obj.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
        return ""
    except Exception:
        return ""

def _tavily_questions(job_desc: str, resume_text: str, api_key: str, top_n: int) -> list[str]:
    try:
        import requests
        q = (job_desc or "")[:200] + "\n" + (resume_text or "")[:200]
        r = requests.post("https://api.tavily.com/search", headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}, data=json.dumps({"query": q, "max_results": max(3, top_n)}), timeout=10)
        if r.status_code == 200:
            obj = r.json()
            items = obj.get("results", [])
            qs = []
            for it in items[:max(1, top_n)]:
                t = str(it.get("title", "")).strip()
                if t:
                    qs.append(f"结合你的经历，谈谈：{t}")
            if qs:
                return qs[:max(1, top_n)]
        return []
    except Exception:
        return []

def generate_questions(job_desc: str, resume_text: str, provider: str | None = None, model: str | None = None, top_n: int = 3) -> list[str]:
    prov = (provider or os.getenv("LLM_PROVIDER", "local")).lower()
    cfg = _read_llm_config()
    if prov == "qwen":
        ak = (cfg.get("qwen", {}).get("api_key") or os.getenv("QWEN_API_KEY", ""))
        mo = model or cfg.get("qwen", {}).get("model") or os.getenv("QWEN_MODEL", "qwen2.5-instruct")
        c = _qwen_call(f"基于如下岗位与简历生成{max(1,int(top_n))}个针对候选人的面试题。岗位：{job_desc}\n简历：{resume_text}", ak, mo)
        if c:
            lines = [x.strip("- • \n") for x in c.split("\n") if x.strip()]
            return lines[:max(1, int(top_n))]
    if prov in {"openai", "ark", "volc", "hunyuan", "qianfan", "kimi"}:
        sect = cfg.get(prov, {})
        url = sect.get("api_url") or os.getenv(f"{prov.upper()}_API_URL", "")
        ak = sect.get("api_key") or os.getenv(f"{prov.upper()}_API_KEY", "")
        mo = model or sect.get("model") or os.getenv(f"{prov.upper()}_MODEL", "")
        if url and ak and mo:
            c = _openai_compat_call(f"基于如下岗位与简历生成{max(1,int(top_n))}个针对候选人的面试题。岗位：{job_desc}\n简历：{resume_text}", url, ak, mo)
            if c:
                lines = [x.strip("- • \n") for x in c.split("\n") if x.strip()]
                return lines[:max(1, int(top_n))]
    if prov == "gemini":
        sect = cfg.get("gemini", {})
        ak = sect.get("api_key") or os.getenv("GEMINI_API_KEY", "")
        mo = model or sect.get("model") or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        if ak and mo:
            c = _gemini_call(f"基于如下岗位与简历生成{max(1,int(top_n))}个针对候选人的面试题。岗位：{job_desc}\n简历：{resume_text}", ak, mo)
            if c:
                lines = [x.strip("- • \n") for x in c.split("\n") if x.strip()]
                return lines[:max(1, int(top_n))]
    if prov == "tavily":
        sect = cfg.get("tavily", {})
        ak = sect.get("api_key") or os.getenv("TAVILY_API_KEY", "")
        if ak:
            qs = _tavily_questions(job_desc, resume_text, ak, max(1, int(top_n)))
            if qs:
                return qs[:max(1, int(top_n))]
    return _local_questions(job_desc, resume_text, top_n)

def generate_analysis(job_desc: str, resume_text: str, provider: str | None = None, model: str | None = None) -> str:
    prov = (provider or os.getenv("LLM_PROVIDER", "local")).lower()
    cfg = _read_llm_config()
    if prov == "qwen":
        ak = (cfg.get("qwen", {}).get("api_key") or os.getenv("QWEN_API_KEY", ""))
        mo = model or cfg.get("qwen", {}).get("model") or os.getenv("QWEN_MODEL", "qwen2.5-instruct")
        c = _qwen_call(f"基于岗位与简历，输出一段候选人的优势与风险分析。岗位：{job_desc}\n简历：{resume_text}", ak, mo)
        if c:
            return c
    if prov in {"openai", "ark", "volc", "hunyuan", "qianfan", "kimi"}:
        sect = cfg.get(prov, {})
        url = sect.get("api_url") or os.getenv(f"{prov.upper()}_API_URL", "")
        ak = sect.get("api_key") or os.getenv(f"{prov.upper()}_API_KEY", "")
        mo = model or sect.get("model") or os.getenv(f"{prov.upper()}_MODEL", "")
        if url and ak and mo:
            c = _openai_compat_call(f"基于岗位与简历，输出一段候选人的优势与风险分析。岗位：{job_desc}\n简历：{resume_text}", url, ak, mo)
            if c:
                return c
    if prov == "gemini":
        sect = cfg.get("gemini", {})
        ak = sect.get("api_key") or os.getenv("GEMINI_API_KEY", "")
        mo = model or sect.get("model") or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        if ak and mo:
            c = _gemini_call(f"基于岗位与简历，输出一段候选人的优势与风险分析。岗位：{job_desc}\n简历：{resume_text}", ak, mo)
            if c:
                return c
    if prov == "tavily":
        return _local_analysis(job_desc, resume_text)
    return _local_analysis(job_desc, resume_text)