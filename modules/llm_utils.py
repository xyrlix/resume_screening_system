from typing import List, Dict
import time
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    _bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    _tok = AutoTokenizer.from_pretrained("alibaba/Qwen-1_8B-Chat", trust_remote_code=True)
    _mdl = AutoModelForCausalLM.from_pretrained("alibaba/Qwen-1_8B-Chat", trust_remote_code=True, quantization_config=_bnb, device_map="auto").eval()
except Exception:
    _tok = None
    _mdl = None

_demands_cache: Dict[str, tuple] = {}
_role_templates = {
    "算法": "你是算法面试官，关注建模方法、评估指标与线上效果复盘。",
    "后端": "你是后端面试官，关注架构设计、性能与稳定性、故障定位与治理。",
    "数据": "你是数据面试官，关注指标口径、数据质量、业务洞察与增长分析。",
}
_industry_templates = {
    "电商": "结合转化率、复购率、客单价与拉新/留存的指标提出问题。",
    "金融": "结合风控、合规、延迟与稳定性要求提出问题。",
    "广告": "结合CTR/CVR、召回与排序、预算与投放ROI提出问题。",
    "医疗": "结合数据隐私、合规与准确性提出问题。",
}

def _load_industry_templates():
    import os, json
    try:
        cfg = os.getenv("INDUSTRY_TEMPLATES_PATH") or os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "industry_templates.json")
        if os.path.isfile(cfg):
            with open(cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return None

_loaded = _load_industry_templates()
if _loaded:
    _industry_templates = _loaded

def get_industry_templates() -> Dict[str, str]:
    return dict(_industry_templates)

def update_industry_templates(data: Dict[str, str]) -> bool:
    import json, os
    cfg = os.getenv("INDUSTRY_TEMPLATES_PATH") or os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "industry_templates.json")
    try:
        with open(cfg, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        global _industry_templates
        _industry_templates = dict(data)
        return True
    except Exception:
        return False

def infer_implicit_demands(job_desc: str) -> List[str]:
    now = time.time()
    ent = _demands_cache.get(job_desc)
    if ent and now - ent[1] < 3600:
        return ent[0]
    keys = ["数据", "算法", "项目", "交付", "协作", "优化", "指标"]
    res = [k for k in keys if k in (job_desc or "")]
    _demands_cache[job_desc] = (res, now)
    return res

def implicit_match_score(text: str, demands: List[str]) -> float:
    t = (text or "").lower()
    hits = 0
    for d in demands:
        if str(d).lower() in t:
            hits += 1
    if len(demands) == 0:
        return 0.5
    return float(hits) / float(len(demands))

def generate_interview_questions(text: str) -> List[str]:
    skills = []
    import re
    skills += re.findall(r"[A-Za-z+#\.\-]{2,}", (text or ""))
    skills = list({s.lower() for s in skills})[:6]
    prompt = (
        "你是资深技术面试官。基于候选人的技能与经历，生成3个高质量、可追问的面试问题，"
        "要求：问题要具体，引用候选人提到的技术关键词，至少包含1个业务结果量化追问。"
        f"技能关键词：{', '.join(skills)}。"
    )
    if _mdl and _tok:
        try:
            inp = _tok(prompt, return_tensors="pt")
            out = _mdl.generate(**inp, max_new_tokens=128, do_sample=False)
            txt = _tok.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            parts = [p.strip() for p in re.split(r"[\n\r]+", txt) if len(p.strip()) > 0]
            qs = []
            for p in parts:
                if len(qs) >= 3:
                    break
                if any(x in p.lower() for x in ["如何", "请举例", "说明", "为什么", "如何验证", "如何量化"]):
                    qs.append(p)
            if qs:
                return qs[:3]
        except Exception:
            pass
    base = [
        f"请举例说明你在项目中用到 {skills[0] if skills else '关键技术'} 的场景与挑战，你如何解决？",
        "针对你提到的一个项目，请量化你的贡献（如性能提升比例、成本下降、用户增长等）并说明验证方式",
        "当系统出现性能瓶颈时，你的定位思路是什么？如何选择优化手段并衡量效果"
    ]
    return base[:3]

def generate_interview_questions_ctx(job_desc: str, text: str) -> List[str]:
    role = "后端"
    import re
    jd = (job_desc or "")
    if re.search(r"算法|模型|推荐|NLP|CV", jd):
        role = "算法"
    elif re.search(r"数据|BI|分析", jd):
        role = "数据"
    role_prompt = _role_templates.get(role, "")
    ind = ""
    for k in _industry_templates.keys():
        if k in jd:
            ind = k
            break
    ind_prompt = _industry_templates.get(ind, "")
    skills = []
    skills += re.findall(r"[A-Za-z+#\.\-]{2,}", (text or ""))
    skills = list({s.lower() for s in skills})[:6]
    prompt = (
        role_prompt + ind_prompt +
        "基于候选人技能与经历，生成3个高质量、可追问的面试问题，引用候选人技术关键词，至少包含1个业务结果量化追问。"
        f"技能关键词：{', '.join(skills)}。"
    )
    if _mdl and _tok:
        try:
            inp = _tok(prompt, return_tensors="pt")
            out = _mdl.generate(**inp, max_new_tokens=128, do_sample=False)
            txt = _tok.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            parts = [p.strip() for p in re.split(r"[\n\r]+", txt) if len(p.strip()) > 0]
            qs = []
            for p in parts:
                if len(qs) >= 3:
                    break
                if any(x in p.lower() for x in ["如何", "请举例", "说明", "为什么", "如何验证", "如何量化"]):
                    qs.append(p)
            if qs:
                return qs[:3]
        except Exception:
            pass
    return generate_interview_questions(text)