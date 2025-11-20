from typing import List, Dict, Any
import time
import os
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    _bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    _tok = AutoTokenizer.from_pretrained("alibaba/Qwen-1_8B-Chat", trust_remote_code=True)
    _mdl = AutoModelForCausalLM.from_pretrained("alibaba/Qwen-1_8B-Chat", trust_remote_code=True, quantization_config=_bnb, device_map="auto").eval()
except Exception:
    _tok = None
    _mdl = None

_demands_cache: Dict[str, tuple] = {}
_qs_cache: Dict[str, tuple] = {}
_use_llm = os.getenv("ENABLE_LLM_GEN", "0") == "1"
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
    k = "base::" + ",".join(skills)
    now = time.time()
    ent = _qs_cache.get(k)
    if ent and now - ent[1] < 3600:
        return ent[0]
    prompt = (
        "你是资深技术面试官。基于候选人的技能与经历，生成3个高质量、可追问的面试问题，"
        "要求：问题要具体，引用候选人提到的技术关键词，至少包含1个业务结果量化追问。"
        f"技能关键词：{', '.join(skills)}。"
    )
    if _mdl and _tok and _use_llm:
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
                _qs_cache[k] = (qs[:3], now)
                return qs[:3]
        except Exception:
            pass
    base = [
        f"请举例说明你在项目中用到 {skills[0] if skills else '关键技术'} 的场景与挑战，你如何解决？",
        "针对你提到的一个项目，请量化你的贡献（如性能提升比例、成本下降、用户增长等）并说明验证方式",
        "当系统出现性能瓶颈时，你的定位思路是什么？如何选择优化手段并衡量效果"
    ]
    _qs_cache[k] = (base[:3], now)
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
    k = "ctx::" + role + "::" + ind + "::" + ",".join(skills)
    now = time.time()
    ent = _qs_cache.get(k)
    if ent and now - ent[1] < 3600:
        return ent[0]
    prompt = (
        role_prompt + ind_prompt +
        "基于候选人技能与经历，生成3个高质量、可追问的面试问题，引用候选人技术关键词，至少包含1个业务结果量化追问。"
        f"技能关键词：{', '.join(skills)}。"
    )
    if _mdl and _tok and _use_llm:
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
                _qs_cache[k] = (qs[:3], now)
                return qs[:3]
        except Exception:
            pass
    res = generate_interview_questions(text)
    _qs_cache[k] = (res, now)
    return res

def set_llm_enabled(flag: bool) -> bool:
    global _use_llm
    _use_llm = bool(flag)
    return _use_llm

def generate_structured_jd(text: str) -> Dict[str, Any]:
    import re
    jd = (text or "")
    title = None
    m = re.search(r"(?:岗位|职位|岗位名称|职位名称)[:：]?\s*(.+)", jd)
    if m:
        title = m.group(1).strip()
    if not title:
        m2 = re.search(r"^(.*?工程师|.*?开发|.*?产品|.*?算法|.*?数据|.*?测试|.*?运维|.*?经理|.*?主管)", jd)
        title = m2.group(1).strip() if m2 else "未命名岗位"
    duties = []
    reqs = []
    for line in [x.strip() for x in re.split(r"[\n\r]+", jd) if x.strip()]:
        if any(k in line for k in ["负责", "参与", "搭建", "维护", "优化", "推进", "协作", "设计", "实施", "统筹", "跟进", "交付"]):
            duties.append(line)
        if any(k in line for k in ["熟悉", "精通", "掌握", "具备", "了解", "至少", "本科", "硕士", "博士", "经验", "能力", "要求", "资格"]):
            reqs.append(line)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#\.\-]{1,}", jd)
    tokens = [t.lower() for t in tokens]
    skills = list({t for t in tokens})[:30]
    frameworks = [t for t in tokens if t in {"spring","springboot","django","flask","fastapi","vue","react","angular","rmq","kafka","spark","hadoop"}]
    tools = [t for t in tokens if t in {"docker","k8s","kubernetes","git","jenkins","maven","gradle","terraform"}]
    languages = [t for t in tokens if t in {"java","python","go","rust","c++","c#","js","ts","sql"}]
    certs = re.findall(r"(PMP|CPA|CFA|ACP|RHCE)", jd, flags=re.I)
    keywords = list({t for t in tokens if t not in frameworks + tools + languages})[:30]
    deg_m = re.search(r"(博士|硕士|本科|大专)", jd)
    degree_required = deg_m.group(1) if deg_m else ""
    years_m = re.findall(r"(\d+)\s*年", jd)
    min_years = None
    if years_m:
        try:
            min_years = min(int(x) for x in years_m)
        except Exception:
            min_years = None
    sal = re.findall(r"(\d+[\.]?\d*)\s*[kK]?\s*[-~到]\s*(\d+[\.]?\d*)\s*[kK]?", jd)
    salary_min = None
    salary_max = None
    if sal:
        try:
            a, b = sal[0]
            salary_min = float(a) * (1000.0 if re.search(r"[kK]", jd) else 1.0)
            salary_max = float(b) * (1000.0 if re.search(r"[kK]", jd) else 1.0)
        except Exception:
            pass
    emp_type_m = re.search(r"(全职|兼职|实习|合同)", jd)
    employment_type = emp_type_m.group(1) if emp_type_m else ""
    loc_m = re.findall(r"(北京|上海|广州|深圳|杭州|南京|成都|武汉|西安|苏州|天津|重庆|合肥|厦门|沈阳|大连|无锡|佛山|宁波|青岛)", jd)
    location = list({x for x in loc_m})
    industry_m = re.findall(r"(互联网|医疗|教育|金融|制造|物流|零售|地产|能源|汽车)", jd)
    industry = list({x for x in industry_m})
    soft_skills = []
    for s in ["沟通","协作","学习","责任","抗压","逻辑","创新"]:
        if s in jd:
            soft_skills.append(s)
    return {
        "title": title,
        "duties": duties[:20],
        "requirements": reqs[:20],
        "skills": skills,
        "frameworks": frameworks,
        "tools": tools,
        "languages": languages,
        "certifications": certs,
        "keywords": keywords,
        "degree_required": degree_required,
        "min_years": min_years,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "employment_type": employment_type,
        "location": location,
        "industry": industry,
        "soft_skills": soft_skills,
    }

def optimize_resume(text: str) -> Dict[str, Dict]:
    import re
    t = (text or "")
    feats = {
        "skills": list({s.lower() for s in re.findall(r"[A-Za-z+#\.\-]{2,}", t)})[:20],
        "years": 0,
        "degree": "",
        "positions": []
    }
    ym = re.findall(r"(\d+)\s*年", t)
    if ym:
        try:
            feats["years"] = max(int(x) for x in ym)
        except Exception:
            feats["years"] = 0
    deg = re.findall(r"博士|硕士|本科|大专", t)
    feats["degree"] = deg[0] if deg else ""
    feats["positions"] = [p for p in re.findall(r"工程师|开发|数据|算法|产品|测试|运维|架构|经理|主管", t)][:10]
    advice = []
    if len(feats["skills"]) < 6:
        advice.append("补充核心技能关键词，突出与目标岗位相关的工具与框架")
    if feats["years"] == 0:
        advice.append("明确总工作年限并量化项目结果（性能提升、成本下降、用户增长等）")
    if not feats["degree"]:
        advice.append("补充最高学历与毕业年份、学校、专业")
    advice.append("为每段经历添加可量化成果与验证方式，避免泛化描述")
    return {"profile": feats, "advice": advice}

def generate_evaluation_report(job_desc: str, resume_text: str) -> Dict[str, str]:
    jd = (job_desc or "")
    rt = (resume_text or "")
    import re
    skills_j = list({s.lower() for s in re.findall(r"[A-Za-z+#\.\-]{2,}", jd)})[:12]
    skills_r = list({s.lower() for s in re.findall(r"[A-Za-z+#\.\-]{2,}", rt)})
    hit = len(set(skills_j) & set(skills_r))
    advs = []
    if hit < max(3, len(skills_j)//3 or 1):
        advs.append("加强岗位核心技能的覆盖与案例描述")
    ym = re.findall(r"(\d+)\s*年", rt)
    if not ym:
        advs.append("补充工作年限并量化关键成果")
    deg = re.findall(r"博士|硕士|本科|大专", rt)
    if not deg:
        advs.append("补充最高学历与院校信息")
    text = (
        f"匹配度分析：命中技能 {hit} 项；建议：" + ("；".join(advs) if advs else "保持现有结构，进一步量化成果与影响")
    )
    return {"report": text}