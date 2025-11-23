from typing import List, Dict, Any
import time
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
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
    "人工智能": "结合模型效果、算法优化与业务指标，关注数据质量与实验设计",
    "新能源": "结合电池技术、能源效率与环保标准，关注技术创新与成本控制",
    "半导体/芯片": "结合芯片设计、制造工艺与性能指标，关注技术难点与质量控制",
    "互联网": "结合用户增长、产品优化与市场竞争力，关注技术架构与用户体验",
    "电子商务": "结合转化率、复购率、客单价、拉新与留存的指标提出问题，并关注AB实验与ROI",
    "金融": "结合风控、合规、延迟与稳定性要求提出问题。",
    "医疗": "结合数据隐私、合规与准确性提出问题。",
    "教育": "结合学习效果评估、个性化推荐与运营指标，关注内容质量与留存",
    "制造": "结合产线数据采集、质量检测与成本控制，关注设备稳定性与预测维护",
    "游戏": "结合DAU/留存/付费与反作弊，关注实时检测与风控策略",
    "出行": "结合路径规划、ETA、供需匹配与高并发稳定性，关注异常处理与容灾",
    "物联网": "结合设备连接、数据采集与边缘计算，关注系统稳定性和安全性",
    "大数据": "结合数据处理、存储与分析，关注数据质量和处理效率",
    "云计算": "结合资源调度、弹性伸缩与成本优化，关注服务稳定性和性能",
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
    industry_m = re.findall(r"(人工智能|新能源|半导体|芯片|互联网|电子商务|金融|医疗|教育|制造|游戏|出行|物联网|大数据|云计算)", jd)
    industry = list({x for x in industry_m})
    if not industry:
        # 兼容旧的行业关键词
        old_industry_m = re.findall(r"(互联网|医疗|教育|金融|制造|物流|零售|地产|能源|汽车|电信|航空|旅游|媒体|政府|非营利组织)", jd)
        industry = list({x for x in old_industry_m})
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

def optimize_resume(text: str) -> Dict[str, Any]:
    import re
    t = (text or "")
    feats: Dict[str, Any] = {
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
    return {"profile": feats, "advice": advice}  # type: ignore

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

def extract_entities_with_llm(text: str) -> List[Dict[str, Any]]:
    """
    使用轻量级LLM提取简历中的实体信息
    提取的实体包括：姓名,性别,出生日期,年龄,联系电话,电子邮箱,现居地,户籍地,政治面貌,
    婚姻状况,期望职位,期望行业,期望工作城市,期望薪资,到岗时间,学校名称,学历层次,专业名称,
    入学时间,毕业时间,是否全日制,GPA/排名/荣誉,公司名称,公司所属行业,在职开始时间,在职结束时间,
    职位名称,工作地点,工作内容关键词,汇报对象,团队规模,项目名称,项目开始时间,项目结束时间,
    项目角色,技术栈,项目成果指标,编程语言,框架/工具,数据库,操作系统/平台,语言能力,证书资质,
    软技能,作品集/个人链接,自我评价关键词,兴趣爱好,总工作经验年限等。
    """
    # 如果有可用的LLM模型，则使用它进行实体提取
    if _mdl is not None and _tok is not None:
        try:
            # 构造提示词，让模型提取实体信息
            prompt = f"""
请从以下简历文本中提取实体信息。提取以下类型的实体：
姓名,性别,出生日期,年龄,联系电话,电子邮箱,现居地,户籍地,政治面貌,婚姻状况,期望职位,期望行业,
期望工作城市,期望薪资,到岗时间,学校名称,学历层次,专业名称,入学时间,毕业时间,是否全日制,
GPA/排名/荣誉,公司名称,公司所属行业,在职开始时间,在职结束时间,职位名称,工作地点,工作内容关键词,
汇报对象,团队规模,项目名称,项目开始时间,项目结束时间,项目角色,技术栈,项目成果指标,编程语言,
框架/工具,数据库,操作系统/平台,语言能力,证书资质,软技能,作品集/个人链接,自我评价关键词,兴趣爱好,总工作经验年限

简历文本：
{text}

请以JSON格式返回结果，格式如下：
[
    {{"type": "实体类型", "text": "实体文本", "start": 起始位置, "end": 结束位置}},
    ...
]
""".strip()

            # 对提示词进行编码
            inputs = _tok(prompt, return_tensors="pt")
            
            # 生成响应
            outputs = _mdl.generate(**inputs, max_new_tokens=1024, do_sample=False)
            
            # 解码响应
            response = _tok.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            # 提取JSON部分
            import json
            import re
            
            # 尝试找到JSON数组
            json_match = re.search(r'\[[^\]]*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    entities = json.loads(json_str)
                    if isinstance(entities, list):
                        return entities
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            # 如果LLM调用失败，继续使用规则方法
            pass
    
    # 如果没有LLM或LLM调用失败，使用规则方法提取实体
    entities = []
    
    # 提取邮箱
    import re
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    for email in emails:
        start = text.find(email)
        if start != -1:
            entities.append({"type": "电子邮箱", "text": email, "start": start, "end": start + len(email)})
    
    # 提取电话号码
    phones = re.findall(r'1[3-9]\d{9}', text)
    for phone in phones:
        start = text.find(phone)
        if start != -1:
            entities.append({"type": "联系电话", "text": phone, "start": start, "end": start + len(phone)})
    
    # 提取学历
    degrees = ["博士", "硕士", "本科", "大专"]
    for degree in degrees:
        start = text.find(degree)
        if start != -1:
            entities.append({"type": "学历层次", "text": degree, "start": start, "end": start + len(degree)})
    
    # 提取工作经验年限
    years = re.findall(r'(\d+)\s*年.*?经验', text)
    for year in years:
        start = text.find(year)
        if start != -1:
            entities.append({"type": "总工作经验年限", "text": f"{year}年", "start": start, "end": text.find("经验", start) + 2})
    
    # 提取技能（简单的关键词匹配）
    skills_keywords = ["Python", "Java", "C++", "JavaScript", "SQL", "React", "Vue", "Angular", 
                      "Django", "Spring", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Docker", 
                      "Kubernetes", "AWS", "Azure", "GCP"]
    
    for skill in skills_keywords:
        start = text.find(skill)
        if start != -1:
            entities.append({"type": "技能", "text": skill, "start": start, "end": start + len(skill)})
    
    return entities

def extract_format_features(text: str, layout_features: Dict[str, Any]) -> Dict[str, float]:
    """
    提取格式特征
    包括：排版工整度、成果量化程度等
    """
    features = {
        "排版工整度": 0.0,
        "成果量化程度": 0.0,
        "关键词密度": 0.0,
        "段落结构清晰度": 0.0
    }
    
    # 计算排版工整度：基于换行符和空格的一致性
    lines = text.split('\n')
    if len(lines) > 1:
        # 计算每行的前导空格数
        leading_spaces = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        if leading_spaces:
            # 计算前导空格的一致性（标准差越小越整齐）
            import statistics
            try:
                std_dev = statistics.stdev(leading_spaces) if len(leading_spaces) > 1 else 0
                # 标准差越小，排版越整齐（0-1之间，1表示最整齐）
                features["排版工整度"] = max(0.0, min(1.0, 1.0 - (std_dev / 10.0)))
            except statistics.StatisticsError:
                features["排版工整度"] = 0.5
    
    # 计算成果量化程度：文本中数字的比例
    if len(text) > 0:
        digit_count = sum(1 for c in text if c.isdigit())
        # 数字越多，量化程度越高
        features["成果量化程度"] = min(1.0, digit_count / (len(text) * 0.1))
    
    # 计算关键词密度：关键词出现频率
    keywords = ["负责", "参与", "完成", "实现", "优化", "提升", "解决", "开发", "设计"]
    keyword_count = sum(text.count(keyword) for keyword in keywords)
    if len(text) > 0:
        features["关键词密度"] = min(1.0, keyword_count / (len(text) * 0.05))
    
    # 计算段落结构清晰度：基于段落数量和长度的一致性
    non_empty_lines = [line for line in lines if line.strip()]
    if len(non_empty_lines) > 1:
        # 计算每行长度的标准差
        line_lengths = [len(line.strip()) for line in non_empty_lines]
        import statistics
        try:
            std_dev = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0
            # 标准差越小，段落长度越一致，结构越清晰
            features["段落结构清晰度"] = max(0.0, min(1.0, 1.0 - (std_dev / 50.0)))
        except statistics.StatisticsError:
            features["段落结构清晰度"] = 0.5
    
    return features

def extract_transferable_skills(text: str, job_description: str) -> Dict[str, Any]:
    """
    提取可迁移能力特征
    基于"岗位-能力映射词典"，提取跨界候选人相关技能特征
    """
    transferable_skills = {
        "项目管理能力": 0.0,
        "沟通协调能力": 0.0,
        "学习能力": 0.0,
        "问题解决能力": 0.0,
        "团队合作能力": 0.0,
        "领导能力": 0.0,
        "创新能力": 0.0,
        "适应能力": 0.0
    }
    
    # 定义关键词映射
    skill_keywords = {
        "项目管理能力": ["项目管理", "项目计划", "项目进度", "项目协调", "项目交付", "里程碑", "甘特图", "风险管理"],
        "沟通协调能力": ["沟通", "协调", "交流", "谈判", "汇报", "演讲", "表达", "倾听"],
        "学习能力": ["学习", "掌握", "研究", "探索", "自学", "培训", "进修", "深造"],
        "问题解决能力": ["解决", "分析", "诊断", "方案", "策略", "优化", "改进", "处理"],
        "团队合作能力": ["团队", "合作", "协作", "配合", "集体", "协同", "互助", "团结"],
        "领导能力": ["领导", "管理", "指导", "带领", "激励", "授权", "决策", "统筹"],
        "创新能力": ["创新", "创意", "革新", "突破", "研发", "发明", "创造", "改进"],
        "适应能力": ["适应", "灵活", "应变", "调整", "转换", "兼容", "融合", "变通"]
    }
    
    # 计算每个能力的得分
    for skill, keywords in skill_keywords.items():
        score = 0.0
        for keyword in keywords:
            # 在简历和岗位描述中查找关键词
            if keyword in text or keyword in job_description:
                score += 0.1  # 每个关键词贡献0.1分
        # 限制最大得分为1.0
        transferable_skills[skill] = min(1.0, score)
        
    return transferable_skills
