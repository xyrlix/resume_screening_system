import os
import json
import re
from typing import List, Dict, Tuple


WEIGHTS = {
    "skills": 0.5,
    "degree": 0.2,
    "years": 0.2,
    "position": 0.1,
    # 新增特征权重（可在 config/matching.json 中覆盖）
    "keywords": 0.1,     # 领域/方向关键词，如 NLP/推荐/风控 等
    "certs": 0.05,       # 证书/资质命中，如 PMP/CKA/AWS 等
    "languages": 0.05,   # 语言能力命中，如 英语六级/托福/雅思 等
}


DEGREE_ORDER = {
    "博士": 4,
    "硕士": 3,
    "研究生": 3,
    "本科": 2,
    "学士": 2,
    "大专": 1,
    "专科": 1,
}


SKILL_DICT = {
    # 常见中文/英文技能关键字
    "python", "java", "c++", "c#", "go", "golang", "sql", "mysql", "postgresql",
    "docker", "k8s", "kubernetes", "linux", "windows", "git", "hadoop", "spark",
    "tensorflow", "pytorch", "scikit-learn", "sklearn", "flask", "fastapi", "django",
    "redis", "mongodb", "elasticsearch", "vue", "react", "angular",
    # 额外扩展
    "aws", "azure", "gcp", "kafka", "rabbitmq", "rest", "grpc", "microservices"
}

# 领域/方向关键词（与技能区分开，更偏业务与算法方向）
KEYWORD_DICT = {
    "nlp", "自然语言处理", "cv", "计算机视觉", "llm", "大模型", "transformer", "transformers",
    "bert", "gpt", "langchain", "rag", "向量数据库", "faiss", "milvus", "pgvector",
    "推荐", "推荐系统", "recommender", "ranking", "ctr", "点击率", "召回", "排序",
    "搜索", "全文检索", "query understanding", "query intent",
    "广告", "投放", "实时竞价", "rtb",
    "风控", "反欺诈", "风控建模",
    "金融", "电商", "医疗", "制造", "物联网",
}

# 常见证书/资质关键词
CERT_DICT = {
    "pmp", "cka", "ckad", "aws saa", "aws sa", "aws devops", "azure az-900", "azure ai-900",
    "gcp pca", "rhce", "hcia", "hcip", "软考中级", "软考高级", "软考",
}

# 语言能力相关关键词
LANG_DICT = {
    "英语四级", "英语六级", "cet-4", "cet4", "cet-6", "cet6",
    "雅思", "托福", "英文流利", "英语可作为工作语言", "英语工作",
    "日语n1", "日语n2", "日语", "jlpt",
    "德语b2", "德语", "韩语topik", "韩语",
}


def _load_matching_config():
    """从配置文件覆盖默认权重与技能词典。
    - 优先读取环境变量 MATCHING_CONFIG_PATH 指定的路径
    - 否则读取项目根目录下 config/matching.json（若存在）
    配置结构示例：
    {
      "weights": {"skills": 0.5, "degree": 0.2, "years": 0.2, "position": 0.1},
      "skills": ["python", "java", ...]
    }
    """
    try:
        import os
        cfg_path = os.getenv("MATCHING_CONFIG_PATH") or os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "matching.json")
        if not os.path.isfile(cfg_path):
            return
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        global WEIGHTS, SKILL_DICT, KEYWORD_DICT, CERT_DICT, LANG_DICT
        # 权重：允许按键名覆盖，未提供的键沿用默认
        if isinstance(cfg.get("weights"), dict):
            w = cfg["weights"]
            for k, v in w.items():
                if k in WEIGHTS and isinstance(v, (int, float)) and v >= 0:
                    WEIGHTS[k] = float(v)
        # 技能词典
        if isinstance(cfg.get("skills"), list) and len(cfg["skills"]) > 0:
            SKILL_DICT = {str(x).lower().strip() for x in cfg["skills"] if str(x).strip()}
        # 领域关键词
        if isinstance(cfg.get("keywords"), list) and len(cfg["keywords"]) > 0:
            KEYWORD_DICT = {str(x).lower().strip() for x in cfg["keywords"] if str(x).strip()}
        # 证书关键词
        if isinstance(cfg.get("certs"), list) and len(cfg["certs"]) > 0:
            CERT_DICT = {str(x).lower().strip() for x in cfg["certs"] if str(x).strip()}
        # 语言关键词
        if isinstance(cfg.get("languages"), list) and len(cfg["languages"]) > 0:
            LANG_DICT = {str(x).lower().strip() for x in cfg["languages"] if str(x).strip()}
        print(
            f"[Matcher] Loaded config from {cfg_path}. Weights={WEIGHTS}, skills={len(SKILL_DICT)}, keywords={len(KEYWORD_DICT)}, certs={len(CERT_DICT)}, languages={len(LANG_DICT)}"
        )
    except Exception as e:
        print(f"[Matcher] Load config failed: {e}")


# 模块导入时尝试加载配置
_load_matching_config()


def normalize(text: str) -> str:
    return text.lower().strip()


def extract_skills(text: str) -> List[str]:
    t = normalize(text)
    found = set()
    for sk in SKILL_DICT:
        if sk in t:
            found.add(sk)
    # 额外捕获“掌握/熟悉/精通”后的技能短语（逗号分隔）
    m = re.findall(r"(?:掌握|熟悉|精通)([^。；\n]+)", text)
    for phrase in m:
        parts = re.split(r"[，,、；;\s]", phrase)
        for p in parts:
            p = p.strip().lower()
            if p and len(p) <= 30:
                # 只收录在字典中的技能，以避免噪声
                if p in SKILL_DICT:
                    found.add(p)
    return sorted(found)


def extract_keywords(text: str) -> List[str]:
    t = normalize(text)
    found = set()
    for kw in KEYWORD_DICT:
        if kw in t:
            found.add(kw)
    return sorted(found)


def extract_certs(text: str) -> List[str]:
    t = normalize(text)
    found = set()
    for c in CERT_DICT:
        if c in t:
            found.add(c)
    # 常见缩略/变体
    if "cet6" in t or "cet-6" in t:
        found.add("cet-6")
    if "cet4" in t or "cet-4" in t:
        found.add("cet-4")
    return sorted(found)


def extract_languages(text: str) -> List[str]:
    t = normalize(text)
    found = set()
    for l in LANG_DICT:
        if l in t:
            found.add(l)
    # 一些常见表达
    if "英文流利" in text or "英语可作为工作语言" in text or "英语工作" in text:
        found.add("英文流利")
    return sorted(found)


def extract_degree(text: str) -> str:
    # 根据出现的最高学历返回映射
    max_deg = 0
    best = ""
    for k, v in DEGREE_ORDER.items():
        if k in text:
            if v > max_deg:
                max_deg = v
                best = k
    return best


def extract_years(text: str) -> int:
    # 匹配“X年工作/开发/经验”，避免年龄误匹配
    m = re.findall(r"(\d+)\s*年(?:(?:工作|开发|项目)?经验)?", text)
    years = 0
    for g in m:
        try:
            y = int(g)
            years = max(years, y)
        except Exception:
            pass
    return years


def extract_position(text: str) -> str:
    # 简易职位提取：捕获“担任XXX”或末尾的职位词
    m = re.search(r"担任([\u4e00-\u9fa5A-Za-z0-9_\-]+)", text)
    if m:
        return m.group(1).strip()
    # 常见职位关键词
    keywords = [
        "工程师", "开发", "后端", "前端", "算法", "数据", "架构", "产品", "测试", "运维",
        "scientist", "developer", "engineer", "architect"
    ]
    for kw in keywords:
        if kw in text:
            return kw
    return ""


def profile_from_text(text: str) -> Dict:
    return {
        "skills": extract_skills(text),
        "degree": extract_degree(text),
        "years": extract_years(text),
        "position": extract_position(text),
        "keywords": extract_keywords(text),
        "certs": extract_certs(text),
        "languages": extract_languages(text),
    }


def degree_meets(require: str, have: str) -> bool:
    if not require:
        return True
    return DEGREE_ORDER.get(have, 0) >= DEGREE_ORDER.get(require, 0)


def match_score(resume: Dict, job: Dict) -> Tuple[float, Dict]:
    # 技能得分：匹配的比例（以岗位技能为基准）
    job_sk = set(job.get("skills", []))
    res_sk = set(resume.get("skills", []))
    matched_skills = sorted(job_sk & res_sk)
    skill_ratio = (len(matched_skills) / len(job_sk)) if len(job_sk) > 0 else 0.5

    # 学历得分：满足最低要求记满，否则0
    deg_req = job.get("degree", "")
    deg_have = resume.get("degree", "")
    degree_score = 1.0 if degree_meets(deg_req, deg_have) else 0.0

    # 年限得分：满足即1，否则按比例 capped 到1
    req_years = job.get("years", 0)
    have_years = resume.get("years", 0)
    years_score = 1.0 if req_years == 0 else min(have_years / max(req_years, 1), 1.0)

    # 职位匹配：若关键词相同或包含，给1，否则0
    pos_req = job.get("position", "")
    pos_have = resume.get("position", "")
    position_score = 0.0
    if pos_req and pos_have:
        if pos_req == pos_have or pos_req in pos_have or pos_have in pos_req:
            position_score = 1.0

    # 关键词（方向）得分：与技能类似，按岗位关键词为基准
    job_kw = set(job.get("keywords", []))
    res_kw = set(resume.get("keywords", []))
    matched_keywords = sorted(job_kw & res_kw)
    keyword_ratio = (len(matched_keywords) / len(job_kw)) if len(job_kw) > 0 else 0.5

    # 证书得分：岗位若有证书要求，命中则 1，否则 0；若无要求则 0.5
    job_certs = set(job.get("certs", []))
    res_certs = set(resume.get("certs", []))
    certs_score = 0.5 if len(job_certs) == 0 else (1.0 if len(job_certs & res_certs) > 0 else 0.0)

    # 语言能力得分：与技能类似，按岗位语言要求为基准
    job_lang = set(job.get("languages", []))
    res_lang = set(resume.get("languages", []))
    matched_languages = sorted(job_lang & res_lang)
    languages_ratio = (len(matched_languages) / len(job_lang)) if len(job_lang) > 0 else 0.5

    total = (
        WEIGHTS["skills"] * skill_ratio +
        WEIGHTS["degree"] * degree_score +
        WEIGHTS["years"] * years_score +
        WEIGHTS["position"] * position_score +
        WEIGHTS.get("keywords", 0.0) * keyword_ratio +
        WEIGHTS.get("certs", 0.0) * certs_score +
        WEIGHTS.get("languages", 0.0) * languages_ratio
    )
    details = {
        "matched_skills": matched_skills,
        "skill_ratio": round(skill_ratio, 4),
        "degree_score": degree_score,
        "years_score": round(years_score, 4),
        "position_score": position_score,
        "matched_keywords": matched_keywords,
        "keyword_ratio": round(keyword_ratio, 4),
        "matched_certs": sorted(job_certs & res_certs),
        "certs_score": certs_score,
        "matched_languages": matched_languages,
        "languages_ratio": round(languages_ratio, 4),
    }
    return round(total, 4), details


def job_profile_from_text(job_text: str) -> Dict:
    # 解析岗位文本得到需求画像
    skills = extract_skills(job_text)
    # 学历要求：优先识别最低门槛词
    deg_req = ""
    if "博士" in job_text:
        deg_req = "博士"
    elif "硕士" in job_text or "研究生" in job_text:
        deg_req = "硕士"
    elif "本科" in job_text or "学士" in job_text:
        deg_req = "本科"
    elif "大专" in job_text or "专科" in job_text:
        deg_req = "大专"

    years = extract_years(job_text)
    position = extract_position(job_text)
    return {
        "skills": skills,
        "degree": deg_req,
        "years": years,
        "position": position,
        "keywords": extract_keywords(job_text),
        "certs": extract_certs(job_text),
        "languages": extract_languages(job_text),
    }


def load_jobs_from_dir(dir_path: str) -> List[Dict]:
    jobs = []
    if not os.path.isdir(dir_path):
        return jobs
    for name in os.listdir(dir_path):
        if not name.lower().endswith(('.txt', '.md')):
            continue
        p = os.path.join(dir_path, name)
        try:
            with open(p, 'r', encoding='utf-8') as f:
                jt = f.read()
            jobs.append({
                "file": name,
                "text": jt,
                "profile": job_profile_from_text(jt)
            })
        except Exception:
            pass
    return jobs


def match_all(resume_json_path: str, jobs_dir: str, out_path: str = "logs/match_results.json"):
    if not os.path.isfile(resume_json_path):
        raise FileNotFoundError(f"简历JSON不存在: {resume_json_path}")
    with open(resume_json_path, 'r', encoding='utf-8') as f:
        resumes = json.load(f)

    jobs = load_jobs_from_dir(jobs_dir)
    if not jobs:
        print("岗位目录为空或未找到岗位文本。可通过 --job_text 进行单次匹配测试。")

    results = []
    for r in resumes:
        rtext = r.get('text', '')
        rprofile = profile_from_text(rtext)
        for j in jobs:
            score, details = match_score(rprofile, j["profile"])
            results.append({
                "resume_id": r.get('id'),
                "job_file": j.get('file'),
                "score": score,
                "resume_profile": rprofile,
                "job_profile": j["profile"],
                "details": details,
            })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"匹配结果已保存到: {out_path}")


def quick_match(resume_text: str, job_text: str) -> Dict:
    rprofile = profile_from_text(resume_text)
    jprofile = job_profile_from_text(job_text)
    score, details = match_score(rprofile, jprofile)
    return {
        "score": score,
        "resume_profile": rprofile,
        "job_profile": jprofile,
        "details": details,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="基于规则的简历-岗位匹配器")
    parser.add_argument("--resume_json", default="data/processed/resumes_for_annotation.json")
    parser.add_argument("--jobs_dir", default="data/raw_jobs")
    parser.add_argument("--out", default="logs/match_results.json")
    parser.add_argument("--resume_text", default=None, help="用于快速测试的简历文本")
    parser.add_argument("--job_text", default=None, help="用于快速测试的岗位文本")
    args = parser.parse_args()

    if args.resume_text and args.job_text:
        res = quick_match(args.resume_text, args.job_text)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        match_all(args.resume_json, args.jobs_dir, args.out)