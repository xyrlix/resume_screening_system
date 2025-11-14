import re
from typing import Dict

SENSITIVE_MAP = {
    r"北京大学|清华大学|复旦大学|上海交通大学|浙江大学": "本科院校",
}

def mask_sensitive(text: str) -> str:
    t = text or ""
    for pat, rep in SENSITIVE_MAP.items():
        t = re.sub(pat, rep, t)
    t = re.sub(r"([\u4e00-\u9fa5]{2,4})(先生|女士)", "候选人", t)
    return t

def collect_fairness_tags(text: str) -> Dict[str, str]:
    t = text or ""
    gender = "未知"
    if re.search(r"先生", t):
        gender = "男"
    elif re.search(r"女士|小姐", t):
        gender = "女"
    school_tier = "未知"
    if re.search(r"985|211|一流大学", t):
        school_tier = "高层次"
    elif re.search(r"本科|大专|专科", t):
        school_tier = "普本/专科"
    gap_type = "未知"
    if re.search(r"空窗|未就业|离职至今", t):
        gap_type = "存在空窗"
    return {"gender": gender, "school_tier": school_tier, "gap_type": gap_type}