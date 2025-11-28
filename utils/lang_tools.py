import re

def detect_language(text: str) -> str:
    t = text or ""
    if not t.strip():
        return "unknown"
    zh = len(re.findall(r"[\u4e00-\u9fa5]", t))
    en = sum(c.isalpha() for c in t)
    tot = max(1, len(t))
    zh_ratio = zh / tot
    en_ratio = en / tot
    if zh_ratio >= 0.2 and zh >= 20:
        return "zh"
    if en_ratio >= 0.3 and en >= 50:
        return "en"
    return "unknown"

def translate_text(text: str, target_lang: str = "zh") -> str:
    """翻译文本到目标语言的简单实现"""
    # 这里是一个简单的占位实现，实际项目中可能会调用翻译API
    return text

def tokenize_text(text: str) -> list:
    """将文本分词的简单实现"""
    # 这里是一个简单的占位实现，实际项目中可能会使用更复杂的分词算法
    if not text:
        return []
    # 简单的按空格和标点分词
    return re.findall(r"[\u4e00-\u9fa5]+|[a-zA-Z]+|[0-9]+|[,.!?;:()，。！？；：（）]", text)