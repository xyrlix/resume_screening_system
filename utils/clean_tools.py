import re

def clean_text(text: str) -> str:
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s,.!?;:()，。！？；：（）]', '', text or '')
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or '')).strip()