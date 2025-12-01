import re
'''
清理工具模块

此模块包含了用于清理文本的各种函数，如移除特殊字符、标准化空白字符等。
'''


def clean_text(text: str) -> str:
    """
    清理文本，移除非中文字符、英文、数字和基本标点符号
    """
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s,.!?;:()，。！？；：（）]', '', text
                  or '')
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def remove_special_chars(text: str) -> str:
    """移除文本中的特殊字符，保留中文、英文、数字和基本标点符号"""
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s,.!?;:()，。！？；：（）]', '', text
                  or '')


def normalize_whitespace(text: str) -> str:
    """
    标准化文本中的空白字符，将多个空格替换为单个空格，并移除首尾空格
    """
    return re.sub(r'\s+', ' ', (text or '')).strip()
