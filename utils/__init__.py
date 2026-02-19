#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块包

此包包含了系统中使用的各种工具函数，如清理工具、语言处理工具、布局解析工具等。
"""

# 版本信息
__version__ = "2.0.0"

# 导出工具模块
from utils.clean_tools import clean_text, remove_special_chars, normalize_whitespace
from utils.lang_tools import translate_text, detect_language, tokenize_text
from utils.logger import get_logger

# 导出模块列表
__all__ = [
    # 清理工具
    'clean_text',
    'remove_special_chars',
    'normalize_whitespace',
    # 语言工具
    'translate_text',
    'detect_language',
    'tokenize_text',
    # 日志工具
    'get_logger'
]


# 版本信息
def get_version():
    """获取工具模块版本信息"""
    return __version__
