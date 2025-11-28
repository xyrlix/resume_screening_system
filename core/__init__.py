#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能简历筛选系统核心模块

此包包含了系统的核心功能模块，包括数据处理、特征工程、匹配模型、LLM链式分析和模型评估等功能。
"""

# 版本信息
__version__ = "2.0.0"
__author__ = "Resume Screening System Team"
__copyright__ = "© 2025 Resume Screening System"

# 导出核心功能模块
from core.data_processor import DataProcessor
from core.feature_engine import FeatureEngine
from core.matcher import ResumeMatcher
from core.llm_chain import LLMChain
from core.evaluator import ModelEvaluator
from core.vectorizer import Vectorizer
from core.file_processor import FileProcessor
from core.industry_job_manager import IndustryJobManager
from core.llm_config_manager import LLMConfigManager
from core.visualizer import Visualizer

# 导出模块列表
__all__ = [
    'DataProcessor',
    'FeatureEngine',
    'ResumeMatcher',
    'LLMChain',
    'ModelEvaluator',
    'Vectorizer',
    'FileProcessor',
    'IndustryJobManager',
    'LLMConfigManager',
    'Visualizer',
]


# 版本信息
def get_version():
    """获取模块版本信息"""
    return __version__


# 打印初始化信息
print(f"正在加载智能简历筛选系统核心模块 v{__version__}")
