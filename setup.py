#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能简历筛选系统 - 安装配置文件
"""

from setuptools import setup, find_packages
import os

# 读取项目版本
version = "1.0.0"

# 读取依赖列表
with open('requirements.txt', 'r', encoding='utf-8') as f:
    install_requires = [
        line.strip() for line in f if line.strip() and not line.startswith('#')
    ]

# 项目描述
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    # 项目基本信息
    name="resume-screening-system",
    version=version,
    description="基于深度学习的智能简历筛选系统，使用BGE-M3向量模型和DeepSeek LLM",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # 项目URL信息
    url="https://github.com/your-username/resume-screening-system",
    project_urls={
        "Bug Tracker":
        "https://github.com/your-username/resume-screening-system/issues",
        "Documentation":
        "https://github.com/your-username/resume-screening-system/wiki",
        "Source Code":
        "https://github.com/your-username/resume-screening-system",
    },

    # 作者信息
    author="xyrlix",
    author_email="xyrlix@outlook.com",

    # 许可证信息
    license="MIT",
    license_files=["LICENSE"],

    # 分类信息
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Human Resources",
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: General",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: Web Environment",
    ],

    # 关键词
    keywords=[
        "resume-screening",
        "talent-acquisition",
        "artificial-intelligence",
        "machine-learning",
        "natural-language-processing",
        "llm",
        "vector-search",
        "huggingface",
        "deepseek",
        "bge-m3",
    ],

    # 包信息
    packages=find_packages(include=[
        "core",
        "core.*",
        "modules",
        "modules.*",
        "scripts",
        "scripts.*",
        "utils",
        "utils.*",
    ]),

    # 依赖信息
    install_requires=install_requires,

    # 额外依赖
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.0.0",
            "mkdocs>=1.0.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-asyncio>=0.20.0",
            "httpx>=0.20.0",
        ],
        "docs": [
            "mkdocs>=1.0.0",
            "mkdocs-material>=8.0.0",
            "mkdocstrings[python]>=0.18.0",
        ],
    },

    # Python版本要求
    python_requires=">=3.8",

    # 入口点
    entry_points={
        "console_scripts": [
            "resume-screening-server=api_server_fixed:main",
            "resume-screening-web=app:main",
            "resume-screening-cli=scripts.cli:main",
        ],
    },

    # 数据文件
    package_data={
        "": [
            "*.md",
            "*.txt",
            "config/*.json",
            "config/*.yml",
            "data/models/*",
            "data/models/**/*",
            "data/*",
            "data/**/*",
        ],
    },

    # 包含额外文件
    include_package_data=True,

    # 排除测试文件
    exclude_package_data={
        "": [
            "tests/*",
            "tests/**/*",
            ".gitignore",
            "*.pyc",
            "__pycache__/*",
            "__pycache__/**/*",
        ],
    },

    # 安装位置
    zip_safe=False,

    # 配置文件
    setup_requires=[
        "setuptools>=40.0.0",
    ],
)
