#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业和岗位管理模块

负责管理行业和岗位信息
"""

import json
import os
from typing import List, Dict, Any, Optional


class IndustryJobManager:
    """
    行业和岗位管理器，负责管理行业和岗位的映射关系，支持配置化管理
    """

    def __init__(self, config_path: str = "config/industry_job_config.json"):
        """
        初始化行业和岗位映射关系，从配置文件加载数据
        
        Args:
            config_path: 配置文件路径，默认为 "config/industry_job_config.json"
        """
        self.config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), config_path)
        self.industries: List[str] = []
        self.jobs: List[str] = []
        self.industry_job_map: Dict[str, List[str]] = {}

        # 从配置文件加载数据
        self._load_config()

    def _load_config(self) -> None:
        """
        从配置文件加载行业和岗位配置
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            self.industries = config.get('industries', [])
            self.jobs = config.get('jobs', [])
            self.industry_job_map = config.get('industry_job_map', {})

            # 确保所有行业在映射关系中都有对应的岗位列表
            for industry in self.industries:
                if industry not in self.industry_job_map:
                    self.industry_job_map[industry] = []

        except FileNotFoundError:
            # 如果配置文件不存在，使用默认配置
            self._initialize_default_config()
            self._save_config()
        except json.JSONDecodeError:
            # 如果配置文件格式错误，使用默认配置
            print(f"警告：配置文件 {self.config_path} 格式错误，使用默认配置")
            self._initialize_default_config()
            self._save_config()

    def _initialize_default_config(self) -> None:
        """
        初始化默认的行业和岗位配置
        """
        # 默认行业列表
        self.industries = ["人工智能", "新能源", "半导体/芯片", "互联网", "电子商务"]

        # 默认岗位列表
        self.jobs = ["算法工程师", "电池研发工程师", "芯片设计工程师", "产品经理", "跨境电商"]

        # 默认行业-岗位映射关系
        self.industry_job_map = {
            "人工智能": ["算法工程师"],
            "新能源": ["电池研发工程师"],
            "半导体/芯片": ["芯片设计工程师"],
            "互联网": ["产品经理"],
            "电子商务": ["跨境电商"]
        }

    def _save_config(self) -> None:
        """
        保存行业和岗位配置到配置文件
        """
        config = {
            'industries': self.industries,
            'jobs': self.jobs,
            'industry_job_map': self.industry_job_map
        }

        # 确保配置文件所在目录存在
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def get_industries(self) -> List[str]:
        """
        获取所有行业列表
        
        Returns:
            List[str]: 行业列表
        """
        return self.industries.copy()

    def get_jobs(self) -> List[str]:
        """
        获取所有岗位列表
        
        Returns:
            List[str]: 岗位列表
        """
        return self.jobs.copy()

    def get_jobs_by_industry(self, industry: str) -> List[str]:
        """
        根据行业获取对应的岗位列表
        
        Args:
            industry: 行业名称
        
        Returns:
            List[str]: 岗位列表，如果行业不存在则返回空列表
        """
        return self.industry_job_map.get(industry, [])

    def get_industry_job_map(self) -> Dict[str, List[str]]:
        """
        获取行业-岗位映射关系
        
        Returns:
            Dict[str, List[str]]: 行业-岗位映射关系
        """
        return self.industry_job_map.copy()

    def is_valid_industry(self, industry: str) -> bool:
        """
        检查行业是否有效
        
        Args:
            industry: 行业名称
        
        Returns:
            bool: 是否有效
        """
        return industry in self.industries

    def is_valid_job(self, job: str) -> bool:
        """
        检查岗位是否有效
        
        Args:
            job: 岗位名称
        
        Returns:
            bool: 是否有效
        """
        return job in self.jobs

    def get_industry_by_job(self, job: str) -> str:
        """
        根据岗位获取对应的行业
        
        Args:
            job: 岗位名称
        
        Returns:
            str: 行业名称，如果岗位不存在则返回空字符串
        """
        for industry, jobs in self.industry_job_map.items():
            if job in jobs:
                return industry
        return ""

    def add_industry(self, industry: str) -> bool:
        """
        添加新的行业
        
        Args:
            industry: 行业名称
        
        Returns:
            bool: 是否添加成功
        """
        if industry not in self.industries:
            self.industries.append(industry)
            self.industry_job_map[industry] = []
            self._save_config()
            return True
        return False

    def add_job(self, job: str, industry: str) -> bool:
        """
        添加新的岗位
        
        Args:
            job: 岗位名称
            industry: 所属行业
        
        Returns:
            bool: 是否添加成功
        """
        if job not in self.jobs:
            self.jobs.append(job)
            if industry not in self.industries:
                self.add_industry(industry)
            self.industry_job_map[industry].append(job)
            self._save_config()
            return True
        return False

    def add_job_to_industry(self, industry: str, job: str) -> bool:
        """
        向指定行业添加岗位（兼容方法）
        
        Args:
            industry: 行业名称
            job: 岗位名称
            
        Returns:
            bool: 是否添加成功
        """
        return self.add_job(job, industry)

    def get_industry_count(self) -> int:
        """
        获取行业数量
        
        Returns:
            int: 行业数量
        """
        return len(self.industries)

    def get_job_count(self) -> int:
        """
        获取岗位数量
        
        Returns:
            int: 岗位数量
        """
        return len(self.jobs)

    def get_industry_job_statistics(self) -> Dict[str, int]:
        """
        获取行业-岗位数量统计
        
        Returns:
            Dict[str, int]: 行业-岗位数量统计
        """
        statistics = {}
        for industry, jobs in self.industry_job_map.items():
            statistics[industry] = len(jobs)
        return statistics

    def remove_industry(self, industry: str) -> bool:
        """
        删除指定行业
        
        Args:
            industry: 行业名称
        
        Returns:
            bool: 是否删除成功
        """
        if industry in self.industries:
            self.industries.remove(industry)
            del self.industry_job_map[industry]
            self._save_config()
            return True
        return False

    def remove_job(self, job: str) -> bool:
        """
        删除指定岗位
        
        Args:
            job: 岗位名称
        
        Returns:
            bool: 是否删除成功
        """
        if job in self.jobs:
            self.jobs.remove(job)
            # 从所有行业的岗位列表中删除该岗位
            for industry in self.industry_job_map:
                if job in self.industry_job_map[industry]:
                    self.industry_job_map[industry].remove(job)
            self._save_config()
            return True
        return False

    def remove_job_from_industry(self, industry: str, job: str) -> bool:
        """
        从指定行业中删除岗位
        
        Args:
            industry: 行业名称
            job: 岗位名称
        
        Returns:
            bool: 是否删除成功
        """
        if industry in self.industry_job_map:
            if job in self.industry_job_map[industry]:
                self.industry_job_map[industry].remove(job)
                self._save_config()
                return True
        return False

    def update_config(self,
                      industries: List[str] = None,
                      jobs: List[str] = None,
                      industry_job_map: Dict[str, List[str]] = None) -> bool:
        """
        批量更新配置
        
        Args:
            industries: 新的行业列表（可选）
            jobs: 新的岗位列表（可选）
            industry_job_map: 新的行业-岗位映射关系（可选）
            
        Returns:
            bool: 是否更新成功
        """
        try:
            if industries is not None:
                self.industries = industries

            if jobs is not None:
                self.jobs = jobs

            if industry_job_map is not None:
                self.industry_job_map = industry_job_map

                # 确保所有行业在映射关系中都有对应的岗位列表
                for industry in self.industries:
                    if industry not in self.industry_job_map:
                        self.industry_job_map[industry] = []

            self._save_config()
            return True
        except Exception as e:
            print(f"更新配置失败: {e}")
            return False

    def is_valid_industry_job(self, industry: str, job: str) -> bool:
        """
        检查行业和岗位的组合是否有效
        
        Args:
            industry: 行业名称
            job: 岗位名称
        
        Returns:
            bool: 是否有效
        """
        if not self.is_valid_industry(industry):
            return False

        jobs = self.get_jobs_by_industry(industry)
        return job in jobs
