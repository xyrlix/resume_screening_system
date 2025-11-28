#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行业和岗位管理模块

负责管理行业和岗位信息
"""

from typing import List, Dict, Any


class IndustryJobManager:
    """
    行业和岗位管理类，负责管理行业和岗位信息
    """

    def __init__(self):
        """
        初始化行业和岗位管理类
        """
        # 热门行业列表
        self.industries = ["人工智能", "新能源", "半导体/芯片", "互联网", "电子商务"]

        # 热门岗位列表
        self.jobs = ["算法工程师", "电池研发工程师", "芯片设计工程师", "产品经理", "跨境电商"]

        # 行业-岗位映射关系
        self.industry_job_map = {
            "人工智能": ["算法工程师"],
            "新能源": ["电池研发工程师"],
            "半导体/芯片": ["芯片设计工程师"],
            "互联网": ["产品经理"],
            "电子商务": ["跨境电商"]
        }

    def get_industries(self) -> List[str]:
        """
        获取所有行业列表
        
        Returns:
            行业列表
        """
        return self.industries.copy()

    def get_jobs(self) -> List[str]:
        """
        获取所有岗位列表
        
        Returns:
            岗位列表
        """
        return self.jobs.copy()

    def get_jobs_by_industry(self, industry: str) -> List[str]:
        """
        根据行业获取对应的岗位列表
        
        Args:
            industry: 行业名称
        
        Returns:
            岗位列表
        """
        return self.industry_job_map.get(industry, [])

    def get_industry_job_map(self) -> Dict[str, List[str]]:
        """
        获取行业-岗位映射关系
        
        Returns:
            行业-岗位映射关系
        """
        return self.industry_job_map.copy()

    def is_valid_industry(self, industry: str) -> bool:
        """
        检查行业是否有效
        
        Args:
            industry: 行业名称
        
        Returns:
            是否有效
        """
        return industry in self.industries

    def is_valid_job(self, job: str) -> bool:
        """
        检查岗位是否有效
        
        Args:
            job: 岗位名称
        
        Returns:
            是否有效
        """
        return job in self.jobs

    def get_industry_by_job(self, job: str) -> str:
        """
        根据岗位获取对应的行业
        
        Args:
            job: 岗位名称
        
        Returns:
            行业名称
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
            是否添加成功
        """
        if industry not in self.industries:
            self.industries.append(industry)
            self.industry_job_map[industry] = []
            return True
        return False

    def add_job(self, job: str, industry: str) -> bool:
        """
        添加新的岗位
        
        Args:
            job: 岗位名称
            industry: 所属行业
        
        Returns:
            是否添加成功
        """
        if job not in self.jobs:
            self.jobs.append(job)
            if industry not in self.industries:
                self.add_industry(industry)
            self.industry_job_map[industry].append(job)
            return True
        return False

    def get_industry_count(self) -> int:
        """
        获取行业数量
        
        Returns:
            行业数量
        """
        return len(self.industries)

    def get_job_count(self) -> int:
        """
        获取岗位数量
        
        Returns:
            岗位数量
        """
        return len(self.jobs)

    def get_industry_job_statistics(self) -> Dict[str, int]:
        """
        获取行业-岗位数量统计
        
        Returns:
            行业-岗位数量统计
        """
        statistics = {}
        for industry, jobs in self.industry_job_map.items():
            statistics[industry] = len(jobs)
        return statistics
