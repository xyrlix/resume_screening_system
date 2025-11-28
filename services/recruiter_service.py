#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
招聘方服务模块

负责处理招聘方的核心业务逻辑
"""

from typing import List, Dict, Any, Tuple
from core.data_processor import DataProcessor
from core.vectorizer import Vectorizer
from core.feature_engine import FeatureEngine
from core.matcher import ResumeMatcher
from core.llm_chain import LLMChain
from core.evaluator import ModelEvaluator
from utils.logger import get_logger

# 初始化日志记录器
logger = get_logger("recruiter_service")


class RecruiterService:
    """
    招聘方服务类，负责处理招聘方的核心业务逻辑
    """

    def __init__(self):
        """
        初始化招聘方服务
        """
        # 初始化核心模块
        self.data_processor = DataProcessor()
        self.vectorizer = Vectorizer()
        self.feature_engine = FeatureEngine(self.vectorizer)
        self.matcher = ResumeMatcher(self.feature_engine)
        self.llm_chain = LLMChain()
        self.evaluator = ModelEvaluator()

        # 存储数据
        self.jobs = []  # 存储JD信息
        self.resumes = []  # 存储简历信息
        self.matching_results = []  # 存储匹配结果

    def add_job(self, jd_text: str, job_id: str = None, meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        添加新的JD
        
        Args:
            jd_text: JD文本
            job_id: 可选的JD ID
        
        Returns:
            添加的JD信息
        """
        logger.info(f"开始处理添加JD请求，job_id: {job_id}")

        # 处理JD文本
        logger.info("正在处理JD文本")
        processed_jd = self.data_processor.process_jd_text(jd_text)

        # 提取特征
        logger.info("正在提取JD特征")
        featured_jd = self.feature_engine.extract_features_from_jd(
            processed_jd)

        # 设置JD ID
        featured_jd['job_id'] = job_id or f"job_{len(self.jobs) + 1}"
        if meta:
            for k, v in meta.items():
                featured_jd[k] = v
        logger.info(f"生成JD ID: {featured_jd['job_id']}")

        # 添加到JD列表
        self.jobs.append(featured_jd)
        logger.info(f"JD添加成功，当前JD总数: {len(self.jobs)}")

        return featured_jd

    def upload_resume(self,
                      resume_text: str,
                      resume_id: str = None,
                      meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        上传简历
        
        Args:
            resume_text: 简历文本
            resume_id: 可选的简历ID
        
        Returns:
            上传的简历信息
        """
        logger.info(f"开始处理上传简历请求，resume_id: {resume_id}")

        # 处理简历文本
        logger.info("正在处理简历文本")
        processed_resume = self.data_processor.process_resume_text(resume_text)

        # 提取特征
        logger.info("正在提取简历特征")
        featured_resume = self.feature_engine.extract_features_from_resume(
            processed_resume)

        # 设置简历ID
        featured_resume[
            'resume_id'] = resume_id or f"resume_{len(self.resumes) + 1}"
        if meta:
            for k, v in meta.items():
                featured_resume[k] = v
        logger.info(f"生成简历ID: {featured_resume['resume_id']}")

        # 添加到简历列表
        self.resumes.append(featured_resume)
        logger.info(f"简历上传成功，当前简历总数: {len(self.resumes)}")

        return featured_resume

    def match_resumes_to_job(
            self,
            job_id: str,
            top_k: int = 10,
            config: Dict[str, Any] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        将简历与指定JD进行匹配
        
        Args:
            job_id: JD ID
            top_k: 返回的匹配结果数量
        
        Returns:
            匹配结果列表，每个元素是(简历, 匹配分数)的元组
        """
        logger.info(f"开始处理简历匹配请求，job_id: {job_id}, top_k: {top_k}")

        # 查找指定的JD
        job = next((j for j in self.jobs if j['job_id'] == job_id), None)
        if not job:
            logger.error(f"未找到ID为 {job_id} 的JD")
            raise ValueError(f"未找到ID为{job_id}的JD")
        logger.info(f"找到JD: {job_id}")

        # 匹配简历
        logger.info(f"开始匹配简历，共 {len(self.resumes)} 份简历")
        if config:
            _results = self.matcher.three_stage_filter(self.resumes, job, top_k, config)
            results = [(resume, score) for resume, score, _, _ in _results]
        else:
            results = self.matcher.match_resumes_to_jd(self.resumes, job, top_k)
        logger.info(f"简历匹配完成，找到 {len(results)} 份匹配简历")

        # 存储匹配结果
        self.matching_results.append({
            'job_id': job_id,
            'results': results,
            'timestamp': self._get_current_timestamp()
        })
        logger.info(f"匹配结果保存成功")

        return results

    def apply_filter_rules(
            self, resumes: List[Dict[str, Any]],
            filter_rules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        应用筛选规则
        
        Args:
            resumes: 简历列表
            filter_rules: 筛选规则
        
        Returns:
            筛选后的简历列表
        """
        filtered_resumes = []

        for resume in resumes:
            if self._matches_filter_rules(resume, filter_rules):
                filtered_resumes.append(resume)

        return filtered_resumes

    def _matches_filter_rules(self, resume: Dict[str, Any],
                              filter_rules: Dict[str, Any]) -> bool:
        """
        检查简历是否符合筛选规则
        
        Args:
            resume: 简历信息
            filter_rules: 筛选规则
        
        Returns:
            是否符合筛选规则
        """
        # 学历要求
        if 'education' in filter_rules and filter_rules['education']:
            if not any(edu in resume['education']
                       for edu in filter_rules['education']):
                return False

        # 工作年限要求
        if 'experience' in filter_rules:
            exp_rules = filter_rules['experience']

            # 提取简历中的工作经验年数
            def extract_years(experience_str: str) -> int:
                import re
                match = re.search(r'\d+', experience_str)
                return int(match.group()) if match else 0

            resume_years = max(
                (extract_years(exp) for exp in resume.get('experience', [])),
                default=0)

            if isinstance(exp_rules, dict):
                min_exp = exp_rules.get('min', 0)
                max_exp = exp_rules.get('max', float('inf'))
                if not (min_exp <= resume_years <= max_exp):
                    return False

        # 技能要求
        if 'skills' in filter_rules and filter_rules['skills']:
            required_skills = filter_rules['skills']
            resume_skills = resume['skills']
            # 要求至少匹配50%的技能
            matching_skills = set(resume_skills) & set(required_skills)
            if len(matching_skills) / len(required_skills) < 0.5:
                return False

        # 位置地点要求
        if 'location' in filter_rules and filter_rules['location']:
            required_locations = filter_rules['location']
            resume_location = resume.get('entities', {}).get('现居地', '')
            if resume_location and not any(loc in resume_location
                                           for loc in required_locations):
                return False

        # 语言要求
        if 'language' in filter_rules and filter_rules['language']:
            required_languages = filter_rules['language']
            resume_languages = resume.get('entities', {}).get('语言能力', '')
            if resume_languages and not any(lang in resume_languages
                                            for lang in required_languages):
                return False

        # 证书要求
        if 'certificates' in filter_rules and filter_rules['certificates']:
            required_certificates = filter_rules['certificates']
            resume_certificates = resume.get('entities', {}).get('证书资质', '')
            if resume_certificates and not any(
                    cert in resume_certificates
                    for cert in required_certificates):
                return False

        # 其他筛选规则...

        return True

    def analyze_resume_with_llm(self, jd_text: str, resume_text: str) -> dict:
        """
        使用LLM链式分析简历和JD
        
        Args:
            jd_text: JD文本
            resume_text: 简历文本
        
        Returns:
            LLM分析结果
        """
        return self.llm_chain.process_resume(jd_text, resume_text)

    def evaluate_matching_results(
            self, job_id: str, actual_scores: List[float]) -> Dict[str, float]:
        """
        评估匹配结果
        
        Args:
            job_id: JD ID
            actual_scores: 实际匹配分数列表
        
        Returns:
            评估结果
        """
        # 查找匹配结果
        matching_result = next(
            (r for r in self.matching_results if r['job_id'] == job_id), None)
        if not matching_result:
            raise ValueError(f"未找到ID为{job_id}的匹配结果")

        # 评估匹配结果
        return self.evaluator.evaluate_matching_results(
            matching_result['results'], actual_scores)

    def get_job_list(self) -> List[Dict[str, Any]]:
        """
        获取JD列表
        
        Returns:
            JD列表
        """
        return self.jobs

    def get_resume_list(self) -> List[Dict[str, Any]]:
        """
        获取简历列表
        
        Returns:
            简历列表
        """
        return self.resumes

    def get_matching_results(self, job_id: str = None) -> List[Dict[str, Any]]:
        """
        获取匹配结果
        
        Args:
            job_id: 可选的JD ID，用于过滤匹配结果
        
        Returns:
            匹配结果列表
        """
        if job_id:
            return [r for r in self.matching_results if r['job_id'] == job_id]
        return self.matching_results

    def _get_current_timestamp(self) -> str:
        """
        获取当前时间戳
        
        Returns:
            当前时间戳字符串
        """
        from datetime import datetime
        return datetime.now().isoformat()
