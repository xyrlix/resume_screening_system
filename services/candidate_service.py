#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
求职者服务模块

负责处理求职者的核心业务逻辑
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
logger = get_logger("candidate_service")


class CandidateService:
    """
    求职者服务类，负责处理求职者的核心业务逻辑
    """

    def __init__(self):
        """
        初始化求职者服务
        """
        # 初始化核心模块
        self.data_processor = DataProcessor()
        self.vectorizer = Vectorizer()
        self.feature_engine = FeatureEngine(self.vectorizer)
        self.matcher = ResumeMatcher(self.feature_engine)
        self.llm_chain = LLMChain()
        self.evaluator = ModelEvaluator()

        # 存储数据
        self.resumes = []
        self.jobs = []
        self.matching_results = []
        self.applications = []

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
        featured_resume['resume_id'] = resume_id or f"resume_{len(self.resumes) + 1}"
        if meta:
            for k, v in meta.items():
                featured_resume[k] = v
        logger.info(f"生成简历ID: {featured_resume['resume_id']}")

        # 添加到简历列表
        self.resumes.append(featured_resume)
        logger.info(f"简历上传成功，当前简历总数: {len(self.resumes)}")

        return featured_resume

    def add_job(self, jd_text: str, job_id: str = None, meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        添加职位信息（用于模拟职位库）
        
        Args:
            jd_text: JD文本
            job_id: 可选的JD ID
        
        Returns:
            添加的职位信息
        """
        logger.info(f"开始处理添加职位请求，job_id: {job_id}")

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
        logger.info(f"生成职位ID: {featured_jd['job_id']}")

        # 添加到JD列表
        self.jobs.append(featured_jd)
        logger.info(f"职位添加成功，当前职位总数: {len(self.jobs)}")

        return featured_jd

    def match_resume_to_jobs(
            self,
            resume_id: str,
            top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        将简历与职位库中的职位进行匹配
        
        Args:
            resume_id: 简历ID
            top_k: 返回的匹配结果数量
        
        Returns:
            匹配结果列表，每个元素是(职位, 匹配分数)的元组
        """
        logger.info(f"开始处理简历匹配职位请求，resume_id: {resume_id}, top_k: {top_k}")

        # 查找指定的简历
        resume = next((r for r in self.resumes if r['resume_id'] == resume_id),
                      None)
        if not resume:
            logger.error(f"未找到ID为 {resume_id} 的简历")
            raise ValueError(f"未找到ID为{resume_id}的简历")
        logger.info(f"找到简历: {resume_id}")

        # 匹配职位
        logger.info(f"开始匹配职位，共 {len(self.jobs)} 个职位")
        results = self.matcher.match_resume_to_jds(resume, self.jobs, top_k)
        logger.info(f"职位匹配完成，找到 {len(results)} 个匹配职位")

        # 存储匹配结果
        self.matching_results.append({
            'resume_id': resume_id,
            'results': results,
            'timestamp': self._get_current_timestamp()
        })
        logger.info(f"匹配结果保存成功")

        return results

    def generate_resume_optimization_suggestions(
            self, resume_id: str, jd_text: str) -> Dict[str, Any]:
        """
        生成简历优化建议
        
        Args:
            resume_id: 简历ID
            jd_text: 职位描述文本
        
        Returns:
            简历优化建议
        """
        logger.info(f"开始处理生成简历优化建议请求，resume_id: {resume_id}")

        # 查找指定的简历
        resume = next((r for r in self.resumes if r['resume_id'] == resume_id),
                      None)
        if not resume:
            logger.error(f"未找到ID为 {resume_id} 的简历")
            raise ValueError(f"未找到ID为{resume_id}的简历")
        logger.info(f"找到简历: {resume_id}")

        # 使用LLM链式分析生成优化建议
        logger.info("开始LLM链式分析")
        analysis = self.llm_chain.process_resume(jd_text,
                                                 resume['cleaned_text'])
        logger.info("LLM链式分析完成")

        # 基于分析结果生成优化建议
        logger.info("开始生成优化建议")
        suggestions = self._generate_suggestions_from_analysis(
            analysis, resume, jd_text)
        logger.info(f"生成优化建议完成，共生成 {len(suggestions)} 条建议")

        return {
            'analysis': analysis,
            'suggestions': suggestions,
            'resume_id': resume_id
        }

    def submit_application(self, resume_id: str, job_id: str) -> Dict[str, Any]:
        app = {
            'application_id': f"app_{len(self.applications)+1}",
            'resume_id': resume_id,
            'job_id': job_id,
            'status': 'submitted',
            'history': []
        }
        self.applications.append(app)
        return app

    def update_application_status(self, application_id: str, status: str, rejection_text: str = None) -> Dict[str, Any]:
        app = next((a for a in self.applications if a['application_id'] == application_id), None)
        if not app:
            raise ValueError(f"未找到ID为{application_id}的投递")
        app['status'] = status
        entry = {'status': status, 'timestamp': self._get_current_timestamp()}
        if rejection_text:
            resume = next((r for r in self.resumes if r['resume_id'] == app['resume_id']), None)
            jd = next((j for j in self.jobs if j['job_id'] == app['job_id']), None)
            analysis = self.llm_chain.analyze_rejection(rejection_text, resume['cleaned_text'] if resume else '', jd['cleaned_text'] if jd else '')
            entry['rejection_analysis'] = analysis
        app['history'].append(entry)
        return app

    def get_applications(self) -> List[Dict[str, Any]]:
        return self.applications

    def generate_learning_path(self, resume_id: str, jd_text: str) -> Dict[str, Any]:
        resume = next((r for r in self.resumes if r['resume_id'] == resume_id), None)
        if not resume:
            raise ValueError(f"未找到ID为{resume_id}的简历")
        analysis = self.llm_chain.process_resume(jd_text, resume['cleaned_text'])
        sm = analysis.get('step3', {}).get('skill_match', {})
        jd_skills = sm.get('jd_skills', [])
        resume_skills = sm.get('resume_skills', [])
        missing = [s for s in jd_skills if s not in resume_skills]
        return self.llm_chain.generate_learning_path(missing, jd_text)

    def _generate_suggestions_from_analysis(self, analysis: dict,
                                            resume: Dict[str, Any],
                                            jd_text: str) -> List[str]:
        """
        基于LLM分析结果生成简历优化建议
        
        Args:
            analysis: LLM分析结果
            resume: 简历信息
            jd_text: 职位描述文本
        
        Returns:
            简历优化建议列表
        """
        suggestions = []

        # 基于技能匹配生成建议
        skill_match = analysis['step3']['skill_match']
        if skill_match['match_rate'] < 1.0:
            missing_skills = set(skill_match['jd_skills']) - set(
                skill_match['resume_skills'])
            if missing_skills:
                suggestions.append(f"建议在简历中添加以下技能：{', '.join(missing_skills)}")

        # 基于匹配分数生成建议
        if analysis['final_score'] < 0.8:
            suggestions.append("建议优化简历内容，使其更符合职位要求")

        # 添加通用建议
        suggestions.append("建议突出与职位相关的工作经验和项目经历")
        suggestions.append("建议使用量化的成果描述工作业绩")

        return suggestions

    def get_resume_list(self) -> List[Dict[str, Any]]:
        """
        获取简历列表
        
        Returns:
            简历列表
        """
        return self.resumes

    def get_job_list(self) -> List[Dict[str, Any]]:
        """
        获取职位列表
        
        Returns:
            职位列表
        """
        return self.jobs

    def get_matching_results(self,
                             resume_id: str = None) -> List[Dict[str, Any]]:
        """
        获取匹配结果
        
        Args:
            resume_id: 可选的简历ID，用于过滤匹配结果
        
        Returns:
            匹配结果列表
        """
        if resume_id:
            return [
                r for r in self.matching_results if r['resume_id'] == resume_id
            ]
        return self.matching_results

    def _get_current_timestamp(self) -> str:
        """
        获取当前时间戳
        
        Returns:
            当前时间戳字符串
        """
        from datetime import datetime
        return datetime.now().isoformat()
