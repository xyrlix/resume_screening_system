#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简历匹配模块

负责简历与JD的匹配，基于向量相似度和其他特征
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from utils.logger import get_logger

# 初始化日志记录器
logger = get_logger("matcher")


class ResumeMatcher:
    """
    简历匹配类，负责计算简历与JD之间的匹配分数
    """

    def __init__(self, feature_engine):
        """
        初始化简历匹配类
        
        Args:
            feature_engine: 特征工程实例
        """
        self.feature_engine = feature_engine

    def calculate_similarity_score(self, resume_vector: np.ndarray,
                                   jd_vector: np.ndarray) -> float:
        """
        计算简历向量与JD向量之间的相似度分数
        
        Args:
            resume_vector: 简历向量
            jd_vector: JD向量
        
        Returns:
            相似度分数，范围[0, 1]
        """
        similarity = self.feature_engine.compute_similarity(
            resume_vector, jd_vector)
        # 将相似度从[-1, 1]映射到[0, 1]
        return (similarity + 1) / 2

    def calculate_skill_match_score(self, resume_skills: List[str],
                                    jd_skills: List[str]) -> float:
        """
        计算技能匹配分数
        
        Args:
            resume_skills: 简历中的技能列表
            jd_skills: JD中的技能列表
        
        Returns:
            技能匹配分数，范围[0, 1]
        """
        if not jd_skills:
            return 1.0

        # 计算匹配的技能数量
        matching_skills = set(resume_skills) & set(jd_skills)
        return len(matching_skills) / len(jd_skills)

    def calculate_education_match_score(self, resume_education: List[str],
                                        jd_education: List[str]) -> float:
        """
        计算教育背景匹配分数
        
        Args:
            resume_education: 简历中的教育背景列表
            jd_education: JD中的教育背景要求
        
        Returns:
            教育背景匹配分数，范围[0, 1]
        """
        if not jd_education:
            return 1.0

        # 教育背景优先级：博士 > 硕士 > 本科 > 大专 > 中专 > 高中
        education_levels = {
            '博士': 5,
            '硕士': 4,
            '本科': 3,
            '大专': 2,
            '中专': 1,
            '高中': 0
        }

        # 获取简历的最高教育水平
        resume_max_level = max(
            (education_levels.get(edu, 0) for edu in resume_education),
            default=0)

        # 获取JD要求的最低教育水平
        jd_min_level = min(
            (education_levels.get(edu, 0) for edu in jd_education), default=0)

        # 如果简历的最高教育水平 >= JD要求的最低教育水平，则匹配
        return 1.0 if resume_max_level >= jd_min_level else 0.0

    def calculate_experience_match_score(self, resume_experience: List[str],
                                         jd_experience: List[str]) -> float:
        """
        计算工作经验匹配分数
        
        Args:
            resume_experience: 简历中的工作经验列表
            jd_experience: JD中的工作经验要求
        
        Returns:
            工作经验匹配分数，范围[0, 1]
        """
        if not jd_experience:
            return 1.0

        # 从工作经验描述中提取年数
        def extract_years(experience_str: str) -> int:
            import re
            match = re.search(r'\d+', experience_str)
            return int(match.group()) if match else 0

        # 获取简历的工作经验年数
        resume_years = max((extract_years(exp) for exp in resume_experience),
                           default=0)

        # 获取JD要求的工作经验年数
        jd_years = min((extract_years(exp) for exp in jd_experience),
                       default=0)

        # 如果简历的工作经验 >= JD要求的工作经验，则匹配
        return 1.0 if resume_years >= jd_years else 0.0

    def calculate_overall_match_score(
            self,
            resume: Dict[str, Any],
            jd: Dict[str, Any],
            weights: Dict[str, float] = None,
            segment_weights: Dict[str, float] = None) -> float:
        """
        计算简历与JD的综合匹配分数
        
        Args:
            resume: 结构化的简历信息
            jd: 结构化的JD信息
            weights: 各匹配维度的权重字典
        
        Returns:
            综合匹配分数，范围[0, 1]
        """
        # 默认权重
        default_weights = {
            'similarity': 0.6,
            'skills': 0.2,
            'education': 0.1,
            'experience': 0.1
        }

        weights = weights or default_weights

        if resume.get('segment_vectors') and jd.get('segment_vectors'):
            seg_w = segment_weights or {
                'experience': 0.5,
                'skills': 0.3,
                'education': 0.2
            }
            sim_parts = []
            total = 0.0
            for k, w in seg_w.items():
                rv = resume['segment_vectors'].get(k)
                jv = jd['segment_vectors'].get(k)
                if rv is not None and jv is not None:
                    sim_parts.append(
                        self.calculate_similarity_score(rv, jv) * w)
                    total += w
            similarity_score = (
                sum(sim_parts) /
                total) if total > 0 else self.calculate_similarity_score(
                    resume['vector'], jd['vector'])
        else:
            similarity_score = self.calculate_similarity_score(
                resume['vector'], jd['vector'])

        skill_score = self.calculate_skill_match_score(resume['skills'],
                                                       jd['skills'])

        education_score = self.calculate_education_match_score(
            resume['education'], jd.get('education', []))

        experience_score = self.calculate_experience_match_score(
            resume['experience'], jd.get('experience', []))

        # 计算加权综合分数
        overall_score = (similarity_score * weights['similarity'] +
                         skill_score * weights['skills'] +
                         education_score * weights['education'] +
                         experience_score * weights['experience'])

        return overall_score

    def three_stage_filter(
        self,
        resumes: List[Dict[str, Any]],
        jd: Dict[str, Any],
        top_k: int = 10,
        config: Dict[str, Any] = None,
        progress_callback: callable = None
    ) -> List[Tuple[Dict[str, Any], float, Dict[str, Any], Dict[str, Any]]]:
        """
        三级漏斗筛选：向量粗筛 → 规则精筛 → LLM补筛
        
        Args:
            resumes: 简历列表
            jd: JD信息
            top_k: 返回的匹配结果数量
        
        Returns:
            匹配结果列表，每个元素是(简历, 匹配分数, 筛选详情, LLM分析结果)的元组
        """
        cfg = config or {}
        filter_details = {
            'stage1': {
                'total': len(resumes),
                'passed': 0,
                'threshold': float(cfg.get('stage1_threshold', 0.3))
            },
            'stage2': {
                'total': 0,
                'passed': 0,
                'rules': {}
            },
            'stage3': {
                'total': 0,
                'passed': 0,
                'enabled': bool(cfg.get('llm_enabled', True))
            }
        }

        # 1. 一级向量粗筛：使用余弦相似度，筛选Top50相关简历
        logger.info(f"开始一级向量粗筛：共 {len(resumes)} 份简历")
        stage1_results = []
        total_resumes = len(resumes)
        for i, resume in enumerate(resumes):
            similarity_score = self.calculate_similarity_score(
                resume['vector'], jd['vector'])
            if similarity_score >= filter_details['stage1']['threshold']:
                stage1_results.append((resume, similarity_score))

            # 每处理10%的简历更新一次进度
            if progress_callback and (i + 1) % max(1,
                                                   total_resumes // 10) == 0:
                progress = min(20 + (i / total_resumes) * 30, 50)  # 1-50%进度
                progress_callback(progress,
                                  f"正在进行向量粗筛... {i + 1}/{total_resumes}")

        # 按相似度分数降序排序，取Top50
        stage1_results.sort(key=lambda x: x[1], reverse=True)
        stage1_results = stage1_results[:50]

        filter_details['stage1']['passed'] = len(stage1_results)
        filter_details['stage2']['total'] = len(stage1_results)
        logger.info(f"一级向量粗筛完成：通过 {len(stage1_results)} 份简历")

        # 2. 二级规则精筛：使用规则过滤，仅保留10份左右
        logger.info(f"开始二级规则精筛：共 {len(stage1_results)} 份简历")
        stage2_results = []

        # 更新进度
        if progress_callback:
            progress_callback(50, "开始二级规则精筛...")

        # 定义筛选规则
        rules = {
            'education': {
                'operator': 'gte',
                'value': jd.get('entities', {}).get('学历要求', '本科'),
                'field': 'education'
            },
            'skills': {
                'operator': 'contains',
                'value': jd.get('skills', []),
                'field': 'skills'
            },
            'experience': {
                'operator':
                'gte',
                'value':
                jd.get('entities', {}).get('工作年限要求',
                                           str(cfg.get('required_years', 3))),
                'field':
                'experience'
            }
        }

        filter_details['stage2']['rules'] = rules

        total_stage1 = len(stage1_results)
        for i, (resume, similarity_score) in enumerate(stage1_results):
            # 检查是否满足所有规则
            match = True
            reasons = []

            # 更新进度
            if progress_callback:
                progress = 50 + (i / total_stage1) * 30  # 50-80%进度
                progress_callback(progress,
                                  f"正在进行规则精筛... {i + 1}/{total_stage1}")

            # 学历规则
            if 'education' in rules:
                edu_rule = rules['education']

                # 处理不同类型的教育要求
                if isinstance(edu_rule, dict):
                    # 来自JD解析的字典格式
                    edu_value = edu_rule.get('value', '')
                    if edu_value:  # 只有当edu_value不为空时才执行筛选
                        resume_edu = resume.get('education', [])

                        # 教育背景优先级
                        education_levels = {
                            '博士': 5,
                            '硕士': 4,
                            '本科': 3,
                            '大专': 2,
                            '中专': 1,
                            '高中': 0
                        }

                        resume_edu_level = max((education_levels.get(edu, 0)
                                                for edu in resume_edu),
                                               default=0)

                        required_edu_level = education_levels.get(edu_value, 3)

                        if resume_edu_level < required_edu_level:
                            match = False
                            reasons.append(
                                f"学历不满足要求（要求：{edu_value}，实际：{', '.join(resume_edu) or '未提取到'}）"
                            )
                elif isinstance(edu_rule, list):
                    # 来自自定义筛选的列表格式
                    required_educations = edu_rule
                    if required_educations:  # 只有当列表不为空时才执行筛选
                        resume_edu = resume.get('education', [])

                        # 检查简历中是否包含至少一个要求的学历
                        if not any(edu in resume_edu
                                   for edu in required_educations):
                            match = False
                            reasons.append(
                                f"学历不满足要求（要求：{', '.join(required_educations)}，实际：{', '.join(resume_edu) or '未提取到'}）"
                            )

            # 技能规则
            if match and 'skills' in rules:
                skill_rule = rules['skills']
                required_skills = skill_rule['value']
                resume_skills = resume.get('skills', [])

                # 检查是否包含所有必需技能
                if required_skills:
                    matching_skills = set(resume_skills) & set(required_skills)
                    match_rate = len(matching_skills) / len(required_skills)
                    min_rate = float(cfg.get('skills_min_rate', 0.3))
                    if match_rate < min_rate:
                        match = False
                        reasons.append(
                            f"技能匹配率不足（要求：{len(required_skills)}个技能，匹配：{len(matching_skills)}个，匹配率：{match_rate:.2%}）"
                        )

            # 工作经验规则
            if match and 'experience' in rules:
                exp_rule = rules['experience']
                required_exp = exp_rule['value']
                resume_exp = resume.get('experience', [])

                # 提取工作经验年数
                def extract_years(experience_str: str) -> int:
                    import re
                    match = re.search(r'\d+', experience_str)
                    return int(match.group()) if match else 0

                resume_years = max((extract_years(exp) for exp in resume_exp),
                                   default=0)

                required_years = int(required_exp) if str(
                    required_exp).isdigit() else int(
                        cfg.get('required_years', 3))

                if resume_years < required_years:
                    match = False
                    reasons.append(
                        f"工作经验不足（要求：{required_years}年，实际：{resume_years}年）")

            if match:
                stage2_results.append((resume, similarity_score))
                # 输出通过日志
                resume_id = resume.get('resume_id', '未知')
                logger.info(
                    f"简历 {resume_id} 通过二级规则精筛，匹配分数: {similarity_score:.4f}")
            else:
                # 输出不匹配原因
                resume_id = resume.get('resume_id', '未知')
                logger.info(
                    f"简历 {resume_id} 未通过二级规则精筛，原因：{'; '.join(reasons)}")

        # 按相似度分数降序排序，取Top10
        stage2_results.sort(key=lambda x: x[1], reverse=True)
        stage2_results = stage2_results[:10]

        filter_details['stage2']['passed'] = len(stage2_results)
        filter_details['stage3']['total'] = len(stage2_results)
        logger.info(f"二级规则精筛完成：通过 {len(stage2_results)} 份简历")

        # 3. 三级LLM补筛：使用轻量级LLM模型进行隐性需求挖掘
        logger.info(f"开始三级LLM补筛：共 {len(stage2_results)} 份简历")
        stage3_results = []

        # 更新进度
        if progress_callback:
            progress_callback(80, "开始三级LLM补筛...")

        # 导入LLMChain
        from core.llm_chain import LLMChain
        llm_chain = LLMChain()

        boundary = cfg.get('llm_boundary', None)
        total_stage2 = len(stage2_results)
        for i, (resume, similarity_score) in enumerate(stage2_results):
            # 更新进度
            if progress_callback:
                progress = 80 + (i / total_stage2) * 20  # 80-100%进度
                progress_callback(progress,
                                  f"正在进行LLM补筛... {i + 1}/{total_stage2}")

            use_llm = filter_details['stage3']['enabled']
            if boundary and isinstance(boundary,
                                       (list, tuple)) and len(boundary) == 2:
                lo, hi = float(boundary[0]), float(boundary[1])
                use_llm = use_llm and (similarity_score >= lo
                                       and similarity_score <= hi)
            if use_llm:
                llm_analysis = llm_chain.process_resume(
                    jd['cleaned_text'], resume['cleaned_text'])
                overall_score = (self.calculate_overall_match_score(
                    resume, jd, segment_weights=cfg.get('segment_weights')) *
                                 0.7 + llm_analysis['final_score'] * 0.3)
                stage3_results.append((resume, overall_score, llm_analysis))
            else:
                overall_score = self.calculate_overall_match_score(
                    resume, jd, segment_weights=cfg.get('segment_weights'))
                llm_analysis = {'final_score': overall_score}
                stage3_results.append((resume, overall_score, llm_analysis))

            # 输出LLM补筛日志
            resume_id = resume.get('resume_id', '未知')
            logger.info(
                f"简历 {resume_id} LLM补筛完成，综合匹配分数: {overall_score:.4f}，LLM分数: {llm_analysis['final_score']:.4f}"
            )

        # 按综合分数降序排序
        stage3_results.sort(key=lambda x: x[1], reverse=True)

        filter_details['stage3']['passed'] = len(stage3_results)
        logger.info(f"三级LLM补筛完成：通过 {len(stage3_results)} 份简历")

        # 最终结果，取Top k
        final_results = []
        for resume, score, llm_analysis in stage3_results[:top_k]:
            final_results.append((resume, score, filter_details, llm_analysis))

        # 完成所有处理
        if progress_callback:
            progress_callback(100, "匹配完成！")

        logger.info(f"三级漏斗筛选完成：最终通过 {len(final_results)} 份简历")
        return final_results

    def match_resumes_to_jd(
            self,
            resumes: List[Dict[str, Any]],
            jd: Dict[str, Any],
            top_k: int = 10,
            use_three_stage: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        将多个简历与单个JD进行匹配，返回匹配度最高的前k个简历
        
        Args:
            resumes: 简历列表
            jd: JD信息
            top_k: 返回的匹配结果数量
            use_three_stage: 是否使用三级漏斗筛选
        
        Returns:
            匹配结果列表，每个元素是(简历, 匹配分数)的元组，按匹配分数降序排列
        """
        if use_three_stage:
            # 使用三级漏斗筛选
            three_stage_results = self.three_stage_filter(resumes, jd, top_k)
            return [(resume, score)
                    for resume, score, _, _ in three_stage_results]
        else:
            # 传统匹配方式
            # 计算每个简历与JD的匹配分数
            match_results = []
            for resume in resumes:
                match_score = self.calculate_overall_match_score(resume, jd)
                match_results.append((resume, match_score))

            # 按匹配分数降序排序，返回前k个结果
            match_results.sort(key=lambda x: x[1], reverse=True)
            return match_results[:top_k]

    def match_resumes_to_jd_with_llm(
        self,
        resumes: List[Dict[str, Any]],
        jd: Dict[str, Any],
        top_k: int = 10,
        config: Dict[str, Any] = None,
        progress_callback: callable = None
    ) -> List[Tuple[Dict[str, Any], float, Dict[str, Any], Dict[str, Any]]]:
        """
        将多个简历与单个JD进行匹配，返回匹配度最高的前k个简历，包含LLM分析结果
        
        Args:
            resumes: 简历列表
            jd: JD信息
            top_k: 返回的匹配结果数量
            config: 匹配配置
            progress_callback: 进度回调函数，接收(progress_percent, status_message)参数
        
        Returns:
            匹配结果列表，每个元素是(简历, 匹配分数, 筛选详情, LLM分析结果)的元组，按匹配分数降序排列
        """
        return self.three_stage_filter(resumes, jd, top_k, config,
                                       progress_callback)

    def match_resume_to_jds(
            self,
            resume: Dict[str, Any],
            jds: List[Dict[str, Any]],
            top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        将单个简历与多个JD进行匹配，返回匹配度最高的前k个JD
        
        Args:
            resume: 简历信息
            jds: JD列表
            top_k: 返回的匹配结果数量
        
        Returns:
            匹配结果列表，每个元素是(JD, 匹配分数)的元组，按匹配分数降序排列
        """
        # 计算简历与每个JD的匹配分数
        match_results = []
        for jd in jds:
            match_score = self.calculate_overall_match_score(resume, jd)
            match_results.append((jd, match_score))

        # 按匹配分数降序排序，返回前k个结果
        match_results.sort(key=lambda x: x[1], reverse=True)
        return match_results[:top_k]
