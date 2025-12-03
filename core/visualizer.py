#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化模块

负责生成各种可视化图表和分析报告
"""

import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple
import numpy as np


class Visualizer:
    """
    可视化类，负责生成各种可视化图表和分析报告
    """

    def __init__(self):
        """
        初始化可视化类
        """
        pass

    def generate_radar_chart(self, resume: Dict[str, Any],
                             jd: Dict[str, Any]) -> go.Figure:
        """
        生成雷达图，展示简历与JD的多维度匹配情况
        
        Args:
            resume: 简历信息
            jd: JD信息
        
        Returns:
            Plotly雷达图对象
        """
        # 定义评估维度
        dimensions = ["技能匹配", "教育背景", "工作经验", "行业匹配", "岗位匹配", "语言能力"]

        # 计算各维度的匹配分数
        scores = []

        # 技能匹配
        jd_skills = jd.get('skills', [])
        resume_skills = resume.get('skills', [])
        skill_match = len(set(jd_skills) & set(resume_skills)) / len(
            jd_skills) if jd_skills else 1.0
        scores.append(skill_match)

        # 教育背景
        education_levels = {
            '博士': 5,
            '硕士': 4,
            '本科': 3,
            '大专': 2,
            '中专': 1,
            '高中': 0
        }
        resume_edu = max((education_levels.get(edu, 0)
                          for edu in resume.get('education', [])),
                         default=0)
        jd_edu = education_levels.get(
            jd.get('entities', {}).get('学历要求', '本科'), 3)
        education_match = 1.0 if resume_edu >= jd_edu else 0.0
        scores.append(education_match)

        # 工作经验
        def extract_years(experience_str: str) -> int:
            import re
            match = re.search(r'\d+', experience_str)
            return int(match.group()) if match else 0

        resume_exp = max(
            (extract_years(exp) for exp in resume.get('experience', [])),
            default=0)
        # 安全地将工作年限要求转换为整数，增加默认值和类型检查

        try:
            jd_exp = int(jd.get('entities', {}).get('工作年限要求', '0') or '0')
        except (ValueError, TypeError):
            jd_exp = 0  # 默认值
        experience_match = min(resume_exp / jd_exp, 1.0) if jd_exp > 0 else 1.0
        scores.append(experience_match)

        # 行业匹配
        industry_match = 1.0  # 简化实现
        scores.append(industry_match)

        # 岗位匹配
        job_match = 1.0  # 简化实现
        scores.append(job_match)

        # 语言能力
        language_match = 1.0  # 简化实现
        scores.append(language_match)

        # 创建雷达图
        fig = go.Figure(data=go.Scatterpolar(
            r=scores, theta=dimensions, fill='toself', name='匹配分数'))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="简历与JD多维度匹配分析",
            showlegend=True)

        return fig

    def generate_pie_chart(
            self, match_results: List[Tuple[Dict[str, Any],
                                            float]]) -> go.Figure:
        """
        生成饼图，展示模型预测结果的匹配情况
        
        Args:
            match_results: 匹配结果列表
        
        Returns:
            Plotly饼图对象
        """
        # 统计不同匹配分数段的简历数量
        score_ranges = {
            "优秀 (0.8-1.0)": 0,
            "良好 (0.6-0.8)": 0,
            "一般 (0.4-0.6)": 0,
            "较差 (0.2-0.4)": 0,
            "很差 (0-0.2)": 0
        }

        for _, score in match_results:
            if score >= 0.8:
                score_ranges["优秀 (0.8-1.0)"] += 1
            elif score >= 0.6:
                score_ranges["良好 (0.6-0.8)"] += 1
            elif score >= 0.4:
                score_ranges["一般 (0.4-0.6)"] += 1
            elif score >= 0.2:
                score_ranges["较差 (0.2-0.4)"] += 1
            else:
                score_ranges["很差 (0-0.2)"] += 1

        # 创建饼图
        fig = go.Figure(data=[
            go.Pie(labels=list(score_ranges.keys()),
                   values=list(score_ranges.values()),
                   hole=.3)
        ])

        fig.update_layout(title="匹配结果分布", showlegend=True)

        return fig

    def generate_interview_questions(self, resume: Dict[str, Any],
                                     jd: Dict[str, Any]) -> List[str]:
        """
        生成面试题，基于简历和JD
        
        Args:
            resume: 简历信息
            jd: JD信息
        
        Returns:
            面试题列表
        """
        # 简化实现，实际项目中应该使用LLM生成
        questions = [
            f"请详细介绍您在{resume.get('entities', {}).get('公司名称', '某公司')}担任{resume.get('entities', {}).get('职位名称', '某职位')}时的主要工作内容和成果。",
            f"根据JD要求，您认为您在{jd.get('entities', {}).get('技能要求', '技能')}方面的优势是什么？",
            f"请描述您在{resume.get('entities', {}).get('项目名称', '某项目')}中遇到的最大挑战以及您是如何解决的。"
        ]

        return questions

    def generate_comprehensive_analysis(self, resume: Dict[str, Any],
                                        jd: Dict[str, Any],
                                        score: float) -> Dict[str, Any]:
        """
        生成候选人的综合分析报告
        
        Args:
            resume: 简历信息
            jd: JD信息
            score: 匹配分数
        
        Returns:
            综合分析报告
        """
        # 简化实现，实际项目中应该使用LLM生成
        analysis = {
            "匹配总分":
            score,
            "优势": [
                f"技能匹配度高，掌握{', '.join(resume.get('skills', []))}等技能",
                f"教育背景符合要求，拥有{', '.join(resume.get('education', []))}学历",
                f"工作经验丰富，具备{', '.join(resume.get('experience', []))}经验"
            ],
            "风险":
            ["需要进一步了解候选人的项目成果细节", "需要评估候选人的团队协作能力", "需要了解候选人的薪资期望是否符合公司预算"],
            "建议": [
                "安排技术面试，深入了解候选人的技术能力", "邀请候选人进行项目分享，评估其项目经验",
                "与候选人沟通薪资期望和职业发展规划"
            ]
        }

        return analysis

    def generate_filter_funnel_chart(
            self, filter_details: Dict[str, Any]) -> go.Figure:
        """
        生成筛选漏斗图，展示三级漏斗筛选的结果
        
        Args:
            filter_details: 筛选详情
        
        Returns:
            Plotly漏斗图对象
        """
        # 定义漏斗阶段
        stages = ["简历总数", "向量粗筛通过", "规则精筛通过", "最终匹配通过"]

        # 获取各阶段的数量
        values = [
            filter_details.get('stage1', {}).get('total', 0),
            filter_details.get('stage1', {}).get('passed', 0),
            filter_details.get('stage2', {}).get('passed', 0),
            filter_details.get('stage3', {}).get('passed', 0)
        ]

        # 创建漏斗图
        fig = go.Figure(
            go.Funnel(
                y=stages,
                x=values,
                textinfo="value+percent initial",
                marker={"color": ["#636EFA", "#00CC96", "#EF553B",
                                  "#AB63FA"]}))

        fig.update_layout(title="三级漏斗筛选结果", showlegend=True)

        return fig
