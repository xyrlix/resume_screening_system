#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
求职者前端应用

使用Streamlit框架实现求职者的核心功能界面
"""

import streamlit as st
import json
from services.candidate_service import CandidateService


# 初始化求职者服务
@st.cache_resource
def init_candidate_service():
    """
    初始化求职者服务
    """
    return CandidateService()


# 页面配置
st.set_page_config(page_title="智能简历筛选系统 - 求职者", page_icon="👤", layout="wide")

# 标题
st.title("👤 智能简历筛选系统 - 求职者")

# 初始化服务
candidate_service = init_candidate_service()

# 侧边栏导航
st.sidebar.title("导航菜单")
page = st.sidebar.radio("选择功能", ["📄 上传简历", "🤝 职位匹配", "✨ 简历优化建议", "📊 匹配结果"])

# 1. 上传简历页面
if page == "📄 上传简历":
    st.header("上传简历")

    # 简历输入区域
    resume_text = st.text_area("请输入简历内容", height=300)

    # 上传按钮
    if st.button("上传简历"):
        if resume_text.strip():
            with st.spinner("处理简历中..."):
                # 上传简历
                resume = candidate_service.upload_resume(resume_text)

                st.success(f"✅ 简历上传成功！")
                st.info(f"简历 ID: {resume['resume_id']}")

                # 显示处理后的简历信息
                with st.expander("查看处理后的简历信息"):
                    st.json(resume)
        else:
            st.error("❌ 请输入简历内容")

    # 显示已上传的简历列表
    st.subheader("已上传的简历列表")
    resumes = candidate_service.get_resume_list()

    if resumes:
        for resume in resumes:
            with st.expander(f"简历 ID: {resume['resume_id']} - 简历信息"):
                st.write(resume['cleaned_text'])
                st.write(f"技能: {', '.join(resume['skills'])}")
                st.write(f"教育背景: {', '.join(resume['education'])}")
                st.write(f"工作经验: {', '.join(resume['experience'])}")
    else:
        st.info("暂无已上传的简历")

# 2. 职位匹配页面
elif page == "🤝 职位匹配":
    st.header("职位匹配")

    # 选择简历
    resumes = candidate_service.get_resume_list()

    if not resumes:
        st.error("❌ 请先上传简历")
    else:
        resume_options = {
            resume['resume_id']: resume['cleaned_text'][:100] + "..."
            for resume in resumes
        }
        selected_resume_id = st.selectbox(
            "选择要匹配的简历",
            list(resume_options.keys()),
            format_func=lambda x: f"{x}: {resume_options[x]}")

        # 设置匹配数量
        top_k = st.slider("返回匹配结果数量", min_value=1, max_value=20, value=10)

        # 匹配按钮
        if st.button("开始职位匹配"):
            jobs = candidate_service.get_job_list()

            if not jobs:
                st.warning("⚠ 职位库为空，正在添加示例职位...")
                # 添加示例职位
                sample_jobs = [
                    "Python开发工程师，需要3-5年工作经验，熟悉Python、Django、MySQL等技术",
                    "Java开发工程师，需要5年以上工作经验，熟悉Java、Spring Boot、微服务等技术",
                    "前端开发工程师，需要2-4年工作经验，熟悉JavaScript、React、Vue等技术",
                    "数据分析师，需要3年以上工作经验，熟悉Python、SQL、Tableau等技术",
                    "机器学习工程师，需要3-5年工作经验，熟悉Python、TensorFlow、PyTorch等技术"
                ]

                for job_text in sample_jobs:
                    candidate_service.add_job(job_text)

                st.success("✅ 示例职位添加完成！")

            with st.spinner("正在匹配职位..."):
                # 进行匹配
                results = candidate_service.match_resume_to_jobs(
                    selected_resume_id, top_k)

                st.success(f"✅ 匹配完成！共找到 {len(results)} 个匹配职位")

                # 显示匹配结果
                st.subheader("匹配结果")
                for i, (job, score) in enumerate(results, 1):
                    with st.expander(
                            f"{i}. 匹配分数: {score:.4f} - 职位 ID: {job['job_id']}"
                    ):
                        st.write(f"**职位描述**: {job['cleaned_text'][:200]}...")
                        st.write(f"**技能要求**: {', '.join(job['skills'])}")

# 3. 简历优化建议页面
elif page == "✨ 简历优化建议":
    st.header("简历优化建议")

    # 选择简历
    resumes = candidate_service.get_resume_list()

    if not resumes:
        st.error("❌ 请先上传简历")
    else:
        resume_options = {
            resume['resume_id']: resume['cleaned_text'][:100] + "..."
            for resume in resumes
        }
        selected_resume_id = st.selectbox(
            "选择要优化的简历",
            list(resume_options.keys()),
            format_func=lambda x: f"{x}: {resume_options[x]}")

        # 职位描述输入
        jd_text = st.text_area("请输入目标职位描述", height=200)

        # 生成建议按钮
        if st.button("生成优化建议"):
            if jd_text.strip():
                with st.spinner("正在生成优化建议..."):
                    # 生成优化建议
                    suggestions = candidate_service.generate_resume_optimization_suggestions(
                        selected_resume_id, jd_text)

                    st.success("✅ 优化建议生成完成！")

                    # 显示优化建议
                    st.subheader("简历优化建议")

                    for i, suggestion in enumerate(suggestions['suggestions'],
                                                   1):
                        st.info(f"{i}. {suggestion}")

                    # 显示匹配分数
                    st.subheader("匹配分数")
                    st.info(
                        f"当前简历与目标职位的匹配分数: {suggestions['analysis']['final_score']:.4f}"
                    )

                    # 显示详细分析（可选）
                    with st.expander("查看详细分析"):
                        st.json(suggestions['analysis'])
            else:
                st.error("❌ 请输入目标职位描述")

# 4. 匹配结果页面
elif page == "📊 匹配结果":
    st.header("匹配结果管理")

    # 获取所有匹配结果
    matching_results = candidate_service.get_matching_results()

    if not matching_results:
        st.info("暂无匹配结果")
    else:
        # 选择匹配结果
        result_options = {
            f"{i+1}. {result['resume_id']} - {result['timestamp'][:19]}":
            result
            for i, result in enumerate(matching_results)
        }

        selected_result_key = st.selectbox("选择匹配结果",
                                           list(result_options.keys()))
        selected_result = result_options[selected_result_key]

        # 显示匹配结果详情
        st.subheader(f"匹配结果详情")
        st.info(f"简历 ID: {selected_result['resume_id']}")
        st.info(f"匹配时间: {selected_result['timestamp'][:19]}")

        # 显示匹配结果列表
        st.subheader("匹配职位列表")
        for i, (job, score) in enumerate(selected_result['results'], 1):
            with st.expander(
                    f"{i}. 匹配分数: {score:.4f} - 职位 ID: {job['job_id']}"):
                st.write(f"**职位描述**: {job['cleaned_text'][:200]}...")
                st.write(f"**技能要求**: {', '.join(job['skills'])}")

# 页脚
st.sidebar.markdown("---")
st.sidebar.info("智能简历筛选系统 v2.0.0")
