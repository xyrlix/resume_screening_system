#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
招聘方前端应用

使用Streamlit框架实现招聘方的核心功能界面
"""

import streamlit as st
import json
from services.recruiter_service import RecruiterService


# 初始化招聘方服务
@st.cache_resource
def init_recruiter_service():
    """
    初始化招聘方服务
    """
    return RecruiterService()


# 页面配置
st.set_page_config(page_title="智能简历筛选系统 - 招聘方", page_icon="🔍", layout="wide")

# 标题
st.title("🔍 智能简历筛选系统 - 招聘方")

# 初始化服务
recruiter_service = init_recruiter_service()

# 侧边栏导航
st.sidebar.title("导航菜单")
page = st.sidebar.radio(
    "选择功能", ["📝 上传JD", "📄 上传简历", "🤝 简历匹配", "🎯 自定义筛选", "🧠 LLM链式分析", "📊 匹配结果"])

# 1. 上传JD页面
if page == "📝 上传JD":
    st.header("上传职位描述 (JD)")

    # JD输入区域
    jd_text = st.text_area("请输入职位描述", height=300)

    # 上传按钮
    if st.button("上传JD"):
        if jd_text.strip():
            with st.spinner("处理JD中..."):
                # 添加JD
                jd = recruiter_service.add_job(jd_text)

                st.success(f"✅ JD上传成功！")
                st.info(f"JD ID: {jd['job_id']}")

                # 显示处理后的JD信息
                with st.expander("查看处理后的JD信息"):
                    st.json(jd)
        else:
            st.error("❌ 请输入职位描述内容")

    # 显示已上传的JD列表
    st.subheader("已上传的JD列表")
    jobs = recruiter_service.get_job_list()

    if jobs:
        for job in jobs:
            with st.expander(f"JD ID: {job['job_id']} - 职位描述"):
                st.write(job['cleaned_text'])
                st.write(f"技能要求: {', '.join(job['skills'])}")
    else:
        st.info("暂无已上传的JD")

# 2. 上传简历页面
elif page == "📄 上传简历":
    st.header("上传简历")

    # 简历输入区域
    resume_text = st.text_area("请输入简历内容", height=300)

    # 上传按钮
    if st.button("上传简历"):
        if resume_text.strip():
            with st.spinner("处理简历中..."):
                # 上传简历
                resume = recruiter_service.upload_resume(resume_text)

                st.success(f"✅ 简历上传成功！")
                st.info(f"简历 ID: {resume['resume_id']}")

                # 显示处理后的简历信息
                with st.expander("查看处理后的简历信息"):
                    st.json(resume)
        else:
            st.error("❌ 请输入简历内容")

    # 显示已上传的简历列表
    st.subheader("已上传的简历列表")
    resumes = recruiter_service.get_resume_list()

    if resumes:
        for resume in resumes:
            with st.expander(f"简历 ID: {resume['resume_id']} - 简历信息"):
                st.write(resume['cleaned_text'])
                st.write(f"技能: {', '.join(resume['skills'])}")
                st.write(f"教育背景: {', '.join(resume['education'])}")
                st.write(f"工作经验: {', '.join(resume['experience'])}")
    else:
        st.info("暂无已上传的简历")

# 3. 简历匹配页面
elif page == "🤝 简历匹配":
    st.header("简历与JD匹配")

    # 选择JD
    jobs = recruiter_service.get_job_list()

    if not jobs:
        st.error("❌ 请先上传JD")
    else:
        job_options = {
            job['job_id']: job['cleaned_text'][:100] + "..."
            for job in jobs
        }
        selected_job_id = st.selectbox(
            "选择要匹配的JD",
            list(job_options.keys()),
            format_func=lambda x: f"{x}: {job_options[x]}")

        # 设置匹配数量
        top_k = st.slider("返回匹配结果数量", min_value=1, max_value=20, value=10)

        # 匹配按钮
        if st.button("开始匹配"):
            resumes = recruiter_service.get_resume_list()

            if not resumes:
                st.error("❌ 请先上传简历")
            else:
                with st.spinner("正在匹配简历..."):
                    # 进行匹配
                    results = recruiter_service.match_resumes_to_job(
                        selected_job_id, top_k)

                    st.success(f"✅ 匹配完成！共找到 {len(results)} 份匹配简历")

                    # 显示匹配结果
                    st.subheader("匹配结果")
                    for i, (resume, score) in enumerate(results, 1):
                        with st.expander(
                                f"{i}. 匹配分数: {score:.4f} - 简历 ID: {resume['resume_id']}"
                        ):
                            st.write(
                                f"**简历内容**: {resume['cleaned_text'][:200]}...")
                            st.write(f"**技能**: {', '.join(resume['skills'])}")
                            st.write(
                                f"**教育背景**: {', '.join(resume['education'])}")
                            st.write(
                                f"**工作经验**: {', '.join(resume['experience'])}")

# 4. 自定义筛选页面
elif page == "🎯 自定义筛选":
    st.header("自定义筛选规则")

    # 初始化筛选规则
    filter_rules = {}

    # 可选硬性条件
    st.subheader("可选硬性条件")

    # 学历要求
    education_options = ["本科", "硕士", "博士", "大专", "中专", "高中"]
    selected_education = st.multiselect("学历要求", education_options)
    if selected_education:
        filter_rules['education'] = selected_education

    # 工作年限要求
    experience = st.text_input("工作年限要求 (例如: 3-5年)")
    if experience.strip():
        filter_rules['experience'] = experience

    # 技能要求
    skills = st.text_input("技能要求 (用逗号分隔)")
    if skills.strip():
        filter_rules['skills'] = [skill.strip() for skill in skills.split(",")]

    # 其他筛选条件可以根据需要添加

    # 显示当前筛选规则
    st.subheader("当前筛选规则")
    if filter_rules:
        st.json(filter_rules)
    else:
        st.info("暂无筛选规则")

    # 应用筛选按钮
    if st.button("应用筛选规则"):
        resumes = recruiter_service.get_resume_list()

        if not resumes:
            st.error("❌ 请先上传简历")
        else:
            with st.spinner("应用筛选规则中..."):
                # 应用筛选规则
                filtered_resumes = recruiter_service.apply_filter_rules(
                    resumes, filter_rules)

                st.success(f"✅ 筛选完成！共找到 {len(filtered_resumes)} 份符合条件的简历")

                # 显示筛选结果
                st.subheader("筛选结果")
                for resume in filtered_resumes:
                    with st.expander(f"简历 ID: {resume['resume_id']}"):
                        st.write(
                            f"**简历内容**: {resume['cleaned_text'][:200]}...")
                        st.write(f"**技能**: {', '.join(resume['skills'])}")
                        st.write(f"**教育背景**: {', '.join(resume['education'])}")
                        st.write(
                            f"**工作经验**: {', '.join(resume['experience'])}")

# 5. LLM链式分析页面
elif page == "🧠 LLM链式分析":
    st.header("LLM链式分析")

    # 选择JD
    jobs = recruiter_service.get_job_list()

    if not jobs:
        st.error("❌ 请先上传JD")
    else:
        job_options = {
            job['job_id']: job['cleaned_text'][:100] + "..."
            for job in jobs
        }
        selected_job_id = st.selectbox(
            "选择JD",
            list(job_options.keys()),
            format_func=lambda x: f"{x}: {job_options[x]}")

        # 选择简历
        resumes = recruiter_service.get_resume_list()

        if not resumes:
            st.error("❌ 请先上传简历")
        else:
            resume_options = {
                resume['resume_id']: resume['cleaned_text'][:100] + "..."
                for resume in resumes
            }
            selected_resume_id = st.selectbox(
                "选择简历",
                list(resume_options.keys()),
                format_func=lambda x: f"{x}: {resume_options[x]}")

            # 开始分析按钮
            if st.button("开始LLM链式分析"):
                # 获取JD和简历内容
                selected_job = next(job for job in jobs
                                    if job['job_id'] == selected_job_id)
                selected_resume = next(
                    resume for resume in resumes
                    if resume['resume_id'] == selected_resume_id)

                with st.spinner("LLM链式分析中..."):
                    # 进行LLM链式分析
                    result = recruiter_service.analyze_resume_with_llm(
                        selected_job['cleaned_text'],
                        selected_resume['cleaned_text'])

                    st.success("✅ LLM链式分析完成！")

                    # 显示分析结果
                    st.subheader("分析结果")

                    # 显示最终匹配分数
                    st.info(f"最终匹配分数: {result['final_score']:.4f}")

                    # 显示链式分析过程
                    st.subheader("LLM链式分析过程")

                    # Step 1: 初步提取
                    with st.expander("Step 1: 初步提取 (Qwen)"):
                        st.json(result['step1'])

                    # Step 2: 验证和修正
                    with st.expander("Step 2: 验证和修正 (DeepSeek)"):
                        st.json(result['step2'])

                    # Step 3: 详细分析
                    with st.expander("Step 3: 详细分析 (OpenAI)"):
                        st.json(result['step3'])

                    # Step 4: 最终评估
                    with st.expander("Step 4: 最终评估 (Openrouter)"):
                        st.json(result['step4'])

# 6. 匹配结果页面
elif page == "📊 匹配结果":
    st.header("匹配结果管理")

    # 获取所有匹配结果
    matching_results = recruiter_service.get_matching_results()

    if not matching_results:
        st.info("暂无匹配结果")
    else:
        # 选择匹配结果
        result_options = {
            f"{i+1}. {result['job_id']} - {result['timestamp'][:19]}": result
            for i, result in enumerate(matching_results)
        }

        selected_result_key = st.selectbox("选择匹配结果",
                                           list(result_options.keys()))
        selected_result = result_options[selected_result_key]

        # 显示匹配结果详情
        st.subheader(f"匹配结果详情")
        st.info(f"JD ID: {selected_result['job_id']}")
        st.info(f"匹配时间: {selected_result['timestamp'][:19]}")

        # 显示匹配结果列表
        st.subheader("匹配简历列表")
        for i, (resume, score) in enumerate(selected_result['results'], 1):
            with st.expander(
                    f"{i}. 匹配分数: {score:.4f} - 简历 ID: {resume['resume_id']}"):
                st.write(f"**简历内容**: {resume['cleaned_text'][:200]}...")
                st.write(f"**技能**: {', '.join(resume['skills'])}")
                st.write(f"**教育背景**: {', '.join(resume['education'])}")
                st.write(f"**工作经验**: {', '.join(resume['experience'])}")

# 页脚
st.sidebar.markdown("---")
st.sidebar.info("智能简历筛选系统 v2.0.0")
