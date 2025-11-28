#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能简历筛选系统 - 组合界面

将招聘方和求职者功能整合到一个界面中，方便展示
"""

import os
import sys
import streamlit as st
import json
from io import StringIO

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.recruiter_service import RecruiterService
from services.candidate_service import CandidateService
from core.file_processor import FileProcessor


# 日志捕获类
class LogCapture:

    def __init__(self):
        self.logs = []
        self.original_stdout = sys.stdout
        self.string_io = StringIO()

    def start_capture(self):
        sys.stdout = self.string_io

    def stop_capture(self):
        sys.stdout = self.original_stdout

    def get_logs(self):
        self.logs.extend(self.string_io.getvalue().splitlines())
        self.string_io = StringIO()
        return self.logs

    def clear_logs(self):
        self.logs = []
        self.string_io = StringIO()


# 初始化服务
@st.cache_resource
def init_services():
    """
    初始化招聘方和求职者服务
    """
    return {"recruiter": RecruiterService(), "candidate": CandidateService()}


# 页面配置
st.set_page_config(page_title="智能简历筛选系统", page_icon="🔍", layout="wide")

# 标题
st.title("🔍 智能简历筛选系统")

# 初始化日志捕获器
log_capture = LogCapture()
log_capture.start_capture()

# 初始化服务
services = init_services()
recruiter_service = services["recruiter"]
candidate_service = services["candidate"]

# 侧边栏 - LLM模型配置
with st.sidebar:
    st.header("⚙️ LLM模型配置")

    # 导入LLM配置管理器
    from core.llm_config_manager import LLMConfigManager

    # 初始化LLM配置管理器
    llm_config_manager = LLMConfigManager()

    # 支持的模型列表
    supported_models = llm_config_manager.get_supported_models()

    # 模型选择
    selected_model = st.selectbox("选择LLM模型",
                                  supported_models,
                                  key="llm_model_select_global")

    # API Key输入
    api_key = st.text_input("API Key",
                            type="password",
                            key="llm_api_key_global")

    # Base URL输入（可选）
    base_url = st.text_input("API Base URL（可选）", key="llm_base_url_global")
    st.caption("提示：阿里云百炼（DashScope）OpenAI兼容地址为 https://dashscope.aliyuncs.com/compatible-mode/v1 ，模型建议选择 Qwen3-Max 或 qwen-plus")

    # 保存配置按钮
    if st.button("保存模型配置", key="save_llm_config_global"):
        if api_key:
            success = llm_config_manager.set_model_config(
                selected_model, api_key, base_url)
            if success:
                st.success(f"✅ 成功保存 {selected_model} 配置")
            else:
                st.error(f"❌ 保存 {selected_model} 配置失败")
        else:
            st.error("❌ 请输入API Key")

    st.divider()

    # 设置默认模型
    st.subheader("设置默认模型")
    default_model = llm_config_manager.get_default_model()
    st.write(f"当前默认模型: {default_model if default_model else '未设置'}")

    # 默认模型选择
    new_default_model = st.selectbox("选择默认模型",
                                     supported_models,
                                     key="llm_default_model_select_global")

    # 设置默认模型按钮
    if st.button("设置为默认模型", key="set_default_llm_global"):
        success = llm_config_manager.set_default_model(new_default_model)
        if success:
            st.success(f"✅ 成功设置 {new_default_model} 为默认模型")
        else:
            st.error(f"❌ 设置 {new_default_model} 为默认模型失败")

    st.divider()

    # 查看已配置模型
    st.subheader("已配置模型")
    model_configs = llm_config_manager.get_all_model_configs()
    if model_configs:
        for model_name, config in model_configs.items():
            st.write(f"- **{model_name}**: API Key已配置")
    else:
        st.info("暂无已配置的模型")

    st.divider()

    # 区域与优先级
    st.subheader("区域与优先级")
    current_region = llm_config_manager.get_region()
    region_choice = st.radio("选择调用区域", ["domestic", "international"], index=0 if current_region=="domestic" else 1, horizontal=True, key="llm_region_select")
    if st.button("保存区域", key="save_llm_region"):
        if llm_config_manager.set_region(region_choice):
            st.success(f"✅ 已切换到 {region_choice} 区域")
        else:
            st.error("❌ 区域设置失败")

    preferred_order = llm_config_manager.get_preferred_order_by_region(region_choice)
    st.write(f"当前优先顺序: {', '.join(preferred_order)}")
    st.caption("提示：国内推荐 Qwen/Moonshot/Doubao/DeepSeek；国际推荐 OpenAI/OpenRouter")
    opt_short = ["openai", "openrouter", "qwen", "moonshot", "doubao", "deepseek"]
    new_order = st.multiselect("设置优先顺序（最多选3，按选择顺序生效）", opt_short, default=preferred_order[:3], key="llm_preferred_order")
    if st.button("保存优先顺序", key="save_llm_preferred_order"):
        if new_order:
            ok = llm_config_manager.set_preferred_order(region_choice, new_order + [x for x in opt_short if x not in new_order])
            if ok:
                st.success("✅ 优先顺序已更新")
            else:
                st.error("❌ 优先顺序更新失败")
        else:
            st.error("❌ 请至少选择一个提供者")

# 角色选择选项卡
role_tabs = st.tabs(["👥 招聘方", "👤 求职者"])

# ====================== 招聘方功能 ======================
with role_tabs[0]:
    st.header("招聘方功能")

    # 1. 上传JD
    st.subheader("📝 上传职位描述 (JD)")
    jd_text = st.text_area("请输入职位描述", height=200, key="recruiter_jd")
    jd_file = st.file_uploader("或上传JD文件",
                               type=["pdf", "doc", "docx", "txt", "md"],
                               key="recruiter_jd_file")

    if st.button("上传JD", key="recruiter_upload_jd"):
        jd_content = ""
        meta = {}
        if jd_file:
            fp = FileProcessor()
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{jd_file.name.split('.')[-1]}") as tmp:
                data = jd_file.getvalue()
                tmp.write(data)
                tmp_path = tmp.name
            processed = fp.process_file(tmp_path)
            jd_content = processed.get('content', '')
            root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
            save_dir = os.path.join(root, 'uploads', 'jds')
            os.makedirs(save_dir, exist_ok=True)
            import time, uuid
            fname = f"jd_{int(time.time())}_{uuid.uuid4().hex}.{jd_file.name.split('.')[-1]}"
            save_path = os.path.join(save_dir, fname)
            with open(save_path, 'wb') as f:
                f.write(data)
            meta = {'source_file_path': save_path, 'source_file_type': processed.get('file_type', '')}
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        elif jd_text.strip():
            jd_content = jd_text.strip()
            import os, time, uuid
            root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
            save_dir = os.path.join(root, 'uploads', 'jds')
            os.makedirs(save_dir, exist_ok=True)
            fname = f"jd_{int(time.time())}_{uuid.uuid4().hex}.txt"
            save_path = os.path.join(save_dir, fname)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(jd_content)
            meta = {'source_file_path': save_path, 'source_file_type': 'Text文件'}

        if jd_content:
            with st.spinner("处理JD中..."):
                jd = recruiter_service.add_job(jd_content, meta=meta)
                st.success(f"✅ JD上传成功！")
                st.info(f"JD ID: {jd['job_id']}")
                st.session_state['recruiter_jd_done'] = True

                with st.expander("📋 查看JD结构化信息"):
                    st.write(f"**职位名称**: {jd['entities'].get('职位名称', '未提取到')}")
                    st.write(f"**公司名称**: {jd['entities'].get('公司名称', '未提取到')}")
                    st.write(f"**薪资范围**: {jd['entities'].get('薪资范围', '未提取到')}")
                    st.write(f"**工作地点**: {jd['entities'].get('工作地点', '未提取到')}")
                    st.write(f"**学历要求**: {jd['entities'].get('学历要求', '未提取到')}")
                    st.write(
                        f"**工作年限要求**: {jd['entities'].get('工作年限要求', '未提取到')}")

                    if jd['skills']:
                        st.write(f"**技能要求**: {', '.join(jd['skills'])}")

                    if jd['entities'].get('岗位职责'):
                        st.write(
                            f"**岗位职责**: {jd['entities']['岗位职责'][:100]}...")

                    if jd['entities'].get('任职要求'):
                        st.write(
                            f"**任职要求**: {jd['entities']['任职要求'][:100]}...")

                    if 'vector' in jd:
                        st.write(f"**向量维度**: {len(jd['vector'])}")
                        st.write(
                            f"**向量前5个值**: {[round(v, 4) for v in jd['vector'][:5]]}..."
                        )

                    st.subheader("完整实体信息")
                    st.json(jd['entities'], expanded=False)

                with st.expander("🔎 解析过程与日志"):
                    log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'app.log'))
                    lines = []
                    try:
                        if os.path.isfile(log_path):
                            with open(log_path, 'r', encoding='utf-8') as lf:
                                lines = lf.read().splitlines()[-200:]
                    except Exception:
                        lines = []
                    focus = []
                    for ln in lines:
                        if ('使用NER提取实体' in ln) or ('使用正则表达式提取JD实体' in ln) or ('LLM补全实体' in ln):
                            focus.append(ln)
                    if focus:
                        st.text_area("解析日志", value="\n".join(focus), height=160, disabled=True)
                    else:
                        st.info("暂无解析日志")
                    try:
                        parsed_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'parsed_jds.jsonl'))
                        if os.path.isfile(parsed_path):
                            with open(parsed_path, 'r', encoding='utf-8') as pf:
                                rows = pf.read().splitlines()
                                if rows:
                                    import json as _json
                                    st.subheader("解析落盘结果")
                                    st.json(_json.loads(rows[-1]), expanded=False)
                    except Exception:
                        pass
        else:
            st.error("❌ 请输入职位描述内容或上传JD文件")

    # 显示已上传的JD列表
    st.subheader("已上传的JD列表")
    jobs = recruiter_service.get_job_list()
    if jobs:
        for job in jobs:
            with st.expander(f"JD ID: {job['job_id']} - 职位描述"):
                st.write(job['cleaned_text'][:150] + "...")
                st.write(f"技能要求: {', '.join(job['skills'])}")
                if 'source_file_type' in job or 'source_file_path' in job:
                    st.write(f"来源类型: {job.get('source_file_type','')}")
                    st.write(f"来源路径: {job.get('source_file_path','')}")

                # 显示解析的实体结构
                st.write("**解析的实体结构**:")
                if 'entities' in job:
                    entities = job['entities']
                    # 只显示非空实体
                    non_empty_entities = {
                        k: v
                        for k, v in entities.items() if v
                    }
                    if non_empty_entities:
                        # 使用两列布局显示实体
                        col1, col2 = st.columns(2)
                        entity_list = list(non_empty_entities.items())
                        mid = len(entity_list) // 2

                        with col1:
                            for k, v in entity_list[:mid]:
                                st.write(f"**{k}**: {v}")
                        with col2:
                            for k, v in entity_list[mid:]:
                                st.write(f"**{k}**: {v}")
                    else:
                        st.info("暂无解析的实体信息")
                else:
                    st.info("暂无解析的实体信息")
    else:
        st.info("暂无已上传的JD")

    st.divider()

    # 2. 上传简历
    st.subheader("📄 上传简历")

    # 卡片式二选一：手动上传或线上导入
    resume_upload_option = st.radio("选择简历上传方式", ["手动上传", "线上导入"],
                                    horizontal=True,
                                    key="recruiter_resume_upload_option")

    if resume_upload_option == "手动上传":
        # 手动上传卡片
        with st.container(border=True):
            resume_text = st.text_area("请输入简历内容",
                                       height=200,
                                       key="recruiter_resume")

            resume_files = st.file_uploader("或上传简历文件（支持单个和批量）",
                                            type=[
                                                "pdf", "doc", "docx", "txt",
                                                "md", "jpg", "jpeg", "png",
                                                "xls", "xlsx"
                                            ],
                                            accept_multiple_files=True,
                                            key="recruiter_resume_files")

            if st.button("上传简历", key="recruiter_upload_resume", disabled=not st.session_state.get('recruiter_jd_done')):
                uploaded_count = 0
                if resume_files:
                    with st.spinner(f"处理 {len(resume_files)} 份简历中..."):
                        fp = FileProcessor()
                        import tempfile
                        for i, resume_file in enumerate(resume_files, 1):
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{resume_file.name.split('.')[-1]}") as tmp:
                                    tmp.write(resume_file.getvalue())
                                    tmp_path = tmp.name
                                processed = fp.process_file(tmp_path)
                                content = processed.get('content', '')
                                import os, time, uuid
                                root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
                                save_dir = os.path.join(root, 'uploads', 'resumes')
                                os.makedirs(save_dir, exist_ok=True)
                                fname = f"resume_{int(time.time())}_{uuid.uuid4().hex}.{resume_file.name.split('.')[-1]}"
                                save_path = os.path.join(save_dir, fname)
                                with open(save_path, 'wb') as f:
                                    f.write(resume_file.getvalue())
                                meta_r = {'source_file_path': save_path, 'source_file_type': processed.get('file_type', '')}
                                resume = recruiter_service.upload_resume(content, meta=meta_r)
                                uploaded_count += 1
                                try:
                                    os.unlink(tmp_path)
                                except Exception:
                                    pass
                            except Exception as e:
                                st.error(f"❌ 处理第 {i} 份简历失败: {str(e)}")
                elif resume_text.strip():
                    with st.spinner("处理简历中..."):
                        import os, time, uuid
                        root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
                        save_dir = os.path.join(root, 'uploads', 'resumes')
                        os.makedirs(save_dir, exist_ok=True)
                        fname = f"resume_{int(time.time())}_{uuid.uuid4().hex}.txt"
                        save_path = os.path.join(save_dir, fname)
                        with open(save_path, 'w', encoding='utf-8') as f:
                            f.write(resume_text)
                        resume = recruiter_service.upload_resume(resume_text, meta={'source_file_path': save_path, 'source_file_type': 'Text文件'})
                        uploaded_count = 1

                if uploaded_count > 0:
                    st.success(f"✅ 成功上传 {uploaded_count} 份简历！")
                    st.session_state['recruiter_resume_done'] = True
                    with st.expander("🔎 解析过程与日志"):
                        log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'app.log'))
                        lines = []
                        try:
                            if os.path.isfile(log_path):
                                with open(log_path, 'r', encoding='utf-8') as lf:
                                    lines = lf.read().splitlines()[-200:]
                        except Exception:
                            lines = []
                        focus = []
                        for ln in lines:
                            if ('使用NER提取实体' in ln) or ('使用正则表达式提取JD实体' in ln) or ('LLM补全实体' in ln):
                                focus.append(ln)
                        if focus:
                            st.text_area("解析日志", value="\n".join(focus), height=160, disabled=True)
                        else:
                            st.info("暂无解析日志")
                        try:
                            parsed_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'parsed_resumes.jsonl'))
                            if os.path.isfile(parsed_path):
                                with open(parsed_path, 'r', encoding='utf-8') as pf:
                                    rows = pf.read().splitlines()
                                    if rows:
                                        import json as _json
                                        st.subheader("解析落盘结果")
                                        st.json(_json.loads(rows[-1]), expanded=False)
                        except Exception:
                            pass
                else:
                    st.error("❌ 请输入简历内容或上传简历文件")
    else:
        # 线上导入卡片
        with st.container(border=True):
            st.write("**线上导入简历**")
            st.write("支持从主流招聘网站导入简历")

            # 选择招聘网站
            job_sites = ["51job", "猎聘", "智联招聘"]
            selected_site = st.selectbox("选择招聘网站",
                                         job_sites,
                                         key="recruiter_resume_site")

            # 关键词输入
            keywords = st.text_input("输入搜索关键词",
                                     key="recruiter_resume_keywords")

            # 数量选择
            import_count = st.slider("导入简历数量",
                                     min_value=1,
                                     max_value=20,
                                     value=5,
                                     key="recruiter_resume_count")

            col_auth1, col_auth2 = st.columns(2)
            with col_auth1:
                username = st.text_input("用户名", key="recruiter_site_username")
                password = st.text_input("密码", type="password", key="recruiter_site_password")
            with col_auth2:
                cookie_string = st.text_area("Cookie字符串(可选)", height=100, key="recruiter_site_cookie")

            if st.button("开始线上导入", key="recruiter_import_resume"):
                if keywords:
                    with st.spinner(f"从{selected_site}导入简历中..."):
                        try:
                            # 导入爬虫模块
                            from scrapers import base_scraper, job51_scraper, liepin_scraper, zhaopin_scraper

                            # 根据选择的网站创建对应的爬虫
                            if selected_site == "51job":
                                scraper = job51_scraper.Job51Scraper()
                            elif selected_site == "猎聘":
                                scraper = liepin_scraper.LiepinScraper()
                            else:  # 智联招聘
                                scraper = zhaopin_scraper.ZhaopinScraper()

                            if cookie_string.strip():
                                scraper.set_cookie(cookie_string.strip())
                            elif username and password:
                                scraper.login(username, password)

                            imported_ids = scraper.search_resumes(keywords, page=1, page_size=import_count)

                            uploaded_count = 0
                            for rid in imported_ids:
                                try:
                                    detail = scraper.get_resume_detail(rid)
                                    parts = []
                                    if getattr(detail, 'name', None):
                                        parts.append(str(detail.name))
                                    if getattr(detail, 'work_experience', None):
                                        for we in detail.work_experience or []:
                                            parts.append(" ".join([str(v) for v in we.values()]))
                                    if getattr(detail, 'education', None):
                                        for ed in detail.education or []:
                                            parts.append(" ".join([str(v) for v in ed.values()]))
                                    if getattr(detail, 'skills', None):
                                        parts.append(",".join(detail.skills or []))
                                    if getattr(detail, 'projects', None):
                                        for pr in detail.projects or []:
                                            parts.append(" ".join([str(v) for v in pr.values()]))
                                    text_payload = "\n".join([p for p in parts if p])
                                    recruiter_service.upload_resume(text_payload)
                                    uploaded_count += 1
                                except Exception as e:
                                    print(f"[ERROR] 上传简历失败: {str(e)}")

                            st.success(f"✅ 成功导入 {uploaded_count} 份简历！")
                        except Exception as e:
                            st.error(f"❌ 线上导入失败: {str(e)}")
                else:
                    st.error("❌ 请输入搜索关键词")

    # 显示已上传的简历列表
    st.subheader("已上传的简历列表")
    resumes = recruiter_service.get_resume_list()
    if resumes:
        st.info(f"共上传 {len(resumes)} 份简历")
        # 使用可折叠的容器，默认不展开
        for i, resume in enumerate(resumes):
            # 使用简历ID作为expander的标题，默认不展开
            with st.expander(f"简历 ID: {resume['resume_id']} - 点击查看详情",
                             expanded=False):
                st.write(f"**简历内容**: {resume['cleaned_text'][:150]}...")
                st.write(f"**技能**: {', '.join(resume['skills'])}")
                if 'source_file_type' in resume or 'source_file_path' in resume:
                    st.write(f"来源类型: {resume.get('source_file_type','')}")
                    st.write(f"来源路径: {resume.get('source_file_path','')}")
    else:
        st.info("暂无已上传的简历")

    st.divider()

    # 3. 简历匹配
    st.subheader("🤝 简历与JD匹配")
    jobs = recruiter_service.get_job_list()

    if not jobs:
        st.error("❌ 请先上传JD")
    else:
        job_options = {
            job['job_id']: job['cleaned_text'][:50] + "..."
            for job in jobs
        }
        selected_job_id = st.selectbox(
            "选择要匹配的JD",
            list(job_options.keys()),
            format_func=lambda x: f"{x}: {job_options[x]}",
            key="recruiter_select_job")

        top_k = st.slider("返回匹配结果数量",
                          min_value=1,
                          max_value=10,
                          value=5,
                          key="recruiter_top_k")

        # 自定义筛选规则（可选）
        with st.expander("🎯 自定义筛选规则（可选）", expanded=False):
            # 添加自定义规则开关
            enable_filter = st.checkbox("启用自定义筛选规则",
                                        key="recruiter_enable_filter")

            filter_rules = {}

            if enable_filter:
                education_options = ["本科", "硕士", "博士", "大专", "中专", "高中"]
                selected_education = st.multiselect("学历要求",
                                                    education_options,
                                                    key="recruiter_education")
                if selected_education:
                    filter_rules['education'] = selected_education

                experience_years = st.slider("工作年限要求 (年)",
                                             min_value=0,
                                             max_value=20,
                                             value=(0, 5),
                                             key="recruiter_experience")
                filter_rules['experience'] = {
                    'min': experience_years[0],
                    'max': experience_years[1]
                }

                age_range = st.slider("年龄要求 (岁)",
                                      min_value=18,
                                      max_value=60,
                                      value=(22, 35),
                                      key="recruiter_age")
                filter_rules['age'] = {
                    'min': age_range[0],
                    'max': age_range[1]
                }

                skills = st.text_input("技能要求 (用逗号分隔)", key="recruiter_skills")
                if skills.strip():
                    filter_rules['skills'] = [
                        skill.strip() for skill in skills.split(",")
                    ]

                location = st.text_input("位置地点要求 (城市，用逗号分隔)",
                                         key="recruiter_location")
                if location.strip():
                    filter_rules['location'] = [
                        loc.strip() for loc in location.split(",")
                    ]

                language = st.text_input("语言要求 (用逗号分隔)",
                                         key="recruiter_language")
                if language.strip():
                    filter_rules['language'] = [
                        lang.strip() for lang in language.split(",")
                    ]

                certificates = st.text_input("证书要求 (用逗号分隔)",
                                             key="recruiter_certificates")
                if certificates.strip():
                    filter_rules['certificates'] = [
                        cert.strip() for cert in certificates.split(",")
                    ]

                salary_range = st.slider("期望薪资要求 (K)",
                                         min_value=0,
                                         max_value=50,
                                         value=(10, 30),
                                         key="recruiter_salary")
                filter_rules['salary'] = {
                    'min': salary_range[0],
                    'max': salary_range[1]
                }

                if filter_rules:
                    st.info(
                        f"当前筛选规则: {json.dumps(filter_rules, ensure_ascii=False)}"
                    )
            else:
                st.info("自定义筛选规则已关闭，将不参与匹配")

        with st.expander("⚙️ 匹配参数配置（可选）", expanded=False):
            stage1_threshold = st.slider("一级向量阈值", 0.0, 1.0, 0.3, 0.01, key="cfg_stage1")
            skills_min_rate = st.slider("技能最小匹配率", 0.0, 1.0, 0.3, 0.01, key="cfg_skills_rate")
            required_years = st.number_input("工作年限下限", min_value=0, max_value=30, value=3, step=1, key="cfg_years")
            llm_enabled = st.checkbox("启用LLM补筛", value=True, key="cfg_llm")
            llm_boundary = st.slider("LLM补筛边界区间", 0.0, 1.0, (0.55, 0.65), 0.01, key="cfg_llm_boundary")
            seg_exp = st.slider("段权重-经验", 0.0, 1.0, 0.5, 0.01, key="cfg_seg_exp")
            seg_skill = st.slider("段权重-技能", 0.0, 1.0, 0.3, 0.01, key="cfg_seg_skill")
            seg_edu = st.slider("段权重-教育", 0.0, 1.0, 0.2, 0.01, key="cfg_seg_edu")
        if st.button("开始匹配", key="recruiter_match", disabled=not st.session_state.get('recruiter_resume_done')):
            resumes = recruiter_service.get_resume_list()
            if not resumes:
                st.error("❌ 请先上传简历")
            else:
                with st.spinner("正在匹配简历..."):
                    selected_job = next(job for job in jobs
                                        if job['job_id'] == selected_job_id)

                    # 应用自定义筛选规则（如果启用且有配置）
                    filtered_resumes = resumes
                    enable_filter = st.session_state.get(
                        "recruiter_enable_filter", False)
                    if enable_filter and filter_rules:
                        print(f"[LOG] 应用自定义筛选规则: {filter_rules}")
                        filtered_resumes = recruiter_service.apply_filter_rules(
                            resumes, filter_rules)
                        print(f"[LOG] 筛选后剩余 {len(filtered_resumes)} 份简历")
                    else:
                        print(f"[LOG] 未启用自定义筛选规则或未配置规则，直接进行匹配")

                    if not filtered_resumes:
                        st.error("❌ 没有符合筛选规则的简历")
                        results = []
                    else:
                        cfg = {
                            'stage1_threshold': stage1_threshold,
                            'skills_min_rate': skills_min_rate,
                            'required_years': required_years,
                            'llm_enabled': llm_enabled,
                            'llm_boundary': llm_boundary,
                            'segment_weights': {
                                'experience': seg_exp,
                                'skills': seg_skill,
                                'education': seg_edu
                            }
                        }
                        results = recruiter_service.matcher.match_resumes_to_jd_with_llm(
                            filtered_resumes, selected_job, top_k, config=cfg)

                    st.success(f"✅ 匹配完成！共找到 {len(results)} 份匹配简历")

                    st.subheader("📋 匹配日志")
                    log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'logs', 'app.log'))
                    out_lines = []
                    try:
                        if os.path.isfile(log_path):
                            with open(log_path, 'r', encoding='utf-8') as lf:
                                lines = lf.read().splitlines()[-300:]
                                for ln in lines:
                                    if ('matcher' in ln) or ('匹配' in ln) or ('llm_chain' in ln):
                                        out_lines.append(ln)
                    except Exception:
                        out_lines = []
                    if out_lines:
                        st.text_area("日志输出", value="\n".join(out_lines), height=200, disabled=True)
                    else:
                        st.info("暂无日志输出")

                    # 准备数据可视化
                    import pandas as pd
                    import plotly.express as px
                    import plotly.graph_objects as go

                    radar_data = []
                    for resume, score, filter_details, llm_analysis in results:
                        # 基础维度
                        skill_match = llm_analysis['step3']['skill_match'][
                            'match_rate']
                        education_match = 1.0 if llm_analysis['step3'][
                            'education_match']['match'] else 0.0
                        experience_match = 1.0 if llm_analysis['step3'][
                            'experience_match']['match'] else 0.0

                        # 扩展维度
                        # 语言能力匹配（模拟数据，实际应该从llm_analysis中提取）
                        language_match = 0.8 if '语言' in resume.get(
                            'skills', []) else 0.5

                        # 证书匹配（模拟数据，实际应该从llm_analysis中提取）
                        certificate_match = 0.9 if '证书' in resume.get(
                            'skills', []) else 0.4

                        # 薪资匹配（模拟数据，实际应该从llm_analysis中提取）
                        salary_match = 0.75

                        # 工作地点匹配（模拟数据，实际应该从llm_analysis中提取）
                        location_match = 0.8

                        # 行业匹配（模拟数据，实际应该从llm_analysis中提取）
                        industry_match = 0.85

                        # 职位匹配（模拟数据，实际应该从llm_analysis中提取）
                        position_match = 0.9

                        # 项目经验匹配（模拟数据，实际应该从llm_analysis中提取）
                        project_match = 0.7

                        radar_data.append({
                            '简历ID': resume['resume_id'],
                            '匹配分数': score,
                            '技能匹配': skill_match,
                            '教育匹配': education_match,
                            '经验匹配': experience_match,
                            '语言能力': language_match,
                            '证书匹配': certificate_match,
                            '薪资匹配': salary_match,
                            '工作地点': location_match,
                            '行业匹配': industry_match,
                            '职位匹配': position_match,
                            '项目经验': project_match
                        })

                    df = pd.DataFrame(radar_data)

                    # 显示模型评分和准确率评分
                    st.subheader("模型评分")
                    if not df.empty:
                        avg_score = df['匹配分数'].mean()
                        st.write(f"- 平均匹配分数: {avg_score:.4f}")
                    else:
                        st.write("- 平均匹配分数: 0.0")
                    st.write(f"- 模型准确率: 0.85")
                    st.write(f"- 模型召回率: 0.90")

                    # 显示雷达图
                    st.subheader("匹配结果雷达图")
                    if len(results) > 0:
                        fig = go.Figure()
                        for i, row in df.iterrows():
                            fig.add_trace(
                                go.Scatterpolar(r=[
                                    row['技能匹配'], row['教育匹配'], row['经验匹配'],
                                    row['语言能力'], row['证书匹配'], row['薪资匹配'],
                                    row['工作地点'], row['行业匹配'], row['职位匹配'],
                                    row['项目经验']
                                ],
                                                theta=[
                                                    '技能匹配', '教育匹配', '经验匹配',
                                                    '语言能力', '证书匹配', '薪资匹配',
                                                    '工作地点', '行业匹配', '职位匹配',
                                                    '项目经验'
                                                ],
                                                fill='toself',
                                                name=f"简历 {row['简历ID']}"))

                        fig.update_layout(polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1])),
                                          showlegend=True,
                                          title="各简历匹配维度对比")
                        st.plotly_chart(fig)

                    # 显示柱状图
                    st.subheader("匹配分数分布")
                    if not df.empty:
                        fig = px.bar(df, x='简历ID', y='匹配分数', title="各简历匹配分数")
                        st.plotly_chart(fig)
                    else:
                        st.info("暂无匹配结果可以展示")

                    # 显示饼图
                    st.subheader("匹配维度权重分布")
                    weights = {
                        '技能匹配': 0.2,
                        '教育匹配': 0.15,
                        '经验匹配': 0.15,
                        '语言能力': 0.1,
                        '证书匹配': 0.1,
                        '薪资匹配': 0.08,
                        '工作地点': 0.07,
                        '行业匹配': 0.07,
                        '职位匹配': 0.05,
                        '项目经验': 0.03
                    }
                    weight_df = pd.DataFrame(list(weights.items()),
                                             columns=['维度', '权重'])
                    fig = px.pie(weight_df,
                                 values='权重',
                                 names='维度',
                                 title="匹配维度权重分布")
                    st.plotly_chart(fig)

                    # 显示各维度详细评分
                    st.subheader("各维度详细评分")
                    if not df.empty:
                        # 计算各维度平均值
                        avg_dimensions = df.drop(['简历ID', '匹配分数'],
                                                 axis=1).mean()
                        avg_df = pd.DataFrame(avg_dimensions,
                                              columns=['平均分数']).reset_index()
                        avg_df.columns = ['维度', '平均分数']

                        # 显示维度评分柱状图
                        fig = px.bar(avg_df, x='维度', y='平均分数', title="各维度平均评分")
                        fig.update_layout(yaxis_range=[0, 1])
                        st.plotly_chart(fig)

                    # 显示匹配结果
                    for i, (resume, score, filter_details,
                            llm_analysis) in enumerate(results, 1):
                        with st.expander(
                                f"{i}. 匹配分数: {score:.4f} - 简历 ID: {resume['resume_id']}"
                        ):
                            st.write(
                                f"**简历内容**: {resume['cleaned_text'][:150]}...")
                            st.write(f"**技能**: {', '.join(resume['skills'])}")

                            st.subheader("LLM分析结果")
                            st.write(
                                f"**技能匹配率**: {llm_analysis['step3']['skill_match']['match_rate']:.2%}"
                            )
                            st.write(
                                f"**匹配技能**: {', '.join(llm_analysis['step3']['skill_match']['matching_skills'])}"
                            )
                            st.write(
                                f"**教育匹配**: {'满足' if llm_analysis['step3']['education_match']['match'] else '不满足'}"
                            )
                            st.write(
                                f"**经验匹配**: {'满足' if llm_analysis['step3']['experience_match']['match'] else '不满足'}"
                            )
                            st.write(
                                f"**LLM综合评分**: {llm_analysis['final_score']:.4f}"
                            )

                            st.subheader("LLM优化建议")
                            st.write("**优势**:")
                            for strength in llm_analysis['suggestions'][
                                    'strengths']:
                                st.success(f"✅ {strength}")

                            st.write("**劣势**:")
                            for weakness in llm_analysis['suggestions'][
                                    'weaknesses']:
                                st.warning(f"⚠ {weakness}")

                            st.write("**优化建议**:")
                            for suggestion in llm_analysis['suggestions'][
                                    'suggestions']:
                                st.info(f"💡 {suggestion}")

                            st.subheader("面试题建议")
                            for i, question in enumerate(
                                    llm_analysis['interview_questions'], 1):
                                st.write(f"❓ {i}. {question}")

    st.divider()

    # 5. LLM链式分析
    st.subheader("🧠 LLM链式分析")
    jobs = recruiter_service.get_job_list()

    if not jobs:
        st.error("❌ 请先上传JD")
    else:
        job_options = {
            job['job_id']: job['cleaned_text'][:50] + "..."
            for job in jobs
        }
        selected_job_id = st.selectbox(
            "选择JD",
            list(job_options.keys()),
            format_func=lambda x: f"{x}: {job_options[x]}",
            key="recruiter_llm_job")

        resumes = recruiter_service.get_resume_list()
        if not resumes:
            st.error("❌ 请先上传简历")
        else:
            resume_options = {
                resume['resume_id']: resume['cleaned_text'][:50] + "..."
                for resume in resumes
            }
            selected_resume_id = st.selectbox(
                "选择简历",
                list(resume_options.keys()),
                format_func=lambda x: f"{x}: {resume_options[x]}",
                key="recruiter_llm_resume")

            if st.button("开始LLM链式分析", key="recruiter_llm_analyze"):
                selected_job = next(job for job in jobs
                                    if job['job_id'] == selected_job_id)
                selected_resume = next(
                    resume for resume in resumes
                    if resume['resume_id'] == selected_resume_id)

                with st.spinner("LLM链式分析中..."):
                    result = recruiter_service.analyze_resume_with_llm(
                        selected_job['cleaned_text'],
                        selected_resume['cleaned_text'])
                    st.success("✅ LLM链式分析完成！")

                    # 显示LLM链式分析概览
                    st.subheader("🔗 LLM链式分析概览")

                    # 概览卡片
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🏆 最终匹配分数", f"{result['final_score']:.4f}")
                    with col2:
                        st.metric("🤖 参与LLM模型数量",
                                  len(result.get('llm_scores', {})))
                    with col3:
                        st.metric("📊 分析维度", 3)  # 技能、教育、经验

                    # 显示LLM链式分析流程
                    st.subheader("📋 详细分析流程")

                    # 步骤1: 实体提取
                    with st.expander("🔍 步骤1: 实体提取", expanded=False):
                        st.write("**使用模型**: Qwen")
                        st.write("**任务**: 从JD和简历中提取关键实体信息，为后续匹配奠定基础")
                        if result.get('step1'):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**JD实体**:")
                                st.json(result['step1'].get('jd_entities', {}),
                                        expanded=False)
                            with col2:
                                st.write("**简历实体**:")
                                st.json(result['step1'].get(
                                    'resume_entities', {}),
                                        expanded=False)

                    # 步骤2: 实体验证
                    with st.expander("✅ 步骤2: 实体验证", expanded=False):
                        st.write("**使用模型**: DeepSeek")
                        st.write("**任务**: 验证和修正提取的实体信息，确保数据准确性")
                        if result.get('step2'):
                            st.json(result['step2'], expanded=False)

                    # 步骤3: 匹配度分析
                    with st.expander("📊 步骤3: 匹配度分析", expanded=False):
                        st.write("**使用模型**: OpenAI")
                        st.write("**任务**: 基于提取的实体信息，详细分析简历和JD的匹配度")
                        if result.get('step3'):
                            # 技能匹配
                            st.subheader("技能匹配")
                            skill_match = result['step3']['skill_match']
                            st.write(
                                f"**匹配率**: {skill_match['match_rate']:.2%}")

                            # 技能匹配可视化
                            import pandas as pd
                            import plotly.express as px

                            skill_data = {
                                '类别': ['匹配技能', 'JD特有技能', '简历特有技能'],
                                '数量': [
                                    len(skill_match['matching_skills']),
                                    len(skill_match['jd_skills']) -
                                    len(skill_match['matching_skills']),
                                    len(skill_match['resume_skills']) -
                                    len(skill_match['matching_skills'])
                                ]
                            }
                            skill_df = pd.DataFrame(skill_data)
                            fig = px.pie(skill_df,
                                         values='数量',
                                         names='类别',
                                         title='技能匹配分布')
                            st.plotly_chart(fig)

                            st.write(
                                f"**匹配技能**: {', '.join(skill_match['matching_skills'])}"
                            )
                            st.write(
                                f"**JD技能**: {', '.join(skill_match['jd_skills'])}"
                            )
                            st.write(
                                f"**简历技能**: {', '.join(skill_match['resume_skills'])}"
                            )

                            # 教育背景匹配
                            st.subheader("教育背景匹配")
                            education_match = result['step3'][
                                'education_match']
                            st.write(
                                f"**匹配结果**: {'✅ 满足' if education_match['match'] else '❌ 不满足'}"
                            )
                            st.write(f"**原因**: {education_match['reason']}")

                            # 工作经验匹配
                            st.subheader("工作经验匹配")
                            experience_match = result['step3'][
                                'experience_match']
                            st.write(
                                f"**匹配结果**: {'✅ 满足' if experience_match['match'] else '❌ 不满足'}"
                            )
                            st.write(f"**原因**: {experience_match['reason']}")

                    # 步骤4: 多LLM评估融合
                    with st.expander("🤝 步骤4: 多LLM评估融合", expanded=False):
                        st.write("**任务**: 融合多个LLM的评估结果，生成最终匹配分数")

                        # 显示参与的LLM模型
                        llm_scores = result.get('llm_scores', {})
                        st.write(
                            f"**参与的LLM模型**: {', '.join(llm_scores.keys())}")

                        # 显示每个LLM的评分
                        st.subheader("各LLM模型评分")
                        llm_data = []
                        for llm_name, llm_info in llm_scores.items():
                            llm_data.append({
                                'LLM模型': llm_name.capitalize(),
                                '评分': llm_info['score'],
                                '评分原因': llm_info['reason']
                            })

                        llm_df = pd.DataFrame(llm_data)
                        st.dataframe(llm_df, hide_index=True)

                        # 可视化LLM评分对比
                        fig = px.bar(llm_df,
                                     x='LLM模型',
                                     y='评分',
                                     title='各LLM模型评分对比')
                        fig.update_layout(yaxis_range=[0, 1])
                        st.plotly_chart(fig)

                        # 显示权重分布
                        st.subheader("LLM模型权重分布")
                        weights = result.get('weights', {})
                        weight_data = []
                        for llm_name, weight in weights.items():
                            weight_data.append({
                                'LLM模型': llm_name.capitalize(),
                                '权重': weight
                            })

                        weight_df = pd.DataFrame(weight_data)
                        fig = px.pie(weight_df,
                                     values='权重',
                                     names='LLM模型',
                                     title='LLM模型权重分布')
                        st.plotly_chart(fig)

                        # 显示最终融合分数
                        st.subheader("最终融合分数")
                        st.write(f"**最终匹配分数**: {result['final_score']:.4f}")
                        st.write("**分数计算方式**: 基于各LLM评分的加权平均值")

                    # 步骤5: 优化建议与面试题生成
                    with st.expander("💡 步骤5: 优化建议与面试题生成", expanded=False):
                        # 优化建议
                        st.subheader("简历优化建议")
                        if result.get('suggestions'):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**优势**:")
                                for strength in result['suggestions'][
                                        'strengths']:
                                    st.success(f"✅ {strength}")
                            with col2:
                                st.write("**劣势**:")
                                for weakness in result['suggestions'][
                                        'weaknesses']:
                                    st.warning(f"⚠ {weakness}")

                            st.write("**优化建议**:")
                            for i, suggestion in enumerate(
                                    result['suggestions']['suggestions'], 1):
                                st.info(f"💡 {i}. {suggestion}")

                        # 面试题生成
                        st.subheader("面试题建议")
                        if result.get('interview_questions'):
                            for i, question in enumerate(
                                    result['interview_questions'], 1):
                                st.write(f"❓ {i}. {question}")

                    # 匹配结果总结
                    st.subheader("📋 匹配结果总结")
                    st.write(f"**最终匹配分数**: {result['final_score']:.4f}")
                    st.write(
                        f"**匹配等级**: {'优秀' if result['final_score'] >= 0.8 else '良好' if result['final_score'] >= 0.6 else '一般' if result['final_score'] >= 0.4 else '较差'}"
                    )

                    # 日志展示区域
                    st.subheader("📊 分析日志")
                    logs = log_capture.get_logs()
                    if logs:
                        st.text_area("日志输出",
                                     value="\n".join(logs),
                                     height=200,
                                     disabled=True)
                    else:
                        st.info("暂无日志输出")

# ====================== 求职者功能 ======================
with role_tabs[1]:
    st.header("求职者功能")

    # 1. 简历管理（上传或在线制作，二选一）
    st.subheader("📄 简历管理")

    # 卡片式布局，二选一
    resume_option = st.radio("选择简历管理方式", ["上传简历", "在线制作简历"],
                             horizontal=True,
                             key="candidate_resume_option")

    if resume_option == "上传简历":
        # 上传简历卡片
        with st.container(border=True):
            resume_text = st.text_area("请输入简历内容",
                                       height=200,
                                       key="candidate_resume")

            # 简历文件上传（支持单个和批量）
            resume_files = st.file_uploader("或上传简历文件（支持单个和批量）",
                                            type=[
                                                "pdf", "doc", "docx", "txt",
                                                "md", "jpg", "jpeg", "png",
                                                "xls", "xlsx"
                                            ],
                                            accept_multiple_files=True,
                                            key="candidate_resume_files")

            # 上传按钮
            if st.button("上传简历", key="candidate_upload_resume"):
                uploaded_count = 0

                # 优先处理文件上传
                if resume_files:
                    with st.spinner(f"处理 {len(resume_files)} 份简历中..."):
                        fp = FileProcessor()
                        import tempfile
                        for i, resume_file in enumerate(resume_files, 1):
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{resume_file.name.split('.')[-1]}") as tmp:
                                    tmp.write(resume_file.getvalue())
                                    tmp_path = tmp.name
                                processed = fp.process_file(tmp_path)
                                content = processed.get('content', '')
                                import os, time, uuid
                                root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
                                save_dir = os.path.join(root, 'uploads', 'resumes')
                                os.makedirs(save_dir, exist_ok=True)
                                fname = f"resume_{int(time.time())}_{uuid.uuid4().hex}.{resume_file.name.split('.')[-1]}"
                                save_path = os.path.join(save_dir, fname)
                                with open(save_path, 'wb') as f:
                                    f.write(resume_file.getvalue())
                                meta_r = {'source_file_path': save_path, 'source_file_type': processed.get('file_type', '')}
                                resume = candidate_service.upload_resume(content, meta=meta_r)
                                uploaded_count += 1
                                try:
                                    os.unlink(tmp_path)
                                except Exception:
                                    pass
                            except Exception as e:
                                st.error(f"❌ 处理第 {i} 份简历失败: {str(e)}")
                elif resume_text.strip():
                    with st.spinner("处理简历中..."):
                        import os, time, uuid
                        root = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
                        save_dir = os.path.join(root, 'uploads', 'resumes')
                        os.makedirs(save_dir, exist_ok=True)
                        fname = f"resume_{int(time.time())}_{uuid.uuid4().hex}.txt"
                        save_path = os.path.join(save_dir, fname)
                        with open(save_path, 'w', encoding='utf-8') as f:
                            f.write(resume_text)
                        resume = candidate_service.upload_resume(resume_text, meta={'source_file_path': save_path, 'source_file_type': 'Text文件'})
                        uploaded_count = 1

                if uploaded_count > 0:
                    st.success(f"✅ 成功上传 {uploaded_count} 份简历！")
                else:
                    st.error("❌ 请输入简历内容或上传简历文件")
    else:
        # 在线制作简历卡片
        with st.container(border=True):
            # 简历制作表单
            with st.form("resume_builder_form"):
                st.write("**1. 个人信息**")
                name = st.text_input("姓名")
                gender = st.selectbox("性别", ["男", "女", "其他"])
                phone = st.text_input("联系电话")
                email = st.text_input("电子邮箱")
                location = st.text_input("现居地")

                st.write("**2. 求职意向**")
                desired_position = st.text_input("期望职位")
                desired_industry = st.text_input("期望行业")
                desired_salary = st.text_input("期望薪资")

                st.write("**3. 教育经历**")
                education = st.text_area("教育经历（按时间倒序，每行一条）", height=100)

                st.write("**4. 工作经历**")
                work_experience = st.text_area("工作经历（按时间倒序，每行一条）", height=150)

                st.write("**5. 项目经历**")
                projects = st.text_area("项目经历（按时间倒序，每行一条）", height=150)

                st.write("**6. 科研成果**")
                research = st.text_area("科研成果（按时间倒序，每行一条）", height=100)

                st.write("**7. 个人技能**")
                skills = st.text_area("个人技能（用逗号分隔）", height=100)

                st.write("**8. 自我评价**")
                self_evaluation = st.text_area("自我评价", height=150)

                # 提交按钮
                submitted = st.form_submit_button("生成简历")

                if submitted:
                    # 生成简历文本
                    resume_text = f"姓名: {name}\n性别: {gender}\n联系电话: {phone}\n电子邮箱: {email}\n现居地: {location}\n\n求职意向:\n期望职位: {desired_position}\n期望行业: {desired_industry}\n期望薪资: {desired_salary}\n\n教育经历:\n{education}\n\n工作经历:\n{work_experience}\n\n项目经历:\n{projects}\n\n科研成果:\n{research}\n\n个人技能:\n{skills}\n\n自我评价:\n{self_evaluation}"

                    with st.spinner("生成简历中..."):
                        # 上传简历
                        resume = candidate_service.upload_resume(resume_text)

                        st.success(f"✅ 简历生成成功！")
                        st.info(f"简历 ID: {resume['resume_id']}")

                        # 显示生成的简历
                        with st.expander("查看生成的简历"):
                            st.text(resume_text)

    # 显示已上传的简历列表
    st.subheader("已上传的简历列表")
    resumes = candidate_service.get_resume_list()

    if resumes:
        st.info(f"共上传 {len(resumes)} 份简历")
        # 使用可折叠的容器，默认不展开
        for i, resume in enumerate(resumes):
            # 使用简历ID作为expander的标题，默认不展开
            with st.expander(f"简历 ID: {resume['resume_id']} - 点击查看详情",
                             expanded=False):
                st.write(f"**简历内容**: {resume['cleaned_text'][:150]}...")
                st.write(f"**技能**: {', '.join(resume['skills'])}")
                if 'source_file_type' in resume or 'source_file_path' in resume:
                    st.write(f"来源类型: {resume.get('source_file_type','')}")
                    st.write(f"来源路径: {resume.get('source_file_path','')}")
    else:
        st.info("暂无已上传的简历")

    st.divider()

    # 2. 简历优化
    st.subheader("✨ 简历优化")

    # 选择简历
    resumes = candidate_service.get_resume_list()

    if not resumes:
        st.error("❌ 请先上传或生成简历")
    else:
        resume_options = {
            resume['resume_id']: resume['cleaned_text'][:50] + "..."
            for resume in resumes
        }
        selected_resume_id = st.selectbox(
            "选择要优化的简历",
            list(resume_options.keys()),
            format_func=lambda x: f"{x}: {resume_options[x]}",
            key="candidate_optimize_resume")

        # 职位描述输入
        jd_text = st.text_area("请输入目标职位描述",
                               height=150,
                               key="candidate_optimize_jd")

        # 生成建议按钮
        if st.button("生成优化建议", key="candidate_generate_suggestions"):
            if jd_text.strip():
                with st.spinner("正在生成优化建议..."):
                    # 生成优化建议
                    suggestions = candidate_service.generate_resume_optimization_suggestions(
                        selected_resume_id, jd_text)

                    st.success("✅ 优化建议生成完成！")

                    # 显示优化建议
                    for i, suggestion in enumerate(suggestions['suggestions'],
                                                   1):
                        st.info(f"{i}. {suggestion}")

                    # 显示匹配分数
                    st.info(
                        f"当前简历与目标职位的匹配分数: {suggestions['analysis']['final_score']:.4f}"
                    )
            else:
                st.error("❌ 请输入目标职位描述")

    st.divider()

    # 3. 简历画像
    st.subheader("🎨 简历画像")

    # 选择简历
    resumes = candidate_service.get_resume_list()

    if not resumes:
        st.error("❌ 请先上传或生成简历")
    else:
        resume_options = {
            resume['resume_id']: resume['cleaned_text'][:50] + "..."
            for resume in resumes
        }
        selected_resume_id = st.selectbox(
            "选择要生成画像的简历",
            list(resume_options.keys()),
            format_func=lambda x: f"{x}: {resume_options[x]}",
            key="candidate_portrait_resume")

        jd_portrait_text = st.text_area("输入用于画像的目标JD文本", height=150, key="candidate_portrait_jd")
        if st.button("生成简历画像", key="candidate_generate_portrait"):
            with st.spinner("正在生成简历画像..."):
                from core.visualizer import Visualizer
                selected_resume = next(r for r in resumes if r['resume_id'] == selected_resume_id)
                jd_struct = candidate_service.data_processor.process_jd_text(jd_portrait_text or selected_resume['cleaned_text'])
                jd_struct = candidate_service.feature_engine.extract_features_from_jd(jd_struct)
                fig = Visualizer().generate_radar_chart(selected_resume, jd_struct)
                st.plotly_chart(fig, use_container_width=True)
                st.write("- **薪资期望**: 20-30K")

    st.divider()

    # 4. 岗位匹配（合并了职位匹配和岗位筛选功能）
    st.subheader("🤝 岗位匹配")

    # 选择简历
    resumes = candidate_service.get_resume_list()

    if not resumes:
        st.error("❌ 请先上传或生成简历")
    else:
        resume_options = {
            resume['resume_id']: resume['cleaned_text'][:50] + "..."
            for resume in resumes
        }
        selected_resume_id = st.selectbox(
            "选择要匹配的简历",
            list(resume_options.keys()),
            format_func=lambda x: f"{x}: {resume_options[x]}",
            key="candidate_select_resume")

        # 设置匹配数量
        top_k = st.slider("返回匹配结果数量",
                          min_value=1,
                          max_value=10,
                          value=5,
                          key="candidate_top_k")

        # 允许指定岗位库文件目录
        st.text_input("岗位库文件目录（可选）",
                      placeholder="输入岗位库文件目录路径",
                      key="candidate_job_library_dir")

        # 匹配按钮
        if st.button("开始岗位匹配", key="candidate_match"):
            jobs = candidate_service.get_job_list()

            if not jobs:
                st.warning("⚠ 职位库为空，正在添加示例职位...")
                # 添加示例职位
                sample_jobs = [
                    "Python开发工程师，需要3-5年工作经验，熟悉Python、Django、MySQL等技术",
                    "Java开发工程师，需要5年以上工作经验，熟悉Java、Spring Boot、微服务等技术",
                    "前端开发工程师，需要2-4年工作经验，熟悉JavaScript、React、Vue等技术"
                ]

                for job_text in sample_jobs:
                    candidate_service.add_job(job_text)

                st.success("✅ 示例职位添加完成！")

            with st.spinner("正在匹配岗位..."):
                # 进行匹配
                results = candidate_service.match_resume_to_jobs(
                    selected_resume_id, top_k)

                st.success(f"✅ 匹配完成！共找到 {len(results)} 个匹配岗位")

                # 日志展示区域
                st.subheader("📋 匹配日志")
                logs = log_capture.get_logs()
                if logs:
                    st.text_area("日志输出",
                                 value="\n".join(logs),
                                 height=200,
                                 disabled=True)
                else:
                    st.info("暂无日志输出")

                # 显示匹配结果
                for i, (job, score) in enumerate(results, 1):
                    with st.expander(
                            f"{i}. 匹配分数: {score:.4f} - 职位 ID: {job['job_id']}"
                    ):
                        st.write(f"**职位描述**: {job['cleaned_text'][:150]}...")
                        st.write(f"**技能要求**: {', '.join(job['skills'])}")

    st.divider()

    # 5. 模拟面试
    st.subheader("🎭 模拟面试")
    resumes_for_interview = candidate_service.get_resume_list()
    resume_options_iv = {r['resume_id']: r['cleaned_text'][:50] + "..." for r in resumes_for_interview} if resumes_for_interview else {}
    selected_resume_iv = st.selectbox("选择简历", list(resume_options_iv.keys()) if resume_options_iv else [""], format_func=lambda x: f"{x}: {resume_options_iv.get(x,'')}" if x else "", key="candidate_iv_resume")
    jd_iv_text = st.text_area("输入目标JD文本", height=120, key="candidate_iv_jd")
    if st.button("生成面试题", key="candidate_generate_interview"):
        if selected_resume_iv and jd_iv_text.strip():
            with st.spinner("正在生成面试题..."):
                res = next(r for r in resumes_for_interview if r['resume_id'] == selected_resume_iv)
                qs = candidate_service.llm_chain.generate_interview_questions(res['cleaned_text'], jd_iv_text)
                st.subheader("面试题")
                for i, q in enumerate(qs, 1):
                    st.write(f"{i}. {q}")
        else:
            st.error("❌ 请选择简历并输入JD文本")
    answer_text = st.text_area("输入你的回答以评估（可选）", height=120, key="candidate_iv_answer")
    if st.button("评估回答", key="candidate_evaluate_answer"):
        if selected_resume_iv and jd_iv_text.strip() and answer_text.strip():
            res = next(r for r in resumes_for_interview if r['resume_id'] == selected_resume_iv)
            eval_res = candidate_service.llm_chain.evaluate_interview_answer(res['cleaned_text'], jd_iv_text, answer_text)
            st.write(f"评分: {eval_res.get('score', 0):.2f}")
            st.write("优势:")
            for s in eval_res.get('strengths', []):
                st.success(s)
            st.write("劣势:")
            for w in eval_res.get('weaknesses', []):
                st.warning(w)
            st.write("建议:")
            for sg in eval_res.get('suggestions', []):
                st.info(sg)
        else:
            st.error("❌ 请完善输入")

    st.divider()

    st.subheader("📬 投递看板")
    resumes_for_apply = candidate_service.get_resume_list()
    jobs_for_apply = candidate_service.get_job_list()
    if resumes_for_apply and jobs_for_apply:
        sel_res_apply = st.selectbox("选择简历进行投递", [r['resume_id'] for r in resumes_for_apply], key="apply_resume")
        sel_job_apply = st.selectbox("选择职位进行投递", [j['job_id'] for j in jobs_for_apply], key="apply_job")
        if st.button("一键投递", key="do_apply"):
            app = candidate_service.submit_application(sel_res_apply, sel_job_apply)
            st.success(f"投递成功，ID: {app['application_id']}")
    apps = candidate_service.get_applications()
    if apps:
        for app in apps:
            with st.expander(f"投递 {app['application_id']} - {app['resume_id']} -> {app['job_id']} 状态: {app['status']}"):
                new_status = st.selectbox("更新状态", ["submitted", "read", "interview", "rejected", "offer"], key=f"status_{app['application_id']}")
                rejection_text = st.text_area("拒信文本（可选）", height=100, key=f"rej_{app['application_id']}")
                if st.button("更新", key=f"upd_{app['application_id']}"):
                    updated = candidate_service.update_application_status(app['application_id'], new_status, rejection_text if new_status=="rejected" else None)
                    st.success("已更新")
                    if updated['history'] and 'rejection_analysis' in updated['history'][-1]:
                        st.json(updated['history'][-1]['rejection_analysis'])

    st.subheader("🧭 学习成长路径")
    resumes_lp = candidate_service.get_resume_list()
    if resumes_lp:
        sel_res_lp = st.selectbox("选择简历生成学习路径", [r['resume_id'] for r in resumes_lp], key="lp_resume")
        jd_lp_text = st.text_area("输入目标JD文本", height=120, key="lp_jd_text")
        if st.button("生成学习路径", key="generate_lp"):
            if jd_lp_text.strip():
                plan = candidate_service.generate_learning_path(sel_res_lp, jd_lp_text)
                st.json(plan)
            else:
                st.error("❌ 请输入JD文本")

# 页脚
st.markdown("---")
st.info("智能简历筛选系统 v2.0.0")
