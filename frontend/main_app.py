#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能简历筛选系统 - 组合界面

将招聘方和求职者功能整合到一个界面中，方便展示
"""

# 必须在import其他模块之前设置环境变量！
import os

# 使用HF镜像源解决下载问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # 允许联网下载
os.environ["HF_HUB_OFFLINE"] = "0"  # 允许HF Hub联网
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '30'  # 超时时间30秒
os.environ['HF_HUB_RETRY'] = '5'  # 重试5次
os.environ['HF_HUB_RETRY_DELAY'] = '2'  # 重试间隔2秒

# 然后再import其他模块
import sys
import streamlit as st
import json
from io import StringIO

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.recruiter_service import RecruiterService
from services.candidate_service import CandidateService
from core.file_processor import FileProcessor

# 异步日志捕获类
import threading
import queue
import time


class AsyncLogCapture:

    def __init__(self, buffer_size=1000):
        self.logs = []
        self.original_stdout = sys.stdout
        self.log_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread = None
        self.buffer_size = buffer_size

    def _async_writer(self):
        """
        异步线程，负责将队列中的日志写入日志列表
        """
        while self.running:
            try:
                # 非阻塞获取队列中的日志，超时100ms
                log_entry = self.log_queue.get(timeout=0.1)
                self.logs.append(log_entry)
                # 限制日志总数，避免内存溢出
                if len(self.logs) > self.buffer_size * 2:
                    self.logs = self.logs[-self.buffer_size:]
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # 确保线程不会因为异常而退出
                try:
                    print(f"异步日志写入错误: {e}")
                except:
                    pass

    class AsyncStringIO:
        """
        异步StringIO类，将写入操作转换为队列操作
        """

        def __init__(self, log_queue):
            self.log_queue = log_queue
            self.buffer = ""

        def write(self, text):
            """
            写入文本，遇到换行符时将整行加入队列
            """
            try:
                if "\n" in text:
                    parts = text.split("\n")
                    if self.buffer:
                        # 处理上一行剩余的内容
                        full_line = self.buffer + parts[0]
                        if full_line.strip():
                            self.log_queue.put(full_line, block=False)
                        self.buffer = ""

                    # 处理中间的完整行
                    for part in parts[1:-1]:
                        if part.strip():
                            self.log_queue.put(part, block=False)

                    # 处理最后一个可能不完整的行
                    if parts[-1]:
                        self.buffer = parts[-1]
                else:
                    # 没有换行符，累积到缓冲区
                    self.buffer += text
            except queue.Full:
                # 队列满时，打印错误但不阻塞
                try:
                    print("警告：日志队列已满，部分日志可能丢失")
                except:
                    pass
            except Exception as e:
                # 确保写入操作不会因为异常而失败
                try:
                    print(f"日志写入错误: {e}")
                except:
                    pass

        def flush(self):
            """
            刷新缓冲区，将剩余内容加入队列
            """
            if self.buffer.strip():
                try:
                    self.log_queue.put(self.buffer, block=False)
                    self.buffer = ""
                except queue.Full:
                    try:
                        print("警告：日志队列已满，部分日志可能丢失")
                    except:
                        pass

        def getvalue(self):
            """
            为了保持兼容性，返回空字符串
            """
            return ""

    def start_capture(self):
        """
        开始异步日志捕获
        """
        if not self.running:
            # 启动异步写入线程
            self.running = True
            self.thread = threading.Thread(target=self._async_writer,
                                           daemon=True)
            self.thread.start()

            # 替换sys.stdout为异步StringIO
            self.async_stdout = self.AsyncStringIO(self.log_queue)
            sys.stdout = self.async_stdout

    def stop_capture(self):
        """
        停止异步日志捕获
        """
        if self.running:
            # 先刷新缓冲区
            if hasattr(self, 'async_stdout'):
                self.async_stdout.flush()

            # 恢复原始stdout
            sys.stdout = self.original_stdout

            # 停止异步线程
            self.running = False
            if self.thread:
                self.thread.join(timeout=1.0)

    def get_logs(self):
        """
        获取当前捕获的日志列表
        """
        # 先刷新缓冲区
        if hasattr(self, 'async_stdout'):
            self.async_stdout.flush()

        # 等待队列中的日志处理完成（短暂等待）
        time.sleep(0.01)

        return self.logs.copy()

    def clear_logs(self):
        """
        清空日志列表
        """
        with threading.Lock():
            self.logs = []


# 为了保持兼容性，保留原类名的引用
LogCapture = AsyncLogCapture


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

# 导入异步任务管理器
from utils.async_task_manager import task_manager

# 侧边栏 - LLM模型配置
with st.sidebar:
    st.header("⚙️ LLM模型配置")

    # 导入LLM配置管理器
    from core.llm_config_manager import LLMConfigManager

    # 初始化LLM配置管理器
    llm_config_manager = LLMConfigManager()

    # 支持的模型列表
    supported_models = llm_config_manager.get_supported_models()

    # 模型操作模式：选择现有模型或添加新模型
    model_mode = st.radio("模型操作", ["选择现有模型", "添加新模型"],
                          key="llm_model_mode_global")

    # 根据选择的模式显示不同的界面
    selected_model = None
    if model_mode == "选择现有模型":
        selected_model = st.selectbox("选择LLM模型",
                                      supported_models,
                                      key="llm_model_select_global")
    else:
        # 添加新模型的输入框
        new_model_name = st.text_input("新模型名称",
                                       placeholder="例如：my-custom-model",
                                       key="llm_new_model_name_global")
        selected_model = new_model_name

    # API Key输入
    api_key = st.text_input("API Key",
                            type="password",
                            key="llm_api_key_global")

    # Base URL输入（可选）
    base_url = st.text_input("API Base URL（可选）", key="llm_base_url_global")

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
    # 获取所有支持的模型
    all_models = llm_config_manager.get_supported_models()
    # 过滤出真正配置了API Key的模型
    configured_models = [
        model for model in all_models
        if llm_config_manager.is_model_configured(model)
    ]
    if configured_models:
        for model_name in configured_models:
            st.write(f"- **{model_name}**: API Key已配置")
    else:
        st.info("暂无已配置的模型")

    st.divider()

    # 区域与优先级
    st.subheader("区域与优先级")
    # 从配置文件中加载当前区域配置
    current_region = llm_config_manager.get_region()
    # 显示当前区域配置
    region_choice = st.radio("选择调用区域", ["domestic", "international"],
                             index=0 if current_region == "domestic" else 1,
                             horizontal=True,
                             key="llm_region_select")
    # 保存区域配置按钮
    if st.button("保存区域", key="save_llm_region"):
        if llm_config_manager.set_region(region_choice):
            st.success(f"✅ 已切换到 {region_choice} 区域")
            # 更新当前区域配置
            current_region = region_choice
            # 重新获取当前区域的优先级顺序
            preferred_order = llm_config_manager.get_preferred_order_by_region(
                current_region)
            st.write(f"当前优先顺序: {', '.join(preferred_order)}")
        else:
            st.error("❌ 区域设置失败")
    else:
        # 初始加载或区域未变化时，根据当前保存的区域配置获取优先级顺序
        preferred_order = llm_config_manager.get_preferred_order_by_region(
            current_region)
        st.write(f"当前优先顺序: {', '.join(preferred_order)}")

    st.caption("提示：国内推荐 Qwen/Moonshot/Doubao/DeepSeek；国际推荐 OpenAI/OpenRouter")
    # 动态获取所有支持的模型键作为选项，不要求必须先配置API Key
    opt_short = list(llm_config_manager.get_model_mappings().keys())
    # 显示当前优先级配置
    # 过滤默认值，确保只包含在选项列表中存在的模型
    valid_defaults = [
        model for model in preferred_order[:3] if model in opt_short
    ]
    new_order = st.multiselect("设置优先顺序（最多选3，按选择顺序生效）",
                               opt_short,
                               default=valid_defaults,
                               key="llm_preferred_order")
    # 保存优先级配置按钮
    if st.button("保存优先顺序", key="save_llm_preferred_order"):
        if new_order:
            ok = llm_config_manager.set_preferred_order(
                current_region,
                new_order + [x for x in opt_short if x not in new_order])
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
            with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f".{jd_file.name.split('.')[-1]}") as tmp:
                data = jd_file.getvalue()
                tmp.write(data)
                tmp_path = tmp.name
            processed = fp.process_file(tmp_path)
            jd_content = processed.get('content', '')
            root = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..'))
            save_dir = os.path.join(root, 'uploads', 'jds')
            os.makedirs(save_dir, exist_ok=True)
            import time, uuid
            fname = f"jd_{int(time.time())}_{uuid.uuid4().hex}.{jd_file.name.split('.')[-1]}"
            save_path = os.path.join(save_dir, fname)
            with open(save_path, 'wb') as f:
                f.write(data)
            meta = {
                'source_file_path': save_path,
                'source_file_type': processed.get('file_type', '')
            }
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        elif jd_text.strip():
            jd_content = jd_text.strip()
            import os, time, uuid
            root = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..'))
            save_dir = os.path.join(root, 'uploads', 'jds')
            os.makedirs(save_dir, exist_ok=True)
            fname = f"jd_{int(time.time())}_{uuid.uuid4().hex}.txt"
            save_path = os.path.join(save_dir, fname)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(jd_content)
            meta = {
                'source_file_path': save_path,
                'source_file_type': 'Text文件'
            }

        if jd_content:
            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("开始处理JD...")

            try:
                # 定义进度回调函数
                def progress_callback(percent, message):
                    progress_bar.progress(percent)
                    status_text.text(message)

                # 添加JD并显示进度
                jd = recruiter_service.add_job(
                    jd_content, meta=meta, progress_callback=progress_callback)

                st.success(f"✅ JD上传成功！")
                st.info(f"JD ID: {jd['job_id']}")
                st.session_state['recruiter_jd_done'] = True
                # 保存当前JD到session_state，用于结构化信息显示
                st.session_state['current_jd'] = jd

                # 清除进度显示
                status_text.empty()
                progress_bar.empty()

                # 使用保存的当前JD显示结构化信息
                current_jd = st.session_state.get('current_jd', {})
                with st.expander("📋 查看JD结构化信息"):
                    st.write(
                        f"**职位名称**: {current_jd['entities'].get('职位名称', '未提取到')}"
                    )
                    st.write(
                        f"**公司名称**: {current_jd['entities'].get('公司名称', '未提取到')}"
                    )
                    st.write(
                        f"**薪资范围**: {current_jd['entities'].get('薪资范围', '未提取到')}"
                    )
                    st.write(
                        f"**工作地点**: {current_jd['entities'].get('工作地点', '未提取到')}"
                    )
                    st.write(
                        f"**学历要求**: {current_jd['entities'].get('学历要求', '未提取到')}"
                    )
                    st.write(
                        f"**工作年限要求**: {current_jd['entities'].get('工作年限要求', '未提取到')}"
                    )

                    if current_jd.get('skills'):
                        st.write(
                            f"**技能要求**: {', '.join(current_jd.get('skills'))}")

                    if current_jd['entities'].get('岗位职责'):
                        st.write(
                            f"**岗位职责**: {current_jd['entities']['岗位职责'][:100]}..."
                        )

                    if current_jd['entities'].get('任职要求'):
                        st.write(
                            f"**任职要求**: {current_jd['entities']['任职要求'][:100]}..."
                        )

                    if 'vector' in current_jd:
                        st.write(f"**向量维度**: {len(current_jd['vector'])}")
                        st.write(
                            f"**向量前5个值**: {[round(v, 4) for v in current_jd['vector'][:5]]}..."
                        )

                    # 使用当前JD的实体信息作为完整实体信息
                    st.subheader("完整实体信息")
                    st.json(current_jd['entities'], expanded=False)
            except Exception as e:
                # 清除进度显示
                status_text.empty()
                progress_bar.empty()
                st.error(f"❌ JD处理失败: {str(e)}")
        else:
            st.error("❌ 请输入职位描述内容或上传JD文件")

    # 显示已上传的JD列表
    st.subheader("已上传的JD列表")
    jobs = recruiter_service.get_job_list()
    if jobs:
        for job in jobs:
            with st.expander(f"JD ID: {job['job_id']} - 职位描述"):
                st.write(job['cleaned_text'][:150] + "...")
                st.write(f"技能要求: {', '.join(job.get('skills', []))}")
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

            if st.button(
                    "上传简历",
                    key="recruiter_upload_resume",
                    disabled=not st.session_state.get('recruiter_jd_done')):
                uploaded_count = 0
                if resume_files:
                    with st.spinner(f"处理 {len(resume_files)} 份简历中..."):
                        fp = FileProcessor()
                        import tempfile
                        for i, resume_file in enumerate(resume_files, 1):
                            try:
                                with tempfile.NamedTemporaryFile(
                                        delete=False,
                                        suffix=
                                        f".{resume_file.name.split('.')[-1]}"
                                ) as tmp:
                                    tmp.write(resume_file.getvalue())
                                    tmp_path = tmp.name
                                processed = fp.process_file(tmp_path)
                                content = processed.get('content', '')
                                import os, time, uuid
                                root = os.path.normpath(
                                    os.path.join(os.path.dirname(__file__),
                                                 '..'))
                                save_dir = os.path.join(
                                    root, 'uploads', 'resumes')
                                os.makedirs(save_dir, exist_ok=True)
                                fname = f"resume_{int(time.time())}_{uuid.uuid4().hex}.{resume_file.name.split('.')[-1]}"
                                save_path = os.path.join(save_dir, fname)
                                with open(save_path, 'wb') as f:
                                    f.write(resume_file.getvalue())
                                meta_r = {
                                    'source_file_path':
                                    save_path,
                                    'source_file_type':
                                    processed.get('file_type', '')
                                }
                                resume = recruiter_service.upload_resume(
                                    content, meta=meta_r)
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
                        root = os.path.normpath(
                            os.path.join(os.path.dirname(__file__), '..'))
                        save_dir = os.path.join(root, 'uploads', 'resumes')
                        os.makedirs(save_dir, exist_ok=True)
                        fname = f"resume_{int(time.time())}_{uuid.uuid4().hex}.txt"
                        save_path = os.path.join(save_dir, fname)
                        with open(save_path, 'w', encoding='utf-8') as f:
                            f.write(resume_text)
                        resume = recruiter_service.upload_resume(
                            resume_text,
                            meta={
                                'source_file_path': save_path,
                                'source_file_type': 'Text文件'
                            })
                        uploaded_count = 1

                if uploaded_count > 0:
                    st.success(f"✅ 成功上传 {uploaded_count} 份简历！")
                    st.session_state['recruiter_resume_done'] = True
                    with st.expander("🔎 解析过程与日志"):
                        log_path = os.path.normpath(
                            os.path.join(os.path.dirname(__file__), '..',
                                         'logs', 'app.log'))
                        lines = []
                        try:
                            if os.path.isfile(log_path):
                                with open(log_path, 'r',
                                          encoding='utf-8') as lf:
                                    lines = lf.read().splitlines()[-200:]
                        except Exception:
                            lines = []
                        focus = []
                        for ln in lines:
                            if ('使用NER提取实体' in ln) or ('使用正则表达式提取JD实体'
                                                       in ln) or ('LLM补全实体'
                                                                  in ln):
                                focus.append(ln)
                        if focus:
                            st.text_area("解析日志",
                                         value="\n".join(focus),
                                         height=160,
                                         disabled=True)
                        else:
                            st.info("暂无解析日志")
                        try:
                            parsed_path = os.path.normpath(
                                os.path.join(os.path.dirname(__file__), '..',
                                             'data', 'processed',
                                             'parsed_resumes.jsonl'))
                            if os.path.isfile(parsed_path):
                                with open(parsed_path, 'r',
                                          encoding='utf-8') as pf:
                                    rows = pf.read().splitlines()
                                    if rows:
                                        import json as _json
                                        st.subheader("解析落盘结果")
                                        st.json(_json.loads(rows[-1]),
                                                expanded=False)
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
                password = st.text_input("密码",
                                         type="password",
                                         key="recruiter_site_password")
            with col_auth2:
                cookie_string = st.text_area("Cookie字符串(可选)",
                                             height=100,
                                             key="recruiter_site_cookie")

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

                            imported_ids = scraper.search_resumes(
                                keywords, page=1, page_size=import_count)

                            uploaded_count = 0
                            for rid in imported_ids:
                                try:
                                    detail = scraper.get_resume_detail(rid)
                                    parts = []
                                    if getattr(detail, 'name', None):
                                        parts.append(str(detail.name))
                                    if getattr(detail, 'work_experience',
                                               None):
                                        for we in detail.work_experience or []:
                                            parts.append(" ".join(
                                                [str(v) for v in we.values()]))
                                    if getattr(detail, 'education', None):
                                        for ed in detail.education or []:
                                            parts.append(" ".join(
                                                [str(v) for v in ed.values()]))
                                    if getattr(detail, 'skills', None):
                                        parts.append(",".join(detail.skills
                                                              or []))
                                    if getattr(detail, 'projects', None):
                                        for pr in detail.projects or []:
                                            parts.append(" ".join(
                                                [str(v) for v in pr.values()]))
                                    text_payload = "\n".join(
                                        [p for p in parts if p])
                                    recruiter_service.upload_resume(
                                        text_payload)
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
            stage1_threshold = st.slider("一级向量阈值",
                                         0.0,
                                         1.0,
                                         0.3,
                                         0.01,
                                         key="cfg_stage1")
            skills_min_rate = st.slider("技能最小匹配率",
                                        0.0,
                                        1.0,
                                        0.3,
                                        0.01,
                                        key="cfg_skills_rate")
            required_years = st.number_input("工作年限下限",
                                             min_value=0,
                                             max_value=30,
                                             value=3,
                                             step=1,
                                             key="cfg_years")
            llm_enabled = st.checkbox("启用LLM补筛", value=True, key="cfg_llm")
            llm_boundary = st.slider("LLM补筛边界区间",
                                     0.0,
                                     1.0, (0.55, 0.65),
                                     0.01,
                                     key="cfg_llm_boundary")
            seg_exp = st.slider("段权重-经验",
                                0.0,
                                1.0,
                                0.5,
                                0.01,
                                key="cfg_seg_exp")
            seg_skill = st.slider("段权重-技能",
                                  0.0,
                                  1.0,
                                  0.3,
                                  0.01,
                                  key="cfg_seg_skill")
            seg_edu = st.slider("段权重-教育",
                                0.0,
                                1.0,
                                0.2,
                                0.01,
                                key="cfg_seg_edu")
        if st.button(
                "开始匹配",
                key="recruiter_match",
                disabled=not st.session_state.get('recruiter_resume_done')):
            resumes = recruiter_service.get_resume_list()
            if not resumes:
                st.error("❌ 请先上传简历")
            else:
                selected_job = next(job for job in jobs
                                    if job['job_id'] == selected_job_id)

                # 应用自定义筛选规则（如果启用且有配置）
                filtered_resumes = resumes
                enable_filter = st.session_state.get("recruiter_enable_filter",
                                                     False)
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

                    # 提交异步任务
                    def match_task():
                        return recruiter_service.matcher.match_resumes_to_jd_with_llm(
                            filtered_resumes, selected_job, top_k, config=cfg)

                    task_id = task_manager.submit_task(match_task)

                    # 显示进度条和状态
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # 轮询任务进度
                    while True:
                        task = task_manager.get_task(task_id)
                        if task:
                            # 更新进度条
                            progress_bar.progress(task.progress)

                            # 更新状态文本
                            status_text.text(
                                f"当前状态: {task.status} - 进度: {task.progress:.1%}"
                            )

                            # 检查任务是否完成
                            if task.status in ["completed", "failed"]:
                                if task.status == "completed":
                                    results = task.result
                                    st.success(
                                        f"✅ 匹配完成！共找到 {len(results)} 份匹配简历")
                                else:
                                    st.error(f"❌ 匹配失败: {task.error}")
                                break

                        # 等待一段时间后再次查询
                        time.sleep(0.1)

                    st.subheader("📋 匹配日志")
                    log_path = os.path.normpath(
                        os.path.join(os.path.dirname(__file__), '..', 'logs',
                                     'app.log'))
                    out_lines = []
                    try:
                        if os.path.isfile(log_path):
                            with open(log_path, 'r', encoding='utf-8') as lf:
                                lines = lf.read().splitlines()[-300:]
                                for ln in lines:
                                    if ('matcher' in ln) or ('匹配' in ln) or (
                                            'llm_chain' in ln):
                                        out_lines.append(ln)
                    except Exception:
                        out_lines = []
                    if out_lines:
                        st.text_area("日志输出",
                                     value="\n".join(out_lines),
                                     height=200,
                                     disabled=True)
                    else:
                        st.info("暂无日志输出")

                    # 准备数据可视化
                    import pandas as pd
                    import plotly.express as px
                    import plotly.graph_objects as go

                    radar_data = []
                    for resume, score, filter_details, llm_analysis in results:
                        # 基础维度 - 添加错误处理，兼容不使用LLM的情况
                        if 'step3' in llm_analysis:
                            skill_match = llm_analysis['step3']['skill_match'][
                                'match_rate']
                            education_match = 1.0 if llm_analysis['step3'][
                                'education_match']['match'] else 0.0
                            experience_match = 1.0 if llm_analysis['step3'][
                                'experience_match']['match'] else 0.0
                        else:
                            # 当没有LLM分析结果时，使用默认值或基于其他数据计算
                            skill_match = score * 0.8  # 使用综合分数近似
                            education_match = 1.0 if 'education' in resume else 0.5
                            experience_match = 1.0 if 'experience' in resume else 0.5

                        # 扩展维度 - 从配置文件加载
                        # 读取匹配维度配置文件
                        config_path = os.path.normpath(
                            os.path.join(os.path.dirname(__file__), '..',
                                         'config',
                                         'matching_dimensions_config.json'))

                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                matching_config = json.load(f)

                            # 合并所有维度配置
                            all_dimensions = {}
                            for dimension_type, dimensions in matching_config[
                                    'dimensions'].items():
                                all_dimensions.update(dimensions)
                        except Exception as e:
                            # 使用默认配置
                            all_dimensions = {
                                '技能匹配': {
                                    'weight': 0.2
                                },
                                '教育匹配': {
                                    'weight': 0.15
                                },
                                '经验匹配': {
                                    'weight': 0.15
                                },
                                '语言能力': {
                                    'weight': 0.1
                                },
                                '证书匹配': {
                                    'weight': 0.1
                                },
                                '薪资匹配': {
                                    'weight': 0.08
                                },
                                '工作地点': {
                                    'weight': 0.07
                                },
                                '行业匹配': {
                                    'weight': 0.07
                                },
                                '职位匹配': {
                                    'weight': 0.05
                                },
                                '项目经验': {
                                    'weight': 0.03
                                }
                            }

                        # 辅助函数：从llm_analysis中提取数据，如果不存在则使用fallback
                        def get_dimension_score(resume, llm_analysis,
                                                dimension_config,
                                                dimension_name):
                            try:
                                # 基础维度特殊处理
                                if dimension_name == '技能匹配' and 'step3' in llm_analysis:
                                    return llm_analysis['step3'][
                                        'skill_match']['match_rate']
                                elif dimension_name == '教育匹配' and 'step3' in llm_analysis:
                                    return 1.0 if llm_analysis['step3'][
                                        'education_match']['match'] else 0.0
                                elif dimension_name == '经验匹配' and 'step3' in llm_analysis:
                                    return 1.0 if llm_analysis['step3'][
                                        'experience_match']['match'] else 0.0

                                # 扩展维度从step4提取
                                if 'step4' in llm_analysis and dimension_name in llm_analysis[
                                        'step4']:
                                    if 'match_rate' in llm_analysis['step4'][
                                            dimension_name]:
                                        return llm_analysis['step4'][
                                            dimension_name]['match_rate']
                                    elif 'match' in llm_analysis['step4'][
                                            dimension_name]:
                                        return 1.0 if llm_analysis['step4'][
                                            dimension_name]['match'] else 0.0

                                # 使用fallback
                                if dimension_name == '语言能力':
                                    return 0.8 if any(
                                        '语言' in skill for skill in resume.get(
                                            'skills', [])) else 0.5
                                elif dimension_name == '证书匹配':
                                    return 0.9 if any(
                                        '证书' in skill for skill in resume.get(
                                            'skills', [])) else 0.4
                            except Exception:
                                pass

                            # 默认值
                            default_values = {
                                '语言能力': 0.5,
                                '证书匹配': 0.4,
                                '薪资匹配': 0.75,
                                '工作地点': 0.8,
                                '行业匹配': 0.85,
                                '职位匹配': 0.9,
                                '项目经验': 0.7
                            }
                            return default_values.get(dimension_name, 0.5)

                        # 获取扩展维度分数
                        language_match = get_dimension_score(
                            resume, llm_analysis,
                            all_dimensions.get('语言能力', {}), '语言能力')
                        certificate_match = get_dimension_score(
                            resume, llm_analysis,
                            all_dimensions.get('证书匹配', {}), '证书匹配')
                        salary_match = get_dimension_score(
                            resume, llm_analysis,
                            all_dimensions.get('薪资匹配', {}), '薪资匹配')
                        location_match = get_dimension_score(
                            resume, llm_analysis,
                            all_dimensions.get('工作地点', {}), '工作地点')
                        industry_match = get_dimension_score(
                            resume, llm_analysis,
                            all_dimensions.get('行业匹配', {}), '行业匹配')
                        position_match = get_dimension_score(
                            resume, llm_analysis,
                            all_dimensions.get('职位匹配', {}), '职位匹配')
                        project_match = get_dimension_score(
                            resume, llm_analysis,
                            all_dimensions.get('项目经验', {}), '项目经验')

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
                    try:
                        from core.evaluator import ModelEvaluator
                        evalr = ModelEvaluator()
                        metrics = evalr.compute_ner_metrics_from_annotations(
                            os.path.normpath(
                                os.path.join(os.path.dirname(__file__), '..',
                                             'data', 'processed',
                                             'ner_annotations.json')))
                        st.write(f"- 模型准确率: {metrics['accuracy']:.4f}")
                        st.write(f"- 模型召回率: {metrics['recall']:.4f}")
                    except Exception:
                        st.write(f"- 模型准确率: --")
                        st.write(f"- 模型召回率: --")

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
                    # 从配置文件加载权重
                    try:
                        config_path = os.path.normpath(
                            os.path.join(os.path.dirname(__file__), '..',
                                         'config',
                                         'matching_dimensions_config.json'))
                        with open(config_path, 'r', encoding='utf-8') as f:
                            matching_config = json.load(f)

                        # 合并所有维度权重
                        weights = {}
                        for dimension_type, dimensions in matching_config[
                                'dimensions'].items():
                            for dim, config in dimensions.items():
                                weights[dim] = config['weight']
                    except Exception:
                        # 使用默认权重
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
                            if 'step3' in llm_analysis:
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
                            else:
                                st.write(
                                    f"**技能匹配率**: {score * 0.8:.2%} (基于综合分数近似)")
                                st.write(f"**匹配技能**: 未使用LLM分析")
                                st.write(
                                    f"**教育匹配**: {'满足' if 'education' in resume else '不满足'}"
                                )
                                st.write(
                                    f"**经验匹配**: {'满足' if 'experience' in resume else '不满足'}"
                                )
                            st.write(
                                f"**LLM综合评分**: {llm_analysis['final_score']:.4f}"
                            )

                            st.subheader("LLM优化建议")
                            if 'suggestions' in llm_analysis:
                                st.write("**优势**:")
                                for strength in llm_analysis[
                                        'suggestions'].get('strengths', []):
                                    st.success(f"✅ {strength}")
                            else:
                                st.write("**优势**: 未使用LLM分析")

                            st.write("**劣势**:")
                            if 'suggestions' in llm_analysis:
                                for weakness in llm_analysis[
                                        'suggestions'].get('weaknesses', []):
                                    st.warning(f"⚠ {weakness}")
                            else:
                                st.write("**劣势**: 未使用LLM分析")

                            st.write("**优化建议**:")
                            if 'suggestions' in llm_analysis:
                                for suggestion in llm_analysis[
                                        'suggestions'].get('suggestions', []):
                                    st.info(f"💡 {suggestion}")
                            else:
                                st.write("**优化建议**: 未使用LLM分析")

                            st.subheader("面试题建议")
                            if 'interview_questions' in llm_analysis:
                                for i, question in enumerate(
                                        llm_analysis['interview_questions'],
                                        1):
                                    st.write(f"❓ {i}. {question}")
                            else:
                                st.write("**面试题建议**: 未使用LLM分析")

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
                                  len(result.get('active_providers', [])))
                    with col3:
                        st.metric("📊 分析维度", 3)  # 技能、教育、经验

                    # 显示LLM链式分析流程
                    st.subheader("📋 详细分析流程")

                    # 步骤1: 实体提取
                    with st.expander("🔍 步骤1: 实体提取", expanded=False):
                        provs = result.get('active_providers', [])
                        st.write(f"**使用模型**: {provs[0] if provs else 'N/A'}")
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
                        provs = result.get('active_providers', [])
                        st.write(
                            f"**使用模型**: {provs[1] if len(provs) > 1 else (provs[0] if provs else 'N/A')} "
                        )
                        st.write("**任务**: 验证和修正提取的实体信息，确保数据准确性")
                        if result.get('step2'):
                            st.json(result['step2'], expanded=False)

                    # 步骤3: 匹配度分析
                    with st.expander("📊 步骤3: 匹配度分析", expanded=False):
                        provs = result.get('active_providers', [])
                        st.write(f"**使用模型**: {provs[0] if provs else 'N/A'}")
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
                                with tempfile.NamedTemporaryFile(
                                        delete=False,
                                        suffix=
                                        f".{resume_file.name.split('.')[-1]}"
                                ) as tmp:
                                    tmp.write(resume_file.getvalue())
                                    tmp_path = tmp.name
                                processed = fp.process_file(tmp_path)
                                content = processed.get('content', '')
                                import os, time, uuid
                                root = os.path.normpath(
                                    os.path.join(os.path.dirname(__file__),
                                                 '..'))
                                save_dir = os.path.join(
                                    root, 'uploads', 'resumes')
                                os.makedirs(save_dir, exist_ok=True)
                                fname = f"resume_{int(time.time())}_{uuid.uuid4().hex}.{resume_file.name.split('.')[-1]}"
                                save_path = os.path.join(save_dir, fname)
                                with open(save_path, 'wb') as f:
                                    f.write(resume_file.getvalue())
                                meta_r = {
                                    'source_file_path':
                                    save_path,
                                    'source_file_type':
                                    processed.get('file_type', '')
                                }
                                resume = candidate_service.upload_resume(
                                    content, meta=meta_r)
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
                        root = os.path.normpath(
                            os.path.join(os.path.dirname(__file__), '..'))
                        save_dir = os.path.join(root, 'uploads', 'resumes')
                        os.makedirs(save_dir, exist_ok=True)
                        fname = f"resume_{int(time.time())}_{uuid.uuid4().hex}.txt"
                        save_path = os.path.join(save_dir, fname)
                        with open(save_path, 'w', encoding='utf-8') as f:
                            f.write(resume_text)
                        resume = candidate_service.upload_resume(
                            resume_text,
                            meta={
                                'source_file_path': save_path,
                                'source_file_type': 'Text文件'
                            })
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

        jd_portrait_text = st.text_area("输入用于画像的目标JD文本",
                                        height=150,
                                        key="candidate_portrait_jd")
        if st.button("生成简历画像", key="candidate_generate_portrait"):
            with st.spinner("正在生成简历画像..."):
                from core.visualizer import Visualizer
                selected_resume = next(r for r in resumes
                                       if r['resume_id'] == selected_resume_id)
                jd_struct = candidate_service.data_processor.process_jd_text(
                    jd_portrait_text or selected_resume['cleaned_text'])
                jd_struct = candidate_service.feature_engine.extract_features_from_jd(
                    jd_struct)
                fig = Visualizer().generate_radar_chart(
                    selected_resume, jd_struct)
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

            # 提交异步任务进行岗位匹配
            def match_task():
                return candidate_service.match_resume_to_jobs(
                    selected_resume_id, top_k)

            task_id = task_manager.submit_task(match_task)

            # 显示进度条和状态
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 轮询任务进度
            while True:
                task = task_manager.get_task(task_id)
                if task:
                    # 更新进度条
                    progress_bar.progress(task.progress)

                    # 更新状态文本
                    status_text.text(
                        f"当前状态: {task.status} - 进度: {task.progress:.1%}")

                    # 检查任务是否完成
                    if task.status in ["completed", "failed"]:
                        if task.status == "completed":
                            results = task.result
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
                                    st.write(
                                        f"**职位描述**: {job['cleaned_text'][:150]}..."
                                    )
                                    st.write(
                                        f"**技能要求**: {', '.join(job['skills'])}"
                                    )
                        else:
                            st.error(f"❌ 匹配失败: {task.error}")
                        break

                # 等待一段时间后再次查询
                time.sleep(0.1)

    st.divider()

    # 5. 模拟面试
    st.subheader("🎭 模拟面试")
    resumes_for_interview = candidate_service.get_resume_list()
    resume_options_iv = {
        r['resume_id']: r['cleaned_text'][:50] + "..."
        for r in resumes_for_interview
    } if resumes_for_interview else {}
    selected_resume_iv = st.selectbox(
        "选择简历",
        list(resume_options_iv.keys()) if resume_options_iv else [""],
        format_func=lambda x: f"{x}: {resume_options_iv.get(x,'')}"
        if x else "",
        key="candidate_iv_resume")
    jd_iv_text = st.text_area("输入目标JD文本", height=120, key="candidate_iv_jd")
    if st.button("生成面试题", key="candidate_generate_interview"):
        if selected_resume_iv and jd_iv_text.strip():
            with st.spinner("正在生成面试题..."):
                res = next(r for r in resumes_for_interview
                           if r['resume_id'] == selected_resume_iv)
                qs = candidate_service.llm_chain.generate_interview_questions(
                    res['cleaned_text'], jd_iv_text)
                st.subheader("面试题")
                for i, q in enumerate(qs, 1):
                    st.write(f"{i}. {q}")
        else:
            st.error("❌ 请选择简历并输入JD文本")
    answer_text = st.text_area("输入你的回答以评估（可选）",
                               height=120,
                               key="candidate_iv_answer")
    if st.button("评估回答", key="candidate_evaluate_answer"):
        if selected_resume_iv and jd_iv_text.strip() and answer_text.strip():
            res = next(r for r in resumes_for_interview
                       if r['resume_id'] == selected_resume_iv)
            eval_res = candidate_service.llm_chain.evaluate_interview_answer(
                res['cleaned_text'], jd_iv_text, answer_text)
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
        sel_res_apply = st.selectbox(
            "选择简历进行投递", [r['resume_id'] for r in resumes_for_apply],
            key="apply_resume")
        sel_job_apply = st.selectbox("选择职位进行投递",
                                     [j['job_id'] for j in jobs_for_apply],
                                     key="apply_job")
        if st.button("一键投递", key="do_apply"):
            app = candidate_service.submit_application(sel_res_apply,
                                                       sel_job_apply)
            st.success(f"投递成功，ID: {app['application_id']}")
    apps = candidate_service.get_applications()
    if apps:
        for app in apps:
            with st.expander(
                    f"投递 {app['application_id']} - {app['resume_id']} -> {app['job_id']} 状态: {app['status']}"
            ):
                new_status = st.selectbox(
                    "更新状态",
                    ["submitted", "read", "interview", "rejected", "offer"],
                    key=f"status_{app['application_id']}")
                rejection_text = st.text_area(
                    "拒信文本（可选）", height=100, key=f"rej_{app['application_id']}")
                if st.button("更新", key=f"upd_{app['application_id']}"):
                    updated = candidate_service.update_application_status(
                        app['application_id'], new_status,
                        rejection_text if new_status == "rejected" else None)
                    st.success("已更新")
                    if updated['history'] and 'rejection_analysis' in updated[
                            'history'][-1]:
                        st.json(updated['history'][-1]['rejection_analysis'])

    st.subheader("🧭 学习成长路径")
    resumes_lp = candidate_service.get_resume_list()
    if resumes_lp:
        sel_res_lp = st.selectbox("选择简历生成学习路径",
                                  [r['resume_id'] for r in resumes_lp],
                                  key="lp_resume")
        jd_lp_text = st.text_area("输入目标JD文本", height=120, key="lp_jd_text")
        if st.button("生成学习路径", key="generate_lp"):
            if jd_lp_text.strip():
                plan = candidate_service.generate_learning_path(
                    sel_res_lp, jd_lp_text)
                st.json(plan)
            else:
                st.error("❌ 请输入JD文本")

# 页脚
st.markdown("---")
st.info("智能简历筛选系统 v2.0.0")
