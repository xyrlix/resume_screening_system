import os
import json
import time
import math
import streamlit as st
import urllib.request
import urllib.error
import importlib.util
from typing import Tuple, List, Dict
try:
    from streamlit_echarts import st_echarts
except Exception:
    st_echarts = None


def page_setup():
    st.set_page_config(page_title="简历筛选系统", page_icon="📄", layout="wide")
    st.title("📄 智能简历筛选可视化")
    st.caption("简历与岗位匹配，可视化展示匹配结果")


def try_health(url: str, timeout: float = 0.8) -> bool:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def detect_api_base() -> str:
    # 优先使用环境变量（例如在部署时配置）
    env_base = os.getenv("API_BASE_URL")
    if env_base and try_health(env_base.rstrip("/") + "/health"):
        return env_base.rstrip("/")
    # 尝试本地端口范围（与后端自动回退一致）
    for port in range(8000, 8006):
        base = f"http://127.0.0.1:{port}"
        if try_health(base + "/health"):
            return base
    # 兜底：默认 8000
    return "http://127.0.0.1:8000"


def api_post(base: str, path: str, payload: dict, timeout: float | None = None):
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    to = timeout if timeout is not None else float(st.session_state.get("api_timeout", 2.5))
    try:
        with urllib.request.urlopen(req, timeout=to) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except urllib.error.HTTPError as e:
        return None, f"HTTPError {e.code}: {e.reason}"
    except Exception as e:
        return None, str(e)

def _download_excel_or_csv_sheets(sheets: Dict[str, List[Dict]], excel_name: str, csv_prefix: str, prefer_excel: bool = True):
    import pandas as pd, io, importlib.util
    try:
        if not prefer_excel or importlib.util.find_spec("xlsxwriter") is None:
            raise ImportError("xlsxwriter not available")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            for name, rows in sheets.items():
                pd.DataFrame(rows).to_excel(w, index=False, sheet_name=name)
        st.download_button("下载Excel", data=buf.getvalue(), file_name=excel_name)
    except Exception:
        for name, rows in sheets.items():
            import pandas as pd
            df = pd.DataFrame(rows)
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(f"下载{name} CSV", data=csv, file_name=f"{csv_prefix}_{name}.csv")

def _post_with_progress(base: str, path: str, payload: dict, timeout: float, label: str):
    import concurrent.futures, time
    bar = st.progress(0.0)
    to = max(float(timeout or 10.0), 1.0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(api_post, base, path, payload, to)
        start = time.time()
        while not fut.done():
            elapsed = time.time() - start
            pct = min(0.95, elapsed / to)
            bar.progress(pct)
            time.sleep(0.1)
        res, err = fut.result()
        bar.progress(1.0)
        return res, err


def _load_module(module_filename: str):
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, module_filename)
    spec = importlib.util.spec_from_file_location(module_filename.replace('.py',''), path)
    mod = importlib.util.module_from_spec(spec)
    if spec and spec.loader:
        spec.loader.exec_module(mod)
    return mod


# 预加载用于本地匹配与文件解析的模块
_matcher_mod = None
_preprocess_mod = None
try:
    _matcher_mod = _load_module('04_matcher_model.py')
except Exception as e:
    _matcher_mod = None
try:
    _preprocess_mod = _load_module('02_data_preprocess.py')
except Exception:
    _preprocess_mod = None


def sidebar():
    if "api_base" not in st.session_state:
        st.session_state.api_base = detect_api_base()
    with st.sidebar:
        st.header("⚙️ 配置")
        api_base = st.text_input("API地址", st.session_state.api_base)
        refresh = st.button("检测API健康")
        if refresh:
            ok = try_health(api_base.rstrip("/") + "/health")
            if ok:
                st.success("API健康正常")
                st.session_state.api_base = api_base.rstrip("/")
            else:
                st.error("API不可用，请检查后端服务或端口")
        st.caption("提示：可通过 API_HOST/API_PORT 控制后端监听；未设置时会自动在 8000–8005 中选择可用端口。")
        st.markdown("---")
        st.subheader("全局设置")
        api_to = st.slider("默认API超时(秒)", 2.0, 30.0, float(st.session_state.get("api_timeout", 12.0)), 0.5, key="sidebar_api_timeout")
        st.session_state["api_timeout"] = float(api_to)
        llm_enabled = st.checkbox("启用大模型面试题(全局)", value=bool(st.session_state.get("llm_global_enabled", False)), key="sidebar_llm_enabled")
        export_csv_default = st.checkbox("默认导出为CSV", value=bool(st.session_state.get("export_csv_default", False)), key="sidebar_export_csv")
        apply = st.button("应用全局设置", key="sidebar_apply")
        if apply:
            st.session_state["export_csv_default"] = bool(export_csv_default)
            st.session_state["llm_global_enabled"] = bool(llm_enabled)
            _ = api_post(st.session_state.api_base, "/config/llm_enabled", {"enabled": bool(llm_enabled)})


def page_health():
    st.subheader("服务概览")
    base = st.session_state.api_base
    col1, col2 = st.columns(2)
    with col1:
        st.write("后端 API:", base)
        ok = try_health(base + "/health")
        st.metric("API健康", "正常" if ok else "异常")
        st.link_button("查看API文档", base + "/docs")
    with col2:
        st.write("匹配配置文件:", "config/matching.json")
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "matching.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            st.json(cfg)
        else:
            st.info("未找到匹配配置文件，使用脚本内置默认配置。")
    st.markdown("---")
    st.subheader("预热向量索引")
    resume_json = os.path.join("data", "processed", "resumes_for_annotation.json")
    run = st.button("从处理后的简历入库向量", key="warmup_vector")
    if run:
        if not os.path.isfile(resume_json):
            st.error("简历JSON不存在")
            return
        with open(resume_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = [{"id": str(i), "text": d.get("text", "")} for i, d in enumerate(data)]
        res, err = api_post(base, "/vector_index", {"items": items}, timeout=12.0)
        if err:
            st.error(f"入库失败：{err}")
        else:
            st.success(f"入库完成：{res.get('count')} 条")


def page_predict():
    st.subheader("实体预测（NER）")
    base = st.session_state.api_base
    text = st.text_area("输入简历文本", height=180, placeholder="在此粘贴简历文本…")
    if st.button("调用 /predict"):
        if not text.strip():
            st.warning("请先输入简历文本")
        else:
            res, err = api_post(base, "/predict", {"text": text})
            if err:
                st.error(f"调用失败：{err}")
            else:
                ents = res.get("entities", [])
                st.success(f"提取到 {len(ents)} 个实体")
                if ents:
                    st.dataframe(ents, use_container_width=True)


def page_match_single():
    st.subheader("单次匹配（Resume vs Job）")
    base = st.session_state.api_base
    c1, c2 = st.columns(2)
    with c1:
        resume_text = st.text_area("简历文本", height=200, placeholder="在此粘贴简历文本…")
        rfile = st.file_uploader("上传简历文件（txt/md/docx/pdf）", type=["txt", "md", "docx", "pdf"], key="resume_file")
        if rfile is not None:
            resume_text = _read_uploaded_text(rfile)
            st.info("已从上传文件填充简历文本")
    with c2:
        job_text = st.text_area("岗位文本", height=200, placeholder="在此粘贴岗位描述…")
        jfile = st.file_uploader("上传岗位文件（txt/md/docx/pdf）", type=["txt", "md", "docx", "pdf"], key="job_file")
        if jfile is not None:
            job_text = _read_uploaded_text(jfile)
            st.info("已从上传文件填充岗位文本")

    use_temp_weights = st.checkbox("使用临时权重在本地计算（不调用后端）", value=False)
    if st.button("调用 /match"):
        if not resume_text.strip() or not job_text.strip():
            st.warning("请同时输入简历与岗位文本")
        else:
            if use_temp_weights and _matcher_mod is not None and "temp_weights" in st.session_state:
                res = _local_match_with_weights(resume_text, job_text, st.session_state["temp_weights"])
                _render_match_result(res)
            else:
                res, err = _post_with_progress(base, "/match", {"resume_text": resume_text, "job_text": job_text}, float(st.session_state.get("api_timeout", 12.0)), "匹配评估")
                if err:
                    st.error(f"调用失败：{err}")
                else:
                    _render_match_result(res)


def page_match_batch():
    st.subheader("批量匹配（目录）")
    base = st.session_state.api_base
    default_resume_json = os.path.join("data", "processed", "resumes_for_annotation.json")
    default_jobs_dir = os.path.join("data", "raw_jobs")
    resume_json = st.text_input("简历JSON路径", default_resume_json)
    jobs_dir = st.text_input("岗位目录", default_jobs_dir)
    use_temp_weights = st.checkbox("使用临时权重在本地计算（不调用后端）", value=False, key="batch_use_local")
    run = st.button("开始批量匹配")
    if run:
        if not os.path.isfile(resume_json):
            st.error("简历JSON不存在")
            return
        if not os.path.isdir(jobs_dir):
            st.error("岗位目录不存在")
            return
        with open(resume_json, "r", encoding="utf-8") as f:
            resumes = json.load(f)
        jobs = []
        for name in os.listdir(jobs_dir):
            if not name.lower().endswith((".txt", ".md")):
                continue
            p = os.path.join(jobs_dir, name)
            try:
                with open(p, "r", encoding="utf-8") as jf:
                    jt = jf.read()
                jobs.append({"file": name, "text": jt})
            except Exception:
                pass
        results = []
        progress = st.progress(0)
        total = max(len(resumes) * max(len(jobs), 1), 1)
        done = 0
        for r in resumes:
            rtext = r.get("text", "")
            for j in jobs:
                if use_temp_weights and _matcher_mod is not None and "temp_weights" in st.session_state:
                    res = _local_match_with_weights(rtext, j.get("text", ""), st.session_state["temp_weights"])
                    results.append({
                        "resume_id": r.get("id"),
                        "job_file": j.get("file"),
                        "score": res.get("score"),
                        "details": res.get("details"),
                    })
                else:
                    payload = {"resume_text": rtext, "job_text": j.get("text", "")}
                    res, err = api_post(base, "/match", payload)
                    if err:
                        results.append({"resume_id": r.get("id"), "job_file": j.get("file"), "error": err})
                    else:
                        results.append({
                            "resume_id": r.get("id"),
                            "job_file": j.get("file"),
                            "score": res.get("score"),
                            "details": res.get("details"),
                        })
                done += 1
                progress.progress(min(done / total, 1.0))
        st.success(f"批量完成，共 {len(results)} 条结果")
        st.dataframe(results, use_container_width=True)
        st.download_button("下载JSON结果", data=json.dumps(results, ensure_ascii=False, indent=2), file_name="match_results.json")


def page_job_search():
    st.subheader("按岗位检索简历（Top N）")
    base = st.session_state.api_base
    # 岗位输入与上传
    c1, c2 = st.columns(2)
    with c1:
        job_text = st.text_area("岗位文本", height=160, placeholder="在此粘贴岗位描述…")
    with c2:
        jfile = st.file_uploader("上传岗位文件（txt/md/docx/pdf）", type=["txt", "md", "docx", "pdf"], key="job_search_file")
        if jfile is not None:
            job_text = _read_uploaded_text(jfile)
            st.info("已从上传文件填充岗位文本")

    source = st.radio("简历来源", ["使用处理后的JSON", "扫描简历目录"], index=0)
    top_k = st.number_input("Top N 输出", min_value=1, value=5, step=1)
    use_temp_weights = st.checkbox("使用临时权重本地计算（不调用后端）", value=False, key="job_search_local")

    resumes = []
    if source == "使用处理后的JSON":
        default_resume_json = os.path.join("data", "processed", "resumes_for_annotation.json")
        resume_json = st.text_input("简历JSON路径", default_resume_json, key="job_search_json")
        if st.button("计算并检索Top N", key="job_search_run_json"):
            if not job_text.strip():
                st.warning("请先填写岗位文本")
                return
            if not os.path.isfile(resume_json):
                st.error("简历JSON不存在")
                return
            with open(resume_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            resumes = [{"id": (i if d.get("id") is None else d.get("id")), "text": d.get("text", "")} for i, d in enumerate(data)]
            _run_job_search(resumes, job_text, base, top_k, use_temp_weights)
    else:
        default_resume_dir = os.path.join("data", "raw_resumes")
        resume_dir = st.text_input("简历目录（docx/pdf/txt/md）", default_resume_dir, key="job_search_dir")
        if st.button("扫描目录并检索Top N", key="job_search_run_dir"):
            if not job_text.strip():
                st.warning("请先填写岗位文本")
                return
            if not os.path.isdir(resume_dir):
                st.error("简历目录不存在")
                return
            # 扫描目录并解析
            items = []
            for name in os.listdir(resume_dir):
                p = os.path.join(resume_dir, name)
                if not os.path.isfile(p):
                    continue
                low = name.lower()
                text = ""
                try:
                    if low.endswith((".txt", ".md")):
                        with open(p, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                    elif low.endswith(".docx"):
                        if _preprocess_mod is not None:
                            text = _preprocess_mod.parse_word(p)
                    elif low.endswith(".pdf"):
                        if _preprocess_mod is not None:
                            text = _preprocess_mod.parse_pdf(p)
                except Exception:
                    text = ""
                if text:
                    items.append({"id": name, "text": text, "path": p})
            if not items:
                st.warning("目录中未解析到简历文本")
                return
            _run_job_search(items, job_text, base, top_k, use_temp_weights)

    # 如果已有检索结果（来自上一次运行），也在页面尾部展示，保证点击“查看”不会丢失数据
    if st.session_state.get("job_search_top"):
        st.markdown("---")
        _render_job_search_results(st.session_state["job_search_top"])

def page_ingest():
    st.subheader("在线采集与入库")
    base = st.session_state.api_base
    plat = st.selectbox("平台", ["通用URL", "Boss直聘", "智联招聘", "猎聘", "LinkedIn"], index=0, key="ingest_platform")
    urls_text = st.text_area("URL（每行一个）", height=120, key="ingest_urls")
    cookie = st.text_input("Cookie（需要账号登录后的Cookie）", value="", key="ingest_cookie")
    if st.button("采集并入库向量", key="ingest_run"):
        urls = [u.strip() for u in (urls_text or "").split("\n") if u.strip()]
        if not urls:
            st.warning("请填写至少一个URL")
            return
        if plat == "通用URL":
            res, err = _post_with_progress(base, "/ingest_online", {"urls": urls}, 12.0, "在线采集")
        elif plat == "Boss直聘":
            if not cookie.strip():
                st.error("Boss直聘需要Cookie")
                return
            res, err = _post_with_progress(base, "/ingest_bosszhipin", {"urls": urls, "cookie": cookie}, 12.0, "在线采集")
        elif plat == "智联招聘":
            if not cookie.strip():
                st.error("智联招聘需要Cookie")
                return
            res, err = _post_with_progress(base, "/ingest_zhilian", {"urls": urls, "cookie": cookie}, 12.0, "在线采集")
        elif plat == "猎聘":
            if not cookie.strip():
                st.error("猎聘需要Cookie")
                return
            res, err = _post_with_progress(base, "/ingest_liepin", {"urls": urls, "cookie": cookie}, 12.0, "在线采集")
        else:
            if not cookie.strip():
                st.error("LinkedIn需要Cookie")
                return
            res, err = _post_with_progress(base, "/ingest_linkedin", {"urls": urls, "cookie": cookie}, 12.0, "在线采集")
        if err:
            st.error(f"采集失败：{err}")
            return
        items = res.get("items", [])
        st.success(f"采集完成，得到 {len(items)} 条")
        rows = [{"url": it.get("url"), "text_len": len(it.get("text", "")), "tags": ",".join(it.get("fairness_tags", []))} for it in items]
        if rows:
            st.dataframe(rows, use_container_width=True)
        push = [{"id": str(i), "text": it.get("text", "")} for i, it in enumerate(items)]
        if push:
            _post_with_progress(base, "/vector_index", {"items": push}, 8.0, "向量入库")
            st.info("已入库到向量索引，可在‘三级漏斗’页面进行筛选")


def _run_job_search(resumes: List[Dict], job_text: str, base: str, top_k: int, use_temp_weights: bool):
    results = []
    progress = st.progress(0)
    total = max(len(resumes), 1)
    done = 0
    for r in resumes:
        rtext = r.get("text", "")
        if not rtext:
            continue
        if use_temp_weights and _matcher_mod is not None and "temp_weights" in st.session_state:
            res = _local_match_with_weights(rtext, job_text, st.session_state["temp_weights"])
            score = res.get("score", 0)
            details = res.get("details", {})
        else:
            res, err = api_post(base, "/match", {"resume_text": rtext, "job_text": job_text})
            if err:
                results.append({"resume_id": r.get("id"), "error": err})
                done += 1
                progress.progress(min(done / total, 1.0))
                continue
            score = res.get("score", 0)
            details = res.get("details", {})
        results.append({
            "resume_id": r.get("id"),
            "score": score,
            "details": details,
            # 便于点击查看原文内容与下载
            "resume_text": rtext,
            "resume_path": r.get("path"),
            "resume_profile": res.get("resume_profile", {}),
            "job_profile": res.get("job_profile", {})
        })
        done += 1
        progress.progress(min(done / total, 1.0))
    # 排序与Top N，并保存到会话状态，避免点击“查看”后丢失数据
    results = sorted([x for x in results if "score" in x], key=lambda x: x.get("score", 0), reverse=True)
    top = results[:top_k]
    st.session_state["job_search_top"] = top
    st.success(f"完成检索，共 {len(results)} 条结果；Top {top_k} 如下")


def _render_job_search_results(top: List[Dict]):
    if not top:
        st.info("暂无结果可展示")
        return
    # 仅展示关键列，避免在表格中显示大文本
    display_top = [{
        "resume_id": t.get("resume_id"),
        "score": t.get("score"),
        "matched_skills": ", ".join(t.get("details", {}).get("matched_skills", []))
    } for t in top]
    st.dataframe(display_top, use_container_width=True)
    st.download_button("下载TopN JSON", data=json.dumps(top, ensure_ascii=False, indent=2), file_name="top_resumes.json")

    # 交互：左右布局（左侧Top N列表，右侧对应简历原文）
    st.markdown("---")
    st.subheader("查看Top N的简历原文")
    ids = [t.get("resume_id") for t in top]
    default_id = ids[0] if ids else None
    if "top_view_select" not in st.session_state or st.session_state["top_view_select"] not in ids:
        st.session_state["top_view_select"] = default_id

    colL, colR = st.columns([1, 2])
    # 左侧：Top N 排序列表与选择
    with colL:
        rank_rows = [{"rank": i + 1, "resume_id": t.get("resume_id"), "score": t.get("score")} for i, t in enumerate(top)]
        st.dataframe(rank_rows, use_container_width=True)
        # 使用大按钮切换选择
        for i, t in enumerate(top):
            rid = t.get("resume_id")
            score = t.get("score")
            label = f"第{i+1}名：{rid}（分数 {score}）"
            if st.button(label, key=f"top_btn_{rid}", type="primary"):
                st.session_state["top_view_select"] = rid

    # 右侧：显示选中简历原文与下载 + 自动生成图表
    with colR:
        selected = st.session_state.get("top_view_select")
        if selected is not None:
            chosen = next((x for x in top if x.get("resume_id") == selected), None)
            if chosen:
                st.info(f"正在查看：{selected}")
                text = chosen.get("resume_text") or ""
                path = chosen.get("resume_path")
                if path and os.path.isfile(path):
                    low = path.lower()
                    try:
                        if low.endswith((".txt", ".md")):
                            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                                text = f.read()
                        elif low.endswith(".docx"):
                            parsed = ""
                            if _preprocess_mod is not None:
                                try:
                                    parsed = _preprocess_mod.parse_word(path)
                                except Exception:
                                    parsed = ""
                            if not parsed:
                                try:
                                    from docx import Document
                                    doc = Document(path)
                                    parsed = "\n".join(p.text for p in doc.paragraphs)
                                except Exception:
                                    parsed = ""
                            text = parsed or text
                        elif low.endswith(".pdf"):
                            parsed = ""
                            if _preprocess_mod is not None:
                                try:
                                    parsed = _preprocess_mod.parse_pdf(path)
                                except Exception:
                                    parsed = ""
                            if not parsed:
                                try:
                                    import PyPDF2
                                    with open(path, "rb") as pf:
                                        reader = PyPDF2.PdfReader(pf)
                                        for page in reader.pages:
                                            page_text = page.extract_text()
                                            if page_text:
                                                parsed += page_text
                                except Exception:
                                    parsed = ""
                            text = parsed or text
                    except Exception:
                        pass
                    try:
                        with open(path, "rb") as bf:
                            st.download_button("下载原文件", data=bf.read(), file_name=os.path.basename(path))
                    except Exception:
                        st.warning("无法提供原文件下载")

                st.text_area("简历内容预览", value=text, height=300)
                st.markdown("---")
                _render_match_charts(chosen.get("details", {}))
                _render_rich_features(chosen.get("details", {}), chosen.get("resume_profile", {}), chosen.get("job_profile", {}))

def _render_rich_features(details: Dict, rprof: Dict, jprof: Dict):
    try:
        import matplotlib.pyplot as plt
        th = _theme()
        fig, ax = plt.subplots(figsize=(5, 3))
        feats = [
            ("skills", float(details.get("skill_ratio", 0))),
            ("degree", float(details.get("degree_score", 0))),
            ("years", float(details.get("years_score", 0))),
            ("position", float(details.get("position_score", 0))),
            ("keywords", float(details.get("keyword_ratio", 0))),
            ("certs", float(details.get("certs_score", 0))),
            ("languages", float(details.get("languages_ratio", 0))),
        ]
        ax.bar([k for k, v in feats], [v for k, v in feats], color=th["primary"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.set_title("特征贡献")
        st.pyplot(fig, use_container_width=False)
    except Exception:
        pass
    # 饼图：技能匹配情况（命中 vs 未命中）
    try:
        import matplotlib.pyplot as plt
        th = _theme()
        job_sk = jprof.get("skills", [])
        matched = details.get("matched_skills", [])
        miss = max(len(job_sk) - len(matched), 0)
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.pie([len(matched), miss], labels=["命中", "未命中"], autopct='%1.0f%%', colors=[th["success"], th["warning"]])
        ax2.set_title("技能命中率")
        st.pyplot(fig2, use_container_width=False)
    except Exception:
        pass

def page_config_view():
    st.subheader("匹配配置查看")
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "matching.json")
    st.write("配置文件路径：", cfg_path)
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        st.json(cfg)
        st.info("提示：当前API在导入匹配脚本时加载配置；如需修改配置生效，请重启后端。")
    else:
        st.warning("未找到配置文件。")

    st.markdown("---")
    st.subheader("临时权重试验（不改文件，仅前端生效）")
    c1, c2, c3, c4 = st.columns(4)
    w_skills = c1.slider("skills", 0.0, 1.0, 0.5, 0.05)
    w_degree = c2.slider("degree", 0.0, 1.0, 0.2, 0.05)
    w_years = c3.slider("years", 0.0, 1.0, 0.2, 0.05)
    w_position = c4.slider("position", 0.0, 1.0, 0.1, 0.05)
    if st.button("应用到本地匹配"):
        st.session_state["temp_weights"] = {
            "skills": float(w_skills),
            "degree": float(w_degree),
            "years": float(w_years),
            "position": float(w_position),
        }
        st.success("已设置临时权重（仅前端本地匹配有效）")
    st.markdown("---")
    st.subheader("行业模板库编辑")
    base = st.session_state.api_base
    tmpl = {}
    try:
        import urllib.request
        with urllib.request.urlopen(base + "/config/industry_templates", timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tmpl = data.get("templates", {})
    except Exception:
        tmpl = {}
    edit_text = st.text_area("编辑JSON", value=json.dumps(tmpl, ensure_ascii=False, indent=2), height=200)
    if st.button("保存行业模板"):
        try:
            new_obj = json.loads(edit_text)
        except Exception:
            st.error("JSON不合法")
            new_obj = None
        if new_obj is not None:
            res, err = api_post(base, "/config/industry_templates", {"templates": new_obj})
            if err:
                st.error(f"保存失败：{err}")
            else:
                st.success("已保存并应用行业模板")


def _read_uploaded_text(file) -> str:
    name = (file.name or "").lower()
    try:
        if name.endswith((".txt", ".md")):
            return file.read().decode("utf-8", errors="ignore")
        if name.endswith(".docx"):
            try:
                from docx import Document
                doc = Document(file)
                return "\n".join(p.text for p in doc.paragraphs)
            except Exception:
                # 回退到预处理模块方法（若存在，需要写入临时文件）
                if _preprocess_mod is not None:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name
                    try:
                        return _preprocess_mod.parse_word(tmp_path)
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                return ""
        if name.endswith(".pdf"):
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                return text
            except Exception:
                # 回退到预处理模块方法（若存在，需要写入临时文件）
                if _preprocess_mod is not None:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name
                    try:
                        return _preprocess_mod.parse_pdf(tmp_path)
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                return ""
        # 其他类型按文本尝试
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _local_match_with_weights(resume_text: str, job_text: str, weights: Dict[str, float]) -> Dict:
    if _matcher_mod is None:
        return {"score": 0, "details": {"error": "matcher module not available"}}
    # 覆盖权重（仅本次计算，不持久化）
    orig = dict(_matcher_mod.WEIGHTS)
    new_w = dict(orig)
    new_w.update({k: float(v) for k, v in weights.items()})
    _matcher_mod.WEIGHTS = new_w
    try:
        rprofile = _matcher_mod.profile_from_text(resume_text)
        jprofile = _matcher_mod.job_profile_from_text(job_text)
        score, details = _matcher_mod.match_score(rprofile, jprofile)
        return {
            "score": score,
            "resume_profile": rprofile,
            "job_profile": jprofile,
            "details": details,
        }
    finally:
        _matcher_mod.WEIGHTS = orig


def _render_match_result(res: Dict):
    st.success("匹配完成")
    score = res.get("score", 0)
    details = res.get("details", {})
    colA, colB, colC = st.columns(3)
    colA.metric("综合得分", f"{score}")
    colB.metric("技能匹配比例", f"{details.get('skill_ratio', 0)}")
    colC.metric("年限匹配", f"{details.get('years_score', 0)}")
    st.write("命中技能：", ", ".join(details.get("matched_skills", [])))
    st.write("学历匹配：", details.get("degree_score", 0))
    st.write("职位匹配：", details.get("position_score", 0))
    st.expander("简历画像").json(res.get("resume_profile", {}))
    st.expander("岗位画像").json(res.get("job_profile", {}))
    _render_match_charts(details)
    _render_match_pipeline(details, float(score))


def _render_match_charts(details: Dict):
    try:
        import matplotlib.pyplot as plt
        # 雷达图：技能、学历、年限、职位
        labels = ["skills", "degree", "years", "position"]
        values = [
            float(details.get("skill_ratio", 0)),
            float(details.get("degree_score", 0)),
            float(details.get("years_score", 0)),
            float(details.get("position_score", 0)),
        ]
        angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
        values += values[:1]
        angles += angles[:1]
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels([])
        st.pyplot(fig, use_container_width=False)
    except Exception:
        st.info("图表渲染失败，已跳过雷达图。")
    # 条形图：技能命中数与岗位技能数（若能获取）
    matched = len(details.get("matched_skills", []))
    try:
        import matplotlib.pyplot as plt
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.bar(["matched"], [matched], color="#4caf50")
        ax2.set_ylabel("count")
        st.pyplot(fig2, use_container_width=False)
    except Exception:
        pass

def _render_match_pipeline(details: Dict, final_score: float):
    hard = (
        float(details.get("degree_score", 0)) +
        float(details.get("years_score", 0)) +
        float(details.get("position_score", 0))
    ) / 3.0
    soft = float(details.get("skill_ratio", 0))
    fmt = float(details.get("format_score", 0))
    comp = 0.4 * hard + 0.5 * soft + 0.1 * fmt
    st.markdown("---")
    st.subheader("阶段评分")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("硬性条件", f"{round(hard, 4)}")
    c2.metric("软性条件", f"{round(soft, 4)}")
    c3.metric("格式规范", f"{round(fmt, 4)}")
    c4.metric("综合加权", f"{round(comp, 4)}")
    st.subheader("综合评价与建议")
    adv = []
    if hard < 0.5:
        adv.append("硬性条件不足，可优先筛选学历/年限更匹配候选")
    if soft < 0.5:
        adv.append("技能匹配较弱，建议关注核心技能训练或转岗可能")
    if fmt < 0.4:
        adv.append("简历格式较弱，建议优化结构与关键经历描述")
    if final_score >= 0.8:
        adv.append("总体高度匹配，建议进入面试流程")
    elif final_score >= 0.6:
        adv.append("总体较为匹配，可考虑电话初筛后决定")
    else:
        adv.append("总体匹配一般，建议放入储备或不进入下一轮")
    st.write("；".join(adv))


def page_ner_eval():
    st.subheader("NER 评估（实体级 Precision/Recall/F1）")
    base = st.session_state.api_base
    data_path = st.text_input("标注数据路径", os.path.join("data", "processed", "entity_train.json"))
    sample_n = st.number_input("评估样本数（0 为全部）", min_value=0, value=0, step=10)
    run = st.button("开始评估")
    if run:
        if not os.path.isfile(data_path):
            st.error("标注数据不存在")
            return
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        if sample_n and sample_n > 0:
            dataset = dataset[:sample_n]
        TP = FP = FN = 0
        rows = []
        trace_prec = []
        trace_rec = []
        trace_f1 = []
        details_rows = []
        import collections
        type_gt = collections.Counter()
        type_pred = collections.Counter()
        cum_type = collections.defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
        trace_type = collections.defaultdict(lambda: {"prec": [], "rec": [], "f1": []})
        fp_examples = []
        fn_examples = []
        prog = st.progress(0.0)
        total = max(len(dataset), 1)
        for i, item in enumerate(dataset):
            text = item.get("text", "")
            gt = {(e.get("type"), e.get("text")) for e in item.get("entities", []) if e.get("type") and e.get("text")}
            res, err = api_post(base, "/predict", {"text": text})
            if err:
                rows.append({"idx": i, "error": err})
                continue
            pred = {(e.get("type"), e.get("text")) for e in res.get("entities", []) if e.get("type") and e.get("text")}
            tp = len(gt & pred)
            fp = len(pred - gt)
            fn = len(gt - pred)
            TP += tp
            FP += fp
            FN += fn
            prec_i = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_i = (2 * prec_i * rec_i) / (prec_i + rec_i) if (prec_i + rec_i) > 0 else 0.0
            rows.append({"idx": i, "tp": tp, "fp": fp, "fn": fn, "gt": len(gt), "pred": len(pred), "precision": round(prec_i, 4), "recall": round(rec_i, 4), "f1": round(f1_i, 4)})
            trace_prec.append((i + 1, (TP / (TP + FP)) if (TP + FP) > 0 else 0.0))
            trace_rec.append((i + 1, (TP / (TP + FN)) if (TP + FN) > 0 else 0.0))
            trace_f1.append((i + 1, ((2 * (TP / (TP + FP)) * (TP / (TP + FN))) / ((TP / (TP + FP)) + (TP / (TP + FN)))) if ((TP + FP) > 0 and (TP + FN) > 0 and ((TP / (TP + FP)) + (TP / (TP + FN))) > 0) else 0.0))
            details_rows.append({"idx": i, "gt": sorted(list(gt)), "pred": sorted(list(pred))})
            for e in item.get("entities", []) or []:
                t = e.get("type")
                if t:
                    type_gt[t] += 1
            for e in res.get("entities", []) or []:
                t = e.get("type")
                if t:
                    type_pred[t] += 1
            for e in (pred - gt):
                fp_examples.append({"idx": i, "type": e[0], "text": e[1]})
            for e in (gt - pred):
                fn_examples.append({"idx": i, "type": e[0], "text": e[1]})
            tset = sorted(set([x[0] for x in gt] + [x[0] for x in pred]))
            for t in tset:
                tp_t = len([x for x in (gt & pred) if x[0] == t])
                fp_t = len([x for x in (pred - gt) if x[0] == t])
                fn_t = len([x for x in (gt - pred) if x[0] == t])
                cum_type[t]["TP"] += tp_t
                cum_type[t]["FP"] += fp_t
                cum_type[t]["FN"] += fn_t
                ctp = cum_type[t]["TP"]
                cfp = cum_type[t]["FP"]
                cfn = cum_type[t]["FN"]
                p_t = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0.0
                r_t = ctp / (ctp + cfn) if (ctp + cfn) > 0 else 0.0
                f_t = (2 * p_t * r_t) / (p_t + r_t) if (p_t + r_t) > 0 else 0.0
                trace_type[t]["prec"].append((i + 1, p_t))
                trace_type[t]["rec"].append((i + 1, r_t))
                trace_type[t]["f1"].append((i + 1, f_t))
            prog.progress(min((i + 1) / float(total), 1.0))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")
        st.metric("F1", f"{f1:.4f}")
        st.dataframe(rows, use_container_width=True)
        st.markdown("---")
        st.subheader("过程曲线")
        try:
            th = _theme()
            if st_echarts:
                xs = [x for x, y in trace_prec]
                opts = {
                    "tooltip": {"trigger": "axis"},
                    "xAxis": {"type": "category", "data": xs, "name": "样本序号"},
                    "yAxis": {"type": "value", "name": "metric"},
                    "series": [
                        {"type": "line", "name": "Precision", "data": [y for x, y in trace_prec], "smooth": True, "itemStyle": {"color": th["primary"]}},
                        {"type": "line", "name": "Recall", "data": [y for x, y in trace_rec], "smooth": True, "itemStyle": {"color": th["success"]}},
                        {"type": "line", "name": "F1", "data": [y for x, y in trace_f1], "smooth": True, "itemStyle": {"color": th["warning"]}},
                    ]
                }
                st_echarts(opts, height=320)
            else:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(6, 3))
                xs = [x for x, y in trace_prec]
                ax.plot(xs, [y for x, y in trace_prec], label="Precision")
                ax.plot(xs, [y for x, y in trace_rec], label="Recall")
                ax.plot(xs, [y for x, y in trace_f1], label="F1")
                ax.set_ylim(0, 1)
                ax.legend()
                st.pyplot(fig, use_container_width=False)
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3))
            th = _theme()
            ax.bar(["TP", "FP", "FN"], [TP, FP, FN], color=[th["success"], th["danger"], th["warning"]])
            st.pyplot(fig, use_container_width=False)
        except Exception:
            pass
        st.markdown("---")
        st.subheader("类型分布")
        try:
            import matplotlib.pyplot as plt
            cats = sorted(set(list(type_gt.keys()) + list(type_pred.keys())))
            vals_gt = [type_gt.get(c, 0) for c in cats]
            vals_pd = [type_pred.get(c, 0) for c in cats]
            th = _theme()
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            w = 0.35
            xs = list(range(len(cats)))
            ax3.bar([x - w/2 for x in xs], vals_gt, width=w, color=th["primary"], label="GT")
            ax3.bar([x + w/2 for x in xs], vals_pd, width=w, color=th["success"], label="Pred")
            ax3.set_xticks(xs)
            ax3.set_xticklabels(cats, rotation=30)
            ax3.legend()
            st.pyplot(fig3, use_container_width=False)
        except Exception:
            pass
        st.markdown("---")
        st.subheader("按类型指标与曲线")
        cats = sorted(set(list(type_gt.keys()) + list(type_pred.keys()) + list(cum_type.keys())))
        type_rows = []
        for t in cats:
            ctp = cum_type[t]["TP"]
            cfp = cum_type[t]["FP"]
            cfn = cum_type[t]["FN"]
            p_t = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0.0
            r_t = ctp / (ctp + cfn) if (ctp + cfn) > 0 else 0.0
            f_t = (2 * p_t * r_t) / (p_t + r_t) if (p_t + r_t) > 0 else 0.0
            type_rows.append({"type": t, "TP": ctp, "FP": cfp, "FN": cfn, "precision": round(p_t, 4), "recall": round(r_t, 4), "f1": round(f_t, 4)})
        if type_rows:
            import pandas as pd
            df_types = pd.DataFrame(type_rows)
            st.dataframe(df_types, use_container_width=True)
            st.download_button("下载按类型指标CSV", data=df_types.to_csv(index=False).encode("utf-8-sig"), file_name="ner_type_metrics.csv")
            sel_type = st.selectbox("选择实体类型查看曲线", cats, index=0)
            tr = trace_type.get(sel_type)
            if tr and (tr.get("prec") or tr.get("rec") or tr.get("f1")):
                try:
                    th = _theme()
                    if st_echarts:
                        xs = [x for x, y in tr.get("prec", [])]
                        opts_t = {
                            "tooltip": {"trigger": "axis"},
                            "xAxis": {"type": "category", "data": xs, "name": "样本序号"},
                            "yAxis": {"type": "value", "name": "metric"},
                            "series": [
                                {"type": "line", "name": "Precision", "data": [y for x, y in tr.get("prec", [])], "smooth": True, "itemStyle": {"color": th["primary"]}},
                                {"type": "line", "name": "Recall", "data": [y for x, y in tr.get("rec", [])], "smooth": True, "itemStyle": {"color": th["success"]}},
                                {"type": "line", "name": "F1", "data": [y for x, y in tr.get("f1", [])], "smooth": True, "itemStyle": {"color": th["warning"]}},
                            ]
                        }
                        st_echarts(opts_t, height=320)
                    else:
                        import matplotlib.pyplot as plt
                        figt, axt = plt.subplots(figsize=(6, 3))
                        xs = [x for x, y in tr.get("prec", [])]
                        axt.plot(xs, [y for x, y in tr.get("prec", [])], label="Precision")
                        axt.plot(xs, [y for x, y in tr.get("rec", [])], label="Recall")
                        axt.plot(xs, [y for x, y in tr.get("f1", [])], label="F1")
                        axt.set_ylim(0, 1)
                        axt.legend()
                        st.pyplot(figt, use_container_width=False)
                except Exception:
                    pass
        st.markdown("---")
        st.subheader("错例浏览与导出")
        if fp_examples:
            import pandas as pd
            df_fp = pd.DataFrame(fp_examples)
            st.write("FP（预测多余）")
            st.dataframe(df_fp, use_container_width=True)
            st.download_button("下载FP JSON", data=json.dumps(fp_examples, ensure_ascii=False, indent=2), file_name="ner_fp.json")
            st.download_button("下载FP CSV", data=df_fp.to_csv(index=False).encode("utf-8-sig"), file_name="ner_fp.csv")
        if fn_examples:
            import pandas as pd
            df_fn = pd.DataFrame(fn_examples)
            st.write("FN（预测缺失）")
            st.dataframe(df_fn, use_container_width=True)
            st.download_button("下载FN JSON", data=json.dumps(fn_examples, ensure_ascii=False, indent=2), file_name="ner_fn.json")
            st.download_button("下载FN CSV", data=df_fn.to_csv(index=False).encode("utf-8-sig"), file_name="ner_fn.csv")
        st.markdown("---")
        st.subheader("评估报告导出")
        import io
        report_html = f"""
        <html><head><meta charset='utf-8'><title>NER评估报告</title></head><body>
        <h2>汇总指标</h2>
        <ul>
        <li>Precision: {precision:.4f}</li>
        <li>Recall: {recall:.4f}</li>
        <li>F1: {f1:.4f}</li>
        </ul>
        <h2>类型指标</h2>
        <table border='1' cellspacing='0' cellpadding='6'>
        <tr><th>type</th><th>TP</th><th>FP</th><th>FN</th><th>precision</th><th>recall</th><th>f1</th></tr>
        {''.join([f"<tr><td>{r['type']}</td><td>{r['TP']}</td><td>{r['FP']}</td><td>{r['FN']}</td><td>{r['precision']}</td><td>{r['recall']}</td><td>{r['f1']}</td></tr>" for r in type_rows])}
        </table>
        </body></html>
        """
        st.download_button("下载评估报告HTML", data=report_html.encode("utf-8"), file_name="ner_eval_report.html")
        try:
            import matplotlib.pyplot as plt
            buf_proc = io.BytesIO()
            xs = [x for x, y in trace_prec]
            figp, axp = plt.subplots(figsize=(6, 3))
            axp.plot(xs, [y for x, y in trace_prec], label="Precision")
            axp.plot(xs, [y for x, y in trace_rec], label="Recall")
            axp.plot(xs, [y for x, y in trace_f1], label="F1")
            axp.set_ylim(0, 1)
            axp.legend()
            figp.savefig(buf_proc, format="png", dpi=160)
            st.download_button("下载过程曲线PNG", data=buf_proc.getvalue(), file_name="ner_proc_curves.png")
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt
            buf_conf = io.BytesIO()
            th = _theme()
            figc, axc = plt.subplots(figsize=(5, 3))
            axc.bar(["TP", "FP", "FN"], [TP, FP, FN], color=[th["success"], th["danger"], th["warning"]])
            figc.savefig(buf_conf, format="png", dpi=160)
            st.download_button("下载混淆计数PNG", data=buf_conf.getvalue(), file_name="ner_confusion_counts.png")
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt
            buf_types = io.BytesIO()
            cats = sorted(set(list(type_gt.keys()) + list(type_pred.keys())))
            vals_gt = [type_gt.get(c, 0) for c in cats]
            vals_pd = [type_pred.get(c, 0) for c in cats]
            th = _theme()
            figt3, axt3 = plt.subplots(figsize=(6, 3))
            w = 0.35
            xs = list(range(len(cats)))
            axt3.bar([x - w/2 for x in xs], vals_gt, width=w, color=th["primary"], label="GT")
            axt3.bar([x + w/2 for x in xs], vals_pd, width=w, color=th["success"], label="Pred")
            axt3.set_xticks(xs)
            axt3.set_xticklabels(cats, rotation=30)
            axt3.legend()
            figt3.savefig(buf_types, format="png", dpi=160)
            st.download_button("下载类型分布PNG", data=buf_types.getvalue(), file_name="ner_type_distribution.png")
        except Exception:
            pass
        st.markdown("---")
        st.subheader("样本详情")
        if details_rows:
            sel = st.number_input("样本序号", min_value=0, max_value=len(details_rows)-1, value=0, step=1)
            chosen = details_rows[int(sel)]
            st.write("GT 实体：")
            st.dataframe([{"type": t, "text": v} for t, v in chosen.get("gt", [])], use_container_width=True)
            st.write("Pred 实体：")
            st.dataframe([{"type": t, "text": v} for t, v in chosen.get("pred", [])], use_container_width=True)


def main():
    page_setup()
    sidebar()

    tabs = st.tabs(["概览", "实体预测", "单次匹配", "批量匹配", "岗位检索", "在线采集与入库", "三级漏斗", "公平性报告", "决策辅助", "配置查看", "NER评估"])
    with tabs[0]:
        page_health()
    with tabs[1]:
        page_predict()
    with tabs[2]:
        page_match_single()
    with tabs[3]:
        page_match_batch()
    with tabs[4]:
        page_job_search()
    with tabs[8]:
        page_decision()
    with tabs[9]:
        page_config_view()
    with tabs[10]:
        page_ner_eval()
    with tabs[6]:
        page_funnel()
    with tabs[7]:
        page_fairness()
    with tabs[5]:
        page_ingest()


 
def page_funnel():
    st.subheader("三级漏斗筛选")
    base = st.session_state.api_base
    job_desc = st.text_area("岗位描述", height=160)
    source = st.radio("候选来源", ["使用处理后的JSON", "向量库已有"], index=0)
    top_k = st.number_input("Top K", min_value=1, value=50, step=1, key="funnel_top_k")
    rules_text = st.text_area("自定义规则(JSON)", value='[{"field":"years","operator":"gt","value":2}]', height=100, key="funnel_rules")
    if st.button("执行漏斗"):
        try:
            rules = json.loads(rules_text) if rules_text.strip() else []
        except Exception:
            st.error("规则JSON不合法")
            return
        if not job_desc.strip():
            st.warning("请填写岗位描述")
            return
        if source == "使用处理后的JSON":
            resume_json = os.path.join("data", "processed", "resumes_for_annotation.json")
            if not os.path.isfile(resume_json):
                st.error("简历JSON不存在")
                return
            with open(resume_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = []
            for i, d in enumerate(data):
                items.append({"id": str(i), "text": d.get("text", "")})
            _post_with_progress(base, "/vector_index", {"items": items}, 8.0, "向量入库")
        res, err = _post_with_progress(base, "/filter", {"job_desc": job_desc, "top_k": int(top_k), "custom_rules": rules}, 12.0, "执行漏斗")
        if err:
            st.error(f"调用失败：{err}")
            return
        results = res.get("results", [])
        st.success(f"完成筛选，返回 {len(results)} 条")
        if results:
            rows = [{"id": r.get("id"), "score": r.get("score"), "base": r.get("base"), "skill": r.get("skill"), "implicit": r.get("implicit"), "format": r.get("format") } for r in results]
            st.dataframe(rows, use_container_width=True)
            if st_echarts:
                xs = [r.get("id") for r in results[:10]]
                ys = [r.get("score") for r in results[:10]]
                opts = {
                    "tooltip": {"trigger": "axis"},
                    "xAxis": {"type": "category", "data": xs},
                    "yAxis": {"type": "value", "name": "score"},
                    "series": [{"type": "bar", "data": ys, "itemStyle": {"color": "#4caf50"}}]
                }
                st_echarts(opts, height=320)
            ids_all = [str(r.get("id")) for r in results]
            default_sel = ids_all[:5]
            sel = st.multiselect("选择对比候选人ID", ids_all[:20], default=default_sel)
            if st_echarts:
                inds = ["base", "skill", "implicit", "format"]
                chosen = [r for r in results if str(r.get("id")) in sel][:20]
                maxs = [1.0, 1.0, 1.0, 1.0]
                if chosen:
                    maxs = [max(0.1, max(float(x.get("base",0)) for x in chosen)), max(0.1, max(float(x.get("skill",0)) for x in chosen)), max(0.1, max(float(x.get("implicit",0)) for x in chosen)), max(0.1, max(float(x.get("format",0)) for x in chosen))]
                ind_cfg = [{"name": n, "max": float(m)} for n, m in zip(inds, maxs)]
                radar_data = []
                for r in chosen:
                    radar_data.append({"value": [float(r.get("base",0)), float(r.get("skill",0)), float(r.get("implicit",0)), float(r.get("format",0))], "name": str(r.get("id"))})
                r_opts = {"legend": {"data": [str(r.get("id")) for r in chosen]}, "radar": {"indicator": ind_cfg}, "series": [{"type": "radar", "data": radar_data}]}
                st_echarts(r_opts, height=360)
            import pandas as pd
            df = pd.DataFrame(rows)
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("下载CSV", data=csv, file_name="funnel_results.csv")

def page_fairness():
    st.subheader("公平性报告")
    dev_thr = st.slider("偏差阈值(%)", min_value=1, max_value=10, value=2, step=1)
    resume_json = os.path.join("data", "processed", "resumes_for_annotation.json")
    if not os.path.isfile(resume_json):
        st.info("未找到预处理输出")
        return
    with open(resume_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    gender_cnt = {}
    tier_cnt = {}
    gap_cnt = {}
    for d in data:
        tags = d.get("fairness_tags", {})
        g = tags.get("gender", "未知")
        t = tags.get("school_tier", "未知")
        gp = tags.get("gap_type", "未知")
        gender_cnt[g] = gender_cnt.get(g, 0) + 1
        tier_cnt[t] = tier_cnt.get(t, 0) + 1
        gap_cnt[gp] = gap_cnt.get(gp, 0) + 1
    # 计算百分比
    def _to_percent(d):
        s = max(1, sum(d.values()))
        return {k: round(v * 100.0 / s, 2) for k, v in d.items()}
    g_pct = _to_percent(gender_cnt)
    t_pct = _to_percent(tier_cnt)
    gp_pct = _to_percent(gap_cnt)
    if st_echarts:
        opts_g = {
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": list(g_pct.keys())},
            "yAxis": {"type": "value", "name": "通过率(%)"},
            "series": [{"type": "bar", "data": list(g_pct.values()), "itemStyle": {"color": "#4caf50"}}],
            "markLine": {"data": [{"yAxis": dev_thr}]}
        }
        st_echarts(opts_g, height=300)
        opts_t = {
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": list(t_pct.keys())},
            "yAxis": {"type": "value", "name": "通过率(%)"},
            "series": [{"type": "bar", "data": list(t_pct.values()), "itemStyle": {"color": "#ff9800"}}],
            "markLine": {"data": [{"yAxis": dev_thr}]}
        }
        st_echarts(opts_t, height=300)
        opts_gp = {
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": list(gp_pct.keys())},
            "yAxis": {"type": "value", "name": "通过率(%)"},
            "series": [{"type": "bar", "data": list(gp_pct.values()), "itemStyle": {"color": "#1976d2"}}],
            "markLine": {"data": [{"yAxis": dev_thr}]}
        }
        st_echarts(opts_gp, height=300)
    else:
        st.json({"gender": g_pct, "school_tier": t_pct, "gap_type": gp_pct})
    # 导出CSV
    import pandas as pd
    df = pd.DataFrame({
        "gender": g_pct,
        "school_tier": t_pct,
        "gap_type": gp_pct,
    })
    csv = df.to_csv().encode("utf-8-sig")
    st.download_button("下载公平性百分比CSV", data=csv, file_name="fairness_report.csv")
def page_decision():
    st.subheader("决策辅助")
    base = st.session_state.api_base
    job_desc = st.text_area("岗位描述（可一行一个支持批量）", height=160)
    rules_text = st.text_area("自定义规则(JSON)", value='[{"field":"years","operator":"gt","value":2}]', height=100, key="decision_rules")
    top_k = st.number_input("Top K", min_value=1, value=50, step=1, key="decision_top_k")
    page = st.number_input("页码", min_value=1, value=1, step=1)
    page_size = st.number_input("每页数量", min_value=5, value=20, step=5)
    llm_flag = st.checkbox("启用大模型面试题", value=bool(st.session_state.get("llm_global_enabled", False)), key="decision_llm")
    csv_flag = st.checkbox("导出为CSV", value=bool(st.session_state.get("export_csv_default", False)), key="decision_export_csv")
    if st.button("应用面试题生成设置", key="decision_llm_apply"):
        _ = api_post(base, "/config/llm_enabled", {"enabled": bool(llm_flag)})
        st.success("已应用设置")
    run = st.button("生成推荐与分数线")
    if run:
        try:
            rules = json.loads(rules_text) if rules_text.strip() else []
        except Exception:
            st.error("规则JSON不合法")
            return
        if not job_desc.strip():
            st.warning("请填写岗位描述")
            return
        lines = [l.strip() for l in job_desc.split("\n") if l.strip()]
        if len(lines) > 1:
            payload = {"job_descs": lines, "top_k": int(top_k), "custom_rules": rules}
            res, err = _post_with_progress(base, "/decision_batch", payload, 20.0, "生成推荐与分数线")
            if err:
                st.error(f"调用失败：{err}")
                return
            items = res.get("items", [])
            for it in items:
                st.markdown("---")
                st.write("岗位：", it.get("job_desc"))
                decision = it.get("decision", {})
                threshold = decision.get("threshold")
                picks = decision.get("recommended", [])
                st.metric("动态分数线", f"{threshold}")
                st.dataframe([{"id": p.get("id"), "score": p.get("score")} for p in picks], use_container_width=True)
            overview_rows = []
            for it in items:
                decision = it.get("decision", {})
                threshold = decision.get("threshold")
                picks = decision.get("recommended", [])
                overview_rows.append({"job_desc": it.get("job_desc"), "threshold": threshold, "recommended_count": len(picks)})
            _download_excel_or_csv_sheets({"overview": overview_rows}, "decision_batch_results.xlsx", "decision_batch", prefer_excel=(not csv_flag))
            return
        payload = {"job_desc": job_desc, "top_k": int(top_k), "custom_rules": rules, "page": int(page), "page_size": int(page_size)}
        res, err = _post_with_progress(base, "/decision", payload, 20.0, "生成推荐与分数线")
        if err:
            st.error(f"调用失败：{err}")
            return
        decision = res.get("decision", {})
        threshold = decision.get("threshold")
        picks = decision.get("recommended", [])
        st.metric("动态分数线", f"{threshold}")
        _summary_cards([("推荐数量", str(len(picks))), ("分数线", f"{threshold}"), ("Top1分数", f"{picks[0].get('score') if picks else 0}")])
        st.write("推荐名单：")
        st.dataframe([{"id": p.get("id"), "score": p.get("score")} for p in picks], use_container_width=True)
        st.write("分页结果：")
        page_rows = res.get("page_results", [])
        st.dataframe(page_rows, use_container_width=True)
        _download_excel_or_csv_sheets({
            "results": res.get("results", []),
            "recommended": [{"id": p.get("id"), "score": p.get("score"), "questions": " | ".join(p.get("interview_questions", []))} for p in picks]
        }, "decision_results.xlsx", "decision", prefer_excel=(not csv_flag))
        if picks and st_echarts:
            xs = [p.get("id") for p in picks]
            ys = [p.get("score") for p in picks]
            opts = {
                "tooltip": {"trigger": "axis"},
                "xAxis": {"type": "category", "data": xs},
                "yAxis": {"type": "value"},
                "series": [{"type": "line", "data": ys, "smooth": True, "itemStyle": {"color": "#1976d2"}}],
                "markLine": {"data": [{"yAxis": threshold}]}
            }
            st_echarts(opts, height=320)
        jrows = []
        for p in picks:
            jrows.append({"id": p.get("id"), "questions": " | ".join(p.get("interview_questions", []))})
        st.dataframe(jrows, use_container_width=True)
        import pandas as pd
        df = pd.DataFrame(res.get("results", []))
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载全量结果CSV", data=csv, file_name="decision_all_results.csv")

if __name__ == "__main__":
    main()