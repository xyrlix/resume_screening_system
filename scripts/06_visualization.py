import os
import json
import time
import math
import streamlit as st
import urllib.request
import urllib.error
import importlib.util
from typing import Tuple, List, Dict


def page_setup():
    st.set_page_config(page_title="简历筛选系统", page_icon="📄", layout="wide")
    st.title("📄 智能简历筛选可视化")
    st.caption("实体识别与人岗匹配（API + 可视化）")


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


def api_post(base: str, path: str, payload: dict, timeout: float = 2.5):
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except urllib.error.HTTPError as e:
        return None, f"HTTPError {e.code}: {e.reason}"
    except Exception as e:
        return None, str(e)


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
                res, err = api_post(base, "/match", {"resume_text": resume_text, "job_text": job_text})
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
            "resume_path": r.get("path")
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
    # 展示第1名的详情图表
    st.markdown("---")
    st.subheader("冠军简历细分项")
    _render_match_charts(top[0].get("details", {}))

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

    # 右侧：显示选中简历原文与下载
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
            rows.append({"idx": i, "tp": tp, "fp": fp, "fn": fn, "gt": len(gt), "pred": len(pred)})
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        st.metric("Precision", f"{precision:.4f}")
        st.metric("Recall", f"{recall:.4f}")
        st.metric("F1", f"{f1:.4f}")
        st.dataframe(rows, use_container_width=True)
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(["TP", "FP", "FN"], [TP, FP, FN], color=["#4caf50", "#f44336", "#ff9800"])
            st.pyplot(fig, use_container_width=False)
        except Exception:
            pass


def main():
    page_setup()
    sidebar()

    tabs = st.tabs(["概览", "实体预测", "单次匹配", "批量匹配", "岗位检索", "配置查看", "NER评估"])
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
    with tabs[5]:
        page_config_view()
    with tabs[6]:
        page_ner_eval()


if __name__ == "__main__":
    main()