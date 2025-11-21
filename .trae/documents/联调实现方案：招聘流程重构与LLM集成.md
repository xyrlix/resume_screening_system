## 改动范围
- 前端：`scripts/06_visualization.py`
- 后端：`scripts/05_api.py`
- 新增：`modules/llm_providers.py`
- 复用：`modules/llm_utils.py`（结构化JD字段已扩展）

## 前端实现（招聘流程导向）
### 排版与步骤强化
- 调整 `page_hr_wizard`：
  - 每步标题采用加粗+色条（HTML片段），分隔线与留白优化。
### 步骤1：岗位需求
- 将“上传JD文件（txt/md/docx/pdf）”置于自然语言输入之前。
- 自然语言输入下方提供“示例JD”一键填充按钮。
- “生成结构化JD”后分组展示：
  - 硬性：`degree_required/min_years/position/salary_min/salary_max/employment_type/location/industry`
  - 软性：`skills/frameworks/tools/languages/soft_skills`
  - 实体特征：`keywords/certifications`
- 提供导出按钮（Excel/CSV）。
### 步骤2：简历来源/在线采集
- 简历来源Tab：支持指定目录路径输入（默认 `data/raw_resumes`），解析常见格式（txt/md/docx/pdf）。
- 在线采集Tab：保留平台选择与Cookie，采集后合并岗位池。
### 步骤3：匹配与评分（Top10）
- 执行匹配后仅保留前10个结果，写入 `st.session_state["hr_match_results"]`。
- 提供列表展示与导出（Excel/CSV）。
### 步骤4：综合匹配输出（整合面试题与分析）
- 顶部：Top10综合雷达图对比（8维）。
- 选择器选择单个候选人后展示卡片：
  - 雷达图+能力占比饼图
  - 细项评分列表（各维分数与命中细节）
  - 面试题（3条）与优势/风险分析（大模型生成）
- 移除原步骤5，将生成题目与分析合并到步骤4。

## 后端实现（LLM Provider适配）
### 适配层
- 新增 `modules/llm_providers.py`：
  - `get_provider()`：读取 `LLM_PROVIDER` 环境变量（`qwen`/`local`）。
  - `generate_questions(job_desc, resume_text)`：通义千问或本地模型生成面试题。
  - `generate_analysis(job_desc, resume_text)`：生成优势/风险分析。
- 环境变量：
  - `LLM_PROVIDER`（默认 `local`）
  - `QWEN_API_KEY`、`QWEN_MODEL`（如 `qwen2.5` 家族）
### 接口联动
- 更新 `/interview_questions` 与 `/evaluation_report` 在 `scripts/05_api.py` 内调用适配层。
- 失败时回退到简易启发式生成，避免阻塞。

## 验证与导出
- 使用默认 `data/raw_resumes` 与 `data/raw_jobs` 完成端到端验证：
  - 结构化JD分组展示与导出
  - 目录解析与在线采集合并
  - 匹配仅Top10
  - 综合匹配雷达+饼图与单人卡片详情、题目与分析生成
- 导出：
  - Top10结果、单人卡片明细、题目与分析（TXT/HTML/Excel）

## 风险与回退
- 云模型不可用时自动回退到本地轻量模型；无密钥不阻塞流程。
- 图表缺失时回退到内置柱状图。