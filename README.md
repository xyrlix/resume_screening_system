# 基于Transformer的智能简历筛选系统

## 项目说明

本系统利用Transformer模型（BERT）实现简历信息的自动提取和人岗匹配，旨在提高招聘效率。

## 环境配置

1.  安装Python 3.9
2.  创建虚拟环境（可选）
3.  安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

## 使用步骤

1.  将原始简历（PDF/Word）放入 `data/raw_resumes` 目录。
2.  将岗位描述（TXT）放入 `data/raw_jobs` 目录。
3.  运行数据预处理脚本：
    ```bash
    python scripts/02_data_preprocess.py
    ```
4.  训练实体提取模型：
    ```bash
    python scripts/03_entity_model.py
    ```
5.  训练人岗匹配模型：
    ```bash
    python scripts/04_matcher_model.py
    ```
6.  启动系统：
    ```bash
    python app.py
    ```

## 快速开始

- 一键运行：`python app.py` 同时启动后端与前端
- 打开前端：访问 `http://localhost:8501`
- 健康检查：`http://127.0.0.1:8000/health`（端口自动在 8000–8005 间回退）

## 按角色使用指南

- 面向招聘方：进入“招聘流程向导”，按步骤完成 JD 输入/采集 → 简历来源解析 → 匹配与评分 → 面试题生成；需要批量筛选时使用“三级漏斗/决策辅助”页面，并支持结果导出。
- 面向求职者：在“简历上传/简历优化/岗位推荐/匹配度分析/面试准备”页面完成从简历优化到岗位推荐与评估报告生成，支持题库下载和练习。
- 管理员：在“流程监控”查看数据预处理、特征提取、训练/测试/评估的曲线与统计；在“配置查看”管理行业模板、匹配权重与 LLM 开关。

### 操作路径 + 小贴士
- 招聘方：
  - 若已有人才库，先“向量入库”再用“三级漏斗”快速筛选；融合权重 α 可在岗位推荐页调节（匹配分 vs 语义相似度）。
  - 在线采集合并岗位池时，按平台填入 URL 与 Cookie；采集完成后在向导中选择岗位进行匹配。
- 求职者：
  - 简历优化后进行岗位推荐并查看前三岗位的匹配度分析；生成评估报告（TXT/HTML）用于自我改进。
  - 面试准备可生成题库并记录练习结果。
- 管理员：
  - 训练/评估日志输出为 JSONL（每行一个样本/步骤），前端自动绘制曲线；缺失时显示友好提示。
  - 关闭向量索引匿名遥测以减少退出噪声（已在代码中关闭）。

## 详细使用指南

### 概述
- 系统能力：从简历文本抽取实体（学历、年限、技能、职位等），并与岗位进行规则化匹配评分；同时提供 API 与可视化界面。
- 主要组件：
  - 数据处理与标注：`scripts/02_data_preprocess.py`、`data/processed/resumes_for_annotation.json`
  - 实体识别模型：`scripts/03_entity_model.py`、`scripts/04_predict.py`
  - 匹配模型：`scripts/04_matcher_model.py`
  - 服务接口：`scripts/05_api.py`
  - 可视化界面：`scripts/06_visualization.py`、启动器 `app.py`

### 环境准备
```bash
python -m venv .venv
./.venv/Scripts/activate
pip install -r requirements.txt
```

### 数据与标注
- 原始简历：`data/raw_resumes/`（Word/PDF），运行预处理后生成待标注文本：
  ```bash
  python scripts/02_data_preprocess.py
  # 输出：data/processed/resumes_for_annotation.json
  ```
- 标注训练集：`data/processed/entity_train.json`（人工在 `resumes_for_annotation.json` 基础上补齐 `entities` 字段）
  - 推荐结构（示例）：
    ```json
    {
      "id": "resume_001",
      "text": "张三，2020年毕业于北京大学计算机专业，硕士学历，熟练掌握Python、TensorFlow，3年算法工程师经验...",
      "entities": [
        {"type": "SCHOOL", "text": "北京大学"},
        {"type": "MAJOR", "text": "计算机"},
        {"type": "DEGREE", "text": "硕士"},
        {"type": "SKILL", "text": "Python"},
        {"type": "SKILL", "text": "TensorFlow"},
        {"type": "YEARS", "text": "3年"},
        {"type": "POSITION", "text": "算法工程师"}
      ]
    }
    ```
  - 注意：`entities.text` 必须与 `text` 中的字符片段逐字匹配，避免偏移错误。

### 训练与预测
- 训练实体识别（需要已标注数据）：
  ```bash
  python scripts/03_entity_model.py
  # 输出模型：models/bert_entity/
  ```
- 单次预测验证：
  ```bash
  python scripts/04_predict.py
  # 输出：原始标签序列与“提取的实体”列表
  ```
- 说明：训练时已启用 fast tokenizer 的 `offset_mapping` 与 `-100` 忽略掩码，保证标签对齐与损失计算合理。

### 岗位匹配
- 快速测试：
  ```bash
  python scripts/04_matcher_model.py \
    --resume_text "张三，硕士毕业于北京大学，拥有5年软件开发经验，熟练掌握Python和Java。曾在ABC公司担任高级软件工程师。" \
    --job_text "招聘高级软件工程师，要求本科及以上学历，5年以上开发经验，熟悉Python、Java，熟悉Linux与Git。"
  ```
  - 输出包含：`score`（0–1）与 `details`（技能命中、学历、年限、职位评分）
- 批量匹配（岗位文件目录）：
  ```bash
  # 在 data/raw_jobs/ 放置 .txt 或 .md 文件（已提供 sample_backend_engineer.txt）
  python scripts/04_matcher_model.py \
    --resume_json data/processed/resumes_for_annotation.json \
    --jobs_dir data/raw_jobs \
    --out logs/match_results.json
  ```
  - 配置化：支持在 `config/matching.json` 中调整匹配权重与技能词典；也可通过环境变量 `MATCHING_CONFIG_PATH` 指定自定义路径。示例：
    ```json
    {
      "weights": {"skills": 0.5, "degree": 0.2, "years": 0.2, "position": 0.1},
      "skills": ["python", "java", "docker", "kubernetes", "aws", "gcp", "kafka", "rabbitmq"]
    }
    ```

### API 服务
- 启动：
  ```bash
  python -m uvicorn scripts.05_api:app --host 127.0.0.1 --port 8000
  ```
- 路由：
  - `GET /`：欢迎信息与端点列表
  - `GET /health`：健康检查
  - `GET /docs`：Swagger UI
  - `POST /predict`：入参 `{ "text": "…简历文本…" }`
  - `POST /match`：入参 `{ "resume_text": "…", "job_text": "…" }`
  - `POST /jd_generate`：入参 `{ "text": "自然语言JD描述" }` 返回结构化JD
  - `POST /resume_optimize`：入参 `{ "text": "简历文本" }` 返回画像与优化建议
  - `POST /interview_questions`：入参 `{ "job_desc": "可选", "resume_text": "简历文本" }` 返回题库
  - `POST /recommend_jobs`：入参 `{ "resume_text": "简历文本", "jobs_dir": "岗位目录", "top_k": 5 }` 返回 TopN 岗位
  - 主机与端口：可用环境变量 `API_HOST`、`API_PORT` 控制监听；当未设置端口或端口占用时，服务会在 `8000–8005` 范围内自动选择可用端口，并在启动日志中提示。

### 可视化界面
- 一键启动（API + Streamlit）：
  ```bash
  python app.py
  ```
- 仅可视化：
  ```bash
  streamlit run scripts/06_visualization.py
  ```

### 项目运行逻辑

- 启动入口
  - `python app.py`：同时启动后端 API 与前端 Streamlit。
  - 仅后端：`python -m uvicorn scripts.05_api:app --host 127.0.0.1 --port 8000`。
  - 仅前端：`streamlit run scripts/06_visualization.py`。

- 后端流程（`scripts/05_api.py`）
  - 动态加载预测与匹配模块：`04_predict.py` 的 `predict` 与 `04_matcher_model.py` 的 `quick_match`。
  - 端口策略：支持环境变量 `API_HOST`、`API_PORT`，若端口占用或未指定，会在 `8000–8005` 间自动选择可用端口。
  - 路由：
    - `GET /health` 健康检查；`GET /docs` 文档。
    - `POST /predict`：输入 `{text}`，返回实体列表。
    - `POST /match`：输入 `{resume_text, job_text}`，返回综合得分与细分项。

- 前端流程（`scripts/06_visualization.py`）
  - 自动发现 API 地址：优先 `API_BASE_URL`，否则轮询 `127.0.0.1:8000–8005/health`。
  - 多页功能：概览、JD智能生成、简历来源与采集、单次/批量匹配、岗位检索与岗位推荐、三级漏斗、决策辅助、配置查看、面试训练、评价记录、NER评估、流程监控。
  - 实体预测页调用 `/predict`；匹配页调用 `/match` 或在前端本地使用临时权重直接匹配（不依赖后端权重）。

- 数据流与算法
  - NER 预测（`scripts/04_predict.py`）：
    - 加载 `models/bert_entity` 的 tokenizer 与模型（BERT Token Classification）。
    - 使用 `offset_mapping` 将标签与字符对齐，输出实体类型与文本片段。
  - 匹配（`scripts/04_matcher_model.py`）：
    - 加载 `config/matching.json`（或 `MATCHING_CONFIG_PATH` 指定文件）以配置权重与技能字典。
    - 解析简历/岗位技能、学历、年限、职位，计算技能比例、学历满足、年限比例、职位关键词匹配，按权重加权得分。

- 配置与端口
  - 匹配策略：修改 `config/matching.json`（权重 `skills/degree/years/position` 与 `skills` 字典），重启后端生效。
  - 端口与地址：后端自动回退端口；前端自动发现 API 地址，也可在侧栏手动设置。

### 文件解析能力

- PDF：优先布局感知解析（`pdfplumber`），回退 `PyPDF2`
- Word：`python-docx`
- 图片OCR：`paddleocr`（中英文）
- Excel/表格：`pandas` 逐行拼接

- 流程图
  ```mermaid
  flowchart LR
    A[app.py (entrypoint)] -->|start uvicorn| B[FastAPI 后端\nscripts/05_api.py]
    A -->|start streamlit| C[Streamlit 前端\nscripts/06_visualization.py]

    subgraph Backend
      B --> D[加载 04_predict.py::predict]
      B --> E[加载 04_matcher_model.py::quick_match]
      B --> F[/health, /docs, /predict, /match]
    end

    subgraph Frontend
      C --> G[自动发现 API (环境变量或 8000–8005)]
      C --> H[页面：概览、实体预测、单次匹配、批量匹配、配置查看、NER评估]
      H --> I[/predict]
      H --> J[/match]
      H --> K[本地临时权重匹配]
    end

    E --- L[config/matching.json\n权重、技能词典]
  ```

### 效果评估
- NER（实体识别）：
  - 观察实体是否合理、覆盖关键点（学历/技能/年限/职位/学校/专业）
  - 指标建议：实体级 Precision/Recall/F1（非逐 token）
  - 若预测全为 `O`：优先检查标注数据的数量与一致性
- 匹配评分：
  - `details.matched_skills` 越多越好；`degree_score` 满足最低学历为 1；`years_score` 满足年限为 1；`position_score` 关键词匹配为 1
  - 可在 `scripts/04_matcher_model.py` 调整 `WEIGHTS` 或扩充 `SKILL_DICT`

### 部署
- 直接运行：
  ```bash
  python -m uvicorn scripts.05_api:app --host 0.0.0.0 --port 8000
  ```
- Windows 服务化：使用计划任务或 NSSM 注册上述命令为服务，设置自动重启与日志输出。
- 反向代理：使用 Nginx/IIS 代理到 `127.0.0.1:8000` 并配置 HTTPS 与访问控制。
- Docker（示例思路）：
  - 基础镜像 `python:3.10-slim`，复制项目并 `pip install -r requirements.txt`
  - 启动命令 `uvicorn scripts.05_api:app --host 0.0.0.0 --port 8000`
  - 映射 `models/bert_entity` 与 `data/processed` 作为卷以持久化模型与数据

### 常见问题
- 访问 `/` 返回 404：已添加根路由；若仍异常，重启服务并访问 `/docs` 验证。
- 训练告警（AdamW/权重初始化）：属信息提示；可切换为 `torch.optim.AdamW`。
- 中文分词对齐：已使用 `BertTokenizerFast` 的 `offset_mapping` 保证字符级裁切。

### 附录：实体类型建议
- 建议统一为：`SCHOOL`、`MAJOR`、`DEGREE`、`YEARS`、`SKILL`、`POSITION`，并在标注时保持表述一致，避免多写/少写导致偏移。