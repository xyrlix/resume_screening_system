# 智能简历筛选系统

## 项目说明

本系统利用Transformer模型（BERT）实现简历信息的自动提取和人岗匹配，旨在提高招聘效率。

## 环境配置

1.  安装Python 3.9
2.  创建虚拟环境（可选）
3.  安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

## 镜像源配置

为了提高依赖包和模型的下载速度，特别是在中国大陆地区，建议配置镜像源。

### Pip 镜像源配置

#### 国内镜像源（推荐在中国大陆使用）
```bash
# 临时使用
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 包名

# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 其他国内镜像源选择：
# 阿里云：https://mirrors.aliyun.com/pypi/simple/
# 中科大：https://pypi.mirrors.ustc.edu.cn/simple/
# 华为云：https://mirrors.huaweicloud.com/repository/pypi/simple/
# 腾讯云：https://mirrors.cloud.tencent.com/pypi/simple/
```

#### 国外镜像源（推荐在海外使用）
```bash
# 默认源（国外服务器）
pip install 包名

# 其他国外镜像源选择：
# PyPI官方：https://pypi.org/simple/
```

### HuggingFace 镜像源配置

#### 国内镜像源（推荐在中国大陆使用）
```bash
# 设置环境变量（Windows PowerShell）
$env:HF_ENDPOINT="https://hf-mirror.com"

# 设置环境变量（Linux/macOS）
export HF_ENDPOINT=https://hf-mirror.com

# 其他国内镜像源选择：
# https://huggingface.co.cn/（由阿里云提供）
```

#### 国外镜像源（推荐在海外使用）
```bash
# 默认源（国外服务器）
# 不需要特殊设置，直接使用

# 其他国外镜像源选择：
# https://huggingface.co/
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

- 面向招聘方：进入"招聘流程向导"，按步骤完成 JD 输入/采集 → 简历来源解析 → 匹配与评分 → 面试题生成；需要批量筛选时使用"三级漏斗/决策辅助"页面，并支持结果导出。
- 面向求职者：在"简历上传/简历优化/岗位推荐/匹配度分析/面试准备"页面完成从简历优化到岗位推荐与评估报告生成，支持题库下载和练习。
- 管理员：在"流程监控"查看数据预处理、特征提取、训练/测试/评估的曲线与统计；在"配置查看"管理行业模板、匹配权重与 LLM 开关。

### 操作路径 + 小贴士
- 招聘方：
  - 若已有人才库，先"向量入库"再用"三级漏斗"快速筛选；融合权重 α 可在岗位推荐页调节（匹配分 vs 语义相似度）。
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
  # 输出：原始标签序列与"提取的实体"列表
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
  - 招聘流程导向：
    - 步骤1：上传JD→自然语言输入（含示例）→结构化JD分组展示（硬性/软性/实体特征）
    - 步骤2：简历来源指定目录与在线采集合并岗位池
    - 步骤3：匹配与评分（Top10）
    - 步骤4：综合匹配输出（雷达/饼图/细项分），并生成面试题与优势/风险分析；支持导出HTML/PDF/Excel
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
### 大模型与配置

- Provider 支持：`local`、`qwen`（阿里云百炼/通义千问）、`ark`（火山方舟）、`volc`（字节火山）、`hunyuan`（腾讯混元）、`qianfan`（百度千帆/文心）、`kimi`（Moonshot/Kimi）、`openai`、`gemini`（Google）、`tavily`（检索生成问题）。
- 开关与生效范围：
  - 前端侧栏"启用大模型(全局)"控制是否调用外部 Provider；不启用时回退到本地生成（`local`）。
  - 后端路由：`POST /interview_questions`、`POST /evaluation_report` 使用所选 Provider。
- 动态配置与持久化：
  - `POST /config/llm_settings` 保存 Provider 的 `api_url`、`api_key`、`model` 到 `config/llm.json`，按 Provider 小节分别持久化。
  - `POST /config/llm_enabled` 切换是否启用 LLM 功能（主要影响面试题生成）。
  - 允许匿名配置开关：设置 `ALLOW_PUBLIC_CONFIG=1` 时上述两个路由不需鉴权；生产建议关闭。
- `config/llm.json` 示例：
  ```json
  {
    "openai": { "api_url": "https://api.openai.com/v1/chat/completions", "api_key": "xxx", "model": "gpt-4o-mini" },
    "ark":    { "api_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions", "api_key": "xxx", "model": "Doubao-pro" },
    "volc":   { "api_url": "https://api.volcengine.com/compatible-mode/v1/chat/completions", "api_key": "xxx", "model": "xxx" },
    "hunyuan":{ "api_url": "https://api.hunyuan.tencent.com/v1/chat/completions", "api_key": "xxx", "model": "xxx" },
    "qianfan":{ "api_url": "https://qianfan.baidu.com/v1/chat/completions", "api_key": "xxx", "model": "ERNIE-Speed-128k" },
    "kimi":   { "api_url": "https://api.moonshot.cn/v1/chat/completions", "api_key": "xxx", "model": "kimi-2.0" },
    "gemini": { "api_key": "xxx", "model": "gemini-1.5-flash" },
    "tavily": { "api_key": "xxx" },
    "qwen":   { "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "api_key": "xxx", "model": "qwen2.5-instruct" }
  }
  ```
  - 未在 `llm.json` 提供的值可通过环境变量临时覆盖（如 `OPENAI_API_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL`）。

### 权限与令牌

- 角色与权限：后端按角色控制路由访问（`admin/hr/interviewer/candidate`）。可通过环境变量 `RBAC_TOKEN_MAP` 或 `AUTH_TOKEN_*` 提供静态令牌与角色映射。
- 令牌签发：`POST /auth/token` 支持本地用户签发 JWT（需设置 `USER_MAP` 或 `USER_DB_PATH`），前端或管理页可将令牌置于 `Authorization: Bearer ...`。
- 开放开关：
  - `ALLOW_PUBLIC_MATCH/FILTER/DECISION/RECOMMEND/CONFIG` 分别控制匹配、漏斗、决策、岗位推荐、配置的匿名访问，默认关闭。
  - 管理页（`/ui`）在存在静态目录时启用，支持行业模板、LLM 开关、入库预热与上传目录入库等操作。

## 增强功能

系统在原有功能基础上，新增了以下增强功能以满足更复杂的需求：

### 多格式文件处理增强
- 支持可编辑PDF/Word、扫描件、图片简历、Excel表格简历等多种格式
- 智能识别文件类型并选择合适的解析方法
- 布局感知解析，保留原文档结构信息

### 特征工程增强
- **文本特征**：使用Sentence-BERT生成768维向量用于向量粗筛
- **实体特征**：通过轻量级LLM提取详细实体信息，包括：
  - 基本信息：姓名、性别、出生日期、年龄、联系电话、电子邮箱等
  - 教育背景：学校名称、学历层次、专业名称、入学时间、毕业时间等
  - 工作经历：公司名称、在职时间、职位名称、工作地点、工作内容等
  - 项目经验：项目名称、项目时间、项目角色、技术栈、项目成果等
  - 技能信息：编程语言、框架/工具、数据库、操作系统、语言能力、证书资质等
- **格式特征**：提取排版工整度、成果量化程度等多模态特征
- **可迁移能力特征**：基于岗位-能力映射词典，提取跨界候选人相关技能特征

### 三级漏斗筛选优化
1. **一级向量粗筛**：
   - 技术选型：ChromaDB向量数据库 + Sentence-BERT模型
   - 逻辑：将简历与JD转换为768维向量，计算余弦相似度，筛选Top50相关简历
   - 阈值：相似度≥0.5进入下一级，淘汰语义相关性低的简历

2. **二级规则精筛**：
   - 技术选型：JSON规则配置 + jsonpath-ng解析
   - 支持算子：in/not_in/gt/lt/eq/contains，覆盖学历、工作年限、核心技能等硬性条件
   - 逻辑：按企业自定义规则过滤，淘汰80%不符简历，仅保留10份左右进入最终筛选

3. **三级LLM补筛**：
   - 技术选型：轻量级LLM模型
   - 核心功能：隐性需求挖掘、空窗期归因、可迁移能力评分
   - 逻辑：计算软性匹配分，综合判断候选人适配度

### 可视化展示增强
- **雷达图**：根据实体特征划分多维度匹配情况
- **饼图**：展示模型预测结果匹配情况分布
- **综合分析**：对匹配度、优势和风险进行全面分析
- **面试题生成**：基于简历和岗位描述自动生成10道面试题

### 角色功能完善
- **管理员**：界面可视化模型训练、生成训练模型、模型评估
  - 数据预处理：从指定目录中读取出训练的简历数据和JD数据，进行数据降噪预处理，缺失值处理，异常值处理，数据不一致处理，最后界面点击预处理输出预处理后的结果。
  - 特征处理：从预处理数据中，提取简历和JD的全部多维实体并界面展示，特征选择准备，转换方法科学有效增强模型的预测能力。
  - 模型训练：模型训练生成模型数据集，提供给预测使用，要求模型阈值准确高度，可靠性强，错误率低。详细界面输出模型训练的过程数据展示。
  - 模型评估：针对模型训练结果，量化评估输出准确度，错误率等评估，是否会过拟合和欠拟合等。
- **招聘方**：多模态解析、文本清洗、特征工程、匹配筛选、可视化展示
  - 多模态解析：可编辑PDF/Word、扫描件/图片简历、表格简历
  - 文本清洗：布局感知排序、文本清洗
  - 特征工程：实体特征、文本特征、格式特征、可迁移能力特征
  - 匹配筛选：三级漏斗筛选（向量粗筛→规则精筛→LLM补筛）
  - 可视化展示：雷达图、饼图展示、面试题生成、综合分析
- **求职者**：简历制作、简历解析优化、简历画像、岗位筛选、模拟面试
  - 简历制作：在线制作完整的简历
  - 简历解析：llm大模型优化简历
  - 简历画像：生成简历画像
  - 岗位筛选：根据简历自动预测匹配出合适的岗位
  - 模拟面试：根据面试岗位，自动生成10道面试题