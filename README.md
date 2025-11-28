# 智能简历筛选系统

基于 BGE-M3 和多 LLM 链式分析的智能简历筛选系统，支持招聘方和求职者两个核心角色，具备完整的简历筛选功能。

## 🌟 功能特点

### 1. 多格式文件支持

- **可编辑 PDF/Word**：使用 PyPDF2、python-docx 提取文本
- **扫描件/图片简历**：使用 PaddleOCR 提取文本，添加图像增强
- **Excel 表格简历**：使用 camelot-py 提取表格数据
- **Markdown 文档**：支持直接读取和处理 Markdown 格式的简历和职位描述

### 2. 行业和岗位支持

- **5 个热门行业**：人工智能、新能源、半导体/芯片、互联网、电子商务
- **5 个热门岗位**：算法工程师、电池研发工程师、芯片设计工程师、产品经理、跨境电商

### 3. LLM 模型配置

- **前端配置界面**：支持配置 API Key
- **模型选择**：支持多种 LLM 模型（gpt-3.5-turbo、gpt-4、qwen-1.8b、deepseek-llm-7b-chat、moonshot-v1-8k）
- **链式组合**：支持 LLM 模型链式调用

### 4. 文件上传功能

- **简历上传**：支持单个和批量上传，支持文本输入
- **JD 上传**：支持单个上传，支持文本输入

### 5. 结构化解析

- **实体提取**：使用 LLM 提取全部实体信息
- **文本向量化**：使用 BGE-M3 生成 768 维向量
- **存储**：使用 ChromaDB 存储向量数据

### 6. 招聘方功能

- **多模态解析**：支持多种格式简历解析
- **文本清洗**：布局感知排序，文本清洗
- **特征工程**：多维特征提取，实体特征提取
- **匹配筛选**：三级漏斗筛选（向量粗筛 → 规则精筛 →LLM 补筛）
- **可视化展示**：雷达图、饼图、面试题生成、综合分析

### 7. 求职者功能

- **简历制作**：在线简历制作
- **简历解析**：LLM 优化简历
- **简历画像**：生成简历画像
- **岗位筛选**：自动匹配合适岗位
- **模拟面试**：生成面试题

## 📦 安装步骤

### 1. 克隆项目

```bash
git clone <项目地址>
cd resume_screening_system
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

```bash
python app.py
```

### 4. 访问系统

在浏览器中访问：http://localhost:8501

## 🚀 使用方法

### 1. 招聘方功能

#### 1.1 上传 JD

- 点击"招聘方"选项卡
- 在"上传 JD"子选项卡中输入或上传 JD
- 点击"上传 JD"按钮

#### 1.2 上传简历

- 在"上传简历"子选项卡中输入或上传简历
- 支持单个和批量上传
- 支持多种文件格式

#### 1.3 简历匹配

- 在"简历匹配"子选项卡中选择要匹配的 JD
- 设置匹配结果数量
- 点击"开始匹配"按钮
- 查看匹配结果

#### 1.4 自定义筛选

- 在"自定义筛选"子选项卡中设置筛选规则
- 支持学历、技能等多种筛选条件
- 点击"应用筛选规则"按钮

#### 1.5 LLM 链式分析

- 在"LLM 链式分析"子选项卡中选择 JD 和简历
- 点击"开始 LLM 链式分析"按钮
- 查看分析结果

#### 1.6 匹配结果

- 在"匹配结果"子选项卡中查看历史匹配结果

### 2. 求职者功能

#### 2.1 上传简历

- 点击"求职者"选项卡
- 在"上传简历"子选项卡中输入或上传简历
- 点击"上传简历"按钮

#### 2.2 职位匹配

- 在"职位匹配"子选项卡中选择要匹配的简历
- 设置匹配结果数量
- 点击"开始职位匹配"按钮
- 查看匹配结果

#### 2.3 简历优化建议

- 在"简历优化建议"子选项卡中选择要优化的简历
- 输入目标职位描述
- 点击"生成优化建议"按钮
- 查看优化建议

#### 2.4 匹配结果

- 在"匹配结果"子选项卡中查看历史匹配结果

## 📁 项目结构

```
resume_screening_system/
├── core/                  # 核心功能模块
│   ├── __init__.py
│   ├── data_processor.py  # 数据收集与处理
│   ├── feature_engine.py  # 特征工程
│   ├── vectorizer.py      # BGE-M3向量化
│   ├── matcher.py         # 简历匹配
│   ├── llm_chain.py       # 多LLM链式分析
│   ├── evaluator.py       # 模型评估
│   ├── file_processor.py  # 多格式文件处理
│   ├── industry_job_manager.py  # 行业和岗位管理
│   └── llm_config_manager.py    # LLM模型配置管理
├── services/              # 业务服务层
│   ├── recruiter_service.py    # 招聘方服务
│   └── candidate_service.py    # 求职者服务
├── frontend/              # 前端界面
│   ├── combined_app.py    # 组合界面
│   ├── recruiter_app.py   # 招聘方界面
│   └── candidate_app.py   # 求职者界面
├── data/                  # 数据目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后的数据
│   └── data/              # 数据和模型文件
├── app.py                 # 主应用入口
├── requirements.txt       # 项目依赖
└── README.md              # 项目说明文档
```

## 🛠️ 技术栈

- **前端框架**：Streamlit
- **后端框架**：Python
- **向量化模型**：BGE-M3
- **LLM 模型**：支持多种 LLM 模型
- **文件处理**：PyPDF2、python-docx、PaddleOCR、camelot-py
- **数据存储**：ChromaDB
- **可视化**：Matplotlib、Plotly

## 🎯 核心模块说明

### 1. 数据处理器 (core/data_processor.py)

- 实现简历和 JD 的数据收集
- 实现数据的初步处理和清洗
- 实现数据的数学处理和分析

### 2. 特征引擎 (core/feature_engine.py)

- 使用 BGE-M3 模型进行向量化
- 实现特征选择和数据正规化
- 支持多语言处理

### 3. 匹配器 (core/matcher.py)

- 实现基于 BGE-M3 的语义匹配
- 实现多 LLM 链式分析
- 支持多种匹配算法

### 4. LLM 链式分析 (core/llm_chain.py)

- 实现多 LLM 链式分析
- 支持多种 LLM 模型
- 实现 LLM 评估融合

### 5. 文件处理器 (core/file_processor.py)

- 支持多种格式的文件处理
- 实现图像增强和 OCR 提取
- 支持批量处理

### 6. 行业和岗位管理 (core/industry_job_manager.py)

- 管理行业和岗位信息
- 实现行业-岗位映射关系
- 支持热门行业和岗位

### 7. LLM 模型配置管理 (core/llm_config_manager.py)

- 管理 LLM 模型配置
- 支持配置 API Key
- 支持链式组合使用

## 🔧 配置说明

### 1. LLM 模型配置

- 在前端界面的"LLM 模型配置"子选项卡中配置
- 支持配置 API Key 和基础 URL
- 支持选择默认模型
- 支持配置链式组合

### 2. 启动参数

```bash
python app.py --port 8501  # 指定端口号
python app.py --info        # 显示系统信息
```

## 📊 性能指标

- **模型加载时间**：< 30 秒
- **简历匹配速度**：< 0.1 秒/份
- **OCR 识别准确率**：≥ 92%
- **支持的文件格式**：PDF、Word、图片、Excel、Markdown

## 🤝 贡献指南

1. Fork 项目
2. 创建分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

如有问题或建议，请联系项目团队。

---

**智能简历筛选系统** - 基于 BGE-M3 和多 LLM 链式分析的智能简历筛选系统
