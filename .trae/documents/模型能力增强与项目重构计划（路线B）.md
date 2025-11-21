## 目标
- 面向复杂简历（中英混排、PDF版式、含量化指标与细粒度证书/领域/方法），提升实体抽取与画像质量，增强漏斗可解释性与匹配效果。
- 在不破坏现有功能的前提下，以可切换的“v2管线”新增更强模型与特征维度，逐步替换或融合现有“v1管线”。

## 现状与痛点
- 实体类型偏少：当前训练脚本固定 10 类（scripts/03_entity_model.py:22），难覆盖“能力层级、项目结果、证书细粒度、语言级别、行业/方法”。
- 模型能力有限：中文 BERT Token Classification 能力足够入门，但对 PDF 布局/英文/跨段关系抽取不足。
- 解释维度有限：漏斗分项 base/skill/implicit/format 可解释，但缺“技能熟练度、项目量化指标、领域/方法命中”等更细粒度项。

## 技术路线（v2管线）
### 1. 数据与预处理升级
- 布局感知解析：在 PDF 场景接入 LayoutLMv3 推理（仅需安装可选依赖），与现有 parse_tools 回退策略兼容。
- 语种检测与分流：中文→RoBERTa-wwm-ext；英文/混合→XLM-R（或多语模型）。
- 标注增强：扩充 entity_train.json 的实体 Schema（片段级 start/end/type），新增示例与半自动预标注流程。

### 2. 实体 Schema 扩充
- 新增类型（建议）：
  - SKILL_PROFICIENCY（技能熟练度：熟练/精通/掌握 + 量化）、YEARS_PER_SKILL（某技能年限）
  - PROJECT、RESULT_METRIC（项目名称/模块、效果指标如性能提升%、成本降低、用户增长等）
  - CERT_TYPE（证书细类，如 AWS SAA/CKA 等）、LANG_LEVEL（IELTS 分数/CET-6 等）、DOMAIN（金融/电商/医疗…）、METHOD（Transformer/RAG/Ranking…）
- 训练标签定义：BIO/BIES 方案保持与 v1 一致，新增类型直接纳入训练。

### 3. 模型升级与任务拆分
- NER 模型
  - 中文：RoBERTa-wwm-ext Token Classification；可选 CRF 层与 label-smoothing。
  - 英文/多语：XLM-R Token Classification。
- 关系抽取（有限目标）
  - 轻量 RE：基于规则或小型分类器关联 SKILL↔PROFICIENCY/YEARS，PROJECT↔RESULT_METRIC。
  - 产出结构化画像（技能→熟练度/年限；项目→结果指标）。

### 4. 画像融合与漏斗分项扩展
- v2 画像生成
  - 在 scripts/04_matcher_model.py 新增 v2 画像合并逻辑：融合 NER 片段与字典命中（保持回退策略）。
- 分项与权重
  - 综合分仍为 base/skill/implicit/format 主干；新增“proficiency/method/domain/metric”的次级权重（默认参与推荐排序但不改变主干权重总和）。
- 解释接口
  - /funnel_explain 增加返回字段：matched_methods/matched_domains/proficiency_score/metric_score 等，让步骤2/步骤3表格直接展示来源与数值。

### 5. 训练与评估体系
- 评估指标
  - 片段级 Precision/Recall/F1（分类型统计），混合语种拆分报告。
  - 关系抽取准确率（skill↔proficiency、project↔metric）。
- 验证集与对照实验
  - v1 与 v2 实体对比、漏斗 TopN 重合率、推荐点击/接受率（若有交互数据）。
- 性能与资源
  - 提供 4-bit/8-bit 量化可选（bitsandbytes），限制显存与加载时间；Windows 环境下兼容 CPU 跑推理（降速但可运行）。

## 代码改进建议
### 配置与可切换
- 新增环境变量：`NER_PIPELINE=v1|v2`、`NER_MODEL_ZH=roberta-wwm-ext`、`NER_MODEL_MUL=xlm-roberta-base`、`ENABLE_LAYOUTLM=0/1`。
- 通过 scripts/05_api.py 统一入口，根据开关选择 v1 或 v2 管线，保证兼容与回退。

### 训练脚本（scripts/03_entity_model.py）
- 抽象 ENTITY_TYPES 为可配置（从 config/entities.json 读取），避免硬编码（现：scripts/03_entity_model.py:22）。
- 支持多模型训练与保存（models/ner_v2_zh、models/ner_v2_mul），引入 CRF/label-smoothing 可选项。

### 推理脚本（scripts/04_predict.py）
- 增加 v2 路径：加载模型与 tokenizer，统一输出片段级实体；同时输出关系对（skill↔proficiency/years、project↔metric）。
- 与后端 `/predict` 保持兼容：若 v2 启用，通过 query 参数或环境变量切换返回中带上 v2 扩展字段（兼容旧前端）。

### 匹配器与漏斗（scripts/04_matcher_model.py、modules/funnel_filter.py、modules/scoring.py）
- 画像融合：优先用 v2 实体；失败时回退 v1 规则与字典。
- 分项扩展：在 modules/scoring.py 增加次级分项（proficiency/method/domain/metric），权重默认较小，可在 config/matching.json 配置调整。
- 解释输出：modules/funnel_filter.py 的 funnel_explain 增加新分项与命中明细，前端步骤2/3统一展示。

### 前端（scripts/06_visualization.py）
- 设置页新增 NER 管线开关与模型选择；步骤3表格新增“熟练度/方法/领域/指标”列，并可以展开查看命中详情（例如 matched_methods/matched_domains/matched_metrics）。
- 仍保持“一键入库并解释”“执行匹配并展示过程”的统一进度与展示逻辑。

## 数据与标注计划
- 半自动预标注
  - 规则与模板：对“熟练度词（熟练/精通）”“结果数字（提升、降低、增长等）”“证书规范名（AWS/GCP/CKA）”“领域词（金融/电商等）”做模式匹配，生成候选标注。
  - LLM 辅助：对难例使用 LLM 生成候选实体，人工确认后入库。
- 标注规范
  - 片段严格对齐；英文与中文统一采用同一 Schema；关系通过锚点（start/end）关联。

## 风险与回退
- 依赖体积与加载时间：RoBERTa/XLM-R/LayoutLM 较重，需提供 CPU 回退与量化选项；默认关闭 LayoutLM。
- 兼容性：前端与后端保持 v1/v2 切换；若 v2 抽取失败，自动回退 v1。
- 标注成本：扩 Schema 需要标注投入；通过半自动预标注与 LLM 辅助降低人工负担。

## 交付与里程碑
- 里程碑1：配置可切换的 v2 管线（模型加载、推理、接口兼容），完成无标注条件下的规则增强与解释扩展。
- 里程碑2：扩充 entity_train.json、训练 v2 NER（中文/英文），上线片段级与关系抽取。
- 里程碑3：评估与调参，完善前端展示与导出，文档更新（README 与管理页说明）。

---
如你确认该计划，我将：
1) 增加 v2 管线的配置与后端入口（不影响现有功能）；
2) 扩展实体 Schema 与训练脚本，提供初版 v2 NER；
3) 扩展漏斗解释分项并更新前端统一展示；
4) 附评估脚本与示例，对照 v1/v2 效果。