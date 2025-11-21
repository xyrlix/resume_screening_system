## 目标
- 在不引入重模型的前提下，扩展特征维度与解释能力，提升匹配与漏斗的可用性与可解释性。
- 主要通过“字典+规则”的方式增强画像与分项，前端统一展示，支持轻量级预标注辅助。

## 改动范围与策略
- 配置驱动：以 `config/matching.json` 为中心扩展字典与权重，避免硬编码。
- 规则增强：在 `scripts/04_matcher_model.py` 的画像提取层与 `modules/rule_engine.py` 规则层补充特征与操作。
- 可解释输出：统一在 `/funnel_explain` 返回新增分项与命中明细，前端步骤2/步骤3同结构展示。

## 实施步骤
### 1. 配置与字典扩展
- 扩展 `config/matching.json`（新增字段）：
  - `domains`: 行业/领域关键词（如 金融/电商/医疗/制造/物联网）
  - `methods`: 技术/方法关键词（如 transformer/bert/gpt/rag/ranking）
  - `certs`: 证书子类（如 AWS SAA/CKA/RHCE/PMP）
  - `languages`: 语言级别（CET-4/6, IELTS, TOEFL, 英文可工作等）
  - `proficiency_keywords`: 熟练度词（熟练/精通/掌握）
  - `metric_keywords`: 结果指标词（提升/降低/增长/减少，含数字/百分比）
  - `weights`: 若需要，将 `proficiency/method/domain/metric` 作为次级权重（默认参与推荐排序但不改变主干权重总和）

### 2. 画像与规则增强
- `scripts/04_matcher_model.py`：
  - 增强 `extract_*`：从文本中抽取 `domains/methods/cert_types/lang_levels/proficiency/metrics`，并加入 `profile_from_text`/`job_profile_from_text` 的画像结构。
  - 保持中文/英文变体兼容（已有英文增强继续沿用）。
- `modules/rule_engine.py`：
  - 在不破坏现有 `in/not_in/gt/lt/eq/contains` 的基础上，支持 `contains_any`（列表中任一命中）与数值型规则的健壮性（对百分比/数字的轻量解析）。

### 3. 漏斗解释统一输出
- `/funnel_explain`（`modules/funnel_filter.py`）：
  - 扩展返回字段：`matched_domains/matched_methods/matched_certs/matched_lang_levels` 与 `proficiency_score/metric_score`（示例评分规则可从字典命中与数字密度近似）。
  - 解释文本：为每候选生成简短解释串（如“方法命中：transformer/rag；领域：金融；证书：CKA；语言：CET-6；熟练度：精通Python”）。
  - 保持综合分计算不变（base/skill/implicit/format），新增分项用于排序调节或纯展示（可通过 `weights` 开关决定是否纳入综合分）。

### 4. 前端统一展示
- 步骤2（岗位池/简历来源）：
  - “入库并解释（一步）”返回表中新增列：`matched_domains/matched_methods/matched_certs/matched_lang_levels/proficiency_score/metric_score`，并保留过滤原因表。
- 步骤3（匹配与评分）：
  - 统一量化表同样加入上述列，合并匹配细项（degree/years/position 与技能命中明细），去除冗余趋势图。
  - 支持导出（Excel/CSV）包含新增列。

### 5. 轻量预标注辅助（可选）
- 新增工具脚本（可选，若你同意我再落地）：`scripts/tools/preannotate.py`
  - 输入 `data/processed/resumes_for_annotation.json`，按照扩展字典与规则生成候选实体（片段级 start/end/type），输出 `data/processed/entity_train_pre.json`。
  - 作为人工标注的初稿，减少纯手工工作量。

### 6. 文档与配置说明
- README 增补“扩展特征字典与解释维度”的章节，给出示例配置与前端展示截图说明（文本为主，不新增图片）。

## 验收与影响
- 对现有功能兼容：不改变已有主分（base/skill/implicit/format）与流程接口；新增字段均为向后兼容的扩展。
- 解释增强：步骤2/3展示一致，字段更丰富；支持导出包含新增列。
- 高可控：全部由配置驱动，便于在数据量有限时快速迭代与灰度开启（调整 `weights` 开关）。

## 风险控制与回退
- 若新增字典较多影响命中噪声：权重默认不纳入综合分，仅展示；在验证通过后再开启权重参与排序。
- 若规则误召回：通过规则引擎与过滤原因表快速定位并调整字典。

---
确认后，我将：
1) 扩展 `config/matching.json` 的字典与示例权重；
2) 增强画像提取与规则引擎；
3) 扩展 `/funnel_explain` 返回字段与解释串；
4) 统一前端步骤2/3表格，新增列并支持导出；
5)（可选）添加预标注辅助脚本并示例运行。