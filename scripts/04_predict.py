import os
import re
import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# 定义实体标签（v1 默认，v2 可通过配置与扩展）
def _load_entity_types():
    try:
        cfg = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "entities.json")
        if os.path.isfile(cfg):
            import json
            with open(cfg, "r", encoding="utf-8") as f:
                obj = json.load(f)
            arr = obj.get("entity_types") or obj.get("types")
            if isinstance(arr, list) and len(arr) > 0:
                return [str(x) for x in arr if str(x).strip()]
    except Exception:
        pass
    return ["姓名", "学历", "专业", "工作年限", "技能", "项目经验", "公司名称", "职位", "毕业院校", "薪资期望"]
ENTITY_TYPES = _load_entity_types()
LABEL_TO_ID = {"O": 0}
for entity in ENTITY_TYPES:
    LABEL_TO_ID[f"B-{entity}"] = len(LABEL_TO_ID)
    LABEL_TO_ID[f"I-{entity}"] = len(LABEL_TO_ID)
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

MODEL_PATH = os.getenv('NER_MODEL_PATH', 'models/bert_entity')
NER_PIPELINE = os.getenv('NER_PIPELINE', 'v1').lower()

# 加载模型和分词器
tokenizer = None
model = None
try:
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    model = BertForTokenClassification.from_pretrained(MODEL_PATH)
except Exception:
    tokenizer = None
    model = None

def _predict_v1(text: str):
    # 使用分词器处理文本，并获取 offset_mapping 以实现精确对齐
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        padding='max_length'
    )
    input_ids = torch.tensor([enc['input_ids']], dtype=torch.long)
    attention_mask = torch.tensor([enc['attention_mask']], dtype=torch.long)
    offsets = enc['offset_mapping']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    pred_ids = torch.argmax(outputs.logits, dim=2)[0].tolist()
    predicted_labels = [ID_TO_LABEL.get(i, 'O') for i in pred_ids]

    # 打印原始预测
    print("原始预测:", predicted_labels)

    # 提取实体（忽略特殊token，其offset为(0,0)）
    entities = []
    current = None
    for i, label in enumerate(predicted_labels):
        s, e = offsets[i]
        if s == 0 and e == 0:
            # 碰到特殊token时，收尾当前实体
            if current:
                entities.append(current)
                current = None
            continue
        if label.startswith('B-'):
            if current:
                entities.append(current)
            current = {"type": label[2:], "start_char": s, "end_char": e}
        elif label.startswith('I-') and current and current["type"] == label[2:]:
            current["end_char"] = e
        else:
            if current:
                entities.append(current)
            current = None
    if current:
        entities.append(current)

    # 转为最终输出
    final_entities = []
    for ent in entities:
        final_entities.append({
            "type": ent["type"],
            "text": text[ent["start_char"]:ent["end_char"]]
        })
    return final_entities

def _extract_v2_rules(text: str):
    t = text
    out = []
    # 能力熟练度
    prof = re.findall(r"(熟练|精通|掌握)\s*([A-Za-z0-9#\-\+]{2,})", t)
    for p in prof[:10]:
        out.append({"type": "SKILL_PROFICIENCY", "text": "".join(p)})
    # 结果指标（百分比/数值提升）
    metrics = re.findall(r"(提升|降低|增长|减少)\s*(\d+[\.]?\d*\s*%?)", t)
    for m in metrics[:10]:
        out.append({"type": "RESULT_METRIC", "text": "".join(m)})
    # 证书细类
    certs = re.findall(r"(AWS\s*(SAA|SA|DevOps)|CKA|CKAD|RHCE|PMP)", t, flags=re.I)
    for c in certs[:10]:
        out.append({"type": "CERT_TYPE", "text": "".join(c)})
    # 语言级别
    lang = re.findall(r"(CET\-?6|CET\-?4|IELTS\s*\d+(?:\.\d+)?|TOEFL\s*\d+|英文可工作|英文流利)", t, flags=re.I)
    for l in lang[:10]:
        out.append({"type": "LANG_LEVEL", "text": l})
    # 领域/方法（基于关键词）
    domains = ["金融", "电商", "医疗", "制造", "物联网"]
    methods = ["transformer", "transformers", "rag", "ranking", "bert", "gpt"]
    for d in domains:
        if d in t.lower() or d in t:
            out.append({"type": "DOMAIN", "text": d})
    for m in methods:
        if m in t.lower():
            out.append({"type": "METHOD", "text": m})
    return out

def predict(text: str):
    if NER_PIPELINE == 'v2':
        # 若无模型则回退规则；若有模型可并行组合
        ents = []
        if tokenizer and model:
            try:
                ents = _predict_v1(text)
            except Exception:
                ents = []
        # 追加 v2 扩展实体（规则回退）
        ents += _extract_v2_rules(text)
        return ents
    # v1 默认
    if tokenizer and model:
        return _predict_v1(text)
    return []

if __name__ == "__main__":
    # 示例文本
    example_text = "张三，硕士毕业于北京大学，拥有5年软件开发经验，熟练掌握Python和Java。曾在ABC公司担任高级软件工程师。"

    # 进行预测
    extracted_entities = predict(example_text)

    # 打印结果
    print(f"示例文本: {example_text}")
    print("提取的实体:")
    for entity in extracted_entities:
        print(f"  - 类型: {entity['type']}, 文本: {entity['text']}")