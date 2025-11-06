import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# 定义实体标签
ENTITY_TYPES = ["姓名", "学历", "专业", "工作年限", "技能", "项目经验", "公司名称", "职位", "毕业院校", "薪资期望"]
LABEL_TO_ID = {"O": 0}
for entity in ENTITY_TYPES:
    LABEL_TO_ID[f"B-{entity}"] = len(LABEL_TO_ID)
    LABEL_TO_ID[f"I-{entity}"] = len(LABEL_TO_ID)
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

MODEL_PATH = 'models/bert_entity'

# 加载模型和分词器
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
model = BertForTokenClassification.from_pretrained(MODEL_PATH)

def predict(text):
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