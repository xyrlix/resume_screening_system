import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
import numpy as np
import os
import sys
import warnings
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
warnings.filterwarnings(
    "ignore",
    message=r"This implementation of AdamW is deprecated",
    category=FutureWarning,
)
import time

# 定义实体标签
ENTITY_TYPES = ["姓名", "学历", "专业", "工作年限", "技能", "项目经验", "公司名称", "职位", "毕业院校", "薪资期望"]
LABEL_TO_ID = {"O": 0}
for entity in ENTITY_TYPES:
    LABEL_TO_ID[f"B-{entity}"] = len(LABEL_TO_ID)
    LABEL_TO_ID[f"I-{entity}"] = len(LABEL_TO_ID)
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label_to_id):
        self.data = data
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        entities = item["entities"]

        # 使用 fast tokenizer 的 offset_mapping 来对齐字符到token
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
            padding='max_length'
        )

        input_ids = torch.tensor(enc['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(enc['attention_mask'], dtype=torch.long)
        offsets = enc['offset_mapping']

        # 初始化labels为-100（忽略），对真实文本token置为O
        labels = np.full(len(input_ids), fill_value=-100, dtype=int)
        for i, (s, e) in enumerate(offsets):
            if not (s == 0 and e == 0):
                labels[i] = self.label_to_id["O"]

        # 为每个实体打 B-/I- 标签
        for ent in entities:
            start_char = ent['start']
            end_char = ent['end']
            ent_type = ent['type']

            began = False
            for i, (s, e) in enumerate(offsets):
                if s == 0 and e == 0:
                    continue  # 跳过特殊token
                if s >= start_char and s < end_char:
                    if not began:
                        labels[i] = self.label_to_id.get(f"B-{ent_type}", self.label_to_id["O"])
                        began = True
                    else:
                        labels[i] = self.label_to_id.get(f"I-{ent_type}", self.label_to_id["O"])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    # 逐步记录批次损失（可选）
    step = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        step += 1
        try:
            os.makedirs('logs', exist_ok=True)
            with open('logs/training_metrics.jsonl', 'a', encoding='utf-8') as lf:
                lf.write(json.dumps({
                    'ts': int(time.time()*1000),
                    'step': step,
                    'loss': float(loss.item()),
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }, ensure_ascii=False) + '\n')
        except Exception:
            pass
    
    print(f"训练损失: {total_loss / len(dataloader)}")

if __name__ == "__main__":
    MODEL_NAME = 'bert-base-chinese'
    DATA_PATH = 'data/processed/entity_train.json'
    MODEL_SAVE_PATH = 'models/bert_entity'

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    train_data = load_data(DATA_PATH)
    train_dataset = NERDataset(train_data, tokenizer, LABEL_TO_ID)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_TO_ID)).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    print("开始训练...")
    # 实际项目中，您需要增加更多的训练轮次
    for epoch in range(3):
        print(f"Epoch {epoch + 1}/3")
        train(model, train_dataloader, optimizer, device)
        # 每轮结束追加一条聚合指标（当前使用损失替代）
        try:
            os.makedirs('logs', exist_ok=True)
            with open('logs/training_metrics.jsonl', 'a', encoding='utf-8') as lf:
                lf.write(json.dumps({
                    'ts': int(time.time()*1000),
                    'step': f'epoch_{epoch+1}',
                    'loss': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }, ensure_ascii=False) + '\n')
        except Exception:
            pass

    print("训练完成。")

    # 保存模型
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"模型已保存到: {MODEL_SAVE_PATH}")