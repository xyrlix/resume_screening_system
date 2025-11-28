import os
import sys
import json
import random
import re
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from torch.optim import AdamW
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.data_processor import DataProcessor
from core.ner_model import BertCRFTagger

class NERDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer: BertTokenizerFast, label2id: Dict[str, int]):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tags = self.labels[idx]
        enc = self.tokenizer(text, return_offsets_mapping=True, return_tensors='pt', truncation=True, max_length=512)
        offsets = enc['offset_mapping'][0].tolist()
        token_tags = []
        for o in offsets:
            token_tags.append('O')
        spans = self._spans_from_tags(text, tags)
        for label, start, end in spans:
            for i, (s, e) in enumerate(offsets):
                if s >= start and e <= end and e > s:
                    token_tags[i] = ('B-' + label) if s == start else ('I-' + label)
        label_ids = torch.tensor([self.label2id.get(t, self.label2id['O']) for t in token_tags], dtype=torch.long)
        return enc['input_ids'][0], enc['attention_mask'][0], label_ids

    def _spans_from_tags(self, text: str, tags: List[Dict[str, Any]]):
        spans = []
        for item in tags:
            label = item['label']
            value = item['value']
            for m in re.finditer(re.escape(value), text):
                spans.append((label, m.start(), m.end()))
        return spans

def collate(batch):
    input_ids = [b[0] for b in batch]
    attn = [b[1] for b in batch]
    labels = [b[2] for b in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return input_ids, attn, labels

def build_silver_annotations(texts: List[str]) -> List[List[Dict[str, Any]]]:
    dp = DataProcessor()
    label_map = {
        '职位名称': 'JobTitle',
        '期望职位': 'JobTitle',
        '公司名称': 'Company',
        '学历层次': 'Degree',
        '学历要求': 'Degree',
        '总工作经验年限': 'Years',
        '工作年限要求': 'Years',
        '技能要求': 'Skill',
        '工作地点': 'Location',
        '现居地': 'Location',
        '薪资范围': 'Salary',
        '期望薪资': 'Salary',
        '联系电话': 'Phone',
        '电子邮箱': 'Email',
        '语言要求': 'Language',
        '语言能力': 'Language',
        '证书要求': 'Certificate',
        '证书资质': 'Certificate'
    }
    results = []
    entity_union = list(set(dp.resume_entities + dp.jd_entities))
    for t in texts:
        entities = dp.extract_entities(t, entity_union, use_llm=True)
        ann = []
        for k, v in entities.items():
            if not v:
                continue
            lab = label_map.get(k)
            if not lab:
                continue
            if k in ['技能要求'] and isinstance(v, str):
                parts = [x.strip() for x in re.split(r'[;,，、\s]+', v) if x.strip()]
                for p in parts:
                    ann.append({'label': lab, 'value': p})
            else:
                for m in re.finditer(re.escape(str(v)), t):
                    ann.append({'label': lab, 'value': t[m.start():m.end()]})
        results.append(ann)
    return results

def main():
    model_dir = os.path.join('data', 'models', 'ner_bert_crf')
    os.makedirs(model_dir, exist_ok=True)
    resume_processed = os.path.join('data', 'processed', 'raw_resumes_processed.json')
    resumes = []
    if os.path.isfile(resume_processed):
        with open(resume_processed, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                t = item.get('text') or item.get('masked_text') or ''
                if t:
                    resumes.append(t)
    jobs_dir = os.path.join('data', 'raw_jobs')
    jobs = []
    if os.path.isdir(jobs_dir):
        for fn in os.listdir(jobs_dir):
            p = os.path.join(jobs_dir, fn)
            if os.path.isfile(p):
                with open(p, 'r', encoding='utf-8') as f:
                    jobs.append(f.read())
    texts = resumes + jobs
    random.seed(42)
    random.shuffle(texts)
    annotations = build_silver_annotations(texts)
    keep_n = min(400, len(texts))
    with open(os.path.join('data', 'processed', 'ner_annotations.json'), 'w', encoding='utf-8') as f:
        json.dump({'texts': texts[:keep_n], 'annotations': annotations[:keep_n]}, f, ensure_ascii=False, indent=2)
    labels = ['O', 'B-JobTitle', 'I-JobTitle', 'B-Company', 'I-Company', 'B-Degree', 'I-Degree', 'B-Years', 'I-Years', 'B-Skill', 'I-Skill', 'B-Location', 'I-Location', 'B-Salary', 'I-Salary', 'B-Phone', 'I-Phone', 'B-Email', 'I-Email', 'B-Language', 'I-Language', 'B-Certificate', 'I-Certificate']
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {str(i): l for i, l in enumerate(labels)}
    with open(os.path.join(model_dir, 'label2id.json'), 'w', encoding='utf-8') as f:
        json.dump(label2id, f, ensure_ascii=False)
    with open(os.path.join(model_dir, 'id2label.json'), 'w', encoding='utf-8') as f:
        json.dump(id2label, f, ensure_ascii=False)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
    dataset = NERDataset(texts, annotations, tokenizer, label2id)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate)
    model = BertCRFTagger(num_labels=len(labels), pretrained_model_name='bert-base-multilingual-cased')
    optim = AdamW(model.parameters(), lr=3e-5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(4):
        model.train()
        for input_ids, attn, y in loader:
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            y = y.to(device)
            loss = model(input_ids, attn, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
    torch.save(model.state_dict(), os.path.join(model_dir, 'pytorch_model.bin'))
    with open(os.path.join(model_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({'pretrained': 'bert-base-multilingual-cased'}, f, ensure_ascii=False)

if __name__ == '__main__':
    main()
