#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练命名实体识别模型脚本
"""

# 必须在import transformers之前设置环境变量！
import os

# 使用HF镜像源解决下载问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # 允许联网下载
os.environ["HF_HUB_OFFLINE"] = "0"  # 允许HF Hub联网
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '30'  # 超时时间30秒
os.environ['HF_HUB_RETRY'] = '5'  # 重试5次
os.environ['HF_HUB_RETRY_DELAY'] = '2'  # 重试间隔2秒

# 导入必要的库
import sys
import json
import random
import re
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from torch.optim import AdamW

# 添加项目根目录到 Python 路径，以便导入自定义模块
try:
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录（假设脚本在scripts目录下）
    project_root = os.path.dirname(script_dir)
    sys.path.append(project_root)
except NameError:
    # 如果__file__不可用，使用当前工作目录
    project_root = os.getcwd()
    sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, 'core'))
    sys.path.append(os.path.join(project_root, 'utils'))
    print(f"使用当前工作目录作为项目根: {project_root}")

# 导入自定义模块，但避免提前加载ner_model
from core.data_processor import DataProcessor


class NERDataset(Dataset):
    """
    命名实体识别(NER)数据集类，用于处理文本和标签数据
    继承自 PyTorch 的 Dataset 类
    """

    def __init__(self, texts: List[str], labels: List[List[Dict[str, Any]]],
                 tokenizer: BertTokenizerFast, label2id: Dict[str, int]):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表，每个元素是一个字典列表
            tokenizer: BERT 分词器
            label2id: 标签到ID的映射字典
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        """返回数据集大小"""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        获取指定索引的数据项
        
        Args:
            idx: 数据项索引
            
        Returns:
            input_ids: 输入ID张量
            attention_mask: 注意力掩码张量
            label_ids: 标签ID张量
        """
        text = self.texts[idx]
        tags = self.labels[idx]
        # 使用分词器处理文本，返回偏移映射用于标记定位
        enc = self.tokenizer(text,
                             return_offsets_mapping=True,
                             return_tensors='pt',
                             truncation=True,
                             max_length=512)
        offsets = enc['offset_mapping'][0].tolist()

        # 初始化所有标记为'O'（无实体）
        token_tags = []
        for o in offsets:
            token_tags.append('O')

        # 从标签中提取实体跨度
        spans = self._spans_from_tags(text, tags)

        # 根据实体跨度为标记分配标签
        for label, start, end in spans:
            for i, (s, e) in enumerate(offsets):
                if s >= start and e <= end and e > s:
                    # 实体开始标记为'B-xxx'，内部标记为'I-xxx'
                    token_tags[i] = ('B-' + label) if s == start else ('I-' +
                                                                       label)

        # 将标签转换为ID
        label_ids = torch.tensor(
            [self.label2id.get(t, self.label2id['O']) for t in token_tags],
            dtype=torch.long)
        return enc['input_ids'][0], enc['attention_mask'][0], label_ids

    def _spans_from_tags(self, text: str, tags: List[Dict[str, Any]]):
        """
        从标签中提取实体跨度信息
        
        Args:
            text: 原始文本
            tags: 标签列表
            
        Returns:
            spans: 实体跨度列表，每个元素为(label, start, end)元组
        """
        spans = []
        for item in tags:
            label = item['label']
            value = item['value']
            # 在文本中查找所有匹配的实体值
            for m in re.finditer(re.escape(value), text):
                spans.append((label, m.start(), m.end()))
        return spans


def collate(batch):
    """
    数据批处理函数，用于将多个数据项组合成一个批次
    主要功能是对不同长度的序列进行填充
    
    Args:
        batch: 数据项列表
        
    Returns:
        input_ids: 填充后的输入ID张量
        attn: 填充后的注意力掩码张量
        labels: 填充后的标签张量
    """
    input_ids = [b[0] for b in batch]
    attn = [b[1] for b in batch]
    labels = [b[2] for b in batch]

    # 对序列进行填充，使其长度一致
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids,
                                                batch_first=True,
                                                padding_value=0)
    attn = torch.nn.utils.rnn.pad_sequence(attn,
                                           batch_first=True,
                                           padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=0)

    return input_ids, attn, labels


def build_silver_annotations(texts: List[str]) -> List[List[Dict[str, Any]]]:
    """
    构建银色标注（使用LLM自动生成的标注）
    
    Args:
        texts: 文本列表
        
    Returns:
        results: 标注列表，每个文本对应一个标注列表
    """
    # 初始化数据处理器
    dp = DataProcessor()

    # 从配置文件加载完整的标签映射
    config_path = os.path.join('config', 'entity_config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        label_map = config.get('label_map', {})
        print(f"使用完整标签映射，共 {len(label_map)} 个映射关系")
    except Exception as e:
        print(f"加载标签映射失败: {e}，将使用默认标签映射")
        # 使用默认标签映射
        label_map = {
            '职位名称': 'JobTitle',
            '期望职位': 'JobTitle',
            '公司名称': 'CompanyName',
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
            '证书资质': 'Certificate',
            '编程语言': 'ProgrammingLanguage'
        }
        print(f"使用默认标签映射，共 {len(label_map)} 个映射关系")

    results = []
    # 获取简历实体和职位描述实体的并集
    entity_union = list(set(dp.resume_entities + dp.jd_entities))

    # 为每个文本生成标注
    for t in texts:
        # 使用数据处理器提取实体，不使用LLM
        entities = dp.extract_entities(t, entity_union, use_llm=False)
        ann = []

        # 处理提取的实体
        for k, v in entities.items():
            if not v:
                continue

            # 获取英文标签
            lab = label_map.get(k)
            if not lab:
                continue

            # 确保v是字符串类型
            entity_values = []
            if isinstance(v, list):
                entity_values = [str(item) for item in v if item]
            else:
                entity_values = [str(v)]

            for entity_value in entity_values:
                if not entity_value or len(entity_value) < 2:  # 过滤掉过短的实体值
                    continue

                # 清理实体值，移除特殊字符和多余空格
                cleaned_value = entity_value.strip()
                if not cleaned_value:
                    continue

                # 特殊处理技能要求，将其拆分为多个技能
                if k in ['技能要求']:
                    parts = [
                        x.strip()
                        for x in re.split(r'[;,，、\s]+', cleaned_value)
                        if x.strip() and len(x.strip()) > 1
                    ]
                    for p in parts:
                        # 确保技能在文本中存在
                        if p in t:
                            ann.append({'label': lab, 'value': p})
                else:
                    # 在文本中查找实体值的位置，确保匹配准确
                    if cleaned_value in t:
                        # 只添加完全匹配的实体
                        ann.append({'label': lab, 'value': cleaned_value})

        # 去重，避免重复标注
        unique_ann = []
        seen = set()
        for item in ann:
            key = f"{item['label']}_{item['value']}"
            if key not in seen:
                seen.add(key)
                unique_ann.append(item)

        results.append(unique_ann)

    return results


def main():
    """
    主函数，用于训练命名实体识别模型
    """
    print("开始执行NER模型训练脚本...")
    # 创建模型保存目录
    model_dir = os.path.join('data', 'models', 'ner_bert_crf')
    os.makedirs(model_dir, exist_ok=True)
    print(f"模型保存目录: {model_dir}")

    # 加载处理后的简历数据
    resume_processed = os.path.join('data', 'processed',
                                    'raw_resumes_processed.json')
    resumes = []
    if os.path.isfile(resume_processed):
        with open(resume_processed, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                t = item.get('text') or item.get('masked_text') or ''
                if t:
                    resumes.append(t)

    # 加载职位描述数据
    jobs_dir = os.path.join('data', 'raw_jobs')
    jobs = []
    if os.path.isdir(jobs_dir):
        for fn in os.listdir(jobs_dir):
            p = os.path.join(jobs_dir, fn)
            if os.path.isfile(p):
                with open(p, 'r', encoding='utf-8') as f:
                    jobs.append(f.read())

    # 合并简历和职位描述数据
    texts = resumes + jobs
    
    # 如果没有足够的数据，添加一些测试数据
    if len(texts) < 10:
        texts = [
            "张三在腾讯科技有限公司担任高级软件工程师，负责Python开发",
            "李四毕业于北京大学计算机科学与技术专业，具有5年Java开发经验",
            "王五期望薪资25K-30K，熟练使用C++和Linux系统",
            "赵六在阿里巴巴集团担任产品经理，负责电商平台的设计和开发"
        ] * 10

    # 随机打乱数据
    random.seed(42)
    random.shuffle(texts)

    # 生成银色标注
    annotations = build_silver_annotations(texts)

    # 限制最大训练样本数为400
    keep_n = min(400, len(texts))

    # 保存标注数据
    with open(os.path.join('data', 'processed', 'ner_annotations.json'),
              'w',
              encoding='utf-8') as f:
        json.dump(
            {
                'texts': texts[:keep_n],
                'annotations': annotations[:keep_n]
            },
            f,
            ensure_ascii=False,
            indent=2)

    # 从配置文件加载标签映射，用于生成完整的标签列表
    config_path = os.path.join('config', 'entity_config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        label_map = config.get('label_map', {})
        # 提取唯一的英文标签
        unique_english_labels = sorted(list(set(label_map.values())))
        # 自动生成标签列表：O + B-xxx + I-xxx
        labels = ['O']
        for label in unique_english_labels:
            labels.extend([f'B-{label}', f'I-{label}'])
        print(f"[OK] 自动生成完整标签列表成功，共 {len(labels)} 个标签")
        print(f"生成的标签: {labels[:20]}...")  # 只打印前20个标签
    except Exception as e:
        print(f"[ERROR] 生成标签列表失败: {str(e)}，将使用默认标签列表")
        # 使用默认标签列表作为 fallback
        labels = [
            'O', 'B-JobTitle', 'I-JobTitle', 'B-Company', 'I-Company',
            'B-Degree', 'I-Degree', 'B-Years', 'I-Years', 'B-Skill', 'I-Skill',
            'B-Location', 'I-Location', 'B-Salary', 'I-Salary', 'B-Phone',
            'I-Phone', 'B-Email', 'I-Email', 'B-Language', 'I-Language',
            'B-Certificate', 'I-Certificate'
        ]

    # 创建标签映射
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {str(i): l for i, l in enumerate(labels)}

    # 保存标签映射
    with open(os.path.join(model_dir, 'label2id.json'), 'w',
              encoding='utf-8') as f:
        json.dump(label2id, f, ensure_ascii=False)
    with open(os.path.join(model_dir, 'id2label.json'), 'w',
              encoding='utf-8') as f:
        json.dump(id2label, f, ensure_ascii=False)

    # 定义预训练模型名称，使用可靠的中文BERT模型
    PRETRAINED_CHINESE_MODEL_NAME = "uer/roberta-base-finetuned-cluener2020-chinese"
    # 用于英文实体识别的预训练模型
    PRETRAINED_ENGLISH_MODEL_NAME = "dslim/bert-base-NER"
    # 默认使用中文模型
    pretrained_model_name = PRETRAINED_CHINESE_MODEL_NAME

    # 加载中文预训练模型的分词器
    tokenizer = BertTokenizerFast.from_pretrained(
        PRETRAINED_CHINESE_MODEL_NAME, local_files_only=False)
    model_name = PRETRAINED_CHINESE_MODEL_NAME
    print(f"成功加载中文预训练分词器: {model_name}")

    # 更新预训练模型名称
    pretrained_model_name = model_name

    # 创建数据集
    dataset = NERDataset(texts, annotations, tokenizer, label2id)

    # 创建数据加载器
    loader = DataLoader(dataset,
                        batch_size=8,
                        shuffle=True,
                        collate_fn=collate)

    # 此时才导入BertCRFTagger，确保使用新生成的标签
    from core.ner_model import BertCRFTagger

    # 初始化模型，确保使用与预训练权重兼容的配置
    model = BertCRFTagger(num_labels=len(labels),
                          pretrained_model_name=pretrained_model_name)

    # 初始化优化器
    optim = AdamW(model.parameters(), lr=3e-5)

    # 选择设备（GPU或CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 训练模型
    print(f"开始训练模型，共 {len(labels)} 个标签，设备: {device}")
    print(f"训练数据样本数: {len(dataset)}")
    print(f"数据加载器批次数量: {len(loader)}")
    
    # 训练4个epoch，保证训练效果
    num_epochs = 4
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        print(f"开始epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (input_ids, attn, y) in enumerate(loader):
            try:
                # 将数据移动到设备
                input_ids = input_ids.to(device)
                attn = attn.to(device)
                y = y.to(device)

                # 计算损失
                loss = model(input_ids, attn, y)
                total_loss += loss.item()
                batch_count += 1

                # 反向传播和参数更新
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                if batch_idx % 5 == 0:
                    print(
                        f"  批次 {batch_idx+1}/{len(loader)}，当前损失: {loss.item():.4f}"
                    )
                    
            except Exception as e:
                print(f"  批次 {batch_idx+1} 训练失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"epoch {epoch+1}/{num_epochs} 完成，平均损失: {avg_loss:.4f}")

    # 保存分词器
    tokenizer_path = os.path.join(model_dir, 'tokenizer')
    print(f"开始保存分词器到: {tokenizer_path}")
    try:
        tokenizer.save_pretrained(tokenizer_path)
        print(f"分词器保存成功，文件列表: {os.listdir(tokenizer_path)}")
    except Exception as e:
        print(f"分词器保存失败: {e}")
        import traceback
        traceback.print_exc()

    # 保存模型参数
    print(f"开始保存模型参数到: {os.path.join(model_dir, 'pytorch_model.bin')}")
    torch.save(model.state_dict(), os.path.join(model_dir,
                                                'pytorch_model.bin'))
    print("模型参数保存成功")

    # 保存模型配置
    config_path = os.path.join(model_dir, 'config.json')
    print(f"开始保存模型配置到: {config_path}")
    config = {
        'pretrained': pretrained_model_name,
        'chinese_model': PRETRAINED_CHINESE_MODEL_NAME,
        'english_model': PRETRAINED_ENGLISH_MODEL_NAME,
        'num_labels': len(labels)
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False)
    print("模型配置保存成功")

    # 打印完成信息
    print("\n========================================")
    print("NER模型训练完成!")
    print(f"使用的预训练模型: {pretrained_model_name}")
    print(f"中文预训练模型: {PRETRAINED_CHINESE_MODEL_NAME}")
    print(f"英文预训练模型: {PRETRAINED_ENGLISH_MODEL_NAME}")
    print(f"标签数量: {len(labels)}")
    print(f"模型保存路径: {model_dir}")
    print(f"最终模型目录内容: {os.listdir(model_dir)}")
    print("========================================")


if __name__ == '__main__':
    print("脚本入口点被调用")
    try:
        main()
        print("main函数执行完成")
    except Exception as e:
        print(f"main函数执行出错: {e}")
        import traceback
        traceback.print_exc()
