#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命名实体识别（NER）模型模块

该模块实现了基于BERT和条件随机场（CRF）的命名实体识别模型，用于从文本中识别和提取关键实体信息。
主要包含：
- CRF：条件随机场实现，用于序列标注
- BertCRFTagger：BERT+CRF的命名实体识别模型
- NERPipeline：NER模型推理管道，提供模型加载和预测接口
- get_ner：单例模式获取NER模型实例
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

# 然后再import其他模块
import json
import torch
from torch import nn
from transformers import BertTokenizerFast, BertModel
from utils.logger import get_logger

logger = get_logger(__name__)


class CRF(nn.Module):
    """
    条件随机场（CRF）实现，用于序列标注任务
    
    主要功能：
    - 计算条件随机场的损失函数
    - 使用Viterbi算法进行序列解码
    - 支持批次处理和掩码操作
    """

    def __init__(self, num_tags: int):
        """
        初始化CRF层
        
        Args:
            num_tags: 标签类别数量
        """
        super().__init__()
        self.num_tags = num_tags
        # 初始化转移矩阵参数
        self.start_transitions = nn.Parameter(
            torch.empty(num_tags))  # 起始状态转移概率
        self.end_transitions = nn.Parameter(torch.empty(num_tags))  # 结束状态转移概率
        self.transitions = nn.Parameter(torch.empty(num_tags,
                                                    num_tags))  # 标签间转移概率矩阵

        # 参数初始化
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _log_sum_exp(self, scores):
        """
        计算对数和指数的数值稳定版本
        
        Args:
            scores: 得分张量
            
        Returns:
            对数和指数结果
        """
        max_score, _ = scores.max(dim=1)
        return max_score + torch.log(
            torch.sum(torch.exp(scores - max_score.unsqueeze(1)), dim=1))

    def forward(self, emissions, tags=None, mask=None, reduction='mean'):
        """
        CRF前向传播
        
        Args:
            emissions: 发射概率，形状为 [batch_size, seq_length, num_tags]
            tags: 真实标签序列（训练时使用），形状为 [batch_size, seq_length]
            mask: 注意力掩码，形状为 [batch_size, seq_length]
            reduction: 损失函数的归约方式（'mean'或'sum'）
            
        Returns:
            如果提供了tags，返回损失值；否则返回解码结果
        """
        if tags is not None:
            # 训练模式：计算损失
            nll = self._compute_loss(emissions, tags, mask)
            if reduction == 'mean':
                return nll.mean()
            elif reduction == 'sum':
                return nll.sum()
            else:
                return nll
        else:
            # 推理模式：执行Viterbi解码
            return self.decode(emissions, mask)

    def _compute_loss(self, emissions, tags, mask):
        """
        计算CRF损失函数
        
        Args:
            emissions: 发射概率，形状为 [batch_size, seq_length, num_tags]
            tags: 真实标签序列，形状为 [batch_size, seq_length]
            mask: 注意力掩码，形状为 [batch_size, seq_length]
            
        Returns:
            负对数似然损失
        """
        batch_size, seq_length, num_tags = emissions.size()

        # 计算真实路径的得分
        score = self.start_transitions[tags[:, 0]] + emissions[:, 0, :].gather(
            1, tags[:, 0].unsqueeze(1)).squeeze(1)

        # 累加每个时间步的得分
        for t in range(1, seq_length):
            mask_t = mask[:, t]
            emit_t = emissions[:, t, :].gather(1,
                                               tags[:,
                                                    t].unsqueeze(1)).squeeze(1)
            trans_t = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + emit_t * mask_t + trans_t * mask_t

        # 添加结束转移得分
        seq_end = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, seq_end.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]

        # 计算分区函数（所有可能路径的总得分）
        partition = self._compute_log_partition(emissions, mask)

        # 负对数似然损失 = 分区函数 - 真实路径得分
        return partition - score

    def _compute_log_partition(self, emissions, mask):
        """
        计算分区函数（所有可能路径的总得分）
        
        Args:
            emissions: 发射概率，形状为 [batch_size, seq_length, num_tags]
            mask: 注意力掩码，形状为 [batch_size, seq_length]
            
        Returns:
            分区函数值
        """
        batch_size, seq_length, num_tags = emissions.size()

        # 初始化前向得分
        alpha = self.start_transitions + emissions[:, 0, :]

        # 递归计算前向得分
        for t in range(1, seq_length):
            emit_t = emissions[:, t, :]
            mask_t = mask[:, t].unsqueeze(1).bool()

            alpha_t = []
            for next_tag in range(num_tags):
                # 计算转移到next_tag的所有可能路径的得分
                score = self.transitions[:, next_tag].unsqueeze(0) + alpha
                alpha_t.append(self._log_sum_exp(score) + emit_t[:, next_tag])

            # 合并所有标签的得分
            alpha_candidate = torch.stack(alpha_t, dim=1)

            # 根据掩码更新得分（保持padding位置的得分不变）
            alpha = torch.where(mask_t, alpha_candidate, alpha)

        # 添加结束转移得分
        alpha = alpha + self.end_transitions

        # 计算所有可能结束标签的总得分
        return self._log_sum_exp(alpha)

    def decode(self, emissions, mask):
        """
        使用Viterbi算法进行序列解码，找到最优标签序列
        
        Args:
            emissions: 发射概率，形状为 [batch_size, seq_length, num_tags]
            mask: 注意力掩码，形状为 [batch_size, seq_length]
            
        Returns:
            解码后的标签序列，形状为 [batch_size, seq_length]
        """
        batch_size, seq_length, num_tags = emissions.size()

        # 初始化Viterbi得分和路径
        viterbi_score = self.start_transitions + emissions[:, 0, :]
        viterbi_path = torch.zeros(batch_size,
                                   seq_length,
                                   num_tags,
                                   dtype=torch.long)

        # 递归计算Viterbi得分和路径
        for t in range(1, seq_length):
            emit_t = emissions[:, t, :]

            # 计算从每个标签转移到当前时间步各个标签的得分
            score_t = viterbi_score.unsqueeze(2) + self.transitions.unsqueeze(
                0)

            # 找到每个时间步每个样本的最优前一标签
            best_score, best_path = score_t.max(dim=1)

            # 更新Viterbi得分和路径
            viterbi_score = best_score + emit_t
            viterbi_path[:, t, :] = best_path

        # 添加结束转移得分并找到最优结束标签
        viterbi_score = viterbi_score + self.end_transitions
        best_last_score, best_last_tag = viterbi_score.max(dim=1)

        # 回溯构建最优路径
        best_paths = []
        for i in range(batch_size):
            # 从最后一个标签开始回溯
            path = [best_last_tag[i].item()]
            for t in range(seq_length - 1, 0, -1):
                # 根据记录的前一标签构建路径
                path.insert(0, viterbi_path[i, t, path[0]].item())
            best_paths.append(path)

        return best_paths


class BertCRFTagger(nn.Module):
    """
    BERT+CRF的命名实体识别模型
    
    主要功能：
    - 使用BERT提取文本特征
    - 通过CRF进行序列标注
    - 支持训练和推理两种模式
    """

    def __init__(self,
                 num_labels: int,
                 pretrained_model_name: str = "bert-base-chinese"):
        """
        初始化BERT+CRF模型
        
        Args:
            num_labels: 标签类别数量
            pretrained_model_name: 预训练BERT模型名称
        """
        super().__init__()
        # 加载预训练BERT模型，优先使用中文模型以提高中文实体识别效果
        try:
            # 尝试从本地加载BERT模型，优先使用中文模型，优先使用本地文件
            self.bert = BertModel.from_pretrained(pretrained_model_name,
                                                  local_files_only=True)
            logger.info(f"从本地成功加载BERT模型: {pretrained_model_name}")
        except Exception as e:
            logger.warning(f"从本地加载BERT模型失败: {e}")
            logger.info("创建随机初始化的BERT模型...")
            # 如果加载失败，创建一个随机初始化的BERT模型
            # 使用与中文BERT模型兼容的配置参数
            from transformers import BertConfig
            config = BertConfig(
                vocab_size=21128,  # 中文BERT模型的词汇表大小
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072)
            self.bert = BertModel(config)

        # 添加dropout层防止过拟合
        self.dropout = nn.Dropout(0.1)
        # 分类器：将BERT输出映射到标签空间
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        # CRF层：用于序列标注
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        模型前向传播
        
        Args:
            input_ids: 输入序列的token ID，形状为 [batch_size, seq_length]
            attention_mask: 注意力掩码，形状为 [batch_size, seq_length]
            labels: 真实标签序列（训练时使用），形状为 [batch_size, seq_length]
            
        Returns:
            如果提供了labels，返回损失值；否则返回解码结果
        """
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 获取BERT最后一层的隐藏状态
        sequence_output = self.dropout(outputs.last_hidden_state)
        # 计算标签发射概率
        emissions = self.classifier(sequence_output)

        if labels is not None:
            # 训练模式：计算CRF损失
            loss = self.crf(emissions,
                            tags=labels,
                            mask=attention_mask.bool(),
                            reduction='mean')
            return loss
        else:
            # 推理模式：执行Viterbi解码
            return self.crf.decode(emissions, mask=attention_mask.bool())


class NERPipeline:
    """
    NER模型推理管道
    
    主要功能：
    - 加载预训练的NER模型和配置
    - 提供文本实体识别接口
    - 处理模型输入输出的格式化
    """

    def __init__(self, model_dir: str):
        """
        初始化NER推理管道
        
        Args:
            model_dir: 模型文件目录路径
        """
        # 加载标签映射
        try:
            with open(os.path.join(model_dir, 'label2id.json'),
                      'r',
                      encoding='utf-8') as f:
                self.label2id = json.load(f)

            with open(os.path.join(model_dir, 'id2label.json'),
                      'r',
                      encoding='utf-8') as f:
                self.id2label = json.load(f)

            logger.info(f"成功加载标签映射，共 {len(self.label2id)} 个标签")
        except Exception as e:
            logger.error(f"加载标签映射失败: {e}")
            # 使用默认的标签映射（适用于简历实体识别）
            self.label2id = {
                'O': 0,
                'B-COMPANY': 1,
                'I-COMPANY': 2,
                'B-POSITION': 3,
                'I-POSITION': 4,
                'B-EDUCATION': 5,
                'I-EDUCATION': 6,
                'B-SKILL': 7,
                'I-SKILL': 8,
                'B-EXPERIENCE': 9,
                'I-EXPERIENCE': 10
            }
            self.id2label = {str(v): k for k, v in self.label2id.items()}
            logger.info("使用默认标签映射")

        # 加载模型配置
        try:
            with open(os.path.join(model_dir, 'config.json'),
                      'r',
                      encoding='utf-8') as f:
                cfg = json.load(f)
                logger.info("成功加载模型配置")
        except Exception as e:
            logger.warning(f"加载模型配置失败: {e}，将使用默认配置")
            cfg = {}

        # 获取预训练模型名称，优先使用中文模型
        pretrained_model_name = cfg.get('pretrained', 'bert-base-chinese')

        # 确保BertTokenizerFast在所有分支都可用
        from transformers import BertTokenizerFast, BertConfig

        # 尝试优先使用本地模型，避免每次都从Hugging Face下载
        try:
            # 首先检查model_dir中是否有本地的tokenizer文件
            tokenizer_path = os.path.join(model_dir, 'tokenizer')
            if os.path.exists(tokenizer_path) and os.path.isdir(
                    tokenizer_path):
                self.tokenizer = BertTokenizerFast.from_pretrained(
                    tokenizer_path)
                logger.info("从模型目录加载分词器成功")
            else:
                # 优先使用中文分词器，提高中文实体识别效果，仅使用本地文件
                self.tokenizer = BertTokenizerFast.from_pretrained(
                    'bert-base-chinese', local_files_only=True)
                logger.info("使用本地中文分词器")
        except Exception as e:
            logger.warning(f"加载中文分词器失败: {e}，将使用内置的简单分词器")
            # 作为最后的备用方案，创建一个简单的分词器配置，不依赖任何外部文件
            logger.info("使用简单分词器配置...")
            # 直接创建分词器，不依赖预训练模型文件
            self.tokenizer = BertTokenizerFast(vocab_file=None,
                                               tokenizer_file=None,
                                               do_lower_case=True,
                                               do_basic_tokenize=True,
                                               never_split=None,
                                               model_max_length=512)
            logger.info("创建简单分词器成功")

        # 添加中文特化的分词器设置
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ['##']})
        logger.info(f"分词器词汇表大小: {self.tokenizer.vocab_size}")

        # 加载模型（BertCRFTagger会在内部处理BERT模型的加载）
        self.model = BertCRFTagger(num_labels=len(self.label2id),
                                   pretrained_model_name=pretrained_model_name)

        # 加载模型权重，处理可能的兼容性问题
        try:
            state_dict_path = os.path.join(model_dir, 'pytorch_model.bin')
            logger.info(f"尝试加载模型权重: {state_dict_path}")
            state_dict = torch.load(state_dict_path, map_location='cpu')

            # 检查并调整权重，确保与当前模型兼容
            model_dict = self.model.state_dict()

            # 只加载兼容的权重
            compatible_state_dict = {
                k: v
                for k, v in state_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            incompatible_keys = {
                k: (v.shape, model_dict[k].shape)
                for k, v in state_dict.items()
                if k in model_dict and v.shape != model_dict[k].shape
            }

            if incompatible_keys:
                logger.warning("发现不兼容的权重:")
                for k, (shape1, shape2) in incompatible_keys.items():
                    logger.warning(f"  {k}: 权重形状 {shape1}, 模型期望 {shape2}")
                logger.info(f"只加载 {len(compatible_state_dict)} 个兼容的权重")

            # 更新模型权重
            model_dict.update(compatible_state_dict)
            self.model.load_state_dict(model_dict, strict=False)
            logger.info("加载兼容的模型权重成功")
        except Exception as e:
            logger.warning(f"加载模型权重失败: {e}")
            logger.info("使用未训练的模型...")

        self.model.eval()

    def predict(self, text: str):
        """
        预测文本中的命名实体
        
        Args:
            text: 输入文本
            
        Returns:
            识别到的实体列表，每个实体包含：
            - label: 实体类型
            - start: 实体起始位置
            - end: 实体结束位置
            - text: 实体文本内容
        """
        # 文本预处理，提高识别效果
        text = text.strip()
        if not text:
            return []

        # 文本编码，获取token偏移量映射，使用更长的max_length以处理长文本
        enc = self.tokenizer(text,
                             return_offsets_mapping=True,
                             return_tensors='pt',
                             max_length=512,
                             truncation=True,
                             padding=False)

        # 模型推理
        with torch.no_grad():
            pred = self.model(enc['input_ids'], enc['attention_mask'])

        # 处理预测结果
        tags = pred[0]
        offsets = enc['offset_mapping'][0].tolist()
        result = []
        current = None

        # 根据标签和偏移量提取实体，改进实体边界处理
        for i, tag_id in enumerate(tags):
            # 跳过特殊标记（如[CLS], [SEP]）
            if offsets[i] == (0, 0):
                continue

            label = self.id2label[str(tag_id)]
            start, end = offsets[i]

            # 确保偏移量有效
            if start >= len(text) or end > len(text):
                continue

            # 如果是O标签（非实体），结束当前实体
            if label == 'O':
                if current and current['text'] and len(
                        current['text'].strip()) > 0:
                    result.append(current)
                    current = None
                continue

            # 如果是B标签（实体开始），开始新实体
            if label.startswith('B-'):
                if current and current['text'] and len(
                        current['text'].strip()) > 0:
                    result.append(current)
                entity_text = text[start:end]
                if entity_text.strip():
                    current = {
                        'label': label[2:],  # 去掉'B-'前缀
                        'start': start,
                        'end': end,
                        'text': entity_text
                    }
            # 如果是I标签（实体内部），扩展当前实体
            elif label.startswith('I-'):
                if current and current['label'] == label[2:]:
                    # 处理BERT分词导致的不连续偏移（如中文词组被分成多个token）
                    if start <= current['end']:  # 确保是连续的或重叠的
                        # 更新结束位置
                        current['end'] = max(current['end'], end)
                        current['text'] = text[current['start']:current['end']]
                else:
                    # 如果当前没有实体或标签类型不匹配，创建新实体
                    entity_text = text[start:end]
                    if entity_text.strip():
                        current = {
                            'label': label[2:],  # 去掉'I-'前缀
                            'start': start,
                            'end': end,
                            'text': entity_text
                        }

        # 添加最后一个实体（如果有）
        if current and current['text'] and len(current['text'].strip()) > 0:
            result.append(current)

        # 后处理：合并重叠或相邻的相同类型实体
        result = self._merge_overlapping_entities(result, text)

        return result

    def _merge_overlapping_entities(self, entities, text):
        """
        合并重叠或相邻的相同类型实体
        
        Args:
            entities: 实体列表
            text: 原始文本
            
        Returns:
            合并后的实体列表
        """
        if not entities:
            return []

        # 按起始位置排序
        entities.sort(key=lambda x: x['start'])

        merged = [entities[0]]

        for current in entities[1:]:
            last = merged[-1]

            # 如果当前实体与上一个实体类型相同，且有重叠或相邻（间隔小于2个字符）
            if (current['label'] == last['label']
                    and current['start'] <= last['end'] + 2):
                # 合并实体
                last['end'] = max(last['end'], current['end'])
                last['text'] = text[last['start']:last['end']]
            else:
                merged.append(current)

        return merged


# NER模型单例实例
_NER_SINGLETON = None


def check_model_needs_retraining(model_dir: str = None) -> bool:
    """
    检查模型是否需要重新训练
    
    Args:
        model_dir: 模型文件目录路径（可选，默认使用'data/models/ner_bert_crf'）
        
    Returns:
        bool: True表示需要重新训练，False表示不需要
    """
    # 设置默认模型路径
    base = os.path.join('data', 'models', 'ner_bert_crf')
    path = model_dir or base

    # 检查模型文件是否存在
    if not os.path.isdir(path):
        logger.warning(f"模型目录不存在: {path}")
        return True

    # 检查关键模型文件是否完整
    required_files = [
        'pytorch_model.bin',  # 模型权重文件
        'config.json',  # 模型配置文件
        'label2id.json',  # 标签映射文件
        'id2label.json'  # ID到标签映射文件
    ]

    for file in required_files:
        file_path = os.path.join(path, file)
        if not os.path.isfile(file_path):
            logger.warning(f"模型文件缺失: {file}")
            return True

    # 分词器目录不是必须的，因为可以使用默认的bert-base-chinese分词器
    tokenizer_path = os.path.join(path, 'tokenizer')
    if not os.path.isdir(tokenizer_path):
        logger.info(f"分词器目录缺失: {tokenizer_path}，将使用默认分词器")

    # 检查模型权重是否有效
    try:
        # 尝试加载模型权重，验证其有效性
        from transformers import BertConfig
        import torch

        # 加载配置
        with open(os.path.join(path, 'config.json'), 'r',
                  encoding='utf-8') as f:
            cfg = json.load(f)
        pretrained_model_name = cfg.get('pretrained', 'bert-base-chinese')

        # 加载标签映射
        with open(os.path.join(path, 'label2id.json'), 'r',
                  encoding='utf-8') as f:
            label2id = json.load(f)
        num_labels = len(label2id)

        # 尝试初始化模型并加载权重
        model = BertCRFTagger(num_labels=num_labels,
                              pretrained_model_name=pretrained_model_name)
        state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'),
                                map_location='cpu')

        # 检查权重是否兼容
        model_dict = model.state_dict()
        compatible_keys = [
            k for k in state_dict.keys()
            if k in model_dict and state_dict[k].shape == model_dict[k].shape
        ]

        # 如果兼容的权重比例低于80%，则需要重新训练
        if len(compatible_keys) < len(model_dict) * 0.8:
            logger.warning(
                f"模型权重兼容性差，只有 {len(compatible_keys)}/{len(model_dict)} 个权重兼容")
            return True

        logger.info("模型状态良好，不需要重新训练")
        return False
    except Exception as e:
        logger.warning(f"模型验证失败: {e}")
        return True


def get_ner(model_dir: str = None):
    """
    获取NER模型实例（单例模式）
    
    Args:
        model_dir: 模型文件目录路径（可选，默认使用'data/models/ner_bert_crf'）
        
    Returns:
        NERPipeline实例或None（如果模型加载失败）
    """
    global _NER_SINGLETON

    # 如果单例已存在，直接返回
    if _NER_SINGLETON is not None:
        return _NER_SINGLETON

    # 设置默认模型路径
    base = os.path.join('data', 'models', 'ner_bert_crf')
    path = model_dir or base

    # 检查模型文件是否存在
    if os.path.isdir(path) and os.path.isfile(
            os.path.join(path, 'pytorch_model.bin')):
        try:
            _NER_SINGLETON = NERPipeline(path)
            logger.info("NER模型加载成功")
            return _NER_SINGLETON
        except Exception as e:
            logger.error(f"NER模型初始化失败: {e}")
            return None
    else:
        logger.warning(f"NER模型文件不存在或不完整: {path}")
        logger.info("尝试使用备用模型路径...")

        # 检查备用模型路径
        alternative_paths = [
            os.path.join('data', 'models', 'ner'),
            os.path.join('models', 'ner_bert_crf'),
            os.path.join('models', 'ner')
        ]

        for alt_path in alternative_paths:
            if os.path.isdir(alt_path) and os.path.isfile(
                    os.path.join(alt_path, 'pytorch_model.bin')):
                try:
                    _NER_SINGLETON = NERPipeline(alt_path)
                    logger.info(f"NER模型从备用路径加载成功: {alt_path}")
                    return _NER_SINGLETON
                except Exception as e:
                    logger.error(f"备用NER模型初始化失败: {e}")
                    continue

    return None
