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
import json
import torch
from torch import nn
import tempfile

# 使用HF镜像源解决下载问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # 允许联网下载
os.environ["HF_HUB_OFFLINE"] = "0"  # 允许HF Hub联网
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '30'  # 超时时间30秒
os.environ['HF_HUB_RETRY'] = '5'  # 重试5次
os.environ['HF_HUB_RETRY_DELAY'] = '2'  # 重试间隔2秒

# 然后再import其他模块
from transformers import BertTokenizerFast, BertModel, pipeline
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
                 pretrained_model_name:
                 str = "uer/roberta-base-finetuned-cluener2020-chinese"):
        """
        初始化BERT+CRF模型
        
        Args:
            num_labels: 标签类别数量
            pretrained_model_name: 预训练BERT模型名称
        """
        super().__init__()
        try:
            # 根据输入模型名称判断语言类型，只使用指定的两个模型
            is_chinese = "chinese" in pretrained_model_name.lower(
            ) or "中文" in pretrained_model_name
            model_name = "uer/roberta-base-finetuned-cluener2020-chinese" if is_chinese else "dslim/bert-base-NER"

            # 直接加载指定模型
            self.bert = BertModel.from_pretrained(
                model_name,
                local_files_only=False,
                resume_download=True  # 支持断点续传
            )
            logger.info(f"成功加载BERT模型: {model_name}")
        except Exception as e:
            logger.error(f"BERT模型加载过程发生异常: {e}")
            logger.info("创建随机初始化的BERT模型...")
            from transformers import BertConfig
            config = BertConfig(vocab_size=21128,
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
        logger.info("初始化NERPipeline...")

        # 从配置文件加载模型配置
        try:
            with open(os.path.join(model_dir, 'config.json'),
                      'r',
                      encoding='utf-8') as f:
                cfg = json.load(f)
                logger.info("成功加载模型配置")
        except Exception as e:
            logger.warning(f"加载模型配置失败: {e}，将使用默认配置")
            cfg = {
                'pretrained': 'uer/roberta-base-finetuned-cluener2020-chinese',
                'chinese_model':
                'uer/roberta-base-finetuned-cluener2020-chinese',
                'english_model': 'dslim/bert-base-NER'
            }

        # 从配置中获取模型名称
        self.pretrained_model = cfg.get(
            'pretrained', 'uer/roberta-base-finetuned-cluener2020-chinese')
        self.chinese_model = cfg.get(
            'chinese_model', 'uer/roberta-base-finetuned-cluener2020-chinese')
        self.english_model = cfg.get('english_model', 'dslim/bert-base-NER')

        logger.info(f"使用预训练模型: {self.pretrained_model}")
        logger.info(f"中文模型: {self.chinese_model}")
        logger.info(f"英文模型: {self.english_model}")

        # 模型根目录
        self.model_root_dir = model_dir
        os.makedirs(self.model_root_dir, exist_ok=True)

        # 初始化模型实例，默认只加载中文模型
        self.ner_model = None
        self.ner_zh = None
        self.ner_en = None

        # 导入所需的transformers组件
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        try:
            logger.info("开始加载NER模型...")

            # 只加载中文模型作为默认模型
            logger.info(f"只加载中文模型: {self.chinese_model}")

            # 中文模型缓存目录
            zh_cache_dir = os.path.join(self.model_root_dir, 'chinese')
            os.makedirs(zh_cache_dir, exist_ok=True)

            # 加载中文模型，使用local_files_only=True优先本地加载，避免重复下载
            try:
                # 优先尝试从本地加载
                logger.info(f"尝试从本地加载中文tokenizer: {self.chinese_model}")
                tokenizer_zh = AutoTokenizer.from_pretrained(
                    self.chinese_model,
                    cache_dir=zh_cache_dir,
                    local_files_only=False)  # 允许从网络下载，确保模型能加载
                logger.info(f"成功加载中文tokenizer: {self.chinese_model}")
            except Exception as e:
                logger.error(f"加载中文tokenizer失败: {e}")
                raise  # 重新抛出异常，确保模型加载失败时能被捕获

            try:
                # 优先尝试从本地加载
                logger.info(f"尝试从本地加载中文模型: {self.chinese_model}")
                model_zh = AutoModelForTokenClassification.from_pretrained(
                    self.chinese_model,
                    cache_dir=zh_cache_dir,
                    local_files_only=False)  # 允许从网络下载，确保模型能加载
                logger.info(f"成功加载中文模型: {self.chinese_model}")
            except Exception as e:
                logger.error(f"加载中文模型失败: {e}")
                raise  # 重新抛出异常，确保模型加载失败时能被捕获

            # 创建中文NER pipeline
            self.ner_zh = pipeline("ner",
                                   model=model_zh,
                                   tokenizer=tokenizer_zh,
                                   aggregation_strategy="simple")
            logger.info("成功创建中文NER pipeline")

            # 设置默认模型为中文
            self.ner_model = self.ner_zh
            logger.info("使用中文模型作为默认模型")

        except Exception as e:
            logger.error(f"加载NER模型失败: {e}")
            logger.error("详细错误信息:", exc_info=True)
            logger.warning("模型加载失败，将使用空结果返回")

            # 尝试使用Hugging Face pipeline直接加载模型，不依赖本地模型文件
            try:
                logger.info("尝试使用Hugging Face pipeline直接加载模型...")
                # 直接使用pipeline加载中文模型，不依赖本地模型文件
                self.ner_zh = pipeline("ner",
                                       model=self.chinese_model,
                                       aggregation_strategy="simple")
                self.ner_model = self.ner_zh
                logger.info(
                    f"成功使用Hugging Face pipeline直接加载中文模型: {self.chinese_model}")
            except Exception as e2:
                logger.error(f"使用Hugging Face pipeline直接加载模型失败: {e2}")

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
        # 文本预处理
        text = text.strip()
        if not text:
            return []

        # 检测文本语言，简单判断：如果包含中文则使用中文模型，否则使用英文模型
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)

        # 选择合适的模型
        selected_model = None
        model_name = ""

        if has_chinese:
            # 使用中文模型
            if self.ner_zh:
                selected_model = self.ner_zh
                model_name = self.chinese_model
                logger.debug("使用中文NER模型进行推理")
            else:
                selected_model = self.ner_model
                model_name = self.pretrained_model if self.ner_model else "无"
                logger.warning("中文模型不可用，使用默认模型")
        else:
            # 使用英文模型，但只在需要时才加载
            if self.ner_en:
                selected_model = self.ner_en
                model_name = self.english_model
                logger.debug("使用英文NER模型进行推理")
            else:
                # 只在需要时才加载英文模型
                logger.info("英文模型未加载，开始加载英文模型...")
                try:
                    from transformers import AutoTokenizer, AutoModelForTokenClassification

                    # 英文模型缓存目录
                    en_cache_dir = os.path.join(self.model_root_dir, 'english')
                    os.makedirs(en_cache_dir, exist_ok=True)

                    # 优先从本地加载英文模型，避免重复下载
                    try:
                        # 加载英文tokenizer
                        tokenizer_en = AutoTokenizer.from_pretrained(
                            self.english_model,
                            cache_dir=en_cache_dir,
                            local_files_only=True)
                        logger.info(
                            f"成功从本地加载英文tokenizer: {self.english_model}")

                        # 加载英文模型
                        model_en = AutoModelForTokenClassification.from_pretrained(
                            self.english_model,
                            cache_dir=en_cache_dir,
                            local_files_only=True)
                        logger.info(f"成功从本地加载英文模型: {self.english_model}")
                    except Exception as e:
                        logger.info(f"本地加载英文模型失败: {e}，尝试从Hugging Face下载一次")
                        # 只在本地没有时下载一次
                        tokenizer_en = AutoTokenizer.from_pretrained(
                            self.english_model,
                            cache_dir=en_cache_dir,
                            local_files_only=False)
                        logger.info(
                            f"成功从Hugging Face下载英文tokenizer: {self.english_model}"
                        )

                        model_en = AutoModelForTokenClassification.from_pretrained(
                            self.english_model,
                            cache_dir=en_cache_dir,
                            local_files_only=False)
                        logger.info(
                            f"成功从Hugging Face下载英文模型: {self.english_model}")

                    # 创建英文NER pipeline
                    self.ner_en = pipeline("ner",
                                           model=model_en,
                                           tokenizer=tokenizer_en,
                                           aggregation_strategy="simple")
                    logger.info("成功创建英文NER pipeline")

                    selected_model = self.ner_en
                    model_name = self.english_model
                except Exception as e:
                    logger.error(f"加载英文模型失败: {e}")
                    selected_model = self.ner_model
                    model_name = self.pretrained_model if self.ner_model else "无"
                    logger.warning("英文模型加载失败，使用默认模型")

        # 检查模型是否可用
        if not selected_model:
            logger.warning("没有可用的NER模型")
            return []

        # 使用选定的模型进行推理
        logger.debug(f"使用模型 {model_name} 进行推理")
        ner_result = selected_model(text)

        # 转换pipeline结果格式为预期格式
        result = []
        for entity in ner_result:
            # 处理实体标签
            label = entity['entity_group']
            # 转换为统一的标签格式
            if label.startswith('B-'):
                label = label[2:]
            elif label.startswith('I-'):
                label = label[2:]

            # 添加到结果列表
            result.append({
                'label': label,
                'start': entity['start'],
                'end': entity['end'],
                'text': entity['word']
            })

        logger.debug(f"最终实体识别结果: {result}")
        return result


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

    try:
        # 尝试直接创建NERPipeline实例，不管模型文件是否存在
        # 因为NERPipeline内部已经实现了模型加载的容错机制
        _NER_SINGLETON = NERPipeline(path)

        # 检查模型是否真正可用
        if _NER_SINGLETON.ner_model or _NER_SINGLETON.ner_zh or _NER_SINGLETON.ner_en:
            logger.info("NER模型加载成功")
            return _NER_SINGLETON
        else:
            logger.warning("NER模型加载成功，但没有可用的模型实例")
            return _NER_SINGLETON
    except Exception as e:
        logger.error(f"NER模型初始化失败: {e}")
        return None
