#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本向量化模块

负责使用BGE-M3模型将文本转换为向量
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
import numpy as np
from typing import List


class Vectorizer:
    """
    文本向量化类，使用BGE-M3模型将文本转换为向量
    
    设计思路：
    1. 优先使用BGE-M3模型，性能最优
    2. 失败时降级使用MiniLM模型
    3. 最终降级使用自定义简单向量化器
    4. 支持模型本地缓存
    5. 使用HF镜像源加速下载
    """

    def __init__(self):
        """
        初始化向量化器，延迟加载BGE-M3模型
        """
        self.model = None
        self.model_name = None
        self._vector_cache = {}  # 添加向量缓存
        # 延迟加载模型，仅在需要时加载
        # self._load_model()

    def _load_model(self):
        """
        加载BGE-M3模型，如果失败则使用备用模型
        实现三级降级策略：BGE-M3 → MiniLM → 简单向量化器
        """
        # 模型列表，按优先级排序
        # 优先使用轻量级模型，减少内存占用
        model_list = [{
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "desc": "MiniLM模型（轻量级）"
        }, {
            "name": "BAAI/bge-m3",
            "desc": "BGE-M3模型（高性能）"
        }]

        from sentence_transformers import SentenceTransformer
        import psutil

        for model_info in model_list:
            try:
                # 检查内存使用情况
                memory = psutil.virtual_memory()
                print(
                    f"[LOG] � 当前内存使用: {memory.percent}%，可用内存: {memory.available / (1024**3):.2f} GB"
                )

                # 如果可用内存小于4GB，优先使用轻量级模型
                if memory.available < 4 * 1024**3 and model_info[
                        "name"] == "BAAI/bge-m3":
                    print(f"[LOG] ⚠ 可用内存不足4GB，跳过BGE-M3模型，尝试轻量级模型")
                    continue

                print(f"[LOG] �🔄 正在加载{model_info['desc']}...")
                self.model = SentenceTransformer(
                    model_info["name"],
                    cache_folder="./data/models",  # 使用本地缓存
                    device="cpu",  # 确保在CPU上运行
                    trust_remote_code=True  # 允许加载远程代码
                )
                self.model_name = model_info["name"]
                print(
                    f"[LOG] ✅ 成功加载{model_info['desc']}: {model_info['name']}")

                # 再次检查内存使用情况
                memory = psutil.virtual_memory()
                print(
                    f"[LOG] 📊 加载模型后内存使用: {memory.percent}%，可用内存: {memory.available / (1024**3):.2f} GB"
                )

                return
            except Exception as e:
                print(f"[LOG] ⚠ 加载{model_info['desc']}失败: {e}")
                continue

        # 如果所有模型都加载失败，使用简单的向量化方案
        print("[LOG] 🔄 尝试使用自定义简单向量化器...")
        self.model = self._create_simple_vectorizer()
        self.model_name = "simple_vectorizer"
        print("[LOG] ✅ 成功使用自定义简单向量化器")

    def _create_simple_vectorizer(self):
        """
        创建一个简单的向量化器作为最后的备用方案
        """

        # 实现一个简单的基于词频的向量化器
        class SimpleVectorizer:

            def __init__(self):
                self.vocab = {}
                self.vocab_size = 384  # 固定维度

            def encode(self, text):
                # 简单的基于词频的向量化
                words = text.lower().split()
                vector = [0.0] * self.vocab_size

                # 构建简单的词袋模型
                for word in words:
                    if word not in self.vocab:
                        if len(self.vocab) < self.vocab_size:
                            self.vocab[word] = len(self.vocab)
                    if word in self.vocab:
                        idx = self.vocab[word]
                        vector[idx] += 1.0

                # 归一化向量
                norm = sum(x * x for x in vector)**0.5
                if norm > 0:
                    vector = [x / norm for x in vector]

                return vector

            def get_sentence_embedding_dimension(self):
                return self.vocab_size

        return SimpleVectorizer()

    def vectorize(self, text: str) -> np.ndarray:
        """
        将单个文本转换为向量，使用缓存加速
        
        Args:
            text: 输入文本
        
        Returns:
            文本向量
        """
        # 检查缓存
        if text in self._vector_cache:
            return self._vector_cache[text]

        # 加载模型（如果未加载）
        if self.model is None:
            self._load_model()

        try:
            vector = self.model.encode(text)
            # 缓存结果
            self._vector_cache[text] = vector
            return vector
        except Exception as e:
            print(f"⚠ 文本向量化失败: {e}")
            # 返回全零向量作为备用
            return np.zeros(384) if hasattr(
                self.model,
                'get_sentence_embedding_dimension') else np.zeros(768)

    def vectorize_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        批量将文本转换为向量，使用缓存加速
        
        Args:
            texts: 输入文本列表
        
        Returns:
            文本向量列表
        """
        # 分离已缓存和未缓存的文本
        cached_texts = {}
        uncached_texts = []

        for i, text in enumerate(texts):
            if text in self._vector_cache:
                cached_texts[i] = self._vector_cache[text]
            else:
                uncached_texts.append((i, text))

        # 加载模型（如果未加载且有未缓存的文本）
        if uncached_texts and self.model is None:
            self._load_model()

        results = [None] * len(texts)

        # 填充已缓存的结果
        for i, vector in cached_texts.items():
            results[i] = vector

        # 处理未缓存的文本
        if uncached_texts:
            try:
                # 提取未缓存的文本
                indices, uncached_text_list = zip(*uncached_texts)
                # 批量编码
                vectors = self.model.encode(uncached_text_list)
                # 填充结果并缓存
                for i, vector, text in zip(indices, vectors,
                                           uncached_text_list):
                    results[i] = vector
                    self._vector_cache[text] = vector
            except Exception as e:
                print(f"⚠ 批量文本向量化失败: {e}")
                # 为失败的文本生成全零向量
                dim = 384 if hasattr(
                    self.model, 'get_sentence_embedding_dimension') else 768
                for i, _ in uncached_texts:
                    results[i] = np.zeros(dim)

        return results

    def get_vector_dimension(self) -> int:
        """
        获取向量的维度
        
        Returns:
            向量维度
        """
        if self.model is None:
            self._load_model()

        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"⚠ 获取向量维度失败: {e}")
            # 默认返回384（MiniLM模型的维度）
            return 384
