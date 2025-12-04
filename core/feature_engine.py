#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程模块

负责文本向量化、特征选择和数据正规化
"""

from typing import List, Dict, Any
import numpy as np


class FeatureEngine:
    """
    特征工程类，负责文本向量化和特征处理
    """

    def __init__(self, vectorizer):
        """
        初始化特征工程类
        
        Args:
            vectorizer: 向量化器实例
        """
        self.vectorizer = vectorizer

    def vectorize_text(self, text: str) -> np.ndarray:
        """
        将文本向量化
        
        Args:
            text: 输入文本
        
        Returns:
            文本向量
        """
        return self.vectorizer.vectorize(text)

    def vectorize_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        将多个文本向量化
        
        Args:
            texts: 输入文本列表
        
        Returns:
            文本向量列表
        """
        return [self.vectorize_text(text) for text in texts]

    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        归一化向量
        
        Args:
            vector: 输入向量
        
        Returns:
            归一化后的向量
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def normalize_vectors(self, vectors: List[np.ndarray]) -> List[np.ndarray]:
        """
        归一化多个向量
        
        Args:
            vectors: 输入向量列表
        
        Returns:
            归一化后的向量列表
        """
        return [self.normalize_vector(vector) for vector in vectors]

    def compute_similarity(self, vector1: np.ndarray,
                           vector2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
        
        Returns:
            余弦相似度，范围[-1, 1]
        """
        # 对齐维度以避免形状不匹配
        len1, len2 = len(vector1), len(vector2)
        if len1 != len2:
            target_len = min(len1, len2)
            vector1 = vector1[:target_len]
            vector2 = vector2[:target_len]

        vector1 = self.normalize_vector(vector1)
        vector2 = self.normalize_vector(vector2)
        return np.dot(vector1, vector2)

    def compute_similarities(self, query_vector: np.ndarray,
                             vectors: List[np.ndarray]) -> List[float]:
        """
        计算查询向量与多个向量的余弦相似度
        
        Args:
            query_vector: 查询向量
            vectors: 向量列表
        
        Returns:
            余弦相似度列表
        """
        # 统一所有向量的维度到最短长度
        target_len = len(query_vector)
        for v in vectors:
            if len(v) < target_len:
                target_len = len(v)

        query_vector = query_vector[:target_len]
        vectors_aligned = [v[:target_len] for v in vectors]

        query_vector = self.normalize_vector(query_vector)
        normalized_vectors = self.normalize_vectors(vectors_aligned)
        return [np.dot(query_vector, vector) for vector in normalized_vectors]

    def select_features(self, features: List[float],
                        feature_importance: List[float],
                        top_k: int) -> List[float]:
        """
        选择重要性最高的k个特征
        
        Args:
            features: 特征列表
            feature_importance: 特征重要性列表
            top_k: 选择的特征数量
        
        Returns:
            选择后的特征列表
        """
        # 按特征重要性排序，选择前k个特征
        sorted_indices = np.argsort(feature_importance)[::-1][:top_k]
        return [features[i] for i in sorted_indices]

    def extract_features_from_resume(self,
                                     resume: Dict[str, Any]) -> Dict[str, Any]:
        """
        从简历中提取特征，优化性能
        
        Args:
            resume: 结构化的简历信息
        
        Returns:
            包含向量特征的简历信息
        """
        # 检查是否已经有向量特征，避免重复计算
        if "vector" not in resume:
            # 向量化简历文本
            resume_vector = self.vectorize_text(resume["cleaned_text"])
            # 添加向量特征到简历信息中
            resume["vector"] = resume_vector

        if "segment_texts" in resume and "segment_vectors" not in resume:
            segs = resume["segment_texts"]
            segment_vectors = {}
            for k, v in segs.items():
                segment_vectors[k] = self.vectorize_text(v or "")
            resume["segment_vectors"] = segment_vectors

        return resume

    def extract_features_from_jd(self, jd: Dict[str, Any]) -> Dict[str, Any]:
        """
        从JD中提取特征，优化性能
        
        Args:
            jd: 结构化的JD信息
        
        Returns:
            包含向量特征的JD信息
        """
        # 检查是否已经有向量特征，避免重复计算
        if "vector" not in jd:
            # 向量化JD文本
            jd_vector = self.vectorize_text(jd["cleaned_text"])
            # 添加向量特征到JD信息中
            jd["vector"] = jd_vector

        if "segment_texts" in jd and "segment_vectors" not in jd:
            segs = jd["segment_texts"]
            segment_vectors = {}
            for k, v in segs.items():
                segment_vectors[k] = self.vectorize_text(v or "")
            jd["segment_vectors"] = segment_vectors

        return jd

    def scale_features(self, features: List[float]) -> List[float]:
        """
        对特征进行缩放，将值缩放到[0, 1]区间
        
        Args:
            features: 特征列表
        
        Returns:
            缩放后的特征列表
        """
        min_val = min(features)
        max_val = max(features)
        if max_val == min_val:
            return [0.5 for _ in features]
        return [(f - min_val) / (max_val - min_val) for f in features]

    def standardize_features(self, features: List[float]) -> List[float]:
        """
        对特征进行标准化，使其均值为0，标准差为1
        
        Args:
            features: 特征列表
        
        Returns:
            标准化后的特征列表
        """
        mean_val = np.mean(features)
        std_val = np.std(features)
        if std_val == 0:
            return [0 for _ in features]
        return [(f - mean_val) / std_val for f in features]
