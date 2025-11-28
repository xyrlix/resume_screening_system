#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估模块

负责评估模型的性能，包括MSE、MAPE等指标的计算
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import json
import os
from core.ner_model import get_ner


class ModelEvaluator:
    """
    模型评估类，负责计算模型的各种评估指标
    """

    def __init__(self):
        """
        初始化模型评估类
        """
        pass

    def calculate_mse(self, actual: List[float],
                      predicted: List[float]) -> float:
        """
        计算均方误差（Mean Squared Error）
        
        Args:
            actual: 实际值列表
            predicted: 预测值列表
        
        Returns:
            均方误差
        """
        if len(actual) != len(predicted):
            raise ValueError("实际值和预测值的长度必须相同")

        actual_array = np.array(actual)
        predicted_array = np.array(predicted)

        return np.mean((actual_array - predicted_array)**2)

    def calculate_mape(self, actual: List[float],
                       predicted: List[float]) -> float:
        """
        计算平均绝对百分比误差（Mean Absolute Percentage Error）
        
        Args:
            actual: 实际值列表
            predicted: 预测值列表
        
        Returns:
            平均绝对百分比误差
        """
        if len(actual) != len(predicted):
            raise ValueError("实际值和预测值的长度必须相同")

        actual_array = np.array(actual)
        predicted_array = np.array(predicted)

        # 避免除以零
        non_zero_mask = actual_array != 0
        if not np.any(non_zero_mask):
            return 0.0

        return np.mean(
            np.abs(
                (actual_array[non_zero_mask] - predicted_array[non_zero_mask])
                / actual_array[non_zero_mask])) * 100

    def calculate_accuracy(self,
                           actual: List[float],
                           predicted: List[float],
                           threshold: float = 0.5) -> float:
        """
        计算分类准确率
        
        Args:
            actual: 实际值列表
            predicted: 预测值列表
            threshold: 分类阈值
        
        Returns:
            准确率
        """
        if len(actual) != len(predicted):
            raise ValueError("实际值和预测值的长度必须相同")

        actual_binary = [1 if val >= threshold else 0 for val in actual]
        predicted_binary = [1 if val >= threshold else 0 for val in predicted]

        correct = sum(1 for a, p in zip(actual_binary, predicted_binary)
                      if a == p)
        return correct / len(actual)

    def calculate_recall(self,
                         actual: List[float],
                         predicted: List[float],
                         threshold: float = 0.5) -> float:
        """
        计算召回率
        
        Args:
            actual: 实际值列表
            predicted: 预测值列表
            threshold: 分类阈值
        
        Returns:
            召回率
        """
        if len(actual) != len(predicted):
            raise ValueError("实际值和预测值的长度必须相同")

        actual_binary = [1 if val >= threshold else 0 for val in actual]
        predicted_binary = [1 if val >= threshold else 0 for val in predicted]

        true_positives = sum(1 for a, p in zip(actual_binary, predicted_binary)
                             if a == 1 and p == 1)
        actual_positives = sum(actual_binary)

        if actual_positives == 0:
            return 0.0

        return true_positives / actual_positives

    def calculate_precision(self,
                            actual: List[float],
                            predicted: List[float],
                            threshold: float = 0.5) -> float:
        """
        计算精确率
        
        Args:
            actual: 实际值列表
            predicted: 预测值列表
            threshold: 分类阈值
        
        Returns:
            精确率
        """
        if len(actual) != len(predicted):
            raise ValueError("实际值和预测值的长度必须相同")

        actual_binary = [1 if val >= threshold else 0 for val in actual]
        predicted_binary = [1 if val >= threshold else 0 for val in predicted]

        true_positives = sum(1 for a, p in zip(actual_binary, predicted_binary)
                             if a == 1 and p == 1)
        predicted_positives = sum(predicted_binary)

        if predicted_positives == 0:
            return 0.0

        return true_positives / predicted_positives

    def calculate_f1_score(self,
                           actual: List[float],
                           predicted: List[float],
                           threshold: float = 0.5) -> float:
        """
        计算F1分数
        
        Args:
            actual: 实际值列表
            predicted: 预测值列表
            threshold: 分类阈值
        
        Returns:
            F1分数
        """
        precision = self.calculate_precision(actual, predicted, threshold)
        recall = self.calculate_recall(actual, predicted, threshold)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def evaluate_model(self,
                       actual: List[float],
                       predicted: List[float],
                       threshold: float = 0.5) -> Dict[str, float]:
        """
        评估模型，计算各种指标
        
        Args:
            actual: 实际值列表
            predicted: 预测值列表
            threshold: 分类阈值
        
        Returns:
            包含各种评估指标的字典
        """
        return {
            'mse': self.calculate_mse(actual, predicted),
            'mape': self.calculate_mape(actual, predicted),
            'accuracy': self.calculate_accuracy(actual, predicted, threshold),
            'precision': self.calculate_precision(actual, predicted,
                                                  threshold),
            'recall': self.calculate_recall(actual, predicted, threshold),
            'f1_score': self.calculate_f1_score(actual, predicted, threshold)
        }

    def evaluate_matching_results(
            self, matching_results: List[Dict[str, Any]],
            actual_scores: List[float]) -> Dict[str, float]:
        """
        评估匹配结果
        
        Args:
            matching_results: 匹配结果列表
            actual_scores: 实际匹配分数列表
        
        Returns:
            包含各种评估指标的字典
        """
        # 从匹配结果中提取预测分数
        predicted_scores = [result[1] for result in matching_results]

        # 计算评估指标
        return self.evaluate_model(actual_scores, predicted_scores)

    def _load_ner_dataset(self, path: str) -> Tuple[List[str], List[List[Dict[str, str]]]]:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = data.get('texts', [])
        annotations = data.get('annotations', [])
        return texts, annotations

    def compute_ner_metrics_from_annotations(self, annotations_path: str) -> Dict[str, float]:
        texts, annotations = self._load_ner_dataset(annotations_path)
        ner = get_ner()

        tp = 0
        fp = 0
        fn = 0

        def norm(s: str) -> str:
            return (s or '').strip()

        for i, text in enumerate(texts):
            gold = annotations[i] if i < len(annotations) else []
            gold_pairs = {(norm(item.get('label')), norm(item.get('value')))
                          for item in gold if item.get('label') and item.get('value')}

            pred_pairs = set()
            if ner:
                spans = ner.predict(text)
                for sp in spans:
                    lab = norm(sp.get('label'))
                    val = norm(sp.get('text'))
                    if lab and val:
                        pred_pairs.add((lab, val))
            else:
                pred_pairs = set()

            matched_gold = set()
            matched_pred = set()

            for plab, pval in pred_pairs:
                candidates = [(glab, gval) for (glab, gval) in gold_pairs if glab == plab]
                hit = False
                for glab, gval in candidates:
                    if pval and gval and (pval in gval or gval in pval):
                        tp += 1
                        matched_gold.add((glab, gval))
                        matched_pred.add((plab, pval))
                        hit = True
                        break
                if not hit:
                    fp += 1

            for g in gold_pairs:
                if g not in matched_gold:
                    fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = precision

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        }

    def generate_evaluation_report(
            self, evaluation_results: Dict[str, float]) -> str:
        """
        生成评估报告
        
        Args:
            evaluation_results: 评估结果字典
        
        Returns:
            评估报告字符串
        """
        report = """模型评估报告
================
"""

        for metric, value in evaluation_results.items():
            if metric == 'mse':
                report += f"均方误差 (MSE): {value:.4f}\n"
            elif metric == 'mape':
                report += f"平均绝对百分比误差 (MAPE): {value:.2f}%\n"
            elif metric == 'accuracy':
                report += f"准确率: {value:.2f}%\n"
            elif metric == 'precision':
                report += f"精确率: {value:.2f}%\n"
            elif metric == 'recall':
                report += f"召回率: {value:.2f}%\n"
            elif metric == 'f1_score':
                report += f"F1分数: {value:.2f}%\n"
            else:
                report += f"{metric}: {value}\n"

        return report
