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

    def _load_ner_dataset(
            self, path: str) -> Tuple[List[str], List[List[Dict[str, str]]]]:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = data.get('texts', [])
        annotations = data.get('annotations', [])
        return texts, annotations

    def __init__(self):
        # 标签映射：将模型输出的标签映射到标注文件使用的标签
        self._label_mapping = {
            'name': ['Name', '姓名'],
            'organization': [
                'Company', 'CompanyName', 'Organization', 'School',
                'University', '毕业院校'
            ],
            'position': ['Position', 'JobTitle', 'Role', '求职意向', '岗位名称'],
            'skill': ['Skill', 'SkillRequirement', '技能', '技能要求'],
            'education': ['Education', 'EducationRequirement', '学历', '学历要求'],
            'industry': ['Industry', 'CompanyIndustry', '所属行业'],
            'year': ['WorkYear', '工作年限', 'years'],
            'time': ['Time'],
            'location': ['Location', '地点'],
            'email': ['Email'],
            'phone': ['Phone'],
            'major': ['Major', '专业'],
            'degree': ['Degree', '学位'],
        }

        # 创建反向映射：将标注文件标签映射到模型标签
        self._reverse_mapping = {}
        for model_label, ann_labels in self._label_mapping.items():
            for ann_label in ann_labels:
                self._reverse_mapping[ann_label.lower()] = model_label

    def _get_model_label(self, annotation_label: str) -> str:
        """将标注标签转换为模型标签"""
        if not annotation_label:
            return annotation_label
        return self._reverse_mapping.get(annotation_label.lower(),
                                         annotation_label.lower())

    def _get_annotation_label(self, model_label: str) -> str:
        """将模型标签转换为标注标签"""
        if not model_label:
            return model_label
        return self._label_mapping.get(model_label, [model_label])[0]

    def _clean_annotation_value(self, value: str) -> str:
        """清理标注值"""
        if not value:
            return value
        value = value.strip()
        if value.startswith('：'):
            value = value[1:].strip()
        return value

    def _norm(self, s: str) -> str:
        """规范化字符串"""
        return (s or '').strip()

    def compute_ner_metrics_from_annotations(
            self,
            annotations_path: str,
            model=None,
            max_samples=None) -> Dict[str, Any]:
        """
        从标注文件计算NER模型的性能指标
        
        Args:
            annotations_path: 标注文件路径
            model: NER模型实例，如果为None则使用默认模型
            max_samples: 最大处理样本数，用于快速测试
            
        Returns:
            dict: 包含各种评估指标的字典
        """
        try:
            # 加载标注数据
            texts, annotations = self._load_ner_dataset(annotations_path)

            # 如果指定了最大样本数，只处理前max_samples个样本
            if max_samples is not None:
                texts = texts[:max_samples]
                annotations = annotations[:max_samples]
                print(f"仅处理前 {max_samples} 个样本以提高速度")

            # 如果没有提供模型，使用默认模型
            if model is None:
                model = get_ner()

            total_annotations = 0
            correct_predictions = 0
            total_predictions = 0

            # 按实体类型统计的指标
            type_metrics = {}

            # 处理每条文本
            for i, (text, ann_list) in enumerate(zip(texts, annotations)):
                # 记录进度
                if (i + 1) % 5 == 0:  # 增加进度显示频率
                    print(f"处理进度: {i + 1}/{len(texts)} 条文本")

                # 跳过过长的文本以提高效率
                if len(text) > 1000:
                    print(f"跳过过长文本（{len(text)}字符），索引: {i}")
                    continue

                try:
                    # 使用NER模型进行预测
                    spans = []
                    if model:
                        spans = model.predict(text)

                    # 转换预测结果格式
                    pred_entities = []
                    for sp in spans:
                        pred_label = self._norm(sp.get('label'))
                        pred_text = self._norm(sp.get('text')).replace(' ', '')
                        pred_entities.append({
                            'label': pred_label,
                            'text': pred_text
                        })

                    # 转换标注结果格式
                    ann_entities = []
                    for ann in ann_list:
                        ann_label = self._norm(ann.get('label'))
                        ann_text = self._clean_annotation_value(
                            ann.get('value'))
                        ann_entities.append({
                            'label': ann_label,
                            'text': ann_text
                        })

                    # 更新总数
                    total_annotations += len(ann_entities)
                    total_predictions += len(pred_entities)

                    # 简化的匹配逻辑 - 只比较标签和文本值
                    matched_preds = set()
                    matched_anns = set()

                    for ann_idx, ann in enumerate(ann_entities):
                        for pred_idx, pred in enumerate(pred_entities):
                            if pred_idx in matched_preds:
                                continue

                            # 检查标签是否匹配（宽松匹配）
                            if self._get_model_label(
                                    ann['label']) == self._get_model_label(
                                        pred['label']):
                                # 检查文本是否匹配（宽松匹配）
                                ann_clean = ann['text'].lower()
                                pred_clean = pred['text'].lower()

                                if (ann_clean in pred_clean
                                        or pred_clean in ann_clean
                                        or ann_clean.strip()
                                        == pred_clean.strip()):
                                    # 认为是正确预测
                                    correct_predictions += 1
                                    matched_preds.add(pred_idx)
                                    matched_anns.add(ann_idx)

                                    # 更新实体类型指标
                                    entity_type = ann['label']
                                    if entity_type not in type_metrics:
                                        type_metrics[entity_type] = {
                                            'tp': 0,
                                            'fp': 0,
                                            'fn': 0
                                        }
                                    type_metrics[entity_type]['tp'] += 1
                                    break

                    # 统计未匹配的预测（FP）
                    for pred_idx, pred in enumerate(pred_entities):
                        if pred_idx not in matched_preds:
                            entity_type = pred['label']
                            if entity_type not in type_metrics:
                                type_metrics[entity_type] = {
                                    'tp': 0,
                                    'fp': 0,
                                    'fn': 0
                                }
                            type_metrics[entity_type]['fp'] += 1

                    # 统计未匹配的标注（FN）
                    for ann_idx, ann in enumerate(ann_entities):
                        if ann_idx not in matched_anns:
                            entity_type = ann['label']
                            if entity_type not in type_metrics:
                                type_metrics[entity_type] = {
                                    'tp': 0,
                                    'fp': 0,
                                    'fn': 0
                                }
                            type_metrics[entity_type]['fn'] += 1

                except Exception as e:
                    print(f"处理文本 {i} 时发生错误: {str(e)}")
                    continue

            # 计算总体指标
            precision = correct_predictions / total_predictions if total_predictions > 0 else 0
            recall = correct_predictions / total_annotations if total_annotations > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (
                precision + recall) > 0 else 0

            # 计算各实体类型的指标
            for entity_type, metrics in type_metrics.items():
                tp = metrics['tp']
                fp = metrics['fp']
                fn = metrics['fn']

                type_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                type_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                type_f1 = 2 * (type_precision * type_recall) / (
                    type_precision + type_recall) if (type_precision +
                                                      type_recall) > 0 else 0

                type_metrics[entity_type]['precision'] = type_precision
                type_metrics[entity_type]['recall'] = type_recall
                type_metrics[entity_type]['f1_score'] = type_f1

            # 生成评估报告
            evaluation_results = {
                'overall': {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'total_annotations': total_annotations,
                    'total_predictions': total_predictions,
                    'correct_predictions': correct_predictions
                },
                'by_entity_type': type_metrics,
                'total_samples': len(texts),
                'processed_samples': len([t for t in texts if len(t) <= 1000])
            }

            return evaluation_results

        except Exception as e:
            print(f"计算NER指标时发生错误: {str(e)}")
            raise

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
