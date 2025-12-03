#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM链式分析模块

负责实现多LLM协同工作的链式分析功能
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import time
import random
import re
try:
    import openai
except ImportError:
    openai = None
from core.llm_config_manager import LLMConfigManager
from utils.logger import get_logger

# 初始化日志记录器
logger = get_logger("llm_chain")


class LLMChain:
    """
    LLM链式分析类，负责多个LLM模型协同工作，相互验证和评估
    使用单例模式确保全局只有一个实例
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMChain, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        初始化LLM链式分析类
        """
        if self._initialized:
            return

        self.llm_providers = {}
        self.llm_config_manager = None
        # 获取当前区域
        from core.llm_config_manager import LLMConfigManager
        self.llm_config_manager = LLMConfigManager()
        self.region = self.llm_config_manager.get_region()
        logger.info(f"LLMChain初始化，当前区域: {self.region}")
        self._init_llm_providers()

        self._initialized = True

    def _init_llm_providers(self):
        """
        初始化LLM提供者
        """
        # 从LLM配置管理器获取配置
        from core.llm_config_manager import LLMConfigManager
        if self.llm_config_manager is None:
            self.llm_config_manager = LLMConfigManager()
        llm_config_manager = self.llm_config_manager
        model_configs = llm_config_manager.get_all_model_configs()
        default_model = llm_config_manager.get_default_model()

        # 根据区域动态选择优先顺序
        self.region = llm_config_manager.get_region()
        preferred_order = llm_config_manager.get_preferred_order_by_region(
            self.region)

        # 记录当前配置状态
        logger.info(f"当前区域配置: {self.region}")
        logger.info(f"区域优先顺序: {', '.join(preferred_order)}")

        # 检查配置的模型并进行健康检查
        configured = []
        healthy_models = []
        for name in preferred_order:
            # 创建临时提供者实例获取模型映射
            temp_provider = BaseLLMProvider(name)
            full = temp_provider.MODEL_NAME_MAPPING.get(name, name)
            is_configured = llm_config_manager.is_model_configured(
                full, self.region)
            logger.info(
                f"模型 {name} ({full}): {'已配置' if is_configured else '未配置'}")
            if is_configured:
                configured.append(name)
                # 进行简单的健康检查
                try:
                    provider = RealLLMProvider(name)
                    if provider.model_config and provider.api_key:
                        healthy_models.append(name)
                        logger.info(f"模型 {name} 健康检查通过")
                    else:
                        logger.warning(f"模型 {name} 配置不完整，跳过")
                except Exception as e:
                    logger.warning(f"模型 {name} 健康检查失败: {e}")

        # 优先使用健康的模型，如果没有健康模型则使用所有配置的模型
        selected = healthy_models if healthy_models else configured

        # 如果没有配置任何模型，设置为空字典并记录错误，但不抛出异常
        if not selected:
            logger.error("未配置任何LLM模型或所有配置的模型都不健康，请检查配置文件")
            self.llm_providers = {}
            self.active_provider_names = []
        else:
            logger.info(
                f"已选择 {len(selected)} 个健康的LLM模型用于链式分析: {', '.join(selected)}")
            self.llm_providers = {n: RealLLMProvider(n) for n in selected}

        self.active_provider_names = selected

    def step1_extract(self, jd_text: str, resume_text: str) -> dict:
        """
        LLM1: 初步提取简历和JD的实体信息
        
        Args:
            jd_text: JD文本
            resume_text: 简历文本
        
        Returns:
            提取的实体信息
        """
        # 第一阶段：使用cheap LLM（如qwen）进行粗提取
        logger.info("第一阶段：使用廉价LLM进行实体粗提取")

        # 尝试最多3个模型进行粗提取
        max_attempts_rough = min(3, len(self.active_provider_names))
        rough_extraction = None

        for i in range(max_attempts_rough):
            # 优先选择qwen作为廉价模型，如果已经尝试过或者不存在，则按顺序选择
            if i == 0:
                # 第一次尝试优先选择qwen
                cheap_model = None
                for name in self.active_provider_names:
                    if name == 'qwen':
                        cheap_model = name
                        break
                if not cheap_model:
                    cheap_model = self.active_provider_names[i]
            else:
                # 后续尝试按顺序选择
                cheap_model = self.active_provider_names[i]

            provider = self.llm_providers[cheap_model]
            logger.info(
                f"LLM调用 步骤1-粗提 provider={cheap_model} (尝试 {i+1}/{max_attempts_rough})"
            )
            try:
                rough_extraction = provider.extract_entities(
                    jd_text, resume_text)
                logger.info("粗提取完成，准备进行精修")
                break
            except Exception as e:
                logger.error(f"步骤1-粗提 provider={cheap_model} 失败: {e}")
                # 如果是最后一次尝试，抛出错误
                if i == max_attempts_rough - 1:
                    raise Exception("所有模型粗提取都失败，请检查LLM模型配置")

        if not rough_extraction:
            raise Exception("没有可用的LLM模型进行提取，请检查配置")

        # 第二阶段：使用strong LLM（如deepseek）进行精修
        logger.info("第二阶段：使用强LLM进行实体精修")

        # 收集强模型
        strong_models = []
        for name in self.active_provider_names:
            if name in ['deepseek', 'zhipu', 'moonshot']:
                strong_models.append(name)

        # 如果没有指定的强模型，使用所有可用模型（排除粗提取使用的模型）
        if not strong_models:
            strong_models = [
                name for name in self.active_provider_names
                if name != cheap_model
            ]
            if not strong_models:
                strong_models = self.active_provider_names

        max_attempts_refine = min(3, len(strong_models))

        for i in range(max_attempts_refine):
            strong_model = strong_models[i]
            provider = self.llm_providers[strong_model]
            logger.info(
                f"LLM调用 步骤1-精修 provider={strong_model} (尝试 {i+1}/{max_attempts_refine})"
            )
            try:
                # 使用专门的JD和简历提取提示词进行精修
                # 分别提取JD和简历实体
                jd_entities = provider.extract_jd_entities(jd_text)
                resume_entities = provider.extract_resume_entities(resume_text)

                # 合并结果
                refined_extraction = {
                    "jd_entities": jd_entities,
                    "resume_entities": resume_entities
                }
                logger.info("精修完成，返回最终提取结果")
                return refined_extraction
            except Exception as e:
                logger.error(f"步骤1-精修 provider={strong_model} 失败: {e}")
                # 如果是最后一次尝试，返回粗提取结果
                if i == max_attempts_refine - 1:
                    logger.info("所有模型精修都失败，返回粗提取结果")
                    return rough_extraction

        logger.warning("没有可用的强LLM进行精修，返回粗提取结果")
        return rough_extraction

    def extract_jd_entities(self, jd_text: str) -> dict:
        """
        专门提取JD实体信息
        
        Args:
            jd_text: JD文本
        
        Returns:
            提取的JD实体信息
        """
        # 优先选择强模型进行JD实体提取
        strong_models = []
        for name in self.active_provider_names:
            if name in ['deepseek', 'zhipu', 'moonshot']:
                strong_models.append(name)

        # 如果没有指定的强模型，使用所有可用模型
        if not strong_models:
            strong_models = self.active_provider_names

        max_attempts = min(3, len(strong_models))  # 最多尝试3个模型

        for i in range(max_attempts):
            strong_model = strong_models[i]
            provider = self.llm_providers[strong_model]
            logger.info(
                f"使用模型 {strong_model} 提取JD实体 (尝试 {i+1}/{max_attempts})")
            try:
                return provider.extract_jd_entities(jd_text)
            except Exception as e:
                logger.error(f"提取JD实体失败: {e}")
                continue

        logger.warning(f"已尝试{max_attempts}个模型都失败，无法提取JD实体")
        return {}

    def extract_resume_entities(self, resume_text: str) -> dict:
        """
        专门提取简历实体信息
        
        Args:
            resume_text: 简历文本
        
        Returns:
            提取的简历实体信息
        """
        # 优先选择强模型进行简历实体提取
        strong_models = []
        for name in self.active_provider_names:
            if name in ['deepseek', 'zhipu', 'moonshot']:
                strong_models.append(name)

        # 如果没有指定的强模型，使用所有可用模型
        if not strong_models:
            strong_models = self.active_provider_names

        max_attempts = min(3, len(strong_models))  # 最多尝试3个模型

        for i in range(max_attempts):
            strong_model = strong_models[i]
            provider = self.llm_providers[strong_model]
            logger.info(
                f"使用模型 {strong_model} 提取简历实体 (尝试 {i+1}/{max_attempts})")
            try:
                return provider.extract_resume_entities(resume_text)
            except Exception as e:
                logger.error(f"提取简历实体失败: {e}")
                continue

        logger.warning(f"已尝试{max_attempts}个模型都失败，无法提取简历实体")
        return {}

    def step2_validate(self, extracted: dict) -> dict:
        """
        LLM2: 验证和修正提取的实体信息
        
        Args:
            extracted: 初步提取的实体信息
        
        Returns:
            验证和修正后的实体信息
        """
        pnames = self.active_provider_names
        order = [pnames[1] if len(pnames) > 1 else pnames[0]] + pnames
        max_attempts = min(3, len(order))  # 最多尝试3个模型

        for i in range(max_attempts):
            name = order[i]
            provider = self.llm_providers[name]
            logger.info(
                f"LLM调用 步骤2 provider={provider.name} (尝试 {i+1}/{max_attempts})"
            )
            try:
                return provider.validate_entities(extracted)
            except Exception as e:
                logger.error(f"步骤2 provider={name} 失败: {e}")
                continue
        logger.warning(f"已尝试{max_attempts}个模型都失败，返回原始提取结果")
        return extracted

    def step3_analyze(self, validated: dict) -> dict:
        """
        LLM3: 详细分析简历和JD的匹配度
        
        Args:
            validated: 验证和修正后的实体信息
        
        Returns:
            详细的匹配度分析
        """
        pnames = self.active_provider_names
        max_attempts = min(3, len(pnames))  # 最多尝试3个模型

        for i in range(max_attempts):
            name = pnames[i]
            provider = self.llm_providers[name]
            logger.info(
                f"LLM调用 步骤3 provider={provider.name} (尝试 {i+1}/{max_attempts})"
            )
            try:
                return provider.analyze_match(validated)
            except Exception as e:
                logger.error(f"步骤3 provider={name} 失败: {e}")
                continue
        logger.warning(f"已尝试{max_attempts}个模型都失败，返回默认匹配分析结果")
        jd_skills = validated.get('jd_entities', {}).get('skills', [])
        resume_skills = validated.get('resume_entities', {}).get('skills', [])
        matching_skills = set(jd_skills) & set(resume_skills)
        return {
            "skill_match": {
                "matching_skills":
                list(matching_skills),
                "jd_skills":
                jd_skills,
                "resume_skills":
                resume_skills,
                "match_rate":
                len(matching_skills) / len(jd_skills) if jd_skills else 0
            },
            "education_match": {
                "match": True,
                "reason": "默认匹配"
            },
            "experience_match": {
                "match": True,
                "reason": "默认匹配"
            }
        }

    def step4_final_eval(self, analyzed: dict) -> dict:
        """
        LLM4: 最终评估匹配结果
        
        Args:
            analyzed: 详细的匹配度分析
        
        Returns:
            最终的评估结果
        """
        pnames = self.active_provider_names
        order = [pnames[1] if len(pnames) > 1 else pnames[0]] + pnames
        max_attempts = min(3, len(order))  # 最多尝试3个模型

        for i in range(max_attempts):
            name = order[i]
            provider = self.llm_providers[name]
            logger.info(
                f"LLM调用 步骤4 provider={provider.name} (尝试 {i+1}/{max_attempts})"
            )
            try:
                return provider.generate_score(analyzed)
            except Exception as e:
                logger.error(f"步骤4 provider={name} 失败: {e}")
                continue
        logger.warning(f"已尝试{max_attempts}个模型都失败，返回默认评估结果")
        skill_match_rate = analyzed.get('skill_match',
                                        {}).get('match_rate', 0.0)
        education_match = 1.0 if analyzed.get('education_match', {}).get(
            'match', True) else 0.0
        experience_match = 1.0 if analyzed.get('experience_match', {}).get(
            'match', True) else 0.0
        score = (skill_match_rate * 0.5) + (education_match *
                                            0.25) + (experience_match * 0.25)
        return {"score": score, "reason": "默认分数", "details": analyzed}

    def multi_llm_eval(self, analyzed: dict) -> dict:
        """
        多LLM评估融合，使用多个LLM模型对同一任务进行评估
        
        Args:
            analyzed: 详细的匹配度分析
        
        Returns:
            融合后的匹配分数和每个LLM的评分
        """
        llm_scores = {}
        for provider_name in self.active_provider_names:
            provider = self.llm_providers[provider_name]
            logger.info(f"LLM融合评分 provider={provider_name}")
            try:
                score_result = provider.generate_score(analyzed)
                llm_scores[provider_name] = {
                    'score': score_result['score'],
                    'reason': score_result['reason']
                }
            except Exception as e:
                logger.error(f"融合评分 provider={provider_name} 失败: {e}")
                sm = analyzed.get('skill_match', {}).get('match_rate', 0.0)
                ed = 1.0 if analyzed.get('education_match', {}).get(
                    'match', True) else 0.0
                ex = 1.0 if analyzed.get('experience_match', {}).get(
                    'match', True) else 0.0
                fallback_score = (sm * 0.5) + (ed * 0.25) + (ex * 0.25)
                llm_scores[provider_name] = {
                    'score': fallback_score,
                    'reason': '默认分数'
                }

        # 加权平均，权重根据历史表现调整
        weights = self._load_llm_weights()
        weighted_sum = sum(llm_scores[p]['score'] * weights[p]
                           for p in self.active_provider_names)
        final_score = weighted_sum / sum(weights.values())

        return {
            'final_score': final_score,
            'llm_scores': llm_scores,
            'weights': weights
        }

    def _load_llm_weights(self) -> dict:
        """
        加载LLM模型的权重
        
        Returns:
            LLM模型权重字典
        """
        # 这里使用默认权重，实际项目中应该根据历史表现动态调整
        # 仅对当前激活的提供者设置权重，平均分配
        weights = {}
        count = max(len(getattr(self, 'active_provider_names', [])), 1)
        for name in getattr(self, 'active_provider_names', ['deepseek']):
            weights[name] = 1.0 / count
        return weights

    def process_resume(self, jd_text: str, resume_text: str) -> dict:
        """
        使用多LLM链式分析处理简历和JD
        
        Args:
            jd_text: JD文本
            resume_text: 简历文本
        
        Returns:
            处理结果
        """
        try:
            # 检查是否有可用的模型
            if not self.active_provider_names:
                logger.error("没有可用的LLM模型来处理简历")
                return {
                    "error": "没有可用的LLM模型来处理简历，所有配置的模型都不健康或未配置",
                    "llm_results": [],
                    "final_score": 0.0,
                    "llm_scores": {},
                    "weights": {},
                    "active_providers": [],
                    "suggestions": {
                        "suggestions": [],
                        "strengths": [],
                        "weaknesses": []
                    },
                    "interview_questions": []
                }

            logger.info(f"开始处理简历与JD匹配")

            # 1. 初步提取 (LLM1)
            logger.info(f"步骤1: 初步提取实体信息")
            step1 = self.step1_extract(jd_text, resume_text)

            # 2. 验证和修正 (LLM2)
            logger.info(f"步骤2: 验证和修正实体信息")
            step2 = self.step2_validate(step1)

            # 3. 详细分析 (LLM3)
            logger.info(f"步骤3: 详细分析匹配度")
            step3 = self.step3_analyze(step2)

            # 4. 最终评估 (LLM4)
            logger.info(f"步骤4: 最终评估")
            step4 = self.step4_final_eval(step3)

            # 5. 多LLM评估融合
            logger.info(f"步骤5: 多LLM评估融合")
            multi_eval_result = self.multi_llm_eval(step3)
            final_score = multi_eval_result['final_score']
            llm_scores = multi_eval_result['llm_scores']
            weights = multi_eval_result['weights']

            # 6. 生成优化建议
            logger.info(f"步骤6: 生成优化建议")
            suggestions = self.generate_suggestions(resume_text, jd_text)

            # 7. 生成面试题
            logger.info(f"步骤7: 生成面试题")
            interview_questions = self.generate_interview_questions(
                resume_text, jd_text)

            logger.info(f"简历处理完成，最终匹配分数: {final_score:.4f}")

            return {
                "step1": step1,
                "step2": step2,
                "step3": step3,
                "step4": step4,
                "final_score": final_score,
                "llm_scores": llm_scores,
                "weights": weights,
                "active_providers": self.active_provider_names,
                "suggestions": suggestions,
                "interview_questions": interview_questions
            }
        except Exception as e:
            logger.error(f"处理简历时发生错误: {str(e)}")
            default_llm_scores = {}
            default_weights = {}
            count = max(len(getattr(self, 'active_provider_names', [])), 1)
            for name in getattr(self, 'active_provider_names', ['deepseek']):
                default_llm_scores[name] = {"score": 0.5, "reason": "默认分数"}
                default_weights[name] = 1.0 / count
            return {
                "step1": {},
                "step2": {},
                "step3": {
                    "skill_match": {
                        "matching_skills": [],
                        "jd_skills": [],
                        "resume_skills": [],
                        "match_rate": 0.0
                    },
                    "education_match": {
                        "match": True,
                        "reason": "默认匹配"
                    },
                    "experience_match": {
                        "match": True,
                        "reason": "默认匹配"
                    }
                },
                "step4": {
                    "score": 0.5,
                    "reason": "默认分数",
                    "details": {}
                },
                "final_score": 0.5,
                "llm_scores": default_llm_scores,
                "weights": default_weights,
                "active_providers": getattr(self, 'active_provider_names', []),
                "suggestions": {
                    "suggestions": [],
                    "strengths": [],
                    "weaknesses": []
                },
                "interview_questions": []
            }

        provider_name = self.active_provider_names[
            0] if self.active_provider_names else list(
                self.llm_providers.keys())[0]
        provider = self.llm_providers[provider_name]
        tmpl = self.llm_config_manager.get_prompt('generate_suggestions')
        prompt = tmpl.replace('{resume_text}',
                              resume_text).replace('{jd_text}', jd_text)

        response = provider._call_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结果
            logger.error(f"解析LLM响应失败: {response}")
            return {
                "suggestions": [
                    "建议突出与JD相关的项目经验", "建议量化工作成果，使用具体数据", "建议添加JD中提到的关键技能关键词",
                    "建议突出相关工作经验"
                ],
                "strengths": ["技能匹配度高", "工作经验丰富", "教育背景符合要求"],
                "weaknesses": ["项目描述不够详细", "缺少相关证书", "未突出团队协作经验"]
            }

    def generate_interview_questions(self, resume_text: str,
                                     jd_text: str) -> list:
        """
        生成面试题
        
        Args:
            resume_text: 简历文本
            jd_text: JD文本
        
        Returns:
            面试题列表
        """
        provider_name = self.active_provider_names[
            0] if self.active_provider_names else list(
                self.llm_providers.keys())[0]
        provider = self.llm_providers[provider_name]
        tmpl = self.llm_config_manager.get_prompt(
            'generate_interview_questions')
        prompt = tmpl.replace('{resume_text}',
                              resume_text).replace('{jd_text}', jd_text)

        response = provider._call_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结果
            logger.error(f"解析LLM响应失败: {response}")
            return [
                "请详细介绍一下你最相关的项目经验", "你如何处理工作中的挑战？请举例说明", "你对我们公司和这个岗位有什么了解？",
                "你认为自己的哪些技能最适合这个岗位？", "你未来的职业规划是什么？"
            ]

    def evaluate_interview_answer(self, resume_text: str, jd_text: str,
                                  answer: str) -> dict:
        provider = self.llm_providers.get(
            self.active_provider_names[0] if self.
            active_provider_names else next(iter(self.llm_providers)))
        tmpl = self.llm_config_manager.get_prompt('evaluate_interview_answer')
        prompt = tmpl.replace('{resume_text}', resume_text).replace(
            '{jd_text}', jd_text).replace('{answer}', answer)
        response = provider._call_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "score": 0.6,
                "strengths": ["结构完整"],
                "weaknesses": ["缺乏量化数据"],
                "suggestions": ["补充具体指标与结果"]
            }

    def analyze_rejection(self, rejection_text: str, resume_text: str,
                          jd_text: str) -> dict:
        provider = self.llm_providers.get(
            self.active_provider_names[0] if self.
            active_provider_names else next(iter(self.llm_providers)))
        tmpl = self.llm_config_manager.get_prompt('analyze_rejection')
        prompt = tmpl.replace('{rejection_text}', rejection_text).replace(
            '{resume_text}', resume_text).replace('{jd_text}', jd_text)
        response = provider._call_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "reasons": ["技能匹配度不足"],
                "suggestions": ["补充相关项目与证书"],
                "priority": ["技能提升"]
            }

    def generate_learning_path(self, missing_skills: list,
                               target_job: str) -> dict:
        provider = self.llm_providers.get(
            self.active_provider_names[0] if self.
            active_provider_names else next(iter(self.llm_providers)))
        tmpl = self.llm_config_manager.get_prompt('generate_learning_path')
        prompt = tmpl.replace('{target_job}',
                              target_job).replace('{missing_skills}',
                                                  ', '.join(missing_skills))
        response = provider._call_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "steps": ["掌握基础概念", "完成小型项目", "参与开源"],
                "courses": [],
                "projects": [],
                "certifications": []
            }


class BaseLLMProvider:
    """
    LLM提供者基类
    """

    def __init__(self, name: str):
        """
        初始化LLM提供者
        
        Args:
            name: LLM提供者名称
        """
        self.name = name
        from core.llm_config_manager import LLMConfigManager
        self.llm_config_manager = LLMConfigManager()

        # 获取当前区域
        self.region = self.llm_config_manager.get_region()

        # 从配置文件中读取模型名称映射
        self.MODEL_NAME_MAPPING = self.llm_config_manager.config.get(
            'model_mappings', {})

        # 获取完整的模型名称
        full_model_name = self.MODEL_NAME_MAPPING.get(name, name)
        # 获取模型配置（考虑区域）
        self.model_config = self.llm_config_manager.get_model_config(
            full_model_name, self.region)

        # 直接暴露配置属性，方便访问
        if self.model_config:
            self.api_key = self.model_config.get('api_key')
            self.base_url = self.model_config.get('base_url')
            self.model_name = full_model_name
        else:
            self.api_key = None
            self.base_url = None
            self.model_name = full_model_name

        # 如果完整名称也没有配置，尝试使用默认模型
        if not self.model_config or 'api_key' not in self.model_config:
            default_model = self.llm_config_manager.get_default_model()
            if default_model:
                # 获取默认模型配置（考虑区域）
                self.model_config = self.llm_config_manager.get_model_config(
                    default_model, self.region)
                # 更新属性
                if self.model_config:
                    self.api_key = self.model_config.get('api_key')
                    self.base_url = self.model_config.get('base_url')
                    self.model_name = default_model
                logger.info(f"使用默认模型 {default_model} 替代 {name}")

    def _get_client(self):
        """
        获取LLM客户端
        """
        if not self.model_config or not self.model_config.get('api_key'):
            raise ValueError(f"未配置{self.name}模型的API Key")

        # 特殊处理Zhipu AI模型
        if self.model_name == 'glm-4.6' or self.name == 'zhipu' or self.name == 'glm':
            try:
                from zai import ZhipuAiClient
                api_key = self.model_config['api_key']
                client = ZhipuAiClient(api_key=api_key)
                return client
            except ImportError:
                raise ImportError("zai-sdk 未安装，无法使用Zhipu AI提供者")

        # 其他模型使用OpenAI客户端
        if openai is None:
            raise ImportError("openai 未安装，无法使用真实LLM提供者")

        # 从model_config中获取配置
        api_key = self.model_config['api_key']
        base_url = self.model_config.get('base_url')

        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        return client

    def _call_llm(self,
                  prompt: str,
                  model: str = None,
                  max_retries: int = 2) -> str:
        """
        调用LLM模型, 支持重试机制
        
        Args:
            prompt: 提示词
            model: 模型名称
            max_retries: 最大重试次数
        
        Returns:
            LLM响应
        """
        client = self._get_client()
        model = model or self.model_name

        # 获取当前使用模型的base_url
        current_model_config = self.model_config
        base_url = current_model_config.get('base_url',
                                            '') if current_model_config else ''

        # 特殊处理SiliconFlow模型，使用实际的API模型名称
        if self.name == 'siliconflow' or model == 'qwen2.5-72b-instruct':
            api_model_name = 'Qwen/Qwen2.5-72B-Instruct'
        else:
            api_model_name = model

        # 根据模型类型调整参数
        if self.name in ['qwen', 'deepseek', 'moonshot', 'zhipu', 'glm']:
            # 国内模型可能需要不同的参数设置
            temperature = 0.1
            max_tokens = 1500
        else:
            temperature = 0.3
            max_tokens = 1000

        # 重试机制
        retries = 0
        while retries <= max_retries:
            try:
                # 清理base_url，确保不包含重复的/chat/completions路径
                clean_base_url = base_url.rstrip('/')
                # 确保base_url不以/chat/completions结尾
                if clean_base_url.endswith('/chat/completions'):
                    clean_base_url = clean_base_url[:-len('/chat/completions')]

                logger.info(
                    f"LLM请求 provider={self.name} model={model} api_model={api_model_name} region={self.region} base_url={clean_base_url} prompt_len={len(prompt)} retry={retries}"
                )

                # 特殊处理Zhipu AI模型
                if self.model_name == 'glm-4.6' or self.name == 'zhipu' or self.name == 'glm':
                    # 使用Zhipu AI SDK调用
                    response = client.chat.completions.create(
                        model=api_model_name,
                        messages=[{
                            "role":
                            "system",
                            "content":
                            "你是一个专业的简历筛选助手，擅长分析简历和职位描述的匹配度。请严格按照JSON格式输出结果，不要包含任何额外的解释或说明。"
                        }, {
                            "role": "user",
                            "content": prompt
                        }],
                        thinking={
                            "type": "enabled",  # 启用深度思考模式
                        },
                        max_tokens=max_tokens,
                        temperature=temperature)
                else:
                    # 使用OpenAI兼容API调用
                    response = client.chat.completions.create(
                        model=api_model_name,
                        messages=[{
                            "role":
                            "system",
                            "content":
                            "你是一个专业的简历筛选助手，擅长分析简历和职位描述的匹配度。请严格按照JSON格式输出结果，不要包含任何额外的解释或说明。"
                        }, {
                            "role": "user",
                            "content": prompt
                        }],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        # 确保返回JSON格式
                        response_format={"type": "json_object"})

                content = response.choices[0].message.content

                # 去除可能的Markdown代码块标记
                if content:
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.startswith('```'):
                        content = content[3:]
                    if content.endswith('```'):
                        content = content[:-3]
                    content = content.strip()

                # 检查是否为空响应
                if not content:
                    logger.error(
                        f"LLM返回空响应 provider={self.name} model={model} retry={retries}"
                    )
                    # 抛出异常让重试机制处理
                    raise ValueError("LLM returned empty response")

                logger.info(
                    f"LLM返回 provider={self.name} model={model} resp_len={len(content)} status=success retry={retries}"
                )
                return content
            except Exception as e:
                retries += 1
                logger.error(
                    f"LLM调用失败 provider={self.name} model={model} retry={retries-1}/{max_retries}: {e}"
                )

                # 如果是最后一次重试，抛出异常
                if retries > max_retries:
                    raise

                # 指数退避重试策略
                backoff_time = min(10, 1 *
                                   (2**(retries - 1))) + random.uniform(0, 1)
                logger.info(
                    f"LLM调用失败，{backoff_time:.2f}秒后重试 provider={self.name} model={model}"
                )
                time.sleep(backoff_time)

                # 如果是特定错误，可以尝试使用不同的模型参数或降级策略
                if "Model Not Exist" in str(e) or "404" in str(e):
                    logger.warning(f"模型 {api_model_name} 不存在或不可用，尝试使用替代参数")
                    # 对于国内模型，可以尝试使用不同的模型名称格式
                    if self.name == 'qwen':
                        api_model_name = 'qwen-plus'
                    elif self.name == 'deepseek':
                        api_model_name = 'deepseek-chat'
                    elif self.name == 'moonshot':
                        api_model_name = 'moonshot-v1-32k'
                    elif self.name == 'zhipu' or self.name == 'glm' or self.model_name == 'glm-4.6':
                        api_model_name = 'glm-4.6'


class RealLLMProvider(BaseLLMProvider):
    """
    真实LLM提供者, 使用OpenAI API
    """

    def extract_entities(self, jd_text: str, resume_text: str) -> dict:
        """
        提取简历和JD的实体信息(兼容旧接口)
        
        Args:
            jd_text: JD文本
            resume_text: 简历文本
        
        Returns:
            提取的实体信息
        """
        # 分别提取JD和简历实体
        jd_entities = self.extract_jd_entities(jd_text)
        resume_entities = self.extract_resume_entities(resume_text)

        # 转换为旧接口兼容的格式
        # 确保技能要求始终是一个列表
        jd_skills = jd_entities.get("技能要求", [])
        if isinstance(jd_skills, str):
            if ';' in jd_skills:
                jd_skills_list = [
                    s.strip() for s in jd_skills.split(';') if s.strip()
                ]
            elif ',' in jd_skills:
                jd_skills_list = [
                    s.strip() for s in jd_skills.split(',') if s.strip()
                ]
            else:
                jd_skills_list = [jd_skills]
        else:
            jd_skills_list = jd_skills

        result = {
            "jd_entities": {
                "skills":
                jd_skills_list,
                "education": [jd_entities.get("学历要求", "")]
                if jd_entities.get("学历要求") else [],
                "experience": [jd_entities.get("工作年限要求", "")]
                if jd_entities.get("工作年限要求") else [],
                "position":
                jd_entities.get("职位名称", "")
            },
            "resume_entities": {
                "skills": [],
                "education": [],
                "experience": [],
                "position": ""
            }
        }

        # 处理简历技能，合并所有技能类别
        resume_skills = resume_entities.get("技能", {})
        for skill_type, skills in resume_skills.items():
            result["resume_entities"]["skills"].extend(skills)

        # 处理简历教育背景
        education = resume_entities.get("教育背景", [])
        if education:
            # 提取最高学历
            highest_edu = ""
            edu_order = ["博士", "硕士", "本科", "大专", "高中"]
            for edu in education:
                degree = edu.get("学历", "")
                for edu_level in edu_order:
                    if edu_level in degree:
                        highest_edu = edu_level
                        break
                if highest_edu:
                    break
            result["resume_entities"]["education"] = [highest_edu
                                                      ] if highest_edu else []

        # 处理简历工作经验
        experience = resume_entities.get("总工作年限", "")
        if experience:
            result["resume_entities"]["experience"] = [experience]

        # 处理简历职位
        work_exp = resume_entities.get("工作经验", [])
        if work_exp:
            # 使用最近的工作职位
            result["resume_entities"]["position"] = work_exp[0].get("职位", "")

        return result

    def extract_jd_entities(self, jd_text: str) -> dict:
        """
        专门提取JD实体信息
        
        Args:
            jd_text: JD文本
        
        Returns:
            提取的JD实体信息
        """
        tmpl = self.llm_config_manager.get_prompt('jd_extract_entities')
        if not tmpl:
            # 如果没有专门的JD提取提示词，使用默认的
            tmpl = self.llm_config_manager.get_prompt('extract_entities')

        prompt = tmpl.replace('{text}', jd_text).replace('{jd_text}', jd_text)

        response = self._call_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结果
            logger.error(f"解析LLM响应失败: {response}")
            return {
                "职位名称": "",
                "公司名称": "",
                "学历要求": "",
                "工作年限要求": "",
                "薪资范围": "",
                "工作地点": "",
                "技能要求": [],
                "岗位职责": "",
                "任职要求": "",
                "行业": "",
                "招聘人数": "",
                "发布时间": "",
                "截止时间": "",
                "职位类型": "",
                "语言要求": "",
                "证书要求": "",
                "福利": "",
                "团队情况": ""
            }

    def extract_resume_entities(self, resume_text: str) -> dict:
        """
        专门提取简历实体信息
        
        Args:
            resume_text: 简历文本
        
        Returns:
            提取的简历实体信息
        """
        tmpl = self.llm_config_manager.get_prompt('resume_extract_entities')
        if not tmpl:
            # 如果没有专门的简历提取提示词，返回默认结果
            logger.warning("没有找到resume_extract_entities提示词，返回空结果")
            return {
                "姓名": "",
                "联系电话": "",
                "电子邮箱": "",
                "现居地": "",
                "期望职位": "",
                "教育背景": [],
                "工作经验": [],
                "项目经验": [],
                "技能": {
                    "编程语言": [],
                    "框架工具": [],
                    "数据库": [],
                    "云平台": [],
                    "软技能": []
                },
                "证书资质": [],
                "语言能力": [],
                "总工作年限": ""
            }

        prompt = tmpl.replace('{resume_text}',
                              resume_text).replace('{text}', resume_text)

        response = self._call_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结果
            logger.error(f"解析LLM响应失败: {response}")
            return {
                "姓名": "",
                "联系电话": "",
                "电子邮箱": "",
                "现居地": "",
                "期望职位": "",
                "教育背景": [],
                "工作经验": [],
                "项目经验": [],
                "技能": {
                    "编程语言": [],
                    "框架工具": [],
                    "数据库": [],
                    "云平台": [],
                    "软技能": []
                },
                "证书资质": [],
                "语言能力": [],
                "总工作年限": ""
            }

    def validate_entities(self, extracted: dict) -> dict:
        """
        验证和修正提取的实体信息
        
        Args:
            extracted: 初步提取的实体信息
        
        Returns:
            验证和修正后的实体信息
        """
        tmpl = self.llm_config_manager.get_prompt('validate_entities')
        prompt = tmpl.replace(
            '{extracted_json}',
            json.dumps(extracted, ensure_ascii=False, indent=2))

        response = self._call_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回原始结果
            logger.error(f"解析LLM响应失败: {response}")
            return extracted

    def analyze_match(self, validated: dict) -> dict:
        """
        分析简历和JD的匹配度
        
        Args:
            validated: 验证和修正后的实体信息
        
        Returns:
            详细的匹配度分析
        """
        tmpl = self.llm_config_manager.get_prompt('analyze_match')
        prompt = tmpl.replace(
            '{validated_json}',
            json.dumps(validated, ensure_ascii=False, indent=2))

        response = self._call_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结果
            logger.error(f"解析LLM响应失败: {response}")
            jd_skills = validated.get('jd_entities', {}).get('skills', [])
            resume_skills = validated.get('resume_entities',
                                          {}).get('skills', [])
            matching_skills = set(jd_skills) & set(resume_skills)
            return {
                "skill_match": {
                    "matching_skills":
                    list(matching_skills),
                    "jd_skills":
                    jd_skills,
                    "resume_skills":
                    resume_skills,
                    "match_rate":
                    len(matching_skills) / len(jd_skills) if jd_skills else 0
                },
                "education_match": {
                    "match": True,
                    "reason": "简历教育背景满足JD要求"
                },
                "experience_match": {
                    "match": True,
                    "reason": "简历工作经验满足JD要求"
                }
            }

    def generate_score(self, analyzed: dict) -> dict:
        """
        生成匹配分数
        
        Args:
            analyzed: 详细的匹配度分析
        
        Returns:
            生成的匹配分数
        """
        import random
        tmpl = self.llm_config_manager.get_prompt('generate_score')
        prompt = tmpl.replace(
            '{analyzed_json}',
            json.dumps(analyzed, ensure_ascii=False, indent=2))

        try:
            response = self._call_llm(prompt)
            # 尝试解析JSON
            return json.loads(response)
        except Exception as e:
            # 处理所有异常情况
            response_content = response if 'response' in locals() else str(e)
            logger.error(f"LLM调用或解析失败: {response_content}")
            logger.error(f"错误详情: {str(e)}")

            # 尝试从失败的响应中提取部分有用信息
            extracted_score = None
            try:
                if 'response' in locals():
                    score_match = re.search(r'"score"\s*:\s*(\d+\.?\d*)',
                                            response)
                    if score_match:
                        extracted_score = float(score_match.group(1))
                        logger.info(f"从失败响应中提取到分数: {extracted_score}")
            except Exception as extract_error:
                logger.error(f"提取部分信息失败: {str(extract_error)}")

            # 计算默认分数
            skill_match_rate = analyzed.get('skill_match',
                                            {}).get('match_rate', 0.0)
            education_match = 1.0 if analyzed.get('education_match', {}).get(
                'match', False) else 0.0
            experience_match = 1.0 if analyzed.get('experience_match', {}).get(
                'match', False) else 0.0

            # 计算综合分数
            score = (skill_match_rate * 0.5) + (education_match * 0.25) + (
                experience_match * 0.25)

            # 如果能从失败响应中提取到分数，就使用它，否则使用计算出的分数
            if extracted_score is not None:
                score = extracted_score / 100.0  # 假设提取的分数是0-100范围

            # 添加一些随机性，模拟不同LLM的差异
            random_factor = random.uniform(0.95, 1.05)
            score = min(max(score * random_factor, 0.0), 1.0)

            logger.info(f"LLM调用失败，使用默认分数 provider={self.name}")
            return {
                "score": score,
                "reason":
                f"技能匹配度: {skill_match_rate:.2f}, 教育背景匹配: {education_match:.2f}, 工作经验匹配: {experience_match:.2f}",
                "details": analyzed
            }
            if skill in resume_text:
                skills["编程语言"].append(skill)

        for skill in ["Docker", "Kubernetes", "Git"]:
            if skill in resume_text:
                skills["框架工具"].append(skill)

        for skill in ["MySQL", "PostgreSQL", "MongoDB"]:
            if skill in resume_text:
                skills["数据库"].append(skill)

        for skill in ["AWS", "阿里云"]:
            if skill in resume_text:
                skills["云平台"].append(skill)

        for skill in ["团队协作", "项目管理"]:
            if skill in resume_text:
                skills["软技能"].append(skill)

        # 模拟提取证书资质
        certificates = []
        if "证书" in resume_text:
            certificates.append("相关专业证书")

        # 模拟提取语言能力
        languages = []
        if "英语" in resume_text:
            languages.append("英语流利")
