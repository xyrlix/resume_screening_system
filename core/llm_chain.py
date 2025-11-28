#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM链式分析模块

负责实现多LLM协同工作的链式分析功能
"""

from typing import List, Dict, Any
import json
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
    """

    def __init__(self):
        """
        初始化LLM链式分析类
        """
        self.llm_providers = {}  # 存储不同的LLM提供者
        self._init_llm_providers()

    def _init_llm_providers(self):
        """
        初始化LLM提供者
        """
        # 从LLM配置管理器获取配置
        from core.llm_config_manager import LLMConfigManager
        llm_config_manager = LLMConfigManager()
        model_configs = llm_config_manager.get_all_model_configs()
        default_model = llm_config_manager.get_default_model()

        # 优先顺序：deepseek, qwen, moonshot, doubao, openrouter
        preferred_order = ['deepseek', 'qwen', 'moonshot', 'doubao', 'openrouter']
        configured = []
        for name in preferred_order:
            full = BaseLLMProvider.MODEL_NAME_MAPPING.get(name, name)
            if llm_config_manager.is_model_configured(full):
                configured.append(name)

        selected = configured[:2]

        if not selected:
            logger.info("未配置优先LLM模型，使用模拟实现")
            # 选择前两个模拟提供者
            selected = preferred_order[:2]
            self.llm_providers = {n: MockLLMProvider(n) for n in selected}
        else:
            logger.info(
                f"已选择 {len(selected)} 个LLM模型用于链式分析: {', '.join(selected)}")
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
        pnames = self.active_provider_names
        provider = self.llm_providers[pnames[0]]
        logger.info(f"LLM调用 步骤1 provider={pnames[0]}")
        return provider.extract_entities(jd_text, resume_text)

    def step2_validate(self, extracted: dict) -> dict:
        """
        LLM2: 验证和修正提取的实体信息
        
        Args:
            extracted: 初步提取的实体信息
        
        Returns:
            验证和修正后的实体信息
        """
        pnames = self.active_provider_names
        provider = self.llm_providers[pnames[1] if len(pnames) > 1 else pnames[0]]
        logger.info(f"LLM调用 步骤2 provider={provider.name}")
        return provider.validate_entities(extracted)

    def step3_analyze(self, validated: dict) -> dict:
        """
        LLM3: 详细分析简历和JD的匹配度
        
        Args:
            validated: 验证和修正后的实体信息
        
        Returns:
            详细的匹配度分析
        """
        pnames = self.active_provider_names
        provider = self.llm_providers[pnames[0]]
        logger.info(f"LLM调用 步骤3 provider={provider.name}")
        return provider.analyze_match(validated)

    def step4_final_eval(self, analyzed: dict) -> dict:
        """
        LLM4: 最终评估匹配结果
        
        Args:
            analyzed: 详细的匹配度分析
        
        Returns:
            最终的评估结果
        """
        pnames = self.active_provider_names
        provider = self.llm_providers[pnames[1] if len(pnames) > 1 else pnames[0]]
        logger.info(f"LLM调用 步骤4 provider={provider.name}")
        return provider.generate_score(analyzed)

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
            score_result = provider.generate_score(analyzed)
            llm_scores[provider_name] = {
                'score': score_result['score'],
                'reason': score_result['reason']
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
                "suggestions": suggestions,
                "interview_questions": interview_questions
            }
        except Exception as e:
            logger.error(f"处理简历时发生错误: {str(e)}")
            # 返回一个默认结果，避免整个流程失败
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
                "llm_scores": {
                    "qwen": {
                        "score": 0.5,
                        "reason": "默认分数"
                    },
                    "deepseek": {
                        "score": 0.5,
                        "reason": "默认分数"
                    },
                    "openai": {
                        "score": 0.5,
                        "reason": "默认分数"
                    },
                    "openrouter": {
                        "score": 0.5,
                        "reason": "默认分数"
                    },
                    "moonshot": {
                        "score": 0.5,
                        "reason": "默认分数"
                    }
                },
                "weights": {
                    "qwen": 0.2,
                    "deepseek": 0.2,
                    "openai": 0.2,
                    "openrouter": 0.2,
                    "moonshot": 0.2
                },
                "suggestions": {
                    "suggestions": [],
                    "strengths": [],
                    "weaknesses": []
                },
                "interview_questions": []
            }

    def generate_suggestions(self, resume_text: str, jd_text: str) -> dict:
        """
        生成简历优化建议
        
        Args:
            resume_text: 简历文本
            jd_text: JD文本
        
        Returns:
            优化建议
        """
        provider = self.llm_providers['openai']
        prompt = f"""请根据以下简历和JD生成优化建议，返回JSON格式。

简历文本：{resume_text}

JD文本：{jd_text}

需要生成的内容包括：
1. 优化建议列表
2. 简历的优势
3. 简历的劣势

返回格式：
{
    "suggestions": ["建议1", "建议2"],
    "strengths": ["优势1", "优势2"],
    "weaknesses": ["劣势1", "劣势2"]
}"""

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
        provider = self.llm_providers['openai']
        prompt = f"""请根据以下简历和JD生成面试题，返回JSON格式。

简历文本：{resume_text}

JD文本：{jd_text}

需要生成5-10道面试题，涵盖技能、经验、项目等方面。

返回格式：
[
    "面试题1",
    "面试题2",
    "面试题3"
]
"""

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


class BaseLLMProvider:
    """
    LLM提供者基类
    """

    # 模型名称映射：简短名称 -> 完整名称
    MODEL_NAME_MAPPING = {
        'qwen': 'qwen-1.8b',
        'deepseek': 'deepseek-chat',
        'openai': 'gpt-3.5-turbo',
        'openrouter': 'gpt-3.5-turbo',
        'moonshot': 'moonshot-v1-8k',
        'doubao': 'Doubao-Seed-1.6'
    }

    def __init__(self, name: str):
        """
        初始化LLM提供者
        
        Args:
            name: LLM提供者名称
        """
        self.name = name
        self.llm_config_manager = LLMConfigManager()

        # 获取完整的模型名称
        full_model_name = self.MODEL_NAME_MAPPING.get(name, name)
        self.model_config = self.llm_config_manager.get_model_config(
            full_model_name)

        # 如果完整名称也没有配置，尝试使用默认模型
        if not self.model_config or 'api_key' not in self.model_config:
            default_model = self.llm_config_manager.get_default_model()
            if default_model:
                self.model_config = self.llm_config_manager.get_model_config(
                    default_model)
                logger.info(f"使用默认模型 {default_model} 替代 {name}")

    def _get_client(self):
        """
        获取OpenAI客户端
        """
        if not self.model_config or 'api_key' not in self.model_config:
            raise ValueError(f"未配置{self.name}模型的API Key")
        if openai is None:
            raise ImportError("openai 未安装，无法使用真实LLM提供者")
        client = openai.OpenAI(api_key=self.model_config['api_key'],
                               base_url=self.model_config.get('base_url'))
        return client

    def _call_llm(self, prompt: str, model: str = None) -> str:
        """
        调用LLM模型
        
        Args:
            prompt: 提示词
            model: 模型名称
        
        Returns:
            LLM响应
        """
        client = self._get_client()
        model = model or self.MODEL_NAME_MAPPING.get(self.name, "gpt-3.5-turbo")
        logger.info(f"LLM请求 provider={self.name} model={model} base_url={self.llm_config_manager.get_model_config(model).get('base_url','')} prompt_len={len(prompt)}")
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "system",
                "content": "你是一个专业的简历筛选助手，擅长分析简历和职位描述的匹配度。"
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.3,
            max_tokens=1000)
        content = response.choices[0].message.content
        logger.info(f"LLM返回 provider={self.name} model={model} resp_len={len(content)}")
        return content


class RealLLMProvider(BaseLLMProvider):
    """
    真实LLM提供者，使用OpenAI API
    """

    def extract_entities(self, jd_text: str, resume_text: str) -> dict:
        """
        提取简历和JD的实体信息
        
        Args:
            jd_text: JD文本
            resume_text: 简历文本
        
        Returns:
            提取的实体信息
        """
        prompt = f"""请从以下职位描述(JD)和简历中提取实体信息，返回JSON格式。

JD文本：{jd_text}

简历文本：{resume_text}

需要提取的JD实体包括：技能、学历要求、工作年限要求、职位名称
需要提取的简历实体包括：技能、教育背景、工作经验、职位

返回格式：
{{
    "jd_entities": {{
        "skills": ["技能1", "技能2"],
        "education": ["学历要求"],
        "experience": ["工作年限要求"],
        "position": "职位名称"
    }},
    "resume_entities": {{
        "skills": ["技能1", "技能2"],
        "education": ["教育背景"],
        "experience": ["工作经验"],
        "position": "职位"
    }}
}}"""

        response = self._call_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结果
            logger.error(f"解析LLM响应失败: {response}")
            return {
                "jd_entities": {
                    "skills": [],
                    "education": [],
                    "experience": [],
                    "position": ""
                },
                "resume_entities": {
                    "skills": [],
                    "education": [],
                    "experience": [],
                    "position": ""
                }
            }

    def validate_entities(self, extracted: dict) -> dict:
        """
        验证和修正提取的实体信息
        
        Args:
            extracted: 初步提取的实体信息
        
        Returns:
            验证和修正后的实体信息
        """
        prompt = f"""请验证并修正以下提取的实体信息，确保信息准确无误。

提取的实体信息：
{json.dumps(extracted, ensure_ascii=False, indent=2)}

请检查：
1. 技能是否准确
2. 学历要求和教育背景是否合理
3. 工作年限要求和工作经验是否匹配
4. 职位名称是否准确

返回修正后的JSON格式：
{{
    "jd_entities": {{
        "skills": ["技能1", "技能2"],
        "education": ["学历要求"],
        "experience": ["工作年限要求"],
        "position": "职位名称"
    }},
    "resume_entities": {{
        "skills": ["技能1", "技能2"],
        "education": ["教育背景"],
        "experience": ["工作经验"],
        "position": "职位"
    }}
}}"""

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
        prompt = f"""请分析以下简历和JD的匹配度，返回JSON格式。

验证后的实体信息：
{json.dumps(validated, ensure_ascii=False, indent=2)}

需要分析的维度：
1. 技能匹配：计算匹配的技能数量和匹配率
2. 教育背景匹配：判断是否满足要求
3. 工作经验匹配：判断是否满足要求

返回格式：
{{
    "skill_match": {{
        "matching_skills": ["匹配的技能1", "匹配的技能2"],
        "jd_skills": ["JD技能1", "JD技能2"],
        "resume_skills": ["简历技能1", "简历技能2"],
        "match_rate": 0.8
    }},
    "education_match": {{
        "match": true,
        "reason": "教育背景满足要求"
    }},
    "experience_match": {{
        "match": true,
        "reason": "工作经验满足要求"
    }}
}}"""

        response = self._call_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结果
            logger.error(f"解析LLM响应失败: {response}")
            jd_skills = validated['jd_entities']['skills']
            resume_skills = validated['resume_entities']['skills']
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

    def generate_score(self, analyzed: dict) -> dict:
        """
        生成匹配分数
        
        Args:
            analyzed: 详细的匹配度分析
        
        Returns:
            生成的匹配分数
        """
        prompt = f"""请根据以下匹配度分析，生成综合匹配分数，返回JSON格式。

匹配度分析：
{json.dumps(analyzed, ensure_ascii=False, indent=2)}

评分标准：
- 技能匹配率权重：50%
- 教育背景匹配权重：25%
- 工作经验匹配权重：25%

返回格式：
{{
    "score": 0.85,
    "reason": "技能匹配度高，教育背景和工作经验满足要求",
    "details": {{
        "skill_weight": 0.5,
        "education_weight": 0.25,
        "experience_weight": 0.25
    }}
}}"""

        response = self._call_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认结果
            logger.error(f"解析LLM响应失败: {response}")
            skill_match_rate = analyzed['skill_match']['match_rate']
            education_match = 1.0 if analyzed['education_match'][
                'match'] else 0.0
            experience_match = 1.0 if analyzed['experience_match'][
                'match'] else 0.0
            score = (skill_match_rate * 0.5) + (education_match * 0.25) + (
                experience_match * 0.25)
            return {
                "score": score,
                "reason":
                f"技能匹配度: {skill_match_rate:.2f}, 教育背景匹配: {education_match:.2f}, 工作经验匹配: {experience_match:.2f}",
                "details": analyzed
            }


class MockLLMProvider(BaseLLMProvider):
    """
    模拟LLM提供者，用于测试和演示
    """

    def extract_entities(self, jd_text: str, resume_text: str) -> dict:
        """
        模拟提取实体信息
        
        Args:
            jd_text: JD文本
            resume_text: 简历文本
        
        Returns:
            提取的实体信息
        """
        logger.info(f"MockLLM 提取实体 provider={self.name}")
        return {
            "jd_entities": {
                "skills": ["Python", "Java", "SQL"],
                "education": ["本科"],
                "experience": ["3-5年"],
                "position": "软件工程师"
            },
            "resume_entities": {
                "skills": ["Python", "SQL", "JavaScript"],
                "education": ["本科"],
                "experience": ["4年"],
                "position": "后端开发工程师"
            }
        }

    def validate_entities(self, extracted: dict) -> dict:
        """
        模拟验证和修正实体信息
        
        Args:
            extracted: 初步提取的实体信息
        
        Returns:
            验证和修正后的实体信息
        """
        # 简单验证，实际项目中应该使用LLM进行验证
        logger.info(f"MockLLM 验证实体 provider={self.name}")
        return extracted

    def analyze_match(self, validated: dict) -> dict:
        """
        模拟分析简历和JD的匹配度
        
        Args:
            validated: 验证和修正后的实体信息
        
        Returns:
            详细的匹配度分析
        """
        jd_skills = validated['jd_entities']['skills']
        resume_skills = validated['resume_entities']['skills']
        matching_skills = set(jd_skills) & set(resume_skills)

        logger.info(f"MockLLM 匹配分析 provider={self.name}")
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
        模拟生成匹配分数
        
        Args:
            analyzed: 详细的匹配度分析
        
        Returns:
            生成的匹配分数
        """
        # 基于分析结果生成模拟分数
        skill_match_rate = analyzed['skill_match']['match_rate']
        education_match = 1.0 if analyzed['education_match']['match'] else 0.0
        experience_match = 1.0 if analyzed['experience_match']['match'] else 0.0

        # 计算综合分数
        score = (skill_match_rate * 0.5) + (education_match *
                                            0.25) + (experience_match * 0.25)

        # 添加一些随机性，模拟不同LLM的差异
        import random
        random_factor = random.uniform(0.95, 1.05)
        score = min(max(score * random_factor, 0.0), 1.0)

        logger.info(f"MockLLM 生成分数 provider={self.name}")
        return {
            "score": score,
            "reason":
            f"技能匹配度: {skill_match_rate:.2f}, 教育背景匹配: {education_match:.2f}, 工作经验匹配: {experience_match:.2f}",
            "details": analyzed
        }
