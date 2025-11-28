#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM模型配置管理模块

负责管理LLM模型配置，支持配置API Key，并且可选链式组合使用
"""

import json
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import List, Dict, Any


class LLMConfigManager:
    """
    LLM模型配置管理类，负责管理LLM模型配置
    """

    def __init__(self, config_file: str = None, encryption_key: str = None):
        """
        初始化LLM模型配置管理类
        
        Args:
            config_file: 配置文件路径
            encryption_key: 加密密钥（可选）
        """
        self.config_file = config_file or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'config',
            'llm_config.json')

        # 密钥文件路径
        self.key_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'config',
            '.encryption_key')

        # 确保配置目录存在
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

        # 支持的LLM模型列表
        self.supported_models = [
            "gpt-3.5-turbo", "gpt-4", "qwen-1.8b", "Qwen3-Max",
            "deepseek-llm-7b-chat", "moonshot-v1-8k", "Doubao-Seed-1.6"
        ]

        # 区域与优先级配置
        self.default_region = 'domestic'

        # 初始化加密器
        self.cipher_suite = self._init_encryption(encryption_key)

        # 加载配置
        self.config = self._load_config()
        self._ensure_prompts()

    def _init_encryption(self, encryption_key: str = None) -> Fernet:
        """
        初始化加密器
        
        Args:
            encryption_key: 加密密钥（可选）
        
        Returns:
            Fernet加密器实例
        """
        if encryption_key:
            # 如果提供了密钥，使用它
            key = self._derive_key(encryption_key)
        elif os.path.exists(self.key_file):
            # 如果存在密钥文件，从文件中读取
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            # 否则生成新密钥并保存
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # 设置密钥文件权限为只读
            os.chmod(self.key_file, 0o400)

        return Fernet(key)

    def _derive_key(self, password: str) -> bytes:
        """
        从密码派生加密密钥
        
        Args:
            password: 用于派生密钥的密码
        
        Returns:
            派生的加密密钥
        """
        salt = b'salt_123456'  # 在实际应用中应该使用随机salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def _encrypt(self, data: str) -> str:
        """
        加密数据
        
        Args:
            data: 要加密的数据
        
        Returns:
            加密后的数据
        """
        return self.cipher_suite.encrypt(data.encode()).decode()

    def _decrypt(self, data: str) -> str:
        """
        解密数据
        
        Args:
            data: 要解密的数据
        
        Returns:
            解密后的数据
        """
        return self.cipher_suite.decrypt(data.encode()).decode()

    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # 解密API密钥
                if 'models' in config:
                    for model_name, model_config in config['models'].items():
                        if 'api_key' in model_config and model_config[
                                'api_key']:
                            try:
                                model_config['api_key'] = self._decrypt(
                                    model_config['api_key'])
                            except Exception as e:
                                print(f"解密{model_name}的API密钥失败: {e}")
                                model_config['api_key'] = ''

                return config
            except Exception as e:
                print(f"加载LLM配置文件失败: {e}")

        # 默认配置
        return {
            'models': {},
            'chains': [],
            'default_model': None,
            'region': self.default_region,
            'preferred_orders': {
                'domestic': ['qwen', 'moonshot', 'doubao', 'deepseek', 'openrouter'],
                'international': ['openai', 'openrouter', 'deepseek', 'moonshot', 'doubao']
            },
            'prompts': {}
        }

    def _ensure_prompts(self) -> None:
        try:
            prompts = self.config.get('prompts') or {}
            defaults = {
                'extract_entities': (
                    "请从以下职位描述(JD)和简历中提取实体信息，返回JSON格式。\n\n"
                    "JD文本：{jd_text}\n\n"
                    "简历文本：{resume_text}\n\n"
                    "需要提取的JD实体包括：技能、学历要求、工作年限要求、职位名称\n"
                    "需要提取的简历实体包括：技能、教育背景、工作经验、职位\n\n"
                    "返回格式：\n{\n    \"jd_entities\": {\n        \"skills\": [\"技能1\", \"技能2\"],\n        \"education\": [\"学历要求\"],\n        \"experience\": [\"工作年限要求\"],\n        \"position\": \"职位名称\"\n    },\n    \"resume_entities\": {\n        \"skills\": [\"技能1\", \"技能2\"],\n        \"education\": [\"教育背景\"],\n        \"experience\": [\"工作经验\"],\n        \"position\": \"职位\"\n    }\n}"
                ),
                'validate_entities': (
                    "请验证并修正以下提取的实体信息，确保信息准确无误。\n\n"
                    "提取的实体信息：\n{extracted_json}\n\n"
                    "请检查：\n1. 技能是否准确\n2. 学历要求和教育背景是否合理\n3. 工作年限要求和工作经验是否匹配\n4. 职位名称是否准确\n\n"
                    "返回修正后的JSON格式：\n{\n    \"jd_entities\": {\n        \"skills\": [\"技能1\", \"技能2\"],\n        \"education\": [\"学历要求\"],\n        \"experience\": [\"工作年限要求\"],\n        \"position\": \"职位名称\"\n    },\n    \"resume_entities\": {\n        \"skills\": [\"技能1\", \"技能2\"],\n        \"education\": [\"教育背景\"],\n        \"experience\": [\"工作经验\"],\n        \"position\": \"职位\"\n    }\n}"
                ),
                'analyze_match': (
                    "请分析以下简历和JD的匹配度，返回JSON格式。\n\n"
                    "验证后的实体信息：\n{validated_json}\n\n"
                    "需要分析的维度：\n1. 技能匹配：计算匹配的技能数量和匹配率\n2. 教育背景匹配：判断是否满足要求\n3. 工作经验匹配：判断是否满足要求\n\n"
                    "返回格式：\n{\n    \"skill_match\": {\n        \"matching_skills\": [\"匹配的技能1\", \"匹配的技能2\"],\n        \"jd_skills\": [\"JD技能1\", \"JD技能2\"],\n        \"resume_skills\": [\"简历技能1\", \"简历技能2\"],\n        \"match_rate\": 0.8\n    },\n    \"education_match\": {\n        \"match\": true,\n        \"reason\": \"教育背景满足要求\"\n    },\n    \"experience_match\": {\n        \"match\": true,\n        \"reason\": \"工作经验满足要求\"\n    }\n}"
                ),
                'generate_score': (
                    "请根据以下匹配度分析，生成综合匹配分数，返回JSON格式。\n\n"
                    "匹配度分析：\n{analyzed_json}\n\n"
                    "评分标准：\n- 技能匹配率权重：50%\n- 教育背景匹配权重：25%\n- 工作经验匹配权重：25%\n\n"
                    "返回格式：\n{\n    \"score\": 0.85,\n    \"reason\": \"技能匹配度高，教育背景和工作经验满足要求\",\n    \"details\": {\n        \"skill_weight\": 0.5,\n        \"education_weight\": 0.25,\n        \"experience_weight\": 0.25\n    }\n}"
                ),
                'generate_suggestions': (
                    "请根据以下简历和JD生成优化建议，返回JSON格式。\n\n"
                    "简历文本：{resume_text}\n\n"
                    "JD文本：{jd_text}\n\n"
                    "需要生成的内容包括：\n1. 优化建议列表\n2. 简历的优势\n3. 简历的劣势\n\n"
                    "返回格式：\n{\n    \"suggestions\": [\"建议1\", \"建议2\"],\n    \"strengths\": [\"优势1\", \"优势2\"],\n    \"weaknesses\": [\"劣势1\", \"劣势2\"]\n}"
                ),
                'generate_interview_questions': (
                    "请根据以下简历和JD生成面试题，返回JSON格式。\n\n"
                    "简历文本：{resume_text}\n\n"
                    "JD文本：{jd_text}\n\n"
                    "需要生成5-10道面试题，涵盖技能、经验、项目等方面。\n\n"
                    "返回格式：\n[\n    \"面试题1\",\n    \"面试题2\",\n    \"面试题3\"\n]"
                ),
                'evaluate_interview_answer': (
                    "请基于以下简历与JD，对候选人的面试回答进行评分并给出改进建议，返回JSON格式。\n\n"
                    "简历：{resume_text}\n\nJD：{jd_text}\n\n回答：{answer}\n\n"
                    "返回格式：\n{\n  \"score\": 0.0,\n  \"strengths\": [\"...\"],\n  \"weaknesses\": [\"...\"],\n  \"suggestions\": [\"...\"]\n}"
                ),
                'analyze_rejection': (
                    "请分析以下拒信文本，归纳拒绝原因并提出改进建议，返回JSON格式。\n\n"
                    "拒信：{rejection_text}\n简历：{resume_text}\nJD：{jd_text}\n\n"
                    "返回格式：\n{\n  \"reasons\": [\"...\"],\n  \"suggestions\": [\"...\"],\n  \"priority\": [\"...\"]\n}"
                ),
                'generate_learning_path': (
                    "请根据缺失技能为候选人生成学习成长路径，返回JSON格式。\n\n"
                    "目标岗位：{target_job}\n缺失技能：{missing_skills}\n\n"
                    "返回格式：\n{\n  \"steps\": [\"...\"],\n  \"courses\": [\"...\"],\n  \"projects\": [\"...\"],\n  \"certifications\": [\"...\"]\n}"
                ),
                'jd_extract_entities': (
                    "请从以下职位描述(JD)中提取完整的实体信息，返回JSON格式。\n\n"
                    "JD文本：{text}\n\n"
                    "需要提取的实体包括：{entity_list}\n\n"
                    "返回格式：\n{\n    \"职位名称\": \"\",\n    \"公司名称\": \"\",\n    \"学历要求\": \"\",\n    \"工作年限要求\": \"\",\n    \"薪资范围\": \"\",\n    \"工作地点\": \"\",\n    \"技能要求\": \"\",\n    \"岗位职责\": \"\",\n    \"任职要求\": \"\",\n    \"行业\": \"\",\n    \"招聘人数\": \"\",\n    \"发布时间\": \"\",\n    \"截止时间\": \"\",\n    \"职位类型\": \"\",\n    \"语言要求\": \"\",\n    \"证书要求\": \"\",\n    \"福利\": \"\",\n    \"团队情况\": \"\"\n}"
                ),
            }
            updated = False
            for k, v in defaults.items():
                if k not in prompts or not prompts.get(k):
                    prompts[k] = v
                    updated = True
            if updated:
                self.config['prompts'] = prompts
                self._save_config()
        except Exception:
            pass

    def get_prompt(self, name: str) -> str:
        return (self.config.get('prompts') or {}).get(name, '')

    def set_prompt(self, name: str, template: str) -> bool:
        try:
            if 'prompts' not in self.config:
                self.config['prompts'] = {}
            self.config['prompts'][name] = template
            self._save_config()
            return True
        except Exception:
            return False

    def _save_config(self) -> None:
        """
        保存配置到文件
        """
        try:
            # 创建配置的副本，用于加密API密钥
            config_to_save = json.loads(json.dumps(self.config))  # 深拷贝

            # 加密API密钥
            if 'models' in config_to_save:
                for model_name, model_config in config_to_save['models'].items(
                ):
                    if 'api_key' in model_config and model_config['api_key']:
                        model_config['api_key'] = self._encrypt(
                            model_config['api_key'])

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存LLM配置文件失败: {e}")

    def get_supported_models(self) -> List[str]:
        """
        获取支持的LLM模型列表
        
        Returns:
            支持的LLM模型列表
        """
        return self.supported_models.copy()

    def get_region(self) -> str:
        return self.config.get('region', self.default_region)

    def set_region(self, region: str) -> bool:
        if region not in ['domestic', 'international']:
            return False
        self.config['region'] = region
        self._save_config()
        return True

    def get_preferred_order_by_region(self, region: str = None) -> List[str]:
        region = region or self.get_region()
        po = self.config.get('preferred_orders', {})
        order = po.get(region)
        if not order:
            order = self.default_preferred_order(region)
        return order

    def set_preferred_order(self, region: str, order: List[str]) -> bool:
        if region not in ['domestic', 'international']:
            return False
        if not order or not isinstance(order, list):
            return False
        self.config.setdefault('preferred_orders', {})[region] = order
        self._save_config()
        return True

    def default_preferred_order(self, region: str) -> List[str]:
        return ['qwen', 'moonshot', 'doubao', 'deepseek', 'openrouter'] if region == 'domestic' else ['openai', 'openrouter', 'deepseek', 'moonshot', 'doubao']

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        获取指定模型的配置
        
        Args:
            model_name: 模型名称
        
        Returns:
            模型配置
        """
        config = self.config['models'].get(model_name, {})
        # 确保返回的是配置的副本，避免直接修改原始配置
        return config.copy()

    def set_model_config(self,
                         model_name: str,
                         api_key: str,
                         base_url: str = None) -> bool:
        """
        设置模型配置
        
        Args:
            model_name: 模型名称
            api_key: API Key
            base_url: 可选的API基础URL
        
        Returns:
            是否设置成功
        """
        if model_name not in self.supported_models:
            return False

        self.config['models'][model_name] = {
            'api_key': api_key,
            'base_url': base_url
        }

        self._save_config()
        return True

    def delete_model_config(self, model_name: str) -> bool:
        """
        删除模型配置
        
        Args:
            model_name: 模型名称
        
        Returns:
            是否删除成功
        """
        if model_name in self.config['models']:
            del self.config['models'][model_name]
            self._save_config()
            return True
        return False

    def get_all_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有模型配置
        
        Returns:
            所有模型配置
        """
        # 返回深拷贝，确保不会意外修改原始配置
        return json.loads(json.dumps(self.config['models']))

    def get_chains(self) -> List[List[str]]:
        """
        获取所有链式组合
        
        Returns:
            链式组合列表
        """
        return self.config['chains'].copy()

    def add_chain(self, chain: List[str]) -> bool:
        """
        添加链式组合
        
        Args:
            chain: 模型链式组合列表
        
        Returns:
            是否添加成功
        """
        # 检查所有模型是否都在支持列表中
        for model_name in chain:
            if model_name not in self.supported_models:
                return False

        # 检查链式组合是否已存在
        if chain not in self.config['chains']:
            self.config['chains'].append(chain)
            self._save_config()
            return True
        return False

    def delete_chain(self, chain: List[str]) -> bool:
        """
        删除链式组合
        
        Args:
            chain: 模型链式组合列表
        
        Returns:
            是否删除成功
        """
        if chain in self.config['chains']:
            self.config['chains'].remove(chain)
            self._save_config()
            return True
        return False

    def get_default_model(self) -> str:
        """
        获取默认模型
        
        Returns:
            默认模型名称
        """
        return self.config['default_model']

    def set_default_model(self, model_name: str) -> bool:
        """
        设置默认模型
        
        Args:
            model_name: 模型名称
        
        Returns:
            是否设置成功
        """
        if model_name in self.supported_models:
            self.config['default_model'] = model_name
            self._save_config()
            return True
        return False

    def is_model_configured(self, model_name: str) -> bool:
        """
        检查模型是否已配置
        
        Args:
            model_name: 模型名称
        
        Returns:
            是否已配置
        """
        if model_name not in self.config.get('models', {}):
            return False
        cfg = self.config['models'][model_name]
        key = cfg.get('api_key')
        return bool(key)

    def get_config_file_path(self) -> str:
        """
        获取配置文件路径
        
        Returns:
            配置文件路径
        """
        return self.config_file
