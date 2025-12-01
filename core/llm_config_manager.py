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
from utils.logger import get_logger

# 初始化日志记录器
logger = get_logger("llm_config_manager")


class LLMConfigManager:
    """
    LLM模型配置管理类，负责管理LLM模型配置
    """
    # 单例模式实现
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_file: str = None, encryption_key: str = None):
        # 防止重复初始化
        if self._initialized:
            return
        """
        初始化LLM模型配置管理类
        
        Args:
            config_file: 配置文件路径
            encryption_key: 加密密钥（可选）
        """
        self.config_file = config_file or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'config',
            'llm_config.json')

        # prompts配置文件路径
        self.prompts_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'config',
            'prompts.json')

        # 密钥文件路径
        self.key_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'config',
            '.encryption_key')

        # 确保配置目录存在
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

        # 区域特定URL的模型列表
        self.region_specific_models = ['qwen-plus', 'deepseek-chat']

        # 区域与优先级配置
        self.default_region = 'domestic'

        # 初始化加密器
        self.cipher_suite = self._init_encryption(encryption_key)

        # 加载配置
        self.config = self._load_config()

        # 加载prompts配置
        self._load_prompts()

        # 从配置文件中读取支持的LLM模型列表
        self.supported_models = self.config.get('supported_models', [])

        # 标记为已初始化
        self._initialized = True

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
        # 默认配置模板
        default_config = {
            'model_mappings': {
                'qwen': 'qwen-plus',
                'deepseek': 'deepseek-chat',
                'openrouter': 'tngtech/tng-r1t-chimera:free',
                'moonshot': 'moonshot-v1-8k',
                'kimi': 'kimi-k2-turbo-preview',
                'siliconflow': 'qwen2.5-72b-instruct',
                'hunyuan': 'hunyuan-lite',
                'spark': 'spark-x',
                'glm': 'glm-4.6',
                'zhipu': 'glm-4.6'
            },
            'supported_models': [
                'qwen-plus', 'deepseek-chat', 'moonshot-v1-8k',
                'kimi-k2-turbo-preview', 'tngtech/tng-r1t-chimera:free',
                'qwen2.5-72b-instruct', 'hunyuan-lite', 'spark-x', 'glm-4.6'
            ],
            'models': {},
            'chains': [],
            'default_model':
            None,
            'region':
            self.default_region,
            'preferred_orders': {
                'domestic': [
                    'deepseek', 'qwen', 'zhipu', 'openrouter', 'moonshot',
                    'doubao'
                ],
                'international': [
                    'moonshot', 'openrouter', 'deepseek', 'zhipu', 'doubao',
                    'qwen'
                ]
            }
        }

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
                                logger.error(f"解密{model_name}的API密钥失败: {e}")
                                model_config['api_key'] = ''

                # 确保配置中包含所有必要的字段
                # 如果字段不存在，从默认配置中获取
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    # 对于嵌套字典，确保所有键都存在
                    elif isinstance(value, dict) and isinstance(
                            config[key], dict):
                        for sub_key, sub_value in value.items():
                            if sub_key not in config[key]:
                                config[key][sub_key] = sub_value
                    # 对于列表，确保不为空（如果默认配置中不为空）
                    elif isinstance(value, list) and value and not config[key]:
                        config[key] = value

                return config
            except Exception as e:
                logger.error(f"加载LLM配置文件失败: {e}")

        # 如果配置文件不存在或加载失败，返回默认配置
        return default_config

    def _load_prompts(self):
        """
        加载prompts配置文件
        """
        try:
            if os.path.exists(self.prompts_file):
                with open(self.prompts_file, 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
                    # 将prompts合并到config中
                    self.config['prompts'] = prompts
                    logger.info(f"成功从 {self.prompts_file} 加载prompts配置")
        except Exception as e:
            logger.error(f"加载prompts配置文件失败: {e}")

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
            logger.info(f"LLM配置文件已成功保存到 {self.config_file}")
        except Exception as e:
            logger.error(f"保存LLM配置文件失败: {e}")

    def get_supported_models(self) -> List[str]:
        """
        获取支持的LLM模型列表
        
        Returns:
            支持的LLM模型列表（包含配置文件中的模型和已配置的模型）
        """
        # 获取配置文件中的支持模型列表和已配置的模型
        models_from_config = list(self.config.get('models', {}).keys())
        # 合并并去重
        all_models = list(set(self.supported_models + models_from_config))
        return all_models.copy()

    def get_model_mappings(self) -> Dict[str, str]:
        """
        获取模型名称映射
        
        Returns:
            模型名称映射字典
        """
        return self.config.get('model_mappings', {}).copy()

    def set_model_mapping(self, short_name: str, full_name: str) -> bool:
        """
        设置模型名称映射
        
        Args:
            short_name: 简短名称
            full_name: 完整名称
        
        Returns:
            是否设置成功
        """
        if 'model_mappings' not in self.config:
            self.config['model_mappings'] = {}

        self.config['model_mappings'][short_name] = full_name
        self._save_config()
        return True

    def delete_model_mapping(self, short_name: str) -> bool:
        """
        删除模型名称映射
        
        Args:
            short_name: 简短名称
        
        Returns:
            是否删除成功
        """
        if 'model_mappings' in self.config and short_name in self.config[
                'model_mappings']:
            del self.config['model_mappings'][short_name]
            self._save_config()
            return True
        return False

    def get_region(self) -> str:
        return self.config.get('region', self.default_region)

    def get_prompt(self, prompt_key):
        """
        获取指定类型的提示词模板
        
        Args:
            prompt_key: 提示词模板的键名
            
        Returns:
            提示词模板字符串，如果未找到则返回空字符串
        """
        return self.config.get("prompts", {}).get(prompt_key, "")

    def set_region(self, region: str) -> bool:
        if region not in ['domestic', 'international']:
            return False
        logger.info(
            f"更新LLM区域配置，从 {self.config.get('region', self.default_region)} 变更为 {region}"
        )
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
        old_order = self.config.get('preferred_orders', {}).get(region, [])
        logger.info(
            f"更新LLM区域{region}的模型优先级顺序，从 {', '.join(old_order)} 变更为 {', '.join(order)}"
        )
        self.config.setdefault('preferred_orders', {})[region] = order
        self._save_config()
        return True

    def default_preferred_order(self, region: str) -> List[str]:
        return ['qwen', 'moonshot', 'deepseek', 'openrouter'
                ] if region == 'domestic' else [
                    'moonshot', 'openrouter', 'deepseek', 'qwen'
                ]

    def get_model_config(self,
                         model_name: str,
                         region: str = None) -> Dict[str, Any]:
        """
        获取指定模型的配置
        
        Args:
            model_name: 模型名称
            region: 可选的区域，不指定则使用当前配置的区域
        
        Returns:
            模型配置，包含根据区域选择的base_url
        """
        config = self.config['models'].get(model_name, {})
        # 确保返回的是配置的副本，避免直接修改原始配置
        config_copy = config.copy()

        # 根据区域选择合适的base_url
        if region is None:
            region = self.get_region()

        # 如果配置了区域特定的base_url，优先使用
        if region in ['domestic', 'international'
                      ] and f'{region}_base_url' in config:
            config_copy['base_url'] = config[f'{region}_base_url']

        return config_copy

    def set_model_config(self,
                         model_name: str,
                         api_key: str,
                         base_url: str = None,
                         domestic_base_url: str = None,
                         international_base_url: str = None) -> bool:
        """
        设置模型配置
        
        Args:
            model_name: 模型名称
            api_key: API Key
            base_url: 可选的默认API基础URL
            domestic_base_url: 可选的国内API基础URL
            international_base_url: 可选的国外API基础URL
        
        Returns:
            是否设置成功
        """
        model_config = {'api_key': api_key}

        # 如果提供了通用base_url，使用它
        if base_url:
            model_config['base_url'] = base_url

        # 如果提供了区域特定的base_url，使用它们
        if domestic_base_url:
            model_config['domestic_base_url'] = domestic_base_url
        if international_base_url:
            model_config['international_base_url'] = international_base_url

        self.config['models'][model_name] = model_config

        # 如果模型不在支持的模型列表中，自动添加
        if model_name not in self.supported_models:
            self.supported_models.append(model_name)
            if 'supported_models' not in self.config:
                self.config['supported_models'] = []
            if model_name not in self.config['supported_models']:
                self.config['supported_models'].append(model_name)

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

    def is_model_configured(self, model_name: str, region: str = None) -> bool:
        """
        检查模型是否已配置
        
        Args:
            model_name: 模型名称
            region: 可选的区域参数（为了保持API兼容性）
        
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
