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
            "gpt-3.5-turbo", "gpt-4", "qwen-1.8b", "deepseek-llm-7b-chat",
            "moonshot-v1-8k"
        ]

        # 初始化加密器
        self.cipher_suite = self._init_encryption(encryption_key)

        # 加载配置
        self.config = self._load_config()

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
        return {'models': {}, 'chains': [], 'default_model': None}

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
        return model_name in self.config[
            'models'] and 'api_key' in self.config['models'][model_name]

    def get_config_file_path(self) -> str:
        """
        获取配置文件路径
        
        Returns:
            配置文件路径
        """
        return self.config_file
