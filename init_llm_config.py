#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
初始化LLM模型配置脚本
使用用户提供的API密钥配置各个LLM模型
"""

import os
import sys
import getpass
from core.llm_config_manager import LLMConfigManager


def init_llm_config(test_mode=False):
    """
    初始化LLM模型配置
    
    Args:
        test_mode: 是否使用测试模式（不进行交互式输入）
    """
    # 创建LLM配置管理器实例
    llm_config = LLMConfigManager()

    # 模型配置信息
    model_info = {
        "qwen-1.8b": {
            "name": "通义千问",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "env_var": "LLM_API_KEY_QWEN"
        },
        "deepseek-llm-7b-chat": {
            "name": "DeepSeek",
            "base_url": "https://api.deepseek.com/v1",
            "env_var": "LLM_API_KEY_DEEPSEEK"
        },
        "moonshot-v1-8k": {
            "name": "Kimi大模型",
            "base_url": "https://api.moonshot.cn/v1",
            "env_var": "LLM_API_KEY_MOONSHOT"
        },
        "gpt-3.5-turbo": {
            "name": "GPT-3.5-Turbo (OpenRouter)",
            "base_url": "https://openrouter.ai/api/v1",
            "env_var": "LLM_API_KEY_OPENROUTER"
        },
        "gpt-4": {
            "name": "GPT-4 (OpenRouter)",
            "base_url": "https://openrouter.ai/api/v1",
            "env_var": "LLM_API_KEY_OPENROUTER"
        }
    }

    if not test_mode:
        print("初始化LLM模型配置")
        print("=" * 50)
        print("请输入各个模型的API密钥（留空表示不配置该模型）")
        print("或按Enter键使用环境变量中的API密钥（如已设置）")
        print("=" * 50)

    # 测试模式：使用示例密钥（仅用于测试）
    if test_mode:
        print("⚠️  使用测试模式，将使用示例API密钥进行配置")
        api_keys = {
            "qwen-1.8b": {
                "api_key": "test_key_qwen",
                "base_url": model_info["qwen-1.8b"]["base_url"]
            }
        }
    else:
        # 获取API密钥
        api_keys = {}
        for model_name, info in model_info.items():
            # 先尝试从环境变量获取
            api_key = os.environ.get(info["env_var"])

            # 如果环境变量没有设置且不是测试模式，则交互式输入
            if not api_key:
                print(f"\n{info['name']} ({model_name}):")
                api_key = getpass.getpass(prompt=f"请输入API密钥（按Enter跳过）: ")

            if api_key:
                api_keys[model_name] = {
                    "api_key": api_key,
                    "base_url": info["base_url"]
                }
            else:
                print(f"跳过 {info['name']} 的配置")

    # 配置各个模型
    if api_keys:
        print("\n开始配置模型...")
        for model_name, config in api_keys.items():
            success = llm_config.set_model_config(model_name=model_name,
                                                  api_key=config["api_key"],
                                                  base_url=config["base_url"])
            if success:
                print(
                    f"✅ 成功配置 {model_info[model_name]['name']} ({model_name})")
            else:
                print(
                    f"❌ 配置 {model_info[model_name]['name']} ({model_name}) 失败")
    else:
        print("\n未配置任何模型。请确保输入了API密钥或设置了环境变量。")
        return

    # 设置默认模型
    if api_keys:
        if test_mode:
            # 测试模式：自动选择第一个模型作为默认模型
            default_model = list(api_keys.keys())[0]
            print(
                f"\n测试模式：自动设置默认模型为：{model_info[default_model]['name']} ({default_model})"
            )
        else:
            print("\n可选：设置默认模型")
            print("已配置的模型：")
            for i, (model_name, info) in enumerate(api_keys.items(), 1):
                print(f"{i}. {model_info[model_name]['name']} ({model_name})")

            default_choice = input("\n请输入默认模型的序号（按Enter使用第一个配置的模型）: ")

            if default_choice and default_choice.isdigit():
                default_index = int(default_choice) - 1
                if 0 <= default_index < len(api_keys):
                    default_model = list(api_keys.keys())[default_index]
                else:
                    print("输入的序号无效，使用第一个配置的模型")
                    default_model = list(api_keys.keys())[0]
            else:
                default_model = list(api_keys.keys())[0]

        if llm_config.set_default_model(default_model):
            print(
                f"✅ 成功设置默认模型: {model_info[default_model]['name']} ({default_model})"
            )
        else:
            print(f"❌ 设置默认模型 {default_model} 失败")

    # 打印配置结果
    print("\n📊 配置结果:")
    all_configs = llm_config.get_all_model_configs()
    for model_name, config in all_configs.items():
        if model_name in model_info:
            print(
                f"- {model_info[model_name]['name']} ({model_name}): API Key已加密配置"
            )
        else:
            print(f"- {model_name}: API Key已加密配置")

    default_model = llm_config.get_default_model()
    if default_model and default_model in model_info:
        print(
            f"\n📌 默认模型: {model_info[default_model]['name']} ({default_model})")
    else:
        print(f"\n📌 默认模型: {default_model}")

    print("\n🔒 注意：所有API密钥已加密存储在配置文件中")
    print("\n环境变量说明：")
    for model_name, info in model_info.items():
        print(f"- {info['name']}: {info['env_var']}")
    print("\n下次运行时可通过设置这些环境变量来自动配置API密钥。")


if __name__ == "__main__":
    # 检查是否有测试模式参数
    test_mode = len(sys.argv) > 1 and sys.argv[1] == "--test"
    init_llm_config(test_mode)
