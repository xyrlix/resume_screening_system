#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用模型管理工具

支持添加、更新、验证和删除LLM模型配置
支持管理模型名称映射和支持的模型列表
用法示例：
  python manage_models.py add model_name api_key base_url
  python manage_models.py update model_name --api_key new_api_key --base_url new_base_url
  python manage_models.py delete model_name
  python manage_models.py list
  python manage_models.py verify model_name
  python manage_models.py mapping add short_name full_name
  python manage_models.py mapping list
  python manage_models.py mapping delete short_name
  python manage_models.py supported list
  python manage_models.py supported add model_name
  python manage_models.py supported delete model_name
"""

import argparse
import json
import sys
from typing import Optional, Dict, Any

# 添加项目根目录到Python路径
sys.path.append('.')

from core.llm_config_manager import LLMConfigManager
from utils.logger import get_logger

logger = get_logger(__name__)


def add_model(args: argparse.Namespace) -> None:
    """
    添加新模型
    """
    manager = LLMConfigManager()

    # 验证必填参数
    if not args.model_name or not args.api_key:
        logger.error("添加模型时，模型名称和API密钥为必填参数")
        return

    # 检查模型是否已存在
    if manager.is_model_configured(args.model_name):
        logger.warning(f"模型 {args.model_name} 已存在，将更新配置")

    # 添加模型配置
    success = manager.set_model_config(
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        domestic_base_url=args.domestic_base_url,
        international_base_url=args.international_base_url)

    if success:
        logger.info(f"模型 {args.model_name} 添加/更新成功")
    else:
        logger.error(f"模型 {args.model_name} 添加/更新失败")


def update_model(args: argparse.Namespace) -> None:
    """
    更新现有模型
    """
    manager = LLMConfigManager()

    # 检查模型是否存在
    if not manager.is_model_configured(args.model_name):
        logger.error(f"模型 {args.model_name} 不存在")
        return

    # 获取现有配置
    model_configs = manager.get_all_model_configs()
    existing_config = model_configs.get(args.model_name, {})

    # 准备更新参数
    update_args = {
        'model_name':
        args.model_name,
        'api_key':
        args.api_key
        if args.api_key is not None else existing_config.get('api_key'),
        'base_url':
        args.base_url
        if args.base_url is not None else existing_config.get('base_url'),
        'domestic_base_url':
        args.domestic_base_url if args.domestic_base_url is not None else
        existing_config.get('domestic_base_url'),
        'international_base_url':
        args.international_base_url if args.international_base_url is not None
        else existing_config.get('international_base_url')
    }

    # 更新模型配置
    success = manager.set_model_config(**update_args)

    if success:
        logger.info(f"模型 {args.model_name} 更新成功")
    else:
        logger.error(f"模型 {args.model_name} 更新失败")


def delete_model(args: argparse.Namespace) -> None:
    """
    删除模型
    """
    manager = LLMConfigManager()

    # 检查模型是否存在
    if not manager.is_model_configured(args.model_name):
        logger.error(f"模型 {args.model_name} 不存在")
        return

    # 删除模型配置
    success = manager.delete_model_config(args.model_name)

    if success:
        logger.info(f"模型 {args.model_name} 删除成功")
    else:
        logger.error(f"模型 {args.model_name} 删除失败")


def list_models(args: argparse.Namespace) -> None:
    """
    列出所有模型
    """
    manager = LLMConfigManager()
    model_configs = manager.get_all_model_configs()
    model_mappings = manager.get_model_mappings()

    if not model_configs:
        logger.info("没有配置任何模型")
    else:
        logger.info(f"\n当前已配置的模型列表 ({len(model_configs)}):")
        logger.info("-" * 100)

        for idx, (model_name, config) in enumerate(model_configs.items(), 1):
            logger.info(f"\n{idx}. {model_name}")
            logger.info(f"   API密钥: {'已配置' if 'api_key' in config else '未配置'}")
            logger.info(f"   Base URL: {config.get('base_url', '未配置')}")
            if 'domestic_base_url' in config:
                logger.info(f"   国内Base URL: {config['domestic_base_url']}")
            if 'international_base_url' in config:
                logger.info(
                    f"   国际Base URL: {config['international_base_url']}")
            logger.info(
                f"   区域: {'国内' if config.get('is_domestic', False) else '国际'}")
            logger.info(
                f"   提示词模板: {'已配置' if config.get('default_prompt') else '未配置'} 个"
            )

    # 列出模型名称映射
    logger.info(f"\n\n当前模型名称映射 ({len(model_mappings)}):")
    logger.info("-" * 100)

    if not model_mappings:
        logger.info("没有配置任何模型名称映射")
    else:
        for idx, (short_name, full_name) in enumerate(model_mappings.items(),
                                                      1):
            logger.info(f"{idx}. {short_name} -> {full_name}")


def verify_model(args: argparse.Namespace) -> None:
    """
    验证模型配置
    """
    manager = LLMConfigManager()

    # 检查模型是否存在
    if not manager.is_model_configured(args.model_name):
        logger.error(f"模型 {args.model_name} 不存在")
        return

    # 获取模型配置
    model_configs = manager.get_all_model_configs()
    config = model_configs.get(args.model_name, {})

    logger.info(f"\n模型 {args.model_name} 配置验证:")
    logger.info("-" * 50)

    # 验证基本配置项
    checks = [
        ("模型名称", args.model_name, True),
        ("API密钥", 'api_key' in config, True),
        ("Base URL", 'base_url' in config, False),
        ("国内Base URL", 'domestic_base_url' in config, False),
        ("国际Base URL", 'international_base_url' in config, False),
    ]

    all_passed = True
    for item, value, is_required in checks:
        status = "✓" if (value if is_required else True) else "✗"
        logger.info(f"{status} {item}: {'已配置' if value else '未配置'}")
        if is_required and not value:
            all_passed = False

    if all_passed:
        logger.info("\n验证结果: 配置完整，可以正常使用")
    else:
        logger.warning("\n验证结果: 缺少必要配置，请补充")


def export_config(args: argparse.Namespace) -> None:
    """
    导出模型配置
    """
    manager = LLMConfigManager()
    model_configs = manager.get_all_model_configs()
    model_mappings = manager.get_model_mappings()
    supported_models = manager.get_supported_models()

    if not model_configs:
        logger.error("没有配置任何模型")
        return

    # 准备导出数据（不包含敏感信息）
    export_data = {
        'models': {},
        'model_mappings': model_mappings,
        'supported_models': supported_models
    }

    for model_name, config in model_configs.items():
        export_data['models'][model_name] = {
            'base_url': config.get('base_url'),
            'domestic_base_url': config.get('domestic_base_url'),
            'international_base_url': config.get('international_base_url'),
            'is_domestic': config.get('is_domestic', False),
            # 不包含API密钥和提示词模板
        }

    # 写入文件
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        logger.info(f"模型配置已导出到 {args.output_file}")
    except Exception as e:
        logger.error(f"导出配置失败: {str(e)}")


def add_mapping(args: argparse.Namespace) -> None:
    """
    添加模型名称映射
    """
    manager = LLMConfigManager()

    success = manager.set_model_mapping(args.short_name, args.full_name)

    if success:
        logger.info(f"✅ 模型名称映射添加成功: {args.short_name} -> {args.full_name}")
    else:
        logger.error(f"❌ 模型名称映射添加失败: {args.short_name} -> {args.full_name}")


def list_mappings(args: argparse.Namespace) -> None:
    """
    列出所有模型名称映射
    """
    manager = LLMConfigManager()
    mappings = manager.get_model_mappings()

    if not mappings:
        logger.info("没有配置任何模型名称映射")
        return

    logger.info(f"\n当前模型名称映射列表 ({len(mappings)}):")
    logger.info("-" * 60)

    for short_name, full_name in mappings.items():
        logger.info(f"{short_name} -> {full_name}")


def delete_mapping(args: argparse.Namespace) -> None:
    """
    删除模型名称映射
    """
    manager = LLMConfigManager()

    success = manager.delete_model_mapping(args.short_name)

    if success:
        logger.info(f"✅ 模型名称映射删除成功: {args.short_name}")
    else:
        logger.error(f"❌ 模型名称映射删除失败: {args.short_name}")


def list_supported_models(args: argparse.Namespace) -> None:
    """
    列出所有支持的模型
    """
    manager = LLMConfigManager()
    supported_models = manager.get_supported_models()

    logger.info(f"\n当前支持的模型列表 ({len(supported_models)}):")
    logger.info("-" * 60)

    for idx, model_name in enumerate(supported_models, 1):
        logger.info(f"{idx}. {model_name}")


def add_supported_model(args: argparse.Namespace) -> None:
    """
    添加支持的模型
    """
    manager = LLMConfigManager()

    # 获取当前支持的模型
    supported_models = manager.get_supported_models()

    if args.model_name in supported_models:
        logger.warning(f"模型 {args.model_name} 已经在支持列表中")
        return

    # 更新配置
    if 'supported_models' not in manager.config:
        manager.config['supported_models'] = []

    manager.config['supported_models'].append(args.model_name)
    manager._save_config()

    logger.info(f"✅ 模型 {args.model_name} 已添加到支持列表")


def delete_supported_model(args: argparse.Namespace) -> None:
    """
    删除支持的模型
    """
    manager = LLMConfigManager()

    # 获取当前支持的模型
    if 'supported_models' not in manager.config:
        logger.error(f"模型 {args.model_name} 不在支持列表中")
        return

    if args.model_name not in manager.config['supported_models']:
        logger.error(f"模型 {args.model_name} 不在支持列表中")
        return

    # 更新配置
    manager.config['supported_models'].remove(args.model_name)
    manager._save_config()

    logger.info(f"✅ 模型 {args.model_name} 已从支持列表中删除")


def main() -> None:
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="通用模型管理工具")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 添加模型命令
    add_parser = subparsers.add_parser('add', help="添加新模型")
    add_parser.add_argument('model_name', help="模型名称")
    add_parser.add_argument('api_key', help="API密钥")
    add_parser.add_argument('base_url', help="基础URL")
    add_parser.add_argument('--domestic_base_url', help="国内基础URL")
    add_parser.add_argument('--international_base_url', help="国际基础URL")
    # 注意：is_domestic参数暂不支持，将在后续版本添加
    add_parser.set_defaults(func=add_model)

    # 更新模型命令
    update_parser = subparsers.add_parser('update', help="更新现有模型")
    update_parser.add_argument('model_name', help="模型名称")
    update_parser.add_argument('--api_key', help="API密钥（可选）")
    update_parser.add_argument('--base_url', help="基础URL（可选）")
    update_parser.add_argument('--domestic_base_url', help="国内基础URL（可选）")
    update_parser.add_argument('--international_base_url', help="国际基础URL（可选）")
    # 注意：is_domestic参数暂不支持，将在后续版本添加
    update_parser.set_defaults(func=update_model)

    # 删除模型命令
    delete_parser = subparsers.add_parser('delete', help="删除模型")
    delete_parser.add_argument('model_name', help="模型名称")
    delete_parser.set_defaults(func=delete_model)

    # 列出模型命令
    list_parser = subparsers.add_parser('list', help="列出所有模型")
    list_parser.set_defaults(func=list_models)

    # 验证模型命令
    verify_parser = subparsers.add_parser('verify', help="验证模型配置")
    verify_parser.add_argument('model_name', help="模型名称")
    verify_parser.set_defaults(func=verify_model)

    # 导出配置命令
    export_parser = subparsers.add_parser('export', help="导出模型配置")
    export_parser.add_argument('--output_file',
                               default='model_configs_export.json',
                               help="导出文件名")
    export_parser.set_defaults(func=export_config)

    # 模型映射子命令
    mapping_parser = subparsers.add_parser('mapping', help="管理模型名称映射")
    mapping_subparsers = mapping_parser.add_subparsers(dest="mapping_command",
                                                       help="模型映射命令")

    # 添加映射命令
    mapping_add_parser = mapping_subparsers.add_parser('add', help="添加模型名称映射")
    mapping_add_parser.add_argument('short_name', help="简短名称")
    mapping_add_parser.add_argument('full_name', help="完整名称")
    mapping_add_parser.set_defaults(func=add_mapping)

    # 列出映射命令
    mapping_list_parser = mapping_subparsers.add_parser('list',
                                                        help="列出所有模型名称映射")
    mapping_list_parser.set_defaults(func=list_mappings)

    # 删除映射命令
    mapping_delete_parser = mapping_subparsers.add_parser('delete',
                                                          help="删除模型名称映射")
    mapping_delete_parser.add_argument('short_name', help="简短名称")
    mapping_delete_parser.set_defaults(func=delete_mapping)

    # 支持的模型子命令
    supported_parser = subparsers.add_parser('supported', help="管理支持的模型列表")
    supported_subparsers = supported_parser.add_subparsers(
        dest="supported_command", help="支持的模型命令")

    # 列出支持的模型命令
    supported_list_parser = supported_subparsers.add_parser('list',
                                                            help="列出所有支持的模型")
    supported_list_parser.set_defaults(func=list_supported_models)

    # 添加支持的模型命令
    supported_add_parser = supported_subparsers.add_parser('add',
                                                           help="添加支持的模型")
    supported_add_parser.add_argument('model_name', help="模型名称")
    supported_add_parser.set_defaults(func=add_supported_model)

    # 删除支持的模型命令
    supported_delete_parser = supported_subparsers.add_parser('delete',
                                                              help="删除支持的模型")
    supported_delete_parser.add_argument('model_name', help="模型名称")
    supported_delete_parser.set_defaults(func=delete_supported_model)

    # 解析参数
    args = parser.parse_args()

    # 执行相应命令
    if args.command:
        if args.command == 'mapping':
            if args.mapping_command:
                args.func(args)
            else:
                mapping_parser.print_help()
        elif args.command == 'supported':
            if args.supported_command:
                args.func(args)
            else:
                supported_parser.print_help()
        else:
            args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
