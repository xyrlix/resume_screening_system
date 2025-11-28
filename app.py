#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能简历筛选系统主入口

功能：
- 一键启动智能简历筛选系统
- 支持配置启动端口
- 提供系统信息和帮助
"""

import os
import sys
import argparse
import subprocess
import time

# 系统信息
SYSTEM_INFO = {
    "name": "智能简历筛选系统",
    "version": "2.0.0",
    "description": "基于BGE-M3和多LLM链式分析的智能简历筛选系统",
    "authors": "xyrlix",
    "copyright": "© 2025 Resume Screening System"
}

# 启动命令模板
MAIN_COMMAND = [
    sys.executable, "-m", "streamlit", "run", "frontend/combined_app.py"
]


def print_system_info():
    """
    打印系统信息
    """
    print(f"\n{SYSTEM_INFO['name']} v{SYSTEM_INFO['version']}")
    print(SYSTEM_INFO['description'])
    print(f"作者: {SYSTEM_INFO['authors']}")
    print(f"版权: {SYSTEM_INFO['copyright']}")
    print()


def start_frontend(port: int = None):
    """
    启动前端界面
    
    Args:
        port: 端口号
    """
    # 如果没有指定端口，使用默认端口8501
    if not port:
        port = 8501

    # 尝试启动应用，如果失败，则尝试其他端口
    for try_port in range(port, port + 10):
        command = MAIN_COMMAND.copy()
        command.extend(["--server.port", str(try_port)])

        print(f"🚀 启动智能简历筛选系统...")
        print(f"📦 命令: {' '.join(command)}")
        print()

        try:
            # 使用PIPE捕获输出
            result = subprocess.run(command,
                                    check=False,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)

            # 检查返回码
            if result.returncode == 0:
                return
            else:
                # 检查输出中是否包含端口被占用
                output = result.stdout + result.stderr
                if "Port" in output and "is already in use" in output:
                    print(f"⚠ 端口 {try_port} 已被占用，正在尝试端口 {try_port + 1}...")
                    print()
                else:
                    # 其他错误，打印输出
                    print(f"❌ 启动智能简历筛选系统失败")
                    print(f"输出: {output}")
                    return
        except KeyboardInterrupt:
            print(f"\n✅ 智能简历筛选系统已停止")
            return
        except Exception as e:
            # 其他异常，直接返回
            print(f"❌ 启动智能简历筛选系统失败: {e}")
            return

    # 尝试了10个端口都失败了
    print(f"❌ 尝试了端口 {port} 到 {port + 9}，都已被占用")
    return


def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description=SYSTEM_INFO['description'])

    # 添加命令行参数
    parser.add_argument("--port",
                        type=int,
                        default=8501,
                        help="指定前端服务端口号，默认8501")
    parser.add_argument("--info", action="store_true", help="显示系统信息")

    # 解析参数
    args = parser.parse_args()

    # 显示系统信息
    print_system_info()

    # 处理命令
    if args.info:
        # 已经显示了系统信息，直接返回
        return
    else:
        # 启动智能简历筛选系统
        start_frontend(args.port)


if __name__ == "__main__":
    main()
