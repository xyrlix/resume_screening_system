#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试NER模型功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath('.'))

from core.ner_model import get_ner


def test_ner_model():
    """测试NER模型是否能正常加载和工作"""
    # 打开文件用于写入测试结果
    with open('ner_test_result.txt', 'w', encoding='utf-8') as f:
        # 重定向标准输出到文件
        original_stdout = sys.stdout
        sys.stdout = f
        
        try:
            print("开始测试NER模型...")
            
            # 获取NER模型实例
            ner = get_ner()
            
            if not ner:
                print("[ERROR] NER模型加载失败")
                return False
            
            print("[OK] NER模型加载成功")
            
            # 测试中文文本
            chinese_text = "张三在腾讯科技有限公司担任高级软件工程师，负责Python开发"
            print(f"测试中文文本: {chinese_text}")
            
            result_chinese = ner.predict(chinese_text)
            print(f"中文识别结果: {result_chinese}")
            
            if result_chinese:
                print("[OK] 中文实体识别成功")
            else:
                print("[WARNING] 中文实体识别结果为空")
            
            # 测试英文文本
            english_text = "John works at Google as a Software Engineer in Mountain View"
            print(f"测试英文文本: {english_text}")
            
            result_english = ner.predict(english_text)
            print(f"英文识别结果: {result_english}")
            
            if result_english:
                print("[OK] 英文实体识别成功")
            else:
                print("[WARNING] 英文实体识别结果为空")
            
            print("\nNER模型测试完成！")
            return True
        finally:
            # 恢复标准输出
            sys.stdout = original_stdout
    
    # 读取并打印测试结果
    with open('ner_test_result.txt', 'r', encoding='utf-8') as f:
        result = f.read()
        print(result)


if __name__ == "__main__":
    test_ner_model()
