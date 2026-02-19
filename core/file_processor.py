#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件处理模块

负责处理不同格式的文件，包括可编辑PDF/Word、扫描件/图片简历和Excel表格简历
"""

import os
import tempfile
from typing import Tuple, List, Dict, Any
from PIL import Image, ImageEnhance, ImageOps


class FileProcessor:
    """
    文件处理类，支持多种格式的文件处理
    """

    def __init__(self):
        """
        初始化文件处理器
        """
        self.supported_formats = {
            'pdf': 'PDF文件',
            'doc': 'Word文件',
            'docx': 'Word文件',
            'jpg': '图片文件',
            'jpeg': '图片文件',
            'png': '图片文件',
            'bmp': '图片文件',
            'tiff': '图片文件',
            'tif': '图片文件',
            'xls': 'Excel文件',
            'xlsx': 'Excel文件',
            'md': 'Markdown文件',
            'txt': 'Text文件'
        }
        self._ocr_instance = None  # 缓存OCR实例
        self._file_cache = {}  # 文件内容缓存

    def get_file_type(self, file_path: str) -> str:
        """
        获取文件类型
        
        Args:
            file_path: 文件路径
        
        Returns:
            文件类型
        """
        ext = os.path.splitext(file_path)[1].lower()[1:]  # 去除点号
        return self.supported_formats.get(ext, '未知文件类型')

    def is_supported(self, file_path: str) -> bool:
        """
        检查文件是否支持
        
        Args:
            file_path: 文件路径
        
        Returns:
            是否支持
        """
        ext = os.path.splitext(file_path)[1].lower()[1:]  # 去除点号
        return ext in self.supported_formats

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        处理文件，根据文件类型选择不同的处理方式，使用缓存加速
        
        Args:
            file_path: 文件路径
        
        Returns:
            处理结果
        """
        # 检查缓存
        if file_path in self._file_cache:
            return self._file_cache[file_path]

        if not self.is_supported(file_path):
            result = {
                'status': 'error',
                'message': f'不支持的文件类型: {os.path.splitext(file_path)[1]}',
                'content': '',
                'file_type': 'unknown'
            }
            self._file_cache[file_path] = result
            return result

        file_type = self.get_file_type(file_path)

        try:
            if file_type == 'PDF文件':
                content = self._process_pdf(file_path)
            elif file_type == 'Word文件':
                content = self._process_word(file_path)
            elif file_type == '图片文件':
                content = self._process_image(file_path)
            elif file_type == 'Excel文件':
                content = self._process_excel(file_path)
            elif file_type == 'Markdown文件':
                content = self._process_markdown(file_path)
            elif file_type == 'Text文件':
                content = self._process_text(file_path)
            else:
                content = ''

            result = {
                'status': 'success',
                'message': f'文件处理成功',
                'content': content,
                'file_type': file_type
            }
            self._file_cache[file_path] = result
            return result
        except Exception as e:
            result = {
                'status': 'error',
                'message': f'文件处理失败: {str(e)}',
                'content': '',
                'file_type': file_type
            }
            self._file_cache[file_path] = result
            return result

    def _process_pdf(self, file_path: str) -> str:
        """
        处理PDF文件
        
        Args:
            file_path: PDF文件路径
        
        Returns:
            提取的文本内容
        """
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
            return text.strip()
        except ImportError:
            # 如果PyPDF2未安装，返回空字符串
            return ''
        except Exception as e:
            # 如果PDF是扫描件，尝试使用OCR
            return self._process_image(file_path)

    def _process_word(self, file_path: str) -> str:
        """
        处理Word文件
        
        Args:
            file_path: Word文件路径
        
        Returns:
            提取的文本内容
        """
        try:
            from docx import Document
            doc = Document(file_path)
            text = ''
            for para in doc.paragraphs:
                text += para.text + '\n'
            return text.strip()
        except ImportError:
            # 如果python-docx未安装，返回空字符串
            return ''

    def _process_image(self, file_path: str) -> str:
        """
        处理图片文件，使用OCR提取文本，优化性能
        
        Args:
            file_path: 图片文件路径
        
        Returns:
            提取的文本内容
        """
        try:
            print(f"[LOG] 🔄 正在处理图片文件: {os.path.basename(file_path)}")

            # 检查OCR结果缓存
            ocr_cache_key = f"ocr:{file_path}"
            if ocr_cache_key in self._file_cache:
                print(f"[LOG] ✅ 从缓存获取OCR结果")
                return self._file_cache[ocr_cache_key]

            # 图像增强（简化处理，减少时间）
            print(f"[LOG] 📸 正在进行图像增强...")
            enhanced_image = self._enhance_image(file_path)

            # 保存增强后的图像到临时文件
            with tempfile.NamedTemporaryFile(suffix='.png',
                                             delete=False) as temp_file:
                enhanced_image.save(temp_file, format='PNG')
                temp_file_path = temp_file.name

            # 使用缓存的OCR实例，避免重复初始化
            if self._ocr_instance is None:
                print(f"[LOG] 📦 正在初始化OCR模型...")
                from paddleocr import PaddleOCR
                # 简化OCR配置，提高速度
                self._ocr_instance = PaddleOCR(use_angle_cls=True,
                                               lang='ch',
                                               show_log=False)

            # 执行OCR
            print(f"[LOG] 正在执行OCR识别...")
            result = self._ocr_instance.ocr(temp_file_path, cls=True)

            # 清理临时文件
            os.unlink(temp_file_path)

            # 提取文本
            text = ''
            for line in result:
                for word_info in line:
                    text += word_info[1][0] + ' '

            # 缓存OCR结果
            self._file_cache[ocr_cache_key] = text.strip()
            print(f"[LOG] OCR处理完成，提取文本长度: {len(text.strip())} 字符")
            return text.strip()
        except ImportError:
            print(f"[LOG] PaddleOCR未安装，无法处理图片文件")
            # 如果PaddleOCR未安装，返回空字符串
            return ''
        except Exception as e:
            print(f"[LOG] OCR处理失败: {str(e)}")
            # 处理失败，返回空字符串
            return ''

    def _process_excel(self, file_path: str) -> str:
        """
        处理Excel文件
        
        Args:
            file_path: Excel文件路径
        
        Returns:
            提取的文本内容
        """
        try:
            import pandas as pd
            # 读取所有工作表
            excel_file = pd.ExcelFile(file_path)
            text = ''

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # 转换为字段-值格式
                for _, row in df.iterrows():
                    for col, value in row.items():
                        if pd.notna(value):
                            text += f'{col}: {value}\n'
                text += '\n'

            return text.strip()
        except ImportError:
            # 如果pandas未安装，返回空字符串
            return ''

    def _enhance_image(self, file_path: str) -> Image.Image:
        """
        图像增强，提高OCR识别率，简化处理步骤以提高速度
        
        Args:
            file_path: 图片文件路径
        
        Returns:
            增强后的图像
        """
        # 打开图像
        img = Image.open(file_path)

        # 简化图像增强：只保留必要的步骤
        # 转换为灰度图
        img = ImageOps.grayscale(img)

        # 仅调整对比度，简化处理
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)  # 降低对比度增强强度，提高速度

        return img

    def _process_markdown(self, file_path: str) -> str:
        """
        处理Markdown文件
        
        Args:
            file_path: Markdown文件路径
        
        Returns:
            提取的文本内容
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content.strip()
        except Exception as e:
            return ''

    def _process_text(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content.strip()
        except Exception as e:
            return ''

    def process_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        批量处理文件
        
        Args:
            file_paths: 文件路径列表
        
        Returns:
            处理结果列表
        """
        results = []
        for file_path in file_paths:
            results.append(self.process_file(file_path))
        return results

    def process_bytes(self, file_bytes: bytes,
                      file_name: str) -> Dict[str, Any]:
        """
        处理文件字节流
        
        Args:
            file_bytes: 文件字节流
            file_name: 文件名
        
        Returns:
            处理结果
        """
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_name)[1],
                                         delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        # 处理文件
        result = self.process_file(temp_file_path)

        # 清理临时文件
        os.unlink(temp_file_path)

        return result

    def get_supported_formats_info(self) -> Dict[str, str]:
        """
        获取支持的文件格式信息
        
        Returns:
            支持的文件格式信息
        """
        return self.supported_formats
