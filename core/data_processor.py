#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据收集与处理模块

负责简历和JD的数据收集、初步处理和清洗
"""

import os
# 使用HF镜像源解决下载问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # 允许联网下载
os.environ["HF_HUB_OFFLINE"] = "0"  # 允许HF Hub联网
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '30'  # 超时时间30秒
os.environ['HF_HUB_RETRY'] = '5'  # 重试5次
os.environ['HF_HUB_RETRY_DELAY'] = '2'  # 重试间隔2秒

import json
import re
from typing import List, Dict, Any, Optional, Pattern
from utils.logger import get_logger
from utils.clean_tools import clean_text as utils_clean_text, remove_special_chars, normalize_whitespace

# 初始化日志记录器
logger = get_logger("data_processor")


# 配置管理器
class ConfigManager:
    """
    配置管理类，负责加载和管理各种配置文件
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.base_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        self.config_dir = os.path.join(self.base_dir, 'config')

        # 加载配置文件
        self.prompts = self._load_prompts()
        self.entity_config = self._load_entity_config()

        self._initialized = True

    def _load_prompts(self) -> Dict[str, str]:
        """加载提示词配置"""
        prompts_path = os.path.join(self.config_dir, 'prompts.json')
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"提示词文件未找到: {prompts_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"提示词JSON解析错误: {prompts_path}")
            raise

    def _load_entity_config(self) -> Dict[str, List[str]]:
        """加载实体配置"""
        config_path = os.path.join(self.config_dir, 'entity_config.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载实体配置失败: {str(e)}")
            # 返回默认配置
            return {
                'resume_entities': [
                    "姓名", "性别", "出生日期", "年龄", "联系电话", "电子邮箱", "现居地", "户籍地",
                    "政治面貌", "婚姻状况", "期望职位", "期望行业", "期望工作城市", "期望薪资", "到岗时间",
                    "学校名称", "学历层次", "专业名称", "入学时间", "毕业时间", "是否全日制",
                    "GPA/排名/荣誉", "公司名称", "公司所属行业", "在职开始时间", "在职结束时间", "职位名称",
                    "工作地点", "工作内容关键词", "汇报对象", "团队规模", "项目名称", "项目开始时间",
                    "项目结束时间", "项目角色", "技术栈", "项目成果指标", "编程语言", "框架/工具", "数据库",
                    "操作系统/平台", "语言能力", "证书资质", "软技能", "作品集/个人链接", "自我评价关键词",
                    "兴趣爱好", "总工作经验年限"
                ],
                'jd_entities': [
                    "职位名称", "行业", "岗位职责", "任职要求", "学历要求", "工作年限要求", "薪资范围",
                    "工作地点", "公司名称", "公司规模", "公司行业", "招聘人数", "发布时间", "截止时间",
                    "职位类型", "技能要求", "语言要求", "证书要求", "福利", "团队情况"
                ]
            }

    @property
    def resume_entities(self) -> List[str]:
        """获取简历实体列表"""
        return self.entity_config.get('resume_entities', [])

    @property
    def jd_entities(self) -> List[str]:
        """获取JD实体列表"""
        return self.entity_config.get('jd_entities', [])


# 文本处理工具
class TextProcessor:
    """
    文本处理工具类，提供文本清理、解析等功能
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """
        清理文本，去除多余的空格、换行符和特殊字符
        使用utils.clean_tools中的实现以避免代码重复
        
        Args:
            text: 原始文本
        
        Returns:
            清理后的文本
        """
        return utils_clean_text(text)

    @staticmethod
    def parse_text_file(file_path: str) -> str:
        """
        解析文本文件，提取文本内容
        
        Args:
            file_path: 文件路径
        
        Returns:
            提取的文本内容
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.txt', '.md']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"解析文件失败 {file_path}: {e}")
                return ""
        else:
            # 对于其他格式，返回空字符串
            logger.warning(f"不支持的文件格式: {ext}")
            return ""


# 实体验证器
class EntityValidator:
    """
    实体验证器，负责验证提取的实体是否有效
    """

    @staticmethod
    def is_valid_entity(label: str, value: Any) -> bool:
        """
        验证实体是否有效
        
        Args:
            label: 实体标签
            value: 实体值
        
        Returns:
            bool: 实体是否有效
        """
        if not value or len(str(value).strip()) < 2:
            return False

        value_str = str(value).strip()

        # 根据不同实体类型进行特定验证
        validation_rules = {
            # 经验/年限相关
            "EXPERIENCE":
            lambda v: bool(re.search(r"\d", v)),
            "Years":
            lambda v: bool(re.search(r"\d", v)),
            "总工作经验年限":
            lambda v: bool(re.search(r"\d", v)),
            "工作年限要求":
            lambda v: bool(re.search(r"\d", v)),

            # 联系方式
            "PHONE":
            lambda v: bool(re.fullmatch(r"1[3-9]\d{9}", v)),
            "Phone":
            lambda v: bool(re.fullmatch(r"1[3-9]\d{9}", v)),
            "联系电话":
            lambda v: bool(re.fullmatch(r"1[3-9]\d{9}", v)),

            # 邮箱
            "EMAIL":
            lambda v: bool(
                re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                             v)),
            "Email":
            lambda v: bool(
                re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                             v)),
            "电子邮箱":
            lambda v: bool(
                re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                             v)),

            # 学历
            "EDUCATION":
            lambda v: bool(re.search(r"博士|硕士|本科|大专|中专|高中|初中", v)),
            "Degree":
            lambda v: bool(re.search(r"博士|硕士|本科|大专|中专|高中|初中", v)),
            "学历层次":
            lambda v: bool(re.search(r"博士|硕士|本科|大专|中专|高中|初中", v)),
            "学历要求":
            lambda v: bool(re.search(r"博士|硕士|本科|大专|中专|高中|初中", v)),

            # 语言
            "LANGUAGE":
            lambda v: bool(
                re.search(r"英语|雅思|托福|CET|日语|德语|法语|韩语|西班牙语|葡萄牙语|俄语|意大利语", v)),
            "Language":
            lambda v: bool(
                re.search(r"英语|雅思|托福|CET|日语|德语|法语|韩语|西班牙语|葡萄牙语|俄语|意大利语", v)),
            "语言能力":
            lambda v: bool(
                re.search(r"英语|雅思|托福|CET|日语|德语|法语|韩语|西班牙语|葡萄牙语|俄语|意大利语", v)),
            "语言要求":
            lambda v: bool(
                re.search(r"英语|雅思|托福|CET|日语|德语|法语|韩语|西班牙语|葡萄牙语|俄语|意大利语", v)),

            # 证书
            "CERTIFICATE":
            lambda v: bool(
                re.search(
                    r"PMP|CKA|CKAD|RHCE|AWS|微软|Oracle|思科|华为|软考|CCNA|CCNP|CFA|CPA|ACCA|FRM|证书",
                    v)),
            "Certificate":
            lambda v: bool(
                re.search(
                    r"PMP|CKA|CKAD|RHCE|AWS|微软|Oracle|思科|华为|软考|CCNA|CCNP|CFA|CPA|ACCA|FRM|证书",
                    v)),
            "证书资质":
            lambda v: bool(
                re.search(
                    r"PMP|CKA|CKAD|RHCE|AWS|微软|Oracle|思科|华为|软考|CCNA|CCNP|CFA|CPA|ACCA|FRM|证书",
                    v)),
            "证书要求":
            lambda v: bool(
                re.search(
                    r"PMP|CKA|CKAD|RHCE|AWS|微软|Oracle|思科|华为|软考|CCNA|CCNP|CFA|CPA|ACCA|FRM|证书",
                    v)),

            # 技能
            "SKILL":
            lambda v: bool(re.search(r"[\u4e00-\u9fa5A-Za-z]{2,}", v)) and
            not re.fullmatch(r"\d+", v),
            "Skill":
            lambda v: bool(re.search(r"[\u4e00-\u9fa5A-Za-z]{2,}", v)
                           ) and not re.fullmatch(r"\d+", v),
            "技能要求":
            lambda v: bool(re.search(r"[\u4e00-\u9fa5A-Za-z]{2,}", v)
                           ) and not re.fullmatch(r"\d+", v),
            "编程语言":
            lambda v: bool(re.search(r"[\u4e00-\u9fa5A-Za-z]{2,}", v)
                           ) and not re.fullmatch(r"\d+", v),
            "框架/工具":
            lambda v: bool(re.search(r"[\u4e00-\u9fa5A-Za-z]{2,}", v)
                           ) and not re.fullmatch(r"\d+", v),
            "数据库":
            lambda v: bool(re.search(r"[\u4e00-\u9fa5A-Za-z]{2,}", v)
                           ) and not re.fullmatch(r"\d+", v),
        }

        # 如果有特定规则，使用特定规则
        if label in validation_rules:
            return validation_rules[label](value_str)

        # 通用验证：至少包含2个汉字或英文单词
        return bool(re.search(r"[\u4e00-\u9fa5]{2,}|[A-Za-z]{2,}", value_str))


# 正则表达式模式管理器
class RegexPatternManager:
    """
    正则表达式模式管理器，负责管理和提供各种正则表达式模式
    """

    def __init__(self):
        # 预编译常用正则表达式模式
        self._patterns = {
            # 职位相关实体
            "职位名称": [
                re.compile(r'招聘职位[:：]?\s*([\u4e00-\u9fa5a-zA-Z&\-\(\)\[\]]+)'),
                re.compile(r'岗位名称[:：]?\s*([\u4e00-\u9fa5a-zA-Z&\-\(\)\[\]]+)'),
                re.compile(r'^([\u4e00-\u9fa5a-zA-Z&\-\(\)\[\]]+?)[\s\-_\|]+'),
                re.compile(r'[招骋]\s*([\u4e00-\u9fa5a-zA-Z&\-\(\)\[\]]+)'),
            ],

            # 公司相关实体
            "公司名称": [
                re.compile(
                    r'([\u4e00-\u9fa5a-zA-Z0-9&\-\(\)\[\]]+)(?:公司|企业|集团|有限公司|股份有限公司)'
                ),
            ],

            # 地点相关实体
            "工作地点": [
                re.compile(r'工作地点[:：]?\s*([\u4e00-\u9fa5a-zA-Z&\-\(\)\[\]]+)'),
                re.compile(
                    r'(北京|上海|广州|深圳|杭州|南京|武汉|成都|西安|天津|重庆|苏州|厦门|长沙|青岛|大连|宁波|济南|沈阳|哈尔滨|福州|昆明|南宁|兰州|太原|石家庄|郑州|合肥|南昌|贵阳|海口|西宁|银川|呼和浩特|乌鲁木齐|拉萨)'
                ),
            ],

            # 技能要求
            "技能要求": [
                re.compile(
                    r'技术专长[:：]?\s*(.*?)(?=\d+\.|[\u4e00-\u9fa5]{2,}[:：]|$)',
                    re.DOTALL),
                re.compile(
                    r'技能要求[:：]?\s*(.*?)(?=\d+\.|[\u4e00-\u9fa5]{2,}[:：]|$)',
                    re.DOTALL),
                re.compile(
                    r'AI技能[:：]?\s*(.*?)(?=\d+\.|[\u4e00-\u9fa5]{2,}[:：]|$)',
                    re.DOTALL),
                re.compile(r'精通[:：]?\s*(.*?)(?=\s*及|\s*或|\s*，|\s*。|$)'),
                re.compile(r'熟练[:：]?\s*(.*?)(?=\s*及|\s*或|\s*，|\s*。|$)'),
                re.compile(
                    r'(Python|PyTorch|TensorFlow|OCR|NLP|FastAPI|SQL|NoSQL|Azure|GPT|Claude|LLM|Machine Learning|Deep Learning)',
                    re.IGNORECASE),
            ],

            # 岗位职责
            "岗位职责": [
                re.compile(
                    r'文档智能处理[:：]?\s*(.*?)(?=\d+\.|[\u4e00-\u9fa5]{2,}[:：]|$)',
                    re.DOTALL),
                re.compile(
                    r'智能内容生成[:：]?\s*(.*?)(?=\d+\.|[\u4e00-\u9fa5]{2,}[:：]|$)',
                    re.DOTALL),
                re.compile(
                    r'端到端系统集成[:：]?\s*(.*?)(?=\d+\.|[\u4e00-\u9fa5]{2,}[:：]|$)',
                    re.DOTALL),
                re.compile(
                    r'岗位职责[:：]?\s*(.*?)(?=\d+\.|[\u4e00-\u9fa5]{2,}[:：]|$)',
                    re.DOTALL),
                re.compile(
                    r'工作内容[:：]?\s*(.*?)(?=\d+\.|[\u4e00-\u9fa5]{2,}[:：]|$)',
                    re.DOTALL),
            ],

            # 任职要求
            "任职要求": [
                re.compile(
                    r'必备资质与经验[:：]?\s*(.*?)(?=\d+\.|[\u4e00-\u9fa5]{2,}[:：]|$)',
                    re.DOTALL),
                re.compile(
                    r'任职要求[:：]?\s*(.*?)(?=\d+\.|[\u4e00-\u9fa5]{2,}[:：]|$)',
                    re.DOTALL),
                re.compile(
                    r'资格要求[:：]?\s*(.*?)(?=\d+\.|[\u4e00-\u9fa5]{2,}[:：]|$)',
                    re.DOTALL),
                re.compile(
                    r'岗位要求[:：]?\s*(.*?)(?=\d+\.|[\u4e00-\u9fa5]{2,}[:：]|$)',
                    re.DOTALL),
            ],
        }

    def get_patterns(self, entity_name: str) -> List[Pattern[str]]:
        """
        获取指定实体的正则表达式模式列表
        
        Args:
            entity_name: 实体名称
        
        Returns:
            List[Pattern]: 正则表达式模式列表
        """
        return self._patterns.get(entity_name, [])


# 实体提取器基类
class BaseEntityExtractor:
    """
    实体提取器基类，定义提取器接口
    """

    def extract(self, text: str, entity_list: List[str],
                entities: Dict[str, Any], stats: Dict[str, int]) -> None:
        """
        提取实体
        
        Args:
            text: 待提取的文本
            entity_list: 要提取的实体列表
            entities: 实体字典（将被更新）
            stats: 统计信息（将被更新）
        """
        raise NotImplementedError("子类必须实现此方法")


# 结构化提取器
class StructuredExtractor(BaseEntityExtractor):
    """
    结构化提取器，基于文本结构提取实体
    """

    def extract(self, text: str, entity_list: List[str],
                entities: Dict[str, Any], stats: Dict[str, int]) -> None:
        """使用结构化方法提取实体"""
        # 提取公司所属行业
        if "公司所属行业" in entity_list and not entities["公司所属行业"]:
            match = re.search(r'所属行业([^\s]+)', text)
            if match:
                industry = match.group(1).strip()
                if EntityValidator.is_valid_entity("公司所属行业", industry):
                    entities["公司所属行业"] = industry
                    stats['struct_count'] += 1
                    logger.debug(f"结构化提取: 行业 = {industry}")

        # 提取学历要求
        if "学历要求" in entity_list and not entities["学历要求"]:
            match = re.search(r'学历要求([^\s]+)', text)
            if match:
                education = match.group(1).strip()
                if EntityValidator.is_valid_entity("学历要求", education):
                    entities["学历要求"] = education
                    stats['struct_count'] += 1
                    logger.debug(f"结构化提取: 学历要求 = {education}")

        # 提取工作年限
        if "工作年限要求" in entity_list and not entities["工作年限要求"]:
            match = re.search(r'工作年限要求([^\s]+)', text)
            if match:
                experience = match.group(1).strip()
                if EntityValidator.is_valid_entity("工作年限要求", experience):
                    entities["工作年限要求"] = experience
                    stats['struct_count'] += 1
                    logger.debug(f"结构化提取: 工作年限要求 = {experience}")

        # 提取技能要求
        if "技能要求" in entity_list and not entities["技能要求"]:
            # 匹配从"技能要求"开始到下一个关键词"岗位职责"为止的所有内容
            match = re.search(r'技能要求(.*?)(?=\s*岗位职责|$)', text, re.DOTALL)
            if match:
                skills = match.group(1).strip()
                if EntityValidator.is_valid_entity("技能要求", skills):
                    entities["技能要求"] = skills
                    stats['struct_count'] += 1
                    logger.debug(f"结构化提取: 技能要求 = {skills}")

        # 提取岗位职责
        if "岗位职责" in entity_list and not entities["岗位职责"]:
            match = re.search(r'岗位职责\s*(-.*?)(?=\s*[\u4e00-\u9fa5]{2,}[:：]|$)',
                              text, re.DOTALL)
            if match:
                responsibilities = match.group(1).strip()
                if EntityValidator.is_valid_entity("岗位职责", responsibilities):
                    entities["岗位职责"] = responsibilities
                    stats['struct_count'] += 1
                    logger.debug(f"结构化提取: 岗位职责 = {responsibilities}")


# 关键词提取器
class KeywordExtractor(BaseEntityExtractor):
    """
    关键词提取器，基于通用模式提取实体
    """

    def extract(self, text: str, entity_list: List[str],
                entities: Dict[str, Any], stats: Dict[str, int]) -> None:
        """使用通用模式提取实体"""
        # 通用提取方法：基于通用模式的提取
        for entity_name in entity_list:
            if entities.get(entity_name):
                continue  # 跳过已提取的实体

            logger.debug(f"尝试使用关键词提取: {entity_name}")

            # 使用通用模式提取实体，不再依赖特定关键词
            # 构建通用匹配模式，匹配实体名后的值
            pattern = f'{entity_name}[:：]?\s*([^\s]+(?:[,，]\s*[^\s]+)*)'

            try:
                # 使用通用模式进行匹配
                match = re.search(pattern, text, re.DOTALL)
                if match and match.group(1):
                    extracted_value = match.group(1).strip()
                    if EntityValidator.is_valid_entity(entity_name,
                                                       extracted_value):
                        entities[entity_name] = extracted_value
                        stats['keyword_count'] += 1
                        logger.debug(
                            f"关键词提取: {entity_name} = {extracted_value}")
            except Exception as e:
                logger.debug(f"关键词提取时出错: {e}")
                continue  # 出错时继续


# 正则表达式提取器
class RegexExtractor(BaseEntityExtractor):
    """
    正则表达式提取器，基于正则表达式提取实体
    """

    def __init__(self):
        self._pattern_manager = RegexPatternManager()

    def extract(self, text: str, entity_list: List[str],
                entities: Dict[str, Any], stats: Dict[str, int]) -> None:
        """使用正则表达式提取实体"""
        for entity_name in entity_list:
            if entities.get(entity_name):
                continue  # 跳过已提取的实体

            logger.debug(f"尝试使用正则表达式提取: {entity_name}")

            # 获取该实体的正则表达式模式
            patterns = self._pattern_manager.get_patterns(entity_name)

            # 遍历模式进行匹配
            for pattern in patterns:
                try:
                    # 严格检查pattern是否为预编译的正则表达式对象
                    if isinstance(pattern, re.Pattern):
                        if entity_name == "技能要求":
                            # 对于技能要求，使用findall提取所有匹配项
                            matches = pattern.findall(text)
                            if matches:
                                # 扁平化匹配结果
                                flat_matches = []
                                for match in matches:
                                    if isinstance(match, tuple):
                                        flat_matches.extend(
                                            [m for m in match if m])
                                    else:
                                        flat_matches.append(match)

                                # 去重并合并
                                unique_skills = list(set(flat_matches))
                                if unique_skills:
                                    entities[entity_name] = ", ".join(
                                        unique_skills)
                                    stats['regex_count'] = stats.get(
                                        'regex_count', 0) + 1
                                    logger.debug(
                                        f"正则表达式提取: {entity_name} = {entities[entity_name]}"
                                    )
                                    break
                        else:
                            # 对于其他实体，使用search提取
                            match = pattern.search(text)
                            if match:
                                # 根据模式获取匹配的组
                                extracted_value = match.group(1).strip(
                                ) if match.groups() else match.group(
                                    0).strip()
                                if EntityValidator.is_valid_entity(
                                        entity_name, extracted_value):
                                    entities[entity_name] = extracted_value
                                    stats['regex_count'] = stats.get(
                                        'regex_count', 0) + 1
                                    logger.debug(
                                        f"正则表达式提取: {entity_name} = {extracted_value}"
                                    )
                                    break
                    else:
                        logger.error(
                            f"无效的正则表达式模式: {type(pattern)}，期望re.Pattern类型")
                        logger.error(f"模式内容: {pattern}")
                        # 跳过无效模式，继续下一个
                        continue
                except Exception as e:
                    logger.error(f"正则表达式提取{entity_name}时出错: {e}")
                    logger.error(f"错误类型: {type(e).__name__}")
                    # 添加更多调试信息
                    logger.debug(f"模式类型: {type(pattern)}, 模式内容: {pattern}")
                    logger.debug(f"文本内容(前100字符): {text[:100]}")
                    # 继续尝试下一个模式


# LLM提取器（按需导入，避免循环导入）
class LLMEntityExtractor(BaseEntityExtractor):
    """
    LLM实体提取器，基于LLM提取实体
    """

    def extract(self, text: str, entity_list: List[str],
                entities: Dict[str, Any], stats: Dict[str, int]) -> None:
        """使用LLM提取实体"""
        try:
            # 延迟导入，避免循环导入
            from core.llm_chain import LLMChain

            logger.info("🔍 开始使用LLM提取实体...")

            # 初始化LLMChain实例（单例模式）
            llm_chain = LLMChain()

            # 调用LLMChain提取实体
            llm_result = llm_chain.extract_jd_entities(text)
            logger.debug(f"LLM调用成功，返回结果长度: {len(str(llm_result))} 字符")

            # 解析LLM结果
            if llm_result:
                self._parse_llm_result(llm_result, entity_list, entities,
                                       stats)
            else:
                logger.warning("⚠️ LLM返回了空结果")

        except Exception as e:
            logger.error(f"LLM调用出错：{str(e)}")

    def _parse_llm_result(self, llm_result: Any, entity_list: List[str],
                          entities: Dict[str, Any], stats: Dict[str,
                                                                int]) -> None:
        """解析LLM返回的结果"""
        try:
            logger.debug(f"原始LLM结果: {llm_result}...")

            # 根据llm_result的类型进行不同处理
            if isinstance(llm_result, dict):
                # 如果已经是字典格式，直接使用
                logger.debug(f"LLM返回的结果已经是字典格式，包含{len(llm_result)}个实体")
                llm_entities = llm_result
            else:
                # 如果是字符串，需要清理和预处理
                processed_result = str(llm_result).strip()

                # 移除可能的代码块标记
                processed_result = re.sub(r'^```json|```$',
                                          '',
                                          processed_result,
                                          flags=re.MULTILINE)
                # 移除多余的空白字符和换行符
                processed_result = processed_result.strip()

                logger.debug(f"处理后的LLM结果: {processed_result[:200]}...")

                # 尝试解析JSON格式的结果
                if processed_result and processed_result.startswith(
                        '{') and processed_result.endswith('}'):
                    try:
                        llm_entities = json.loads(processed_result)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON解析失败: {e}，尝试修复格式")
                        # 尝试简单的格式修复
                        try:
                            # 移除可能的尾随逗号
                            processed_result = re.sub(r',\s*}', '}',
                                                      processed_result)
                            processed_result = re.sub(r',\s*\]', ']',
                                                      processed_result)
                            llm_entities = json.loads(processed_result)
                        except:
                            # 如果修复失败，尝试使用更健壮的解析方式
                            llm_entities = self._parse_as_key_value_pairs(
                                processed_result)
                else:
                    # 不是有效的JSON格式，尝试使用键值对解析
                    logger.warning(
                        f"LLM返回的结果不是有效的JSON格式: {processed_result[:100]}...")
                    llm_entities = self._parse_as_key_value_pairs(
                        processed_result)

            # 更新实体字典
            if isinstance(llm_entities, dict):
                for entity_name, value in llm_entities.items():
                    if entity_name in entity_list and value and EntityValidator.is_valid_entity(
                            entity_name, value):
                        entities[entity_name] = value
                        stats['llm_count'] += 1
                        logger.debug(f"LLM提取: {entity_name} = {value}")
            else:
                logger.warning(f"LLM返回的结果不是字典格式: {type(llm_entities)}")

        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}，尝试使用增强的文本解析方式")
            # 使用更健壮的键值对解析方式
            self._parse_as_key_value_pairs(str(llm_result), entity_list,
                                           entities, stats)
        except Exception as e:
            logger.error(f"解析LLM结果时出错: {e}")

    def _parse_as_key_value_pairs(self, text: str, entity_list: List[str],
                                  entities: Dict[str, Any],
                                  stats: Dict[str, int]) -> Dict[str, str]:
        """将文本解析为键值对"""
        result = {}

        # 首先尝试提取所有可能的JSON对象
        json_pattern = r'\{[^\{\}]*\}'
        json_matches = re.findall(json_pattern, text)

        if json_matches:
            # 尝试解析每个找到的JSON片段
            for json_str in json_matches:
                try:
                    fragment = json.loads(json_str)
                    if isinstance(fragment, dict):
                        for entity_name, value in fragment.items():
                            if entity_name in entity_list and value and EntityValidator.is_valid_entity(
                                    entity_name, value):
                                entities[entity_name] = value
                                stats['llm_count'] += 1
                                logger.debug(
                                    f"片段解析提取: {entity_name} = {value}")
                                result[entity_name] = value
                except:
                    continue

        # 最后尝试传统的键值对解析
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if ':' in line and len(line) > 3:
                try:
                    entity_name, value = line.split(':', 1)
                    entity_name = entity_name.strip()
                    value = value.strip()

                    # 清理可能的JSON格式残留
                    entity_name = re.sub(r'["\{\}\[\]]', '', entity_name)
                    value = re.sub(r'["\{\},\[\]]', '', value)

                    # 进一步清理和验证
                    entity_name = entity_name.strip()
                    value = value.strip()

                    if entity_name in entity_list and value and EntityValidator.is_valid_entity(
                            entity_name, value):
                        entities[entity_name] = value
                        stats['llm_count'] += 1
                        logger.debug(f"文本解析提取: {entity_name} = {value}")
                        result[entity_name] = value
                except:
                    continue

        return result


# NER实体提取器
class NerExtractor:
    """
    基于BERT+CRF的实体提取器，作为LLM的降级方案
    """

    def __init__(self):
        """
        初始化NER提取器
        """
        self._ner_model = None
        self._ner_loaded = False
        # 简化的标签映射，只保留最通用的映射关系
        self._label_map = {
            # 公司相关
            'CompanyName': ['公司名称', '公司'],
            'Company': ['公司名称', '公司'],
            'COMPANY': ['公司名称', '公司'],

            # 职位相关
            'JobTitle': ['职位名称', '岗位名称', '职位'],
            'POSITION': ['职位名称', '岗位名称', '职位'],

            # 教育相关
            'EducationLevel': ['学历层次', '学历'],
            'Degree': ['学历要求', '学历'],
            'EDUCATION': ['学历要求', '学历'],

            # 技能相关
            'Skill': ['技能', '技能要求'],
            'SKILL': ['技能', '技能要求'],

            # 经验相关
            'TotalWorkExperience': ['总工作经验年限'],
            'ExperienceRequirement': ['工作年限要求', '工作年限'],
            'EXPERIENCE': ['工作年限', '经验']
        }

    def _load_ner_model(self):
        """
        加载NER模型（懒加载）
        """
        if not self._ner_loaded:
            try:
                from core.ner_model import get_ner
                self._ner_model = get_ner()
                self._ner_loaded = True
                if self._ner_model:
                    logger.info("NER模型加载成功")
                else:
                    logger.warning("NER模型加载失败，返回了None值")
            except Exception as e:
                logger.warning(f"NER模型加载失败: {e}")
                self._ner_loaded = True  # 避免重复尝试加载

    def extract(self, text: str, entity_list: List[str],
                entities: Dict[str, Any], stats: Dict[str, int]):
        """
        使用NER模型提取实体
        
        Args:
            text: 待提取的文本
            entity_list: 要提取的实体列表
            entities: 已提取的实体字典（将被更新）
            stats: 提取统计信息
        """
        # 加载NER模型（懒加载）
        self._load_ner_model()

        if not self._ner_model:
            logger.warning("NER模型不可用，跳过NER提取")
            return

        try:
            # 使用NER模型预测实体
            ner_results = self._ner_model.predict(text)

            # 处理NER结果，映射到实体列表
            for entity in ner_results:
                ner_label = entity.get('label')
                ner_text = entity.get('text')

                if not ner_label or not ner_text:
                    continue

                # 处理BIO格式标签（如'B-CompanyName'、'I-CompanyName'）
                if '-' in ner_label:
                    bio_tag, entity_type = ner_label.split('-', 1)
                    # 只处理开始标签（B-），忽略内部标签（I-）以避免重复
                    if bio_tag != 'B':
                        continue
                    ner_label = entity_type

                # 将NER标签映射到系统实体名称
                # 首先尝试直接映射
                mapped = False
                for sys_entity in entity_list:
                    if sys_entity in self._label_map.get(ner_label, []):
                        # 只有在实体为空时才更新（降级策略）
                        if not entities.get(sys_entity):
                            entities[sys_entity] = ner_text
                            stats['ner_count'] = stats.get('ner_count', 0) + 1
                            logger.debug(f"NER提取: {sys_entity} = {ner_text}")
                        mapped = True
                        break

                # 如果直接映射失败，尝试反向映射（英文标签到中文实体）
                if not mapped:
                    for sys_entity in entity_list:
                        for en_label, cn_labels in self._label_map.items():
                            if sys_entity in cn_labels and en_label.lower(
                            ) == ner_label.lower():
                                if not entities.get(sys_entity):
                                    entities[sys_entity] = ner_text
                                    stats['ner_count'] = stats.get(
                                        'ner_count', 0) + 1
                                    logger.debug(
                                        f"NER提取（反向映射）: {sys_entity} = {ner_text}"
                                    )
                                mapped = True
                                break
                        if mapped:
                            break

                # 针对关键词匹配生成的实体类型进行特殊处理
                if not mapped:
                    special_mapping = {
                        'CompanyName': '公司名称',
                        'JobTitle': '职位名称',
                        'ProgrammingLanguage': '编程语言',
                        'Name': '姓名',
                        'Degree': '学历层次',
                        'Location': '工作地点',
                        'Skill': '技能要求'
                    }

                    if ner_label in special_mapping:
                        sys_entity = special_mapping[ner_label]
                        if sys_entity in entity_list and not entities.get(
                                sys_entity):
                            entities[sys_entity] = ner_text
                            stats['ner_count'] = stats.get('ner_count', 0) + 1
                            logger.debug(
                                f"NER提取（特殊映射）: {sys_entity} = {ner_text}")
        except Exception as e:
            logger.error(f"NER提取过程中发生错误: {e}")


# 实体提取协调器
class EntityExtractionCoordinator:
    """
    实体提取协调器，协调不同的提取器按顺序工作，实现降级策略
    """

    def __init__(self):
        # 初始化各种提取器
        self._extractors = [
            StructuredExtractor(),
            KeywordExtractor(),
            RegexExtractor(),
        ]

        # 初始化NER提取器（作为LLM的降级方案）
        self._ner_extractor = NerExtractor()

        # LLM提取器在需要时才初始化，避免不必要的依赖
        self._llm_extractor = None

    def extract_entities(self,
                         text: str,
                         entity_list: List[str],
                         use_llm: bool = True) -> Dict[str, Any]:
        """
        从文本中提取实体信息，实现降级策略：
        1. 优先使用LLM进行提取（如果启用）
        2. 然后使用NER模型进行补充（作为LLM的降级方案）
        3. 最后使用其他提取器（结构化、关键词、正则）进行进一步补充和验证
        
        Args:
            text: 待提取的文本
            entity_list: 要提取的实体列表
            use_llm: 是否使用LLM提取实体，默认为True
        
        Returns:
            Dict[str, Any]: 提取的实体字典
        """
        # 初始化所有实体为空
        entities = {entity: "" for entity in entity_list}

        # 记录提取过程的统计信息
        extraction_stats = {
            'llm_count': 0,
            'keyword_count': 0,
            'struct_count': 0,
            'regex_count': 0,
            'ner_count': 0
        }

        # 清理和标准化文本
        cleaned_text = TextProcessor.clean_text(text)

        # 1. 首先使用LLM提取（如果启用）
        if use_llm:
            try:
                self._llm_extractor = LLMEntityExtractor()
                self._llm_extractor.extract(cleaned_text, entity_list,
                                            entities, extraction_stats)
                logger.info(f"LLM提取完成，提取了{extraction_stats['llm_count']}个实体")
            except Exception as e:
                logger.warning(f"LLM提取失败，将使用NER作为降级方案: {e}")
                # LLM失败时，继续使用NER进行提取

        # 2. 使用NER模型进行补充（作为LLM的降级方案）
        self._ner_extractor.extract(cleaned_text, entity_list, entities,
                                    extraction_stats)
        logger.info(f"NER提取完成，提取了{extraction_stats['ner_count']}个实体")

        # 3. 最后使用其他提取器进行进一步补充和验证
        for extractor in self._extractors:
            extractor.extract(cleaned_text, entity_list, entities,
                              extraction_stats)

        logger.info(f"实体提取完成，统计信息: {extraction_stats}")
        return entities


# 主数据处理器
class DataProcessor:
    """
    数据处理类，负责简历和JD的数据收集、清洗和初步处理
    使用单例模式确保全局只有一个实例
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataProcessor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        初始化数据处理类
        """
        if self._initialized:
            return

        # 初始化各个组件
        self.config_manager = ConfigManager()
        self.text_processor = TextProcessor()
        self.entity_extractor = EntityExtractionCoordinator()

        # 获取实体配置
        self.resume_entities = self.config_manager.resume_entities
        self.jd_entities = self.config_manager.jd_entities

        logger.info(
            f"成功初始化DataProcessor，简历实体数量: {len(self.resume_entities)}，JD实体数量: {len(self.jd_entities)}"
        )

        self._initialized = True

    def parse_resume_file(self, file_path: str) -> str:
        """
        解析简历文件，提取文本内容
        
        Args:
            file_path: 简历文件路径
        
        Returns:
            提取的文本内容
        """
        return self.text_processor.parse_text_file(file_path)

    def parse_jd_file(self, file_path: str) -> str:
        """
        解析JD文件，提取文本内容
        
        Args:
            file_path: JD文件路径
        
        Returns:
            提取的文本内容
        """
        return self.text_processor.parse_text_file(file_path)

    def extract_entities(self,
                         text: str,
                         entity_list: List[str],
                         use_llm: bool = True) -> Dict[str, Any]:
        """
        从文本中提取实体信息
        
        Args:
            text: 待提取的文本
            entity_list: 要提取的实体列表
            use_llm: 是否使用LLM提取实体
        
        Returns:
            提取的实体字典
        """
        return self.entity_extractor.extract_entities(text, entity_list,
                                                      use_llm)

    def process_resume(self,
                       resume_text: str,
                       use_llm: bool = True) -> Dict[str, Any]:
        """
        处理简历文本，提取所有简历实体
        
        Args:
            resume_text: 简历文本
            use_llm: 是否使用LLM提取实体
        
        Returns:
            提取的简历实体字典
        """
        logger.info("开始处理简历...")
        return self.extract_entities(resume_text, self.resume_entities,
                                     use_llm)

    def process_jd(self, jd_text: str, use_llm: bool = True) -> Dict[str, Any]:
        """
        处理JD文本，提取所有JD实体
        
        Args:
            jd_text: JD文本
            use_llm: 是否使用LLM提取实体
        
        Returns:
            提取的JD实体字典
        """
        logger.info("开始处理JD...")
        return self.extract_entities(jd_text, self.jd_entities, use_llm)

    def process_resume_text(self,
                            resume_text: str,
                            use_llm: bool = True) -> Dict[str, Any]:
        """
        处理简历文本，提供与原始API兼容的方法
        
        Args:
            resume_text: 简历文本
            use_llm: 是否使用LLM提取实体
        
        Returns:
            处理后的简历数据，包含原始文本、清理后的文本和提取的实体
        """
        cleaned_text = self.text_processor.clean_text(resume_text)
        entities = self.process_resume(resume_text, use_llm)

        # 构建与原始API兼容的返回格式
        result = {
            "raw_text": resume_text,
            "cleaned_text": cleaned_text,
            "entities": entities
        }

        # 从实体中提取结构化信息
        if "姓名" in entities and entities["姓名"]:
            result["姓名"] = entities["姓名"]
        if "联系电话" in entities and entities["联系电话"]:
            result["联系方式"] = entities["联系电话"]
        if "电子邮箱" in entities and entities["电子邮箱"]:
            if "联系方式" in result:
                result["联系方式"] += f", {entities['电子邮箱']}"
            else:
                result["联系方式"] = entities["电子邮箱"]
        if "总工作经验年限" in entities and entities["总工作经验年限"]:
            result["工作经验"] = [entities["总工作经验年限"]]
        if "学历层次" in entities and entities["学历层次"]:
            result["教育经历"] = [entities["学历层次"]]
        if "学校名称" in entities and entities["学校名称"]:
            if "教育机构" not in result:
                result["教育机构"] = []
            result["教育机构"].append(entities["学校名称"])
        if "语言能力" in entities and entities["语言能力"]:
            result["语言技能"] = [entities["语言能力"]]
        if "证书资质" in entities and entities["证书资质"]:
            result["证书"] = [entities["证书资质"]]

        return result

    def process_jd_text(self,
                        jd_text: str,
                        use_llm: bool = True) -> Dict[str, Any]:
        """
        处理JD文本，提供与原始API兼容的方法
        
        Args:
            jd_text: JD文本
            use_llm: 是否使用LLM提取实体
        
        Returns:
            处理后的JD数据，包含原始文本、清理后的文本和提取的实体
        """
        cleaned_text = self.text_processor.clean_text(jd_text)
        entities = self.process_jd(jd_text, use_llm)

        # 构建与原始API兼容的返回格式
        result = {
            "raw_text": jd_text,
            "cleaned_text": cleaned_text,
            "entities": entities
        }

        # 从实体中提取结构化信息
        if "职位名称" in entities and entities["职位名称"]:
            result["职位名称"] = [entities["职位名称"]]
        if "薪资范围" in entities and entities["薪资范围"]:
            result["薪资"] = [entities["薪资范围"]]
        if "工作地点" in entities and entities["工作地点"]:
            result["工作地点"] = [entities["工作地点"]]
        if "工作年限要求" in entities and entities["工作年限要求"]:
            result["工作年限要求"] = [entities["工作年限要求"]]
        if "公司名称" in entities and entities["公司名称"]:
            result["公司信息"] = [entities["公司名称"]]

        return result
