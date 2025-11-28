#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据收集与处理模块

负责简历和JD的数据收集、初步处理和清洗
"""

import os
import json
import re
from typing import List, Dict, Any
from utils.logger import get_logger

# 初始化日志记录器
logger = get_logger("data_processor")


class DataProcessor:
    """
    数据处理类，负责简历和JD的数据收集、清洗和初步处理
    """

    def __init__(self):
        """
        初始化数据处理类
        """
        # 定义需要提取的实体列表
        self.resume_entities = [
            "姓名", "性别", "出生日期", "年龄", "联系电话", "电子邮箱", "现居地", "户籍地", "政治面貌",
            "婚姻状况", "期望职位", "期望行业", "期望工作城市", "期望薪资", "到岗时间", "学校名称", "学历层次",
            "专业名称", "入学时间", "毕业时间", "是否全日制", "GPA/排名/荣誉", "公司名称", "公司所属行业",
            "在职开始时间", "在职结束时间", "职位名称", "工作地点", "工作内容关键词", "汇报对象", "团队规模",
            "项目名称", "项目开始时间", "项目结束时间", "项目角色", "技术栈", "项目成果指标", "编程语言",
            "框架/工具", "数据库", "操作系统/平台", "语言能力", "证书资质", "软技能", "作品集/个人链接",
            "自我评价关键词", "兴趣爱好", "总工作经验年限"
        ]

        self.jd_entities = [
            "职位名称", "行业", "岗位职责", "任职要求", "学历要求", "工作年限要求", "薪资范围", "工作地点",
            "公司名称", "公司规模", "公司行业", "招聘人数", "发布时间", "截止时间", "职位类型", "技能要求",
            "语言要求", "证书要求", "福利", "团队情况"
        ]

    @staticmethod
    def clean_text(text: str) -> str:
        """
        清理文本，去除多余的空格、换行符和特殊字符
        
        Args:
            text: 原始文本
        
        Returns:
            清理后的文本
        """
        # 去除多余的空格和换行符
        text = re.sub(r'\s+', ' ', text)
        # 去除特殊字符
        text = re.sub(r'[^一-龥a-zA-Z0-9\s.,!?;:()\[\]{}\-_+=@#$%^&*]', '', text)
        return text.strip()

    @staticmethod
    def parse_resume_file(file_path: str) -> str:
        """
        解析简历文件，提取文本内容
        
        Args:
            file_path: 简历文件路径
        
        Returns:
            提取的文本内容
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # 对于其他格式，返回空字符串
            return ""

    @staticmethod
    def parse_jd_file(file_path: str) -> str:
        """
        解析JD文件，提取文本内容
        
        Args:
            file_path: JD文件路径
        
        Returns:
            提取的文本内容
        """
        return DataProcessor.parse_resume_file(file_path)

    def extract_entities(self,
                         text: str,
                         entity_list: List[str],
                         use_llm: bool = True) -> Dict[str, Any]:
        """
        从文本中提取实体信息
        
        Args:
            text: 待提取的文本
            entity_list: 要提取的实体列表
            use_llm: 是否使用LLM提取实体，默认为True
        
        Returns:
            提取的实体字典
        """
        # 初始化所有实体为空
        entities = {entity: "" for entity in entity_list}

        # LLM 补全将在正则之后执行

        from core.ner_model import get_ner
        ner = get_ner()
        if ner:
            try:
                spans = ner.predict(text)
                agg = {}
                for s in spans:
                    lab = s['label']
                    val = s['text']
                    if lab not in agg:
                        agg[lab] = []
                    agg[lab].append(val)
                def _valid(label: str, value: str) -> bool:
                    if not value or len(value.strip()) < 2:
                        return False
                    if label in ["Years", "Salary"]:
                        return bool(re.search(r"\d", value))
                    if label in ["Phone"]:
                        return bool(re.fullmatch(r"1[3-9]\d{9}", value))
                    if label in ["Email"]:
                        return bool(re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", value))
                    if label in ["Degree"]:
                        return value in ["博士","硕士","本科","大专","中专","高中"]
                    if label in ["Language"]:
                        return bool(re.search(r"英语|雅思|托福|CET", value))
                    if label in ["Certificate"]:
                        return bool(re.search(r"PMP|CKA|CKAD|RHCE|AWS", value))
                    # Company/JobTitle/Location/Skill: basic character check
                    return bool(re.search(r"[\u4e00-\u9fa5A-Za-z]{2,}", value))
                def assign(key, lab):
                    if key in entity_list and lab in agg and agg[lab]:
                        vals = [v.strip() for v in agg[lab] if _valid(lab, v.strip())]
                        if vals:
                            entities[key] = ", ".join(sorted(list(set(vals))))
                assign("职位名称", "JobTitle")
                assign("期望职位", "JobTitle")
                assign("公司名称", "Company")
                assign("学历层次", "Degree")
                assign("学历要求", "Degree")
                assign("总工作经验年限", "Years")
                assign("工作年限要求", "Years")
                assign("技能要求", "Skill")
                assign("工作地点", "Location")
                assign("现居地", "Location")
                assign("薪资范围", "Salary")
                assign("期望薪资", "Salary")
                assign("联系电话", "Phone")
                assign("电子邮箱", "Email")
                assign("语言要求", "Language")
                assign("语言能力", "Language")
                assign("证书要求", "Certificate")
                assign("证书资质", "Certificate")
                logger.info(
                    f"使用NER提取实体，共提取 {sum(1 for v in entities.values() if v)} 个非空实体")
            except Exception as e:
                logger.error(f"使用NER提取实体失败: {str(e)}")
        # 如果不使用LLM或NER提取失败，使用正则表达式提取实体
        # 增强规则提取，提高实体提取准确率

        # 1. 提取职位名称（JD和简历通用）
        position_patterns = [
            r'^([\u4e00-\u9fa5a-zA-Z&\-]+)',  # 职位名称在开头
            r'职位名称[:：]?\s*([\u4e00-\u9fa5a-zA-Z&\-]+)',
            r'招聘[:：]?\s*([\u4e00-\u9fa5a-zA-Z&\-]+)',
            r'[招聘]\s*([\u4e00-\u9fa5a-zA-Z&\-]+)',
        ]
        for pattern in position_patterns:
            position_match = re.search(pattern, text)
            if position_match and not entities.get("职位名称"):
                entities["职位名称"] = position_match.group(1)
                break

        # 2. 提取公司名称（JD和简历通用）
        company_patterns = [
            r'公司名称[:：]?\s*([\u4e00-\u9fa5a-zA-Z0-9&\-]+)',
            r'[招聘]\s*([\u4e00-\u9fa5a-zA-Z0-9&\-]+)\s*[招聘]',
            r'任职于[:：]?\s*([\u4e00-\u9fa5a-zA-Z0-9&\-]+)',
            r'在[:：]?\s*([\u4e00-\u9fa5a-zA-Z0-9&\-]+)\s*工作',
        ]
        for pattern in company_patterns:
            company_match = re.search(pattern, text)
            if company_match and not entities.get("公司名称"):
                entities["公司名称"] = company_match.group(1)
                break

        # 3. 提取学历（JD和简历通用）
        education_pattern = r'(博士|硕士|本科|大专|中专|高中)'
        education_match = re.search(education_pattern, text)
        if education_match:
            if not entities.get("学历层次"):
                entities["学历层次"] = education_match.group(1)
            if not entities.get("学历要求"):
                entities["学历要求"] = education_match.group(1)

        # 4. 提取工作经验（JD和简历通用）
        experience_match = re.search(r'(\d+)年', text)
        if experience_match:
            if not entities.get("总工作经验年限"):
                entities["总工作经验年限"] = experience_match.group(1)
            if not entities.get("工作年限要求"):
                entities["工作年限要求"] = experience_match.group(1)

        # 5. JD特定实体提取
        if "薪资范围" in entity_list:
            # 提取薪资范围
            salary_patterns = [
                r'薪资[:：]?\s*(\d+)[kK]?\s*[-—]\s*(\d+)[kK]?',
                r'薪资[:：]?\s*(\d+)[kK]?',
                r'待遇[:：]?\s*(\d+)[kK]?\s*[-—]\s*(\d+)[kK]?',
            ]
            for pattern in salary_patterns:
                salary_match = re.search(pattern, text)
                if salary_match and not entities.get("薪资范围"):
                    if len(salary_match.groups()) == 2:
                        entities["薪资范围"] = f"{salary_match.group(1)}K-{salary_match.group(2)}K"
                    else:
                        entities["薪资范围"] = f"{salary_match.group(1)}K"
                    break

        if "工作地点" in entity_list:
            # 提取工作地点
            location_patterns = [
                r'工作地点[:：]?\s*([\u4e00-\u9fa5]+)',
                r'地点[:：]?\s*([\u4e00-\u9fa5]+)',
                r'[在]\s*([\u4e00-\u9fa5]+)\s*[工作]',
            ]
            for pattern in location_patterns:
                location_match = re.search(pattern, text)
                if location_match and not entities.get("工作地点"):
                    entities["工作地点"] = location_match.group(1)
                    break

        if "技能要求" in entity_list:
            # 提取技能要求
            skill_patterns = [
                r'(Python|Java|C\+\+|JavaScript|Go|SQL|MySQL|PostgreSQL|MongoDB|Redis|Kafka|Spark|Hadoop|Docker|Kubernetes|AWS|Azure|GCP|Oracle|SQLite|Elasticsearch|Flask|Django|Spring|React|Vue|Angular|Node\.js|TypeScript|HTML|CSS|Sass|Less|Webpack|Vite|Git|SVN|Linux|Windows|MacOS|iOS|Android|Swift|Kotlin|React Native|Flutter|TensorFlow|PyTorch|Scikit-learn|XGBoost|LightGBM|NLTK|spaCy|Transformers|GraphQL|REST|gRPC|Thrift|ZooKeeper|HBase|Hive|Pig|Storm|Flink|Airflow|Jenkins|CI/CD|敏捷开发|Scrum|Kanban|TDD|BDD|微服务|分布式|高并发|高可用|性能优化|安全|加密|区块链|物联网|云计算|边缘计算|大数据|人工智能|机器学习|深度学习|自然语言处理|计算机视觉|语音识别|推荐系统|搜索算法|排序算法|数据挖掘|数据分析|数据可视化|ETL|数据仓库|数据湖)',
            ]
            skills = []
            for pattern in skill_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                skills.extend(matches)

            # 从岗位职责和任职要求中提取技能
            if "岗位职责" in entities and entities["岗位职责"]:
                duty_skills = re.findall(skill_patterns[0], entities["岗位职责"],
                                         re.IGNORECASE)
                skills.extend(duty_skills)
            if "任职要求" in entities and entities["任职要求"]:
                req_skills = re.findall(skill_patterns[0], entities["任职要求"],
                                        re.IGNORECASE)
                skills.extend(req_skills)

            if skills and not entities.get("技能要求"):
                entities["技能要求"] = ", ".join(list(set(skills)))

        if "岗位职责" in entity_list:
            # 提取岗位职责
            duty_pattern = r'(岗位职责|工作内容|职位描述)[:：]?([\s\S]*?)(?=(任职要求|资格要求|岗位要求|福利待遇|薪资|工作地点|$))'
            duty_match = re.search(duty_pattern, text)
            if duty_match and duty_match.group(2) and not entities.get("岗位职责"):
                entities["岗位职责"] = duty_match.group(2).strip()

        if "任职要求" in entity_list:
            # 提取任职要求
            req_pattern = r'(任职要求|资格要求|岗位要求|招聘要求)[:：]?([\s\S]*?)(?=(福利待遇|薪资|工作地点|$))'
            req_match = re.search(req_pattern, text)
            if req_match and req_match.group(2) and not entities.get("任职要求"):
                entities["任职要求"] = req_match.group(2).strip()

        # 6. 提取联系方式（简历特定）
        if "联系电话" in entity_list:
            phone_match = re.search(r'1[3-9]\d{9}', text)
            if phone_match and not entities.get("联系电话"):
                entities["联系电话"] = phone_match.group(0)

        if "电子邮箱" in entity_list:
            email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
            if email_match and not entities.get("电子邮箱"):
                entities["电子邮箱"] = email_match.group(0)

        # 7. 提取期望相关信息（简历特定）
        if "期望职位" in entity_list:
            expected_position_pattern = r'期望职位[:：]?\s*([\u4e00-\u9fa5a-zA-Z&\-]+)'
            expected_position_match = re.search(expected_position_pattern, text)
            if expected_position_match and not entities.get("期望职位"):
                entities["期望职位"] = expected_position_match.group(1)

        if "期望薪资" in entity_list:
            salary_pattern = r'期望薪资[:：]?\s*(\d+)[kK]?'
            salary_match = re.search(salary_pattern, text)
            if salary_match and not entities.get("期望薪资"):
                entities["期望薪资"] = salary_match.group(1) + "K"

        # 使用LLM进行最后补全
        if use_llm:
            try:
                from core.llm_chain import LLMChain
                llm_chain = LLMChain()
                from core.llm_config_manager import LLMConfigManager
                tmpl = LLMConfigManager().get_prompt('jd_extract_entities')
                prompt = tmpl.replace('{text}', text).replace('{entity_list}', ', '.join(entity_list))
                providers = [
                    llm_chain.llm_providers[name]
                    for name in getattr(llm_chain, 'active_provider_names', [])
                    if name in llm_chain.llm_providers
                ] or list(llm_chain.llm_providers.values())
                import json
                for p in providers:
                    try:
                        response = p._call_llm(prompt)
                        llm_entities = json.loads(response)
                        for key, value in llm_entities.items():
                            if key in entities and not entities.get(key):
                                entities[key] = value
                        logger.info(
                            f"LLM补全实体，共提取 {sum(1 for v in entities.values() if v)} 个非空实体"
                        )
                        break
                    except Exception as e:
                        logger.error(f"LLM补全实体失败: {str(e)}")
                        continue
            except Exception as e:
                logger.error(f"LLM补全实体失败: {str(e)}")
        logger.info(f"使用正则表达式提取JD实体，共提取 {sum(1 for v in entities.values() if v)} 个非空实体")
        return entities

    def process_resume_text(self, resume_text: str) -> Dict[str, Any]:
        """
        处理简历文本，提取结构化信息
        
        Args:
            resume_text: 简历文本
        
        Returns:
            结构化的简历信息
        """
        logger.info(f"📄 开始处理简历文本，原始长度: {len(resume_text)} 字符")

        # 清理文本
        logger.info(f"🧹 正在清理文本...")
        cleaned_text = self.clean_text(resume_text)
        logger.info(f"✅ 文本清理完成，清理后长度: {len(cleaned_text)} 字符")

        # 简单的结构化处理，提取关键词
        skills = []
        experience = []
        education = []

        # 提取技能关键词
        logger.info(f"🔍 正在提取技能关键词...")
        skill_patterns = [
            r'(Python|Java|C\+\+|JavaScript|Go|SQL|MySQL|PostgreSQL|MongoDB|Redis|Kafka|Spark|Hadoop|Docker|Kubernetes|AWS|Azure|GCP)',
            r'(深度学习|机器学习|人工智能|自然语言处理|计算机视觉|大数据|云计算|DevOps|全栈开发|前端开发|后端开发|移动开发)'
        ]

        for pattern in skill_patterns:
            matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
            skills.extend(matches)

        skills = list(set(skills))
        logger.info(f"✅ 技能提取完成，共找到 {len(skills)} 个技能关键词")

        # 提取工作经验
        logger.info(f"🔍 正在提取工作经验...")
        experience_pattern = r'(\d+\s*年.*?经验|工作.*?\d+\s*年)'
        experience_matches = re.findall(experience_pattern, cleaned_text)
        experience.extend(experience_matches)
        logger.info(f"✅ 工作经验提取完成，共找到 {len(experience)} 条经验记录")

        # 提取教育背景
        logger.info(f"🔍 正在提取教育背景...")
        education_pattern = r'(本科|硕士|博士|大专|中专|高中)'
        education_matches = re.findall(education_pattern, cleaned_text)
        education.extend(education_matches)
        logger.info(f"✅ 教育背景提取完成，共找到 {len(education)} 条教育记录")

        # 提取实体信息
        logger.info(f"🔍 正在提取实体信息...")
        entities = self.extract_entities(cleaned_text, self.resume_entities, use_llm=True)
        # 统计非空实体数量
        non_empty_entities = sum(1 for v in entities.values() if v)
        logger.info(f"✅ 实体信息提取完成，共提取 {non_empty_entities} 个非空实体")

        result = {
            "raw_text": resume_text,
            "cleaned_text": cleaned_text,
            "skills": skills,
            "experience": experience,
            "education": education,
            "entities": entities,
            "segment_texts": {
                "skills": ", ".join(skills),
                "experience": "\n".join(experience),
                "education": ", ".join(education)
            }
        }

        # 落盘
        try:
            os.makedirs(os.path.join('data','processed'), exist_ok=True)
            with open(os.path.join('data','processed','parsed_resumes.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            pass
        logger.info(f"📄 简历处理完成")
        return result

    def process_jd_text(self, jd_text: str) -> Dict[str, Any]:
        """
        处理JD文本，提取结构化信息
        
        Args:
            jd_text: JD文本
        
        Returns:
            结构化的JD信息
        """
        logger.info(f"开始处理JD文本，原始长度: {len(jd_text)} 字符")

        # 清理文本
        logger.info(f"正在清理文本...")
        cleaned_text = self.clean_text(jd_text)
        logger.info(f"文本清理完成，清理后长度: {len(cleaned_text)} 字符")

        # 简单的结构化处理，提取关键词
        skills = []
        requirements = []

        # 提取技能关键词
        logger.info(f"🔍 正在提取技能关键词...")
        skill_patterns = [
            r'(Python|Java|C\+\+|JavaScript|Go|SQL|MySQL|PostgreSQL|MongoDB|Redis|Kafka|Spark|Hadoop|Docker|Kubernetes|AWS|Azure|GCP)',
            r'(深度学习|机器学习|人工智能|自然语言处理|计算机视觉|大数据|云计算|DevOps|全栈开发|前端开发|后端开发|移动开发)'
        ]

        for pattern in skill_patterns:
            matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
            skills.extend(matches)

        skills = list(set(skills))
        logger.info(f"✅ 技能提取完成，共找到 {len(skills)} 个技能关键词")

        # 提取岗位要求
        logger.info(f"🔍 正在提取岗位要求...")
        requirement_pattern = r'(要求|需要|具备|熟悉|精通|掌握).*?'
        requirement_matches = re.findall(requirement_pattern, cleaned_text)
        requirements.extend(requirement_matches)
        logger.info(f"✅ 岗位要求提取完成，共找到 {len(requirements)} 条要求")

        # 提取实体信息
        logger.info(f"🔍 正在提取实体信息...")
        entities = self.extract_entities(cleaned_text, self.jd_entities, use_llm=True)
        # 统计非空实体数量
        non_empty_entities = sum(1 for v in entities.values() if v)
        logger.info(f"✅ 实体信息提取完成，共提取 {non_empty_entities} 个非空实体")

        result = {
            "raw_text": jd_text,
            "cleaned_text": cleaned_text,
            "skills": skills,
            "requirements": requirements,
            "entities": entities,
            "segment_texts": {
                "skills": ", ".join(skills),
                "requirements": "\n".join(requirements),
                "education": entities.get("学历要求", "")
            }
        }

        # 落盘
        try:
            os.makedirs(os.path.join('data','processed'), exist_ok=True)
            with open(os.path.join('data','processed','parsed_jds.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        except Exception as e:
            pass
        logger.info(f"📄 JD处理完成")
        return result

    def collect_resumes_from_dir(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        从目录中收集简历
        
        Args:
            dir_path: 简历目录路径
        
        Returns:
            收集的简历列表
        """
        resumes = []
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                raw_text = self.parse_resume_file(file_path)
                if raw_text:
                    processed_resume = self.process_resume_text(raw_text)
                    processed_resume["filename"] = filename
                    resumes.append(processed_resume)
        return resumes

    def collect_jds_from_dir(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        从目录中收集JD
        
        Args:
            dir_path: JD目录路径
        
        Returns:
            收集的JD列表
        """
        jds = []
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                raw_text = self.parse_jd_file(file_path)
                if raw_text:
                    processed_jd = self.process_jd_text(raw_text)
                    processed_jd["filename"] = filename
                    jds.append(processed_jd)
        return jds

    def save_processed_data(self, data: List[Dict[str, Any]],
                            file_path: str) -> None:
        """
        保存处理后的数据到文件
        
        Args:
            data: 处理后的数据
            file_path: 保存文件路径
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_processed_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        从文件加载处理后的数据
        
        Args:
            file_path: 数据文件路径
        
        Returns:
            加载的数据列表
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
