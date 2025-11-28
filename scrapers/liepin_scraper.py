"""
猎聘网爬虫
用于从猎聘网站获取简历信息
"""

import json
import re
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from .base_scraper import BaseScraper, ResumeData


class LiepinScraper(BaseScraper):
    """猎聘网简历爬虫"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://hr.liepin.com"
        self.login_url = "https://passport.liepin.com/login"
        self.logged_in = False
    
    def login(self, username: str, password: str) -> bool:
        """登录猎聘网"""
        try:
            # 第一步：访问登录页面获取必要的Cookie和token
            response = self._request("GET", self.login_url)
            
            # 构建登录请求数据
            payload = {
                "username": username,
                "password": password,
                "rememberLogin": "0"
            }
            
            # 提交登录请求
            login_api = "https://passport.liepin.com/c/login"
            headers = {
                "Content-Type": "application/json",
                "X-Requested-With": "XMLHttpRequest"
            }
            response = self._request("POST", login_api, json=payload, headers=headers)
            
            # 解析响应
            result = response.json()
            
            if result.get('status') == 1:
                self.logged_in = True
                print("猎聘网登录成功")
                return True
            else:
                print(f"猎聘网登录失败: {result.get('msg', '未知错误')}")
                return False
        except Exception as e:
            print(f"猎聘网登录异常: {e}")
            return False
    
    def search_resumes(self, keyword: str, **kwargs) -> List[str]:
        """搜索简历，返回简历ID列表"""
        if not self.logged_in:
            raise ValueError("请先登录")
        
        resume_ids = []
        page = kwargs.get('page', 1)
        page_size = kwargs.get('page_size', 20)
        
        try:
            # 构建搜索URL
            search_url = "https://hr.liepin.com/resume/searchResumeList.json"
            
            # 构建搜索参数
            payload = {
                "keyword": keyword,
                "pageNo": page,
                "pageSize": page_size,
                "sort": "1"
            }
            
            headers = {
                "Content-Type": "application/json",
                "X-Requested-With": "XMLHttpRequest"
            }
            
            response = self._request("POST", search_url, json=payload, headers=headers)
            data = response.json()
            
            # 提取简历ID
            if data.get('status') == 1 and 'data' in data and 'list' in data['data']:
                for item in data['data']['list']:
                    resume_id = item.get('resumeId')
                    if resume_id:
                        resume_ids.append(resume_id)
            
            print(f"找到 {len(resume_ids)} 份简历")
            return resume_ids
        except Exception as e:
            print(f"搜索简历失败: {e}")
            return []
    
    def get_resume_detail(self, resume_id: str) -> ResumeData:
        """获取简历详情"""
        if not self.logged_in:
            raise ValueError("请先登录")
        
        try:
            # 获取简历详情页面
            detail_url = "https://hr.liepin.com/resume/getResumeDetail.json"
            params = {
                "resumeId": resume_id
            }
            
            headers = {
                "X-Requested-With": "XMLHttpRequest"
            }
            
            response = self._request("GET", detail_url, params=params, headers=headers)
            data = response.json()
            
            if data.get('status') != 1:
                raise Exception(f"获取简历失败: {data.get('msg', '未知错误')}")
            
            # 提取简历数据
            resume_info = data.get('data', {})
            
            resume_data = ResumeData(
                source="liepin",
                resume_id=resume_id,
                raw_data=resume_info
            )
            
            # 提取基本信息
            if 'basic' in resume_info:
                basic = resume_info['basic']
                
                # 提取姓名
                resume_data.name = basic.get('name', '')
                
                # 提取性别
                gender_map = {'1': '男', '2': '女'}
                resume_data.gender = gender_map.get(basic.get('gender', ''), '')
                
                # 提取年龄
                resume_data.age = basic.get('age', 0)
            
            # 提取教育经历
            resume_data.education = self._extract_education(resume_info)
            
            # 提取工作经历
            resume_data.work_experience = self._extract_work_experience(resume_info)
            
            # 提取技能
            resume_data.skills = self._extract_skills(resume_info)
            
            # 提取项目经历
            resume_data.projects = self._extract_projects(resume_info)
            
            # 提取自我介绍
            resume_data.self_intro = self._extract_self_intro(resume_info)
            
            # 提取联系方式
            resume_data.contact_info = self._extract_contact_info(resume_info)
            
            return resume_data
        except Exception as e:
            print(f"获取简历详情失败: {e}")
            raise
    
    def _extract_education(self, resume_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """提取教育经历"""
        education_list = []
        
        # 从API返回的数据中提取教育经历
        if 'education' in resume_info and isinstance(resume_info['education'], list):
            for edu_item in resume_info['education']:
                edu_info = {
                    'school': edu_item.get('school', ''),
                    'major': edu_item.get('major', ''),
                    'degree': edu_item.get('education', ''),
                    'start_time': edu_item.get('startTime', ''),
                    'end_time': edu_item.get('endTime', ''),
                    'time': f"{edu_item.get('startTime', '')} - {edu_item.get('endTime', '')}"
                }
                education_list.append(edu_info)
        
        # 如果没有结构化数据，尝试解析raw_data中的HTML
        elif 'resumeHtml' in resume_info:
            soup = BeautifulSoup(resume_info['resumeHtml'], 'html.parser')
            edu_sections = soup.select('.education-section .edu-item')
            
            for section in edu_sections:
                edu_info = {}
                
                # 提取学校名称
                school_elem = section.select_one('.school')
                if school_elem:
                    edu_info['school'] = school_elem.text.strip()
                
                # 提取专业
                major_elem = section.select_one('.major')
                if major_elem:
                    edu_info['major'] = major_elem.text.strip()
                
                # 提取学历
                degree_elem = section.select_one('.degree')
                if degree_elem:
                    edu_info['degree'] = degree_elem.text.strip()
                
                # 提取时间
                time_elem = section.select_one('.time')
                if time_elem:
                    time_text = time_elem.text.strip()
                    edu_info['time'] = time_text
                    # 分离起止时间
                    if ' - ' in time_text:
                        start_time, end_time = time_text.split(' - ', 1)
                        edu_info['start_time'] = start_time
                        edu_info['end_time'] = end_time
                
                if any(edu_info.values()):
                    education_list.append(edu_info)
        
        return education_list
    
    def _extract_work_experience(self, resume_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """提取工作经历"""
        work_list = []
        
        # 从API返回的数据中提取工作经历
        if 'workExperience' in resume_info and isinstance(resume_info['workExperience'], list):
            for work_item in resume_info['workExperience']:
                work_info = {
                    'company': work_item.get('company', ''),
                    'position': work_item.get('title', ''),
                    'start_time': work_item.get('startTime', ''),
                    'end_time': work_item.get('endTime', ''),
                    'time': f"{work_item.get('startTime', '')} - {work_item.get('endTime', '')}",
                    'content': work_item.get('description', '')
                }
                work_list.append(work_info)
        
        # 如果没有结构化数据，尝试解析raw_data中的HTML
        elif 'resumeHtml' in resume_info:
            soup = BeautifulSoup(resume_info['resumeHtml'], 'html.parser')
            work_sections = soup.select('.work-experience-section .work-item')
            
            for section in work_sections:
                work_info = {}
                
                # 提取公司名称
                company_elem = section.select_one('.company')
                if company_elem:
                    work_info['company'] = company_elem.text.strip()
                
                # 提取职位名称
                position_elem = section.select_one('.position')
                if position_elem:
                    work_info['position'] = position_elem.text.strip()
                
                # 提取时间
                time_elem = section.select_one('.time')
                if time_elem:
                    time_text = time_elem.text.strip()
                    work_info['time'] = time_text
                    # 分离起止时间
                    if ' - ' in time_text:
                        start_time, end_time = time_text.split(' - ', 1)
                        work_info['start_time'] = start_time
                        work_info['end_time'] = end_time
                
                # 提取工作内容
                content_elem = section.select_one('.content')
                if content_elem:
                    work_info['content'] = content_elem.text.strip()
                
                if any(work_info.values()):
                    work_list.append(work_info)
        
        return work_list
    
    def _extract_skills(self, resume_info: Dict[str, Any]) -> List[str]:
        """提取技能"""
        skills = []
        
        # 从API返回的数据中提取技能
        if 'skill' in resume_info:
            if isinstance(resume_info['skill'], str):
                # 如果是字符串，尝试分割
                skills = [s.strip() for s in resume_info['skill'].split('、') if s.strip()]
            elif isinstance(resume_info['skill'], list):
                skills = resume_info['skill']
        
        # 如果没有结构化数据或技能列表为空，尝试解析raw_data中的HTML
        if not skills and 'resumeHtml' in resume_info:
            soup = BeautifulSoup(resume_info['resumeHtml'], 'html.parser')
            
            # 查找技能标签
            skill_tags = soup.select('.skills-section .skill-tag')
            for tag in skill_tags:
                skill = tag.text.strip()
                if skill and len(skill) > 1:
                    skills.append(skill)
            
            # 如果没有找到标签，尝试提取文本
            if not skills:
                skill_section = soup.find('div', string=re.compile(r'技能|专业技能'))
                if skill_section and skill_section.parent:
                    skill_text = skill_section.parent.text.strip()
                    # 尝试分割技能
                    potential_skills = re.findall(r'[\u4e00-\u9fa5a-zA-Z+#\.-]{2,}', skill_text)
                    # 过滤掉常见的非技能词汇
                    exclude_words = ['技能', '专业技能', '熟练', '精通', '掌握', '了解']
                    skills = [s for s in potential_skills if s not in exclude_words and len(s) > 1][:20]
        
        # 去重并限制数量
        return list(set(skills))[:20]
    
    def _extract_projects(self, resume_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """提取项目经历"""
        projects_list = []
        
        # 从API返回的数据中提取项目经历
        if 'project' in resume_info and isinstance(resume_info['project'], list):
            for project_item in resume_info['project']:
                project_info = {
                    'name': project_item.get('projectName', ''),
                    'role': project_item.get('role', ''),
                    'start_time': project_item.get('startTime', ''),
                    'end_time': project_item.get('endTime', ''),
                    'time': f"{project_item.get('startTime', '')} - {project_item.get('endTime', '')}",
                    'description': project_item.get('description', ''),
                    'tech_stack': project_item.get('technologies', '')
                }
                projects_list.append(project_info)
        
        # 如果没有结构化数据，尝试解析raw_data中的HTML
        elif 'resumeHtml' in resume_info:
            soup = BeautifulSoup(resume_info['resumeHtml'], 'html.parser')
            project_sections = soup.select('.project-section .project-item')
            
            for section in project_sections:
                project_info = {}
                
                # 提取项目名称
                project_elem = section.select_one('.project-name')
                if project_elem:
                    project_info['name'] = project_elem.text.strip()
                
                # 提取角色
                role_elem = section.select_one('.role')
                if role_elem:
                    project_info['role'] = role_elem.text.strip()
                
                # 提取时间
                time_elem = section.select_one('.time')
                if time_elem:
                    time_text = time_elem.text.strip()
                    project_info['time'] = time_text
                    # 分离起止时间
                    if ' - ' in time_text:
                        start_time, end_time = time_text.split(' - ', 1)
                        project_info['start_time'] = start_time
                        project_info['end_time'] = end_time
                
                # 提取项目描述
                desc_elem = section.select_one('.description')
                if desc_elem:
                    project_info['description'] = desc_elem.text.strip()
                
                # 提取技术栈
                tech_elem = section.select_one('.tech-stack')
                if tech_elem:
                    project_info['tech_stack'] = tech_elem.text.strip()
                
                if any(project_info.values()):
                    projects_list.append(project_info)
        
        return projects_list
    
    def _extract_self_intro(self, resume_info: Dict[str, Any]) -> Optional[str]:
        """提取自我介绍"""
        # 从API返回的数据中提取自我介绍
        if 'selfEvaluation' in resume_info:
            return resume_info['selfEvaluation']
        
        # 如果没有结构化数据，尝试解析raw_data中的HTML
        if 'resumeHtml' in resume_info:
            soup = BeautifulSoup(resume_info['resumeHtml'], 'html.parser')
            
            # 查找自我介绍部分
            intro_section = soup.find('div', string=re.compile(r'自我评价|自我介绍|个人评价'))
            if intro_section and intro_section.parent:
                content_elem = intro_section.parent.find('div', class_=re.compile(r'content|text'))
                if content_elem:
                    return content_elem.text.strip()
                
                # 尝试获取所有文本内容
                texts = []
                for elem in intro_section.parent.find_all('p'):
                    if elem.text.strip() not in ['自我评价', '自我介绍', '个人评价']:
                        texts.append(elem.text.strip())
                
                if texts:
                    return '\n'.join(texts)
        
        return None
    
    def _extract_contact_info(self, resume_info: Dict[str, Any]) -> Dict[str, str]:
        """提取联系方式"""
        contact_info = {}
        
        # 从API返回的数据中提取联系方式
        if 'basic' in resume_info:
            basic = resume_info['basic']
            
            # 提取电话
            if 'phone' in basic:
                contact_info['phone'] = basic['phone']
            
            # 提取邮箱
            if 'email' in basic:
                contact_info['email'] = basic['email']
            
            # 提取地址
            if 'location' in basic:
                contact_info['address'] = basic['location']
        
        # 如果没有结构化数据，尝试解析raw_data中的HTML
        if not contact_info and 'resumeHtml' in resume_info:
            soup = BeautifulSoup(resume_info['resumeHtml'], 'html.parser')
            
            # 提取电话
            if 'phone' not in contact_info:
                phone_match = re.search(r'1[3-9]\d{9}', soup.text)
                if phone_match:
                    contact_info['phone'] = phone_match.group(0)
            
            # 提取邮箱
            if 'email' not in contact_info:
                email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', soup.text)
                if email_match:
                    contact_info['email'] = email_match.group(0)
            
            # 提取地址
            if 'address' not in contact_info:
                address_elem = soup.select_one('.contact-section .address')
                if address_elem:
                    contact_info['address'] = address_elem.text.strip()
        
        return contact_info