"""
智联招聘爬虫
用于从智联招聘网站获取简历信息
"""

import json
import re
import time
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from .base_scraper import BaseScraper, ResumeData


class ZhaopinScraper(BaseScraper):
    """智联招聘简历爬虫"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://rd5.zhaopin.com"
        self.login_url = "https://passport.zhaopin.com/org/login"
        self.logged_in = False
    
    def login(self, username: str, password: str) -> bool:
        """登录智联招聘"""
        try:
            # 模拟登录流程
            # 实际使用时需要根据网站的登录机制进行调整
            payload = {
                "LoginName": username,
                "Password": password,
                "remember": "false"
            }
            
            response = self._request("POST", self.login_url, data=payload, allow_redirects=True)
            
            # 检查是否登录成功
            if "zhaopin.com" in response.url and "login" not in response.url:
                self.logged_in = True
                print("智联招聘登录成功")
                return True
            else:
                print("智联招聘登录失败")
                return False
        except Exception as e:
            print(f"智联招聘登录异常: {e}")
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
            search_url = f"{self.base_url}/resume/search/resumeList.json"
            
            # 搜索参数
            params = {
                "keyword": keyword,
                "page": page,
                "pageSize": page_size,
                "_": str(int(time.time() * 1000))
            }
            
            response = self._request("GET", search_url, params=params)
            data = response.json()
            
            # 提取简历ID
            if data.get('status') == 200 and data.get('data'):
                for item in data['data'].get('resumeList', []):
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
            detail_url = f"{self.base_url}/resume/search/viewNewResume"
            params = {
                "resumeId": resume_id,
                "_": str(int(time.time() * 1000))
            }
            
            response = self._request("GET", detail_url, params=params)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取简历数据
            resume_data = ResumeData(
                source="zhaopin",
                resume_id=resume_id,
                raw_data={"html": response.text}
            )
            
            # 提取姓名
            name_elem = soup.select_one('.name')
            if name_elem:
                resume_data.name = name_elem.text.strip()
            
            # 提取基本信息
            basic_info_elem = soup.select_one('.basic-info')
            if basic_info_elem:
                basic_info_text = basic_info_elem.text.strip()
                # 提取性别、年龄等信息
                if "男" in basic_info_text:
                    resume_data.gender = "男"
                elif "女" in basic_info_text:
                    resume_data.gender = "女"
                
                # 提取年龄
                age_match = re.search(r'(\d+)岁', basic_info_text)
                if age_match:
                    resume_data.age = int(age_match.group(1))
            
            # 提取教育经历
            resume_data.education = self._extract_education(soup)
            
            # 提取工作经历
            resume_data.work_experience = self._extract_work_experience(soup)
            
            # 提取技能
            resume_data.skills = self._extract_skills(soup)
            
            # 提取项目经历
            resume_data.projects = self._extract_projects(soup)
            
            # 提取自我介绍
            resume_data.self_intro = self._extract_self_intro(soup)
            
            # 提取联系方式
            resume_data.contact_info = self._extract_contact_info(soup)
            
            return resume_data
        except Exception as e:
            print(f"获取简历详情失败: {e}")
            raise
    
    def _extract_education(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """提取教育经历"""
        education_list = []
        edu_sections = soup.select('.education-section .edu-item')
        
        for section in edu_sections:
            edu_info = {}
            
            # 提取学校名称
            school_elem = section.select_one('.school-name')
            if school_elem:
                edu_info['school'] = school_elem.text.strip()
            
            # 提取专业和学历
            major_elem = section.select_one('.major')
            if major_elem:
                major_text = major_elem.text.strip()
                # 分离专业和学历
                edu_info['major'] = major_text
                # 提取学历
                for degree in ['博士', '硕士', '本科', '大专']:
                    if degree in major_text:
                        edu_info['degree'] = degree
                        break
            
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
            
            if edu_info:
                education_list.append(edu_info)
        
        return education_list
    
    def _extract_work_experience(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """提取工作经历"""
        work_list = []
        work_sections = soup.select('.work-experience-section .work-item')
        
        for section in work_sections:
            work_info = {}
            
            # 提取公司名称
            company_elem = section.select_one('.company-name')
            if company_elem:
                work_info['company'] = company_elem.text.strip()
            
            # 提取职位和部门
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
            
            if work_info:
                work_list.append(work_info)
        
        return work_list
    
    def _extract_skills(self, soup: BeautifulSoup) -> List[str]:
        """提取技能"""
        skills = []
        skill_elems = soup.select('.skills-section .skill-tag')
        
        for elem in skill_elems:
            skill = elem.text.strip()
            if skill:
                skills.append(skill)
        
        # 如果没有找到技能标签，尝试从其他地方提取
        if not skills:
            skill_text = soup.find(string=re.compile(r'技能|专业技能'))
            if skill_text and skill_text.parent:
                parent = skill_text.parent
                # 提取技能文本
                skill_content = parent.find_next_sibling()
                if skill_content:
                    # 简单分割技能
                    potential_skills = re.findall(r'[\u4e00-\u9fa5a-zA-Z+#\.-]{2,}', skill_content.text)
                    skills = list(set(potential_skills))[:20]  # 去重并限制数量
        
        return skills
    
    def _extract_projects(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """提取项目经历"""
        projects_list = []
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
            
            if project_info:
                projects_list.append(project_info)
        
        return projects_list
    
    def _extract_self_intro(self, soup: BeautifulSoup) -> Optional[str]:
        """提取自我介绍"""
        # 查找自我介绍部分
        intro_elem = soup.select_one('.self-intro-section .intro-content')
        if intro_elem:
            return intro_elem.text.strip()
        
        # 尝试其他可能的位置
        intro_text = soup.find(string=re.compile(r'自我介绍|个人评价'))
        if intro_text and intro_text.parent:
            parent = intro_text.parent
            next_elem = parent.find_next_sibling()
            if next_elem:
                return next_elem.text.strip()
        
        return None
    
    def _extract_contact_info(self, soup: BeautifulSoup) -> Dict[str, str]:
        """提取联系方式"""
        contact_info = {}
        
        # 尝试提取电话
        phone_match = re.search(r'1[3-9]\d{9}', soup.text)
        if phone_match:
            contact_info['phone'] = phone_match.group(0)
        
        # 尝试提取邮箱
        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', soup.text)
        if email_match:
            contact_info['email'] = email_match.group(0)
        
        # 尝试提取地址
        address_elem = soup.select_one('.contact-section .address')
        if address_elem:
            contact_info['address'] = address_elem.text.strip()
        
        return contact_info