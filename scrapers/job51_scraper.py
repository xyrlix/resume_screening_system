"""
前程无忧爬虫
用于从前程无忧网站获取简历信息
"""

import json
import re
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from .base_scraper import BaseScraper, ResumeData


class Job51Scraper(BaseScraper):
    """前程无忧简历爬虫"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "https://ehire.51job.com"
        self.login_url = "https://ehire.51job.com/MainLogin.aspx"
        self.logged_in = False
    
    def login(self, username: str, password: str) -> bool:
        """登录前程无忧"""
        try:
            # 第一步：访问登录页面获取必要的Cookie和表单信息
            response = self._request("GET", self.login_url)
            
            # 构建登录表单数据
            payload = {
                "ctl00$ContentPlaceHolder1$txtUserName": username,
                "ctl00$ContentPlaceHolder1$txtPassword": password,
                "ctl00$ContentPlaceHolder1$btnLogin": "登录"
            }
            
            # 提交登录请求
            response = self._request("POST", self.login_url, data=payload, allow_redirects=True)
            
            # 检查是否登录成功
            if "MainPage.aspx" in response.url:
                self.logged_in = True
                print("前程无忧登录成功")
                return True
            else:
                print("前程无忧登录失败")
                return False
        except Exception as e:
            print(f"前程无忧登录异常: {e}")
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
            search_url = f"{self.base_url}/Resume/SearchResume.aspx"
            
            # 搜索参数
            params = {
                "KeyWord": keyword,
                "Page": str(page),
                "PageSize": str(page_size)
            }
            
            response = self._request("GET", search_url, params=params)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取简历ID
            resume_links = soup.select('a[href*="ViewResume.aspx?"]')
            for link in resume_links:
                href = link.get('href')
                if href:
                    # 提取简历ID
                    match = re.search(r'Rid=(\d+)', href)
                    if match:
                        resume_id = match.group(1)
                        if resume_id not in resume_ids:
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
            detail_url = f"{self.base_url}/Resume/ViewResume.aspx"
            params = {
                "Rid": resume_id
            }
            
            response = self._request("GET", detail_url, params=params)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取简历数据
            resume_data = ResumeData(
                source="51job",
                resume_id=resume_id,
                raw_data={"html": response.text}
            )
            
            # 提取姓名
            name_elem = soup.select_one('.cName')
            if name_elem:
                resume_data.name = name_elem.text.strip()
            
            # 提取基本信息
            basic_info_elem = soup.select_one('.BaseInfo')
            if basic_info_elem:
                basic_info_text = basic_info_elem.text.strip()
                # 提取性别
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
        
        # 查找教育经历部分
        edu_section = soup.find('div', string=re.compile(r'教育背景|教育经历'))
        if edu_section:
            edu_parent = edu_section.find_parent()
            edu_items = edu_parent.find_all('div', class_=re.compile(r'edu-item|education-item'))
            
            for item in edu_items:
                edu_info = {}
                
                # 提取学校和专业信息
                school_major_text = item.text.strip()
                
                # 提取时间范围
                time_match = re.search(r'(\d{4}\.\d{1,2})[\s-]+(\d{4}\.\d{1,2}|至今)', school_major_text)
                if time_match:
                    edu_info['start_time'] = time_match.group(1)
                    edu_info['end_time'] = time_match.group(2)
                    edu_info['time'] = time_match.group(0)
                
                # 提取学历
                for degree in ['博士', '硕士', '本科', '大专', '高中']:
                    if degree in school_major_text:
                        edu_info['degree'] = degree
                        break
                
                # 提取学校和专业
                # 简单的提取逻辑，实际情况可能需要更复杂的处理
                parts = re.split(r'[\s]+', school_major_text)
                for part in parts:
                    if '大学' in part or '学院' in part:
                        edu_info['school'] = part
                    elif any(x in part for x in ['专业', '工程', '技术', '管理']):
                        edu_info['major'] = part
                
                if edu_info:
                    education_list.append(edu_info)
        
        return education_list
    
    def _extract_work_experience(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """提取工作经历"""
        work_list = []
        
        # 查找工作经历部分
        work_section = soup.find('div', string=re.compile(r'工作经验|工作经历'))
        if work_section:
            work_parent = work_section.find_parent()
            work_items = work_parent.find_all('div', class_=re.compile(r'work-item|exp-item'))
            
            for item in work_items:
                work_info = {}
                
                # 提取公司名称
                company_elem = item.select_one('.company-name, .cName')
                if company_elem:
                    work_info['company'] = company_elem.text.strip()
                
                # 提取职位名称
                position_elem = item.select_one('.position-name, .pName')
                if position_elem:
                    work_info['position'] = position_elem.text.strip()
                
                # 提取工作时间
                time_elem = item.select_one('.time, .period')
                if time_elem:
                    time_text = time_elem.text.strip()
                    work_info['time'] = time_text
                    # 分离起止时间
                    if ' - ' in time_text:
                        start_time, end_time = time_text.split(' - ', 1)
                        work_info['start_time'] = start_time
                        work_info['end_time'] = end_time
                
                # 提取工作内容
                content_elem = item.select_one('.content, .desc')
                if content_elem:
                    work_info['content'] = content_elem.text.strip()
                
                if work_info:
                    work_list.append(work_info)
        
        # 如果没有找到结构化的工作经历，尝试从文本中提取
        if not work_list and '工作经验' in soup.text:
            # 这是一个简单的备用方法，实际使用时可能需要更复杂的处理
            work_text = soup.text
            work_blocks = re.split(r'\n\s*\n', work_text)
            
            for block in work_blocks:
                if any(x in block for x in ['公司', '职位', '职责']):
                    work_info = {}
                    
                    # 简单提取公司名称
                    company_match = re.search(r'公司[：:](.+)', block)
                    if company_match:
                        work_info['company'] = company_match.group(1).strip()
                    
                    # 简单提取职位
                    position_match = re.search(r'职位[：:](.+)', block)
                    if position_match:
                        work_info['position'] = position_match.group(1).strip()
                    
                    # 简单提取时间
                    time_match = re.search(r'(\d{4})[年.](\d{1,2})[月.][\s-]+(\d{4})[年.](\d{1,2})[月.]|至今', block)
                    if time_match:
                        work_info['time'] = time_match.group(0)
                    
                    if work_info:
                        work_list.append(work_info)
        
        return work_list
    
    def _extract_skills(self, soup: BeautifulSoup) -> List[str]:
        """提取技能"""
        skills = []
        
        # 查找技能部分
        skill_section = soup.find('div', string=re.compile(r'技能|专业技能'))
        if skill_section:
            skill_parent = skill_section.find_parent()
            
            # 查找技能标签
            skill_tags = skill_parent.find_all('span', class_=re.compile(r'skill|tag'))
            for tag in skill_tags:
                skill = tag.text.strip()
                if skill and len(skill) > 1:
                    skills.append(skill)
            
            # 如果没有找到标签，尝试提取文本
            if not skills:
                skill_text = skill_parent.text.strip()
                # 尝试分割技能
                potential_skills = re.findall(r'[\u4e00-\u9fa5a-zA-Z+#\.-]{2,}', skill_text)
                # 过滤掉常见的非技能词汇
                exclude_words = ['技能', '专业技能', '熟练', '精通', '掌握', '了解']
                skills = [s for s in potential_skills if s not in exclude_words and len(s) > 1][:20]
        
        # 如果还是没有找到，尝试从整个页面提取
        if not skills:
            # 尝试提取常见技术词汇
            tech_keywords = ['Python', 'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'SQL', 'MySQL', 
                             'Oracle', 'MongoDB', 'Redis', 'Docker', 'Kubernetes', 'Git', 'Linux',
                             '数据分析', '机器学习', '深度学习', '人工智能', '算法', '架构',
                             '前端', '后端', '全栈', '开发', '测试', '运维', '产品', '设计']
            
            page_text = soup.text
            for keyword in tech_keywords:
                if keyword in page_text:
                    skills.append(keyword)
            
            # 去重
            skills = list(set(skills))[:20]
        
        return skills
    
    def _extract_projects(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """提取项目经历"""
        projects_list = []
        
        # 查找项目经历部分
        project_section = soup.find('div', string=re.compile(r'项目经验|项目经历'))
        if project_section:
            project_parent = project_section.find_parent()
            project_items = project_parent.find_all('div', class_=re.compile(r'project-item|proj-item'))
            
            for item in project_items:
                project_info = {}
                
                # 提取项目名称
                project_elem = item.select_one('.project-name, .pName')
                if project_elem:
                    project_info['name'] = project_elem.text.strip()
                
                # 提取角色
                role_elem = item.select_one('.role, .position')
                if role_elem:
                    project_info['role'] = role_elem.text.strip()
                
                # 提取时间
                time_elem = item.select_one('.time, .period')
                if time_elem:
                    time_text = time_elem.text.strip()
                    project_info['time'] = time_text
                    # 分离起止时间
                    if ' - ' in time_text:
                        start_time, end_time = time_text.split(' - ', 1)
                        project_info['start_time'] = start_time
                        project_info['end_time'] = end_time
                
                # 提取项目描述
                desc_elem = item.select_one('.description, .content')
                if desc_elem:
                    project_info['description'] = desc_elem.text.strip()
                
                # 提取技术栈
                tech_elem = item.select_one('.tech-stack, .skills')
                if tech_elem:
                    project_info['tech_stack'] = tech_elem.text.strip()
                
                if project_info:
                    projects_list.append(project_info)
        
        return projects_list
    
    def _extract_self_intro(self, soup: BeautifulSoup) -> Optional[str]:
        """提取自我介绍"""
        # 查找自我介绍部分
        intro_section = soup.find('div', string=re.compile(r'自我评价|自我介绍|个人评价'))
        if intro_section:
            intro_parent = intro_section.find_parent()
            content_elem = intro_parent.find('div', class_=re.compile(r'content|text'))
            if content_elem:
                return content_elem.text.strip()
            # 如果没有找到内容元素，尝试获取父元素的所有文本
            return ''.join([text for text in intro_parent.stripped_strings if text not in ['自我评价', '自我介绍', '个人评价']])
        
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
        address_section = soup.find('div', string=re.compile(r'所在地|地址'))
        if address_section and address_section.parent:
            address_text = address_section.parent.text.strip()
            # 提取地址信息
            address_parts = re.split(r'[：:]', address_text, 1)
            if len(address_parts) > 1:
                contact_info['address'] = address_parts[1].strip()
        
        return contact_info