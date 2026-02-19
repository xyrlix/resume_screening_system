"""
基础爬虫类
定义所有简历爬虫的通用接口和方法
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import random
from dataclasses import dataclass
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class ResumeData:
    """简历数据类"""
    source: str  # 来源网站
    resume_id: str  # 简历ID
    name: Optional[str] = None  # 姓名
    gender: Optional[str] = None  # 性别
    age: Optional[int] = None  # 年龄
    education: Optional[List[Dict[str, str]]] = None  # 教育经历
    work_experience: Optional[List[Dict[str, str]]] = None  # 工作经历
    skills: Optional[List[str]] = None  # 技能列表
    projects: Optional[List[Dict[str, str]]] = None  # 项目经历
    self_intro: Optional[str] = None  # 自我介绍
    contact_info: Optional[Dict[str, str]] = None  # 联系方式
    raw_data: Optional[Dict[str, Any]] = None  # 原始数据


class BaseScraper(ABC):
    """爬虫基类，定义通用接口"""
    
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
            "Connection": "keep-alive",
        }
        self.timeout = kwargs.get('timeout', 30)
        self.max_retries = kwargs.get('max_retries', 3)
        self.proxy = kwargs.get('proxy')
        self.session = self._create_session()
        self.delay_range = kwargs.get('delay_range', (1, 3))  # 请求间隔时间范围（秒）
        self.logged_in = False
    
    def set_cookie(self, cookie_string: str) -> None:
        """设置cookie进行认证"""
        if not hasattr(self, '_session') or self._session is None:
            self._session = self._create_session()
        
        # 解析cookie字符串并设置到session中
        cookies = {}
        for cookie in cookie_string.split(';'):
            if '=' in cookie:
                key, value = cookie.split('=', 1)
                cookies[key.strip()] = value.strip()
        
        self._session.cookies.update(cookies)
        self.logged_in = True
        print(f"已设置cookie，当前登录状态: {self.logged_in}")
    
    def _create_session(self) -> Session:
        """创建requests会话，配置重试和连接池"""
        session = Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=20, pool_maxsize=50)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(self.headers)
        return session
    
    def _request(self, method: str, url: str, **kwargs) -> Any:
        """发送HTTP请求，带重试和错误处理"""
        method = method.upper()
        kwargs.setdefault('timeout', self.timeout)
        
        if self.proxy:
            kwargs.setdefault('proxies', {'http': self.proxy, 'https': self.proxy})
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                wait_time = random.uniform(self.delay_range[0], self.delay_range[1]) * (attempt + 1)
                print(f"请求失败: {e}, 将在 {wait_time:.2f} 秒后重试 (尝试 {attempt + 1}/{self.max_retries})")
                time.sleep(wait_time)
    
    def _random_delay(self) -> None:
        """随机延迟，避免被网站反爬"""
        delay = random.uniform(self.delay_range[0], self.delay_range[1])
        time.sleep(delay)
    
    @abstractmethod
    def login(self, username: str, password: str) -> bool:
        """登录网站"""
        pass
    
    @abstractmethod
    def search_resumes(self, keyword: str, **kwargs) -> List[str]:
        """搜索简历，返回简历ID列表"""
        pass
    
    @abstractmethod
    def get_resume_detail(self, resume_id: str) -> ResumeData:
        """获取简历详情"""
        pass
    
    def batch_get_resume_details(self, resume_ids: List[str], max_workers: int = 5) -> List[ResumeData]:
        """批量获取简历详情"""
        resumes = []
        for resume_id in resume_ids:
            try:
                resume = self.get_resume_detail(resume_id)
                resumes.append(resume)
                self._random_delay()
            except Exception as e:
                print(f"获取简历 {resume_id} 失败: {e}")
        return resumes
    
    def close(self):
        """关闭会话"""
        if self.session:
            self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()