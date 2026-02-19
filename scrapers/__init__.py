"""
网联简历爬虫模块
提供从各个招聘网站获取简历信息的功能
"""

from .base_scraper import BaseScraper
from .zhaopin_scraper import ZhaopinScraper
from .liepin_scraper import LiepinScraper
from .job51_scraper import Job51Scraper

__all__ = [
    'BaseScraper',
    'ZhaopinScraper',
    'LiepinScraper',
    'Job51Scraper'
]