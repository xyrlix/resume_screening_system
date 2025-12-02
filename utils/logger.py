import logging
import os
from logging.handlers import RotatingFileHandler
'''
日志工具模块

此模块包含了用于配置和获取日志记录器的函数。
'''


def get_logger(name: str):
    os.makedirs("logs", exist_ok=True)
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s")
        fh = RotatingFileHandler(os.path.join("logs", "app.log"),
                                 maxBytes=2 * 1024 * 1024,
                                 backupCount=3,
                                 encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    return logger
