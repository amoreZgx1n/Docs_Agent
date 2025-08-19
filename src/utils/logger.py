"""
日志管理模块
提供统一的日志记录功能
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional
from .config import get_config


def setup_logger(
    name: str = "data_analyze_agent",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径
        max_bytes: 单个日志文件最大大小（字节）
        backup_count: 备份文件数量
    
    Returns:
        配置好的日志记录器
    """
    config = get_config()
    
    # 使用配置中的默认值
    if level is None:
        level = config.logging.level
    if log_file is None:
        log_file = config.logging.file_path
    if max_bytes is None:
        max_bytes = config.logging.max_file_size_mb * 1024 * 1024
    if backup_count is None:
        backup_count = config.logging.backup_count
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(config.logging.format)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（带轮转）
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建轮转文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "data_analyze_agent") -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
    
    Returns:
        日志记录器实例
    """
    logger = logging.getLogger(name)
    
    # 如果日志记录器还没有配置处理器，则进行配置
    if not logger.handlers:
        setup_logger(name)
    
    return logger


class LoggerMixin:
    """日志记录器混入类，为其他类提供日志功能"""
    
    def __init__(self, logger_name: Optional[str] = None):
        """
        初始化日志记录器
        
        Args:
            logger_name: 日志记录器名称，默认为类名
        """
        if logger_name is None:
            logger_name = self.__class__.__name__
        self.logger = get_logger(logger_name)
    
    def log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(message)
    
    def log_error(self, message: str, exc_info: bool = True):
        """记录错误日志"""
        self.logger.error(message, exc_info=exc_info)
    
    def log_debug(self, message: str):
        """记录调试日志"""
        self.logger.debug(message)


def log_function_call(func):
    """函数调用日志装饰器"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"调用函数: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"函数 {func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {e}", exc_info=True)
            raise
    return wrapper


def log_async_function_call(func):
    """异步函数调用日志装饰器"""
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"调用异步函数: {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"异步函数 {func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"异步函数 {func.__name__} 执行失败: {e}", exc_info=True)
            raise
    return wrapper 