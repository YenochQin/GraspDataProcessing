#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id: progress_manager.py
@date: 2025/01/22
@author: YenochQin (秦毅)
@description: 统一的进度条管理模块
'''

import sys
from typing import Optional, Any, Dict, Union
from tqdm import tqdm
from .environment_config import get_environment_config

class ProgressManager:
    """统一的进度条管理器，根据环境自动配置tqdm行为"""
    
    def __init__(self):
        self.env_config = get_environment_config()
        self.base_config = self.env_config.get_progress_config()
    
    def create_progress_bar(
        self, 
        iterable=None, 
        total: Optional[int] = None,
        desc: Optional[str] = None,
        unit: str = 'it',
        unit_scale: bool = False,
        override_disable: Optional[bool] = None,
        **kwargs
    ) -> tqdm:
        """
        创建进度条，自动根据环境配置
        
        Args:
            iterable: 可迭代对象
            total: 总数量
            desc: 描述文本
            unit: 单位
            unit_scale: 是否自动缩放单位
            override_disable: 强制覆盖disable设置
            **kwargs: 其他tqdm参数
            
        Returns:
            配置好的tqdm对象
        """
        # 合并配置
        config = self.base_config.copy()
        config.update(kwargs)
        
        # 允许手动覆盖disable设置
        if override_disable is not None:
            config['disable'] = override_disable
        
        # 生产模式下的特殊处理
        if self.env_config.is_production_mode and not config.get('disable', False):
            # 在生产模式下，即使不完全禁用，也要简化输出
            config.update({
                'ncols': 80,
                'ascii': True,
                'bar_format': '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
            })
        
        return tqdm(
            iterable=iterable,
            total=total,
            desc=desc,
            unit=unit,
            unit_scale=unit_scale,
            **config
        )
    
    def log_stage_start(self, stage_name: str, logger=None):
        """记录阶段开始"""
        message = f"🔧 开始阶段: {stage_name}"
        if logger:
            logger.info(message)
        elif not self.env_config.is_production_mode:
            print(message)
    
    def log_stage_end(self, stage_name: str, logger=None, **metrics):
        """记录阶段结束"""
        if metrics:
            metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
            message = f"✅ 完成阶段: {stage_name} ({metrics_str})"
        else:
            message = f"✅ 完成阶段: {stage_name}"
            
        if logger:
            logger.info(message)
        elif not self.env_config.is_production_mode:
            print(message)

# 全局进度管理器实例
_progress_manager = ProgressManager()

def create_progress_bar(*args, **kwargs) -> tqdm:
    """快捷函数：创建进度条"""
    return _progress_manager.create_progress_bar(*args, **kwargs)

def log_stage_start(stage_name: str, logger=None):
    """快捷函数：记录阶段开始"""
    _progress_manager.log_stage_start(stage_name, logger)

def log_stage_end(stage_name: str, logger=None, **metrics):
    """快捷函数：记录阶段结束"""
    _progress_manager.log_stage_end(stage_name, logger, **metrics)

def wrap_iterator(iterable, desc: Optional[str] = None, **kwargs):
    """
    包装迭代器，自动应用环境相关的进度条配置
    
    Args:
        iterable: 要包装的可迭代对象
        desc: 进度条描述
        **kwargs: 传递给tqdm的其他参数
        
    Returns:
        包装后的迭代器
    """
    return create_progress_bar(iterable, desc=desc, **kwargs)

def progress_range(n: int, desc: Optional[str] = None, **kwargs):
    """
    创建range的进度条版本
    
    Args:
        n: 范围大小
        desc: 进度条描述
        **kwargs: 传递给tqdm的其他参数
        
    Returns:
        带进度条的range迭代器
    """
    return create_progress_bar(range(n), desc=desc, total=n, **kwargs)

class ProgressContext:
    """进度条上下文管理器，用于处理嵌套进度条"""
    
    def __init__(self, desc: str, total: Optional[int] = None, **kwargs):
        self.desc = desc
        self.total = total
        self.kwargs = kwargs
        self.pbar = None
        
    def __enter__(self):
        self.pbar = create_progress_bar(desc=self.desc, total=self.total, **self.kwargs)
        return self.pbar
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()

def progress_context(desc: str, total: Optional[int] = None, **kwargs):
    """
    创建进度条上下文管理器
    
    用法:
        with progress_context("处理数据", total=100) as pbar:
            for i in range(100):
                # 处理逻辑
                pbar.update(1)
    """
    return ProgressContext(desc, total, **kwargs) 