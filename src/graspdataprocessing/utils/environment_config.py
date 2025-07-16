#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id: environment_config.py
@date: 2025/01/22
@author: YenochQin (秦毅)
@description: 统一的环境检测和配置模块
'''

import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path

class EnvironmentConfig:
    """环境配置管理器，用于检测运行环境和设置相应的配置"""
    
    def __init__(self):
        self._is_slurm = self._detect_slurm_environment()
        self._is_debug = self._detect_debug_mode()
        self._cpu_count = os.cpu_count() or 4
        
    def _detect_slurm_environment(self) -> bool:
        """检测是否在SLURM环境中运行"""
        slurm_indicators = [
            'SLURM_JOB_ID',
            'SLURM_PROCID', 
            'SLURM_LOCALID',
            'SLURM_TASK_PID'
        ]
        return any(env_var in os.environ for env_var in slurm_indicators)
    
    def _detect_debug_mode(self) -> bool:
        """检测是否处于调试模式"""
        # 检查环境变量
        if os.environ.get('DEBUG', '').lower() in ('1', 'true', 'yes'):
            return True
        if os.environ.get('PYTHON_DEBUG', '').lower() in ('1', 'true', 'yes'):
            return True
        
        # 检查命令行参数
        if '--debug' in sys.argv or '-d' in sys.argv:
            return True
            
        # 检查是否通过交互式环境运行
        if hasattr(sys, 'ps1') or sys.flags.interactive:
            return True
            
        return False
    
    @property
    def is_slurm_environment(self) -> bool:
        """是否在SLURM环境中运行"""
        return self._is_slurm
    
    @property
    def is_debug_mode(self) -> bool:
        """是否处于调试模式"""
        return self._is_debug
    
    @property
    def is_production_mode(self) -> bool:
        """是否处于生产模式（SLURM环境且非调试模式）"""
        return self._is_slurm and not self._is_debug
    
    @property
    def cpu_count(self) -> int:
        """系统CPU核心数"""
        return self._cpu_count
    
    def get_environment_info(self) -> Dict[str, Any]:
        """获取环境信息摘要"""
        return {
            'is_slurm': self.is_slurm_environment,
            'is_debug': self.is_debug_mode,
            'is_production': self.is_production_mode,
            'cpu_count': self.cpu_count,
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'slurm_task_pid': os.environ.get('SLURM_TASK_PID')
        }
    
    def get_progress_config(self) -> Dict[str, Any]:
        """获取进度条配置"""
        if self.is_production_mode:
            # 生产模式：关闭进度条
            return {
                'disable': True,
                'leave': False,
                'dynamic_ncols': False,
                'file': None  # 不输出到任何地方
            }
        else:
            # 调试模式：启用进度条
            return {
                'disable': False,
                'leave': True,
                'dynamic_ncols': True,
                'file': sys.stderr,
                'colour': 'green'
            }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        if self.is_production_mode:
            # 生产模式：结构化日志，关注关键信息
            return {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s',
                'show_progress_logs': False,
                'highlight_stages': True
            }
        else:
            # 调试模式：详细日志
            return {
                'level': 'DEBUG',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'show_progress_logs': True,
                'highlight_stages': True
            }

# 全局环境配置实例
_env_config = EnvironmentConfig()

def get_environment_config() -> EnvironmentConfig:
    """获取全局环境配置实例"""
    return _env_config

def is_slurm_environment() -> bool:
    """快捷函数：检查是否在SLURM环境"""
    return _env_config.is_slurm_environment

def is_debug_mode() -> bool:
    """快捷函数：检查是否在调试模式"""
    return _env_config.is_debug_mode

def is_production_mode() -> bool:
    """快捷函数：检查是否在生产模式"""
    return _env_config.is_production_mode 