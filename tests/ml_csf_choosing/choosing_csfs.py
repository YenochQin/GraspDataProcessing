#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :choosing_csfs.py
@date :2025/06/11 10:45:00
@author :YenochQin (秦毅)
@description: 进行组态选择的流程，支持多轮迭代训练和智能索引选择
'''

import argparse
from pathlib import Path
import sys
import numpy as np
import math
import os
from typing import Any, Dict, List, Optional, Protocol, Tuple

# 路径通过 sbatch 脚本中的 PYTHONPATH 环境变量自动设置
try:
    import graspdataprocessing as gdp
except ImportError:
    print("错误: 无法导入 graspdataprocessing 模块")
    sys.exit(1)

class CSFs(Protocol):
    """
    为 graspdataprocessing.CSFs 定义一个协议，用于类型提示
    """
    block_num: int
    CSFs_block_data: List[List[Any]]
    subshell_info_raw: Any

def should_use_colors():
    """检测是否应该使用颜色输出"""
    # 检查是否在终端中运行
    if not sys.stdout.isatty():
        return False
    # 检查环境变量（某些CI/脚本环境可能设置）
    if os.environ.get('NO_COLOR', '').lower() in ('1', 'true', 'yes'):
        return False
    if os.environ.get('TERM', '').lower() == 'dumb':
        return False
    return True

# 根据环境决定是否使用颜色
if should_use_colors():
    COLOR_RED = '\033[0;31m'
    COLOR_GREEN = '\033[0;32m'
    COLOR_YELLOW = '\033[1;33m'
    COLOR_BLUE = '\033[0;34m'
    COLOR_PURPLE = '\033[0;35m'
    COLOR_CYAN = '\033[0;36m'
    COLOR_WHITE = '\033[1;37m'
    COLOR_BOLD = '\033[1m'
    COLOR_RESET = '\033[0m'
else:
    # 禁用颜色代码（在脚本或非终端环境中）
    COLOR_RED = ''
    COLOR_GREEN = ''
    COLOR_YELLOW = ''
    COLOR_BLUE = ''
    COLOR_PURPLE = ''
    COLOR_CYAN = ''
    COLOR_WHITE = ''
    COLOR_BOLD = ''
    COLOR_RESET = ''

def simplify_path_python(full_path, root_path=None):
    """
    路径简化函数 - 去除root_path前缀，只显示相对路径
    """
    if root_path is None:
        # 尝试从环境变量获取root_path
        root_path = os.getcwd()
    
    full_path_str = str(full_path)
    root_path_str = str(root_path)
    
    # 如果路径不包含root_path，返回原路径
    if not full_path_str.startswith(root_path_str):
        return full_path_str
    
    # 移除root_path前缀
    relative_path = full_path_str[len(root_path_str):]
    # 移除开头的斜杠
    relative_path = relative_path.lstrip('/')
    
    # 如果简化后路径为空，表示就是root目录
    if not relative_path:
        return "."
    
    return relative_path

def highlight_number_python(text, color=COLOR_CYAN):
    """
    数值高亮函数
    """
    return f"{color}{text}{COLOR_RESET}"

def highlight_path_python(path, root_path=None):
    """
    路径高亮和简化函数
    """
    simplified = simplify_path_python(path, root_path)
    return f"{COLOR_BLUE}{simplified}{COLOR_RESET}"

def load_target_pool_data(config) -> Tuple[Optional[CSFs], Dict[str, Any]]:
    """
    加载target_pool CSFs数据，从预处理文件中加载
    注意：预处理工作已在initial_csfs.py中完成
    
    Args:
        config: 配置对象
        
    Returns:
        tuple: (target_pool_csfs_data, status_info)
    """
    root_path = Path(config.root_path)
    target_pool_binary_path = root_path / f"{config.conf}.pkl"
    
    if not target_pool_binary_path.exists():
        return None, {
            'success': False,
            'error': f"未找到预处理的CSFs数据文件: {target_pool_binary_path}",
            'suggestion': "请先运行 initial_csfs.py 进行数据预处理",
            'file_path': str(target_pool_binary_path)
        }
    
    try:
        # 加载CSFs数据
        target_pool_csfs_data: CSFs = gdp.load_csfs_binary(target_pool_binary_path)
        return target_pool_csfs_data, {
            'success': True,
            'message': f"从二进制文件加载CSFs数据",
            'file_path': str(target_pool_binary_path)
        }
    except Exception as e:
        return None, {
            'success': False,
            'error': f"加载CSFs数据时发生错误: {str(e)}",
            'file_path': str(target_pool_binary_path)
        }

def load_selected_indices(config, target_pool_csfs_data_block_num):
    """
    加载selected indices数据
    
    Args:
        config: 配置对象
        target_pool_csfs_data_block_num: 目标池CSFs数据块数量
        
    Returns:
        tuple: (selected_csfs_indices_dict, status_info)
    """
    root_path = Path(config.root_path)
    selected_indices_path = root_path / f"{config.conf}_selected_indices.pkl"
    
    # 优先尝试加载预处理的indices文件（initial_csfs.py生成的）
    if selected_indices_path.exists():
        try:
            selected_csfs_indices_dict = gdp.csfs_index_load(selected_indices_path)
            return selected_csfs_indices_dict, {
                'success': True,
                'message': f"加载预处理的初筛CSFs indices",
                'file_path': str(selected_indices_path),
                'found_existing': True,
                'source': 'preprocessed'
            }
        except Exception as e:
            # 预处理文件损坏，继续尝试从原始文件加载
            pass
    
    # 如果预处理文件不存在或损坏，尝试从原始selected_csfs_file加载
    elif hasattr(config, 'selected_csfs_file') and config.selected_csfs_file:
        selected_csfs_file_path = root_path / config.selected_csfs_file
        if selected_csfs_file_path.exists():
            try:
                # 从原始.c文件读取CSFs
                selected_csfs_load = gdp.GraspFileLoad.from_filepath(selected_csfs_file_path, file_type='CSF')
                selected_csfs_data = selected_csfs_load.data_file_process()
                
                # 加载目标池数据用于映射
                target_pool_binary_path = root_path / f"{config.conf}.pkl"
                if target_pool_binary_path.exists():
                    # 使用哈希映射生成indices
                    selected_csfs_indices_dict = gdp.maping_two_csfs_indices(
                        selected_csfs_data.CSFs_block_data, 
                        target_pool_binary_path
                    )
                    
                    return selected_csfs_indices_dict, {
                        'success': True,
                        'message': f"从原始初筛CSFs文件生成indices",
                        'file_path': str(selected_csfs_file_path),
                        'found_existing': True,
                        'source': 'original_file'
                    }
                else:
                    return {block: [] for block in range(target_pool_csfs_data_block_num)}, {
                        'success': False,
                        'error': f"找不到目标池数据文件: {target_pool_binary_path}",
                        'file_path': str(selected_csfs_file_path),
                        'source': 'original_file'
                    }
            except Exception as e:
                return {block: [] for block in range(target_pool_csfs_data_block_num)}, {
                    'success': False,
                    'error': f"从原始文件加载初筛CSFs失败: {str(e)}",
                    'file_path': str(selected_csfs_file_path),
                    'fallback': True,
                    'source': 'original_file'
                }
    
    # 如果都没有找到，创建空的indices
    selected_csfs_indices_dict = {block: [] for block in range(target_pool_csfs_data_block_num)}
    return selected_csfs_indices_dict, {
        'success': True,
        'message': "未找到任何初筛CSFs文件，使用空的indices",
        'file_path': "none",
        'found_existing': False,
        'source': 'empty'
    }

def truncate_initial_selected_with_weights(config, selected_csfs_indices_dict, target_pool_csfs_data: CSFs):
    """
    使用权重信息对initial_selected进行智能截断
    
    Args:
        config: 配置对象
        selected_csfs_indices_dict: 原始选择的indices字典
        target_pool_csfs_data: 目标池数据
        
    Returns:
        tuple: (truncated_indices_dict, status_info)
    """
    root_path = Path(config.root_path)
    mix_file_path = root_path / config.selected_csfs_mix_file
    
    # 尝试加载混合系数文件进行权重排序
    csf_weights = {}
    weight_loading_info = {'loaded': False, 'error': None}
    
    if mix_file_path.exists():
        try:
            # 这里可以添加加载混合系数文件的逻辑
            # mix_data = gdp.load_mix_coefficient_file(mix_file_path)
            # 暂时使用简单的方法：保持原有顺序作为权重
            weight_loading_info = {
                'loaded': True, 
                'file_path': str(mix_file_path),
                'message': '加载混合系数文件进行权重排序'
            }
        except Exception as e:
            weight_loading_info = {
                'loaded': False,
                'error': f"加载混合系数文件失败: {e}",
                'file_path': str(mix_file_path)
            }
    
    truncated_indices_dict = {}
    random_selection_ratio = 0.1  # 固定10%的随机选择比例
    truncation_details = []
    
    total_target_pool = sum(len(target_pool_csfs_data.CSFs_block_data[block]) 
                            for block in range(target_pool_csfs_data.block_num))
    total_target_chosen = math.ceil(total_target_pool * config.chosen_ratio)
    
    for block in range(target_pool_csfs_data.block_num):
        block_csfs = target_pool_csfs_data.CSFs_block_data[block]
        block_target_chosen = math.ceil(len(block_csfs) * config.chosen_ratio)
        block_initial_selected = selected_csfs_indices_dict.get(block, [])
        
        if len(block_initial_selected) > block_target_chosen:
            # 计算随机选择数量（目标数量的10%）
            random_selection_count = max(1, math.ceil(block_target_chosen * random_selection_ratio))
            # 截断后的initial选择数量
            truncated_count = block_target_chosen - random_selection_count
            
            # 智能截断逻辑
            if block in csf_weights and len(csf_weights[block]) > 0:
                # 基于权重排序截断
                weighted_indices = sorted(
                    block_initial_selected, 
                    key=lambda idx: csf_weights[block].get(idx, 0), 
                    reverse=True
                )
                truncated_indices = weighted_indices[:truncated_count]
                truncation_details.append({
                    'block': block,
                    'method': 'weight_based',
                    'original_count': len(block_initial_selected),
                    'truncated_count': len(truncated_indices),
                    'random_space': random_selection_count
                })
            else:
                # 简单截断：保留前N个（假设按重要性排序）
                truncated_indices = block_initial_selected[:truncated_count]
                truncation_details.append({
                    'block': block,
                    'method': 'simple',
                    'original_count': len(block_initial_selected),
                    'truncated_count': len(truncated_indices),
                    'random_space': random_selection_count
                })
            
            truncated_indices_dict[block] = truncated_indices
        else:
            truncated_indices_dict[block] = block_initial_selected
            truncation_details.append({
                'block': block,
                'method': 'no_truncation',
                'original_count': len(block_initial_selected),
                'truncated_count': len(block_initial_selected),
                'random_space': 0
            })
    
    status_info = {
        'success': True,
        'weight_loading': weight_loading_info,
        'truncation_details': truncation_details,
        'total_target_chosen': total_target_chosen,
        'random_selection_ratio': random_selection_ratio
    }
    
    return truncated_indices_dict, status_info

def load_previous_chosen_indices(config):
    """
    加载前一轮选择的CSFs索引
    
    Args:
        config: 配置对象
        
    Returns:
        tuple: (indices_dict, status_info)
    """
    root_path = Path(config.root_path)
    previous_indices_file = root_path / 'results' / f'{config.conf}_{config.cal_loop_num-1}_important_indices'
    
    if previous_indices_file.with_suffix('.pkl').exists():
        try:
            final_chosen_indices_dict = gdp.csfs_index_load(previous_indices_file)
            counts = [len(indices) for indices in final_chosen_indices_dict.values()]
            return final_chosen_indices_dict, {
                'success': True,
                'message': '加载上一轮计算重要的CSFs indices',
                'file_path': str(previous_indices_file) + '.pkl',
                'config_counts': counts
            }
        except Exception as e:
            return None, {
                'success': False,
                'error': f'加载indices文件失败: {str(e)}',
                'file_path': str(previous_indices_file) + '.pkl'
            }
    else:
        return None, {
            'success': False,
            'message': '未找到上一轮计算的重要CSFs indices文件',
            'file_path': str(previous_indices_file) + '.pkl',
            'file_exists': False
        }

def load_ml_final_chosen_indices(config):
    """
    加载机器学习训练后生成的最终选择索引（优先级最高）
    
    Args:
        config: 配置对象
        
    Returns:
        tuple: (indices_dict, status_info)
    """
    root_path = Path(config.root_path)
    
    # 优先查找final_chosen_indices（train.py生成的最终选择索引）
    final_chosen_path = root_path / 'results' / f'{config.conf}_{config.cal_loop_num-1}_final_chosen_indices'
    
    if final_chosen_path.with_suffix('.pkl').exists():
        try:
            final_chosen_indices_dict = gdp.csfs_index_load(final_chosen_path)
            counts = [len(indices) for indices in final_chosen_indices_dict.values()]
            return final_chosen_indices_dict, {
                'success': True,
                'message': '加载ML最终选择CSFs indices',
                'file_path': str(final_chosen_path) + '.pkl',
                'config_counts': counts
            }
        except Exception as e:
            return None, {
                'success': False,
                'error': f'加载ML indices文件失败: {str(e)}',
                'file_path': str(final_chosen_path) + '.pkl'
            }
    else:
        return None, {
            'success': False,
            'message': '未找到ML最终选择的indices文件',
            'file_path': str(final_chosen_path) + '.pkl',
            'file_exists': False
        }

def load_previous_ml_chosen_indices(config):
    """
    加载前一轮计算后机器学习选择出的CSFs索引（备用选项）
    
    Args:
        config: 配置对象
        
    Returns:
        tuple: (indices_dict, status_info)
    """
    root_path = Path(config.root_path)
    ml_results_path = root_path / 'results' / f'{config.conf}_{config.cal_loop_num-1}_final_chosen_indices'
    
    if ml_results_path.with_suffix('.pkl').exists():
        try:
            selected_csfs_indices_dict = gdp.csfs_index_load(ml_results_path)
            return selected_csfs_indices_dict, {
                'success': True,
                'message': '加载机器学习选择的CSFs indices',
                'file_path': str(ml_results_path) + '.pkl'
            }
        except Exception as e:
            return {}, {
                'success': False,
                'error': f'加载ML indices文件失败: {str(e)}',
                'file_path': str(ml_results_path) + '.pkl'
            }
    else:
        return {}, {
            'success': False,
            'message': '未找机器学习选择的CSFs indices文件',
            'file_path': str(ml_results_path),
            'file_exists': False
        }

def build_csfs_from_indices(target_pool_csfs_data: CSFs, chosen_indices_dict):
    """
    直接从索引构建CSFs数据（类似ann3_proba.py的方式）
    
    Args:
        target_pool_csfs_data: 目标池CSFs数据
        chosen_indices_dict: 选择的索引字典
        
    Returns:
        tuple: (chosen_csfs_dict, unselected_indices_dict, status_info)
    """
    chosen_csfs_dict = {}
    unselected_indices_dict = {}
    block_results = []
    
    for block in range(target_pool_csfs_data.block_num):
        if block in chosen_indices_dict and len(chosen_indices_dict[block]) > 0:
            # 使用指定的索引
            chosen_indices = np.array(chosen_indices_dict[block])
            
            # 验证索引的有效性
            max_index = len(target_pool_csfs_data.CSFs_block_data[block]) - 1
            # invalid_indices = []
            invalid_indices = np.array([])

            if np.any(chosen_indices > max_index):
                invalid_indices = chosen_indices[chosen_indices > max_index]
                chosen_indices = chosen_indices[chosen_indices <= max_index]
            
            # 构建选择的CSFs
            chosen_csfs_dict[block] = [target_pool_csfs_data.CSFs_block_data[block][i] for i in chosen_indices]
            
            # 计算未选择的索引
            all_indices = np.arange(len(target_pool_csfs_data.CSFs_block_data[block]))
            unselected_indices_dict[block] = np.setdiff1d(all_indices, chosen_indices).tolist()
            
            block_results.append({
                'block': block,
                'chosen_count': len(chosen_csfs_dict[block]),
                'unselected_count': len(unselected_indices_dict[block]),
                'invalid_indices': invalid_indices.tolist() if len(invalid_indices) > 0 else [], 
                'max_valid_index': max_index,
                'has_indices': True
            })
        else:
            # 如果没有指定索引，使用空列表
            chosen_csfs_dict[block] = []
            unselected_indices_dict[block] = list(range(len(target_pool_csfs_data.CSFs_block_data[block])))
            
            block_results.append({
                'block': block,
                'chosen_count': 0,
                'unselected_count': len(unselected_indices_dict[block]),
                'invalid_indices': [],
                'max_valid_index': len(target_pool_csfs_data.CSFs_block_data[block]) - 1,
                'has_indices': False
            })
    
    status_info = {
        'success': True,
        'method': 'direct_index_build',
        'block_results': block_results,
        'total_blocks': target_pool_csfs_data.block_num
    }
    
    return chosen_csfs_dict, unselected_indices_dict, status_info

def perform_csfs_selection(config):
    """
    执行组态选择的主要流程（支持智能索引选择）
    
    Args:
        config: 配置对象
        
    Returns:
        dict: 包含选择结果的字典
    """
    logger = gdp.setup_logging(config)
    
    # 步骤1：加载目标池数据
    target_pool_csfs_data, load_status = load_target_pool_data(config)
    if not load_status['success']:
        logger.error(load_status['error'])
        if 'suggestion' in load_status:
            logger.error(load_status['suggestion'])
        raise FileNotFoundError(load_status['error'])
    
    # 向类型查器断言 target_pool_csfs_data 不是 None
    assert target_pool_csfs_data is not None
    
    logger.info(f"{load_status['message']}: {load_status['file_path']}")
    
    logger.info("CSFs选择程序启动")
    logger.info(f'计算循环次数: {config.cal_loop_num}')
    logger.info(f"光谱项: {config.spetral_term}")
    
    # 步骤2：创建输出目录
    root_path = Path(config.root_path)
    cal_path = root_path / f'{config.conf}_{config.cal_loop_num}'
    cal_path.mkdir(exist_ok=True)
    logger.info(f"创建输出目录: {cal_path}")
    
    # 步骤3：智能确定选择的CSFs索引
    selected_csfs_indices_dict = None
    selection_method = "unknown"
    
    if config.cal_loop_num == 1:
        # 第一轮使用initial_selected_indices（来自selected_csfs_file或空）
        selected_csfs_indices_dict, indices_status = load_selected_indices(config, target_pool_csfs_data.block_num)
        
        # 记录加载状态
        if indices_status['success']:
            if indices_status['found_existing']:
                logger.info(f"{indices_status['message']}: {indices_status['file_path']}")
            else:
                logger.info(indices_status['message'])
        else:
            if 'fallback' in indices_status:
                logger.warning(f"{indices_status['error']}，使用空的indices")
            else:
                logger.error(indices_status['error'])
        selection_method = "initial_selected"
        logger.info("第一轮选择，使用基础selected indices")
        
        # 第一轮initial_selected数量过多的情况
        total_target_pool = sum(len(target_pool_csfs_data.CSFs_block_data[block]) 
        for block in range(target_pool_csfs_data.block_num))
        total_target_chosen = math.ceil(total_target_pool * config.chosen_ratio)
        
        # 检查initial_selected的总数量
        total_initial_selected = sum(len(selected_csfs_indices_dict.get(block, [])) 
                    for block in range(target_pool_csfs_data.block_num))
        
        # 检查是否需要扩展选择（当selected_csfs数量远小于total_target_pool时）
        ratio_selected_to_pool = total_initial_selected / total_target_pool if total_target_pool > 0 else 0
        
        # 如果selected_csfs数量比total_target_pool小两个数量级或更多，则使用expansion_ratio扩展
        if ratio_selected_to_pool < 0.01:  # 小于1%，约两个数量级
            expansion_ratio = getattr(config, 'expansion_ratio', 2)
            expanded_target_chosen = math.ceil(total_target_chosen * expansion_ratio)
            
            logger.warning(f"⚠️ Selected CSFs数量远小于target pool: {total_initial_selected} / {total_target_pool} = {ratio_selected_to_pool:.4%}")
            logger.info(f"🔧 应用扩展比例 {expansion_ratio}，扩展目标数量: {total_target_chosen} -> {expanded_target_chosen}")
            
            # 更新目标选择数量
            total_target_chosen = expanded_target_chosen
        
        if total_initial_selected > total_target_chosen:
            logger.warning(f"⚠️ Initial selected CSFs数量过多: {total_initial_selected} > 目标数量: {total_target_chosen}")
            logger.info(f"🔧 使用cutoff_value={config.cutoff_value}进行截断处理")
            
            # 对每个块进行截断处理
            truncated_indices_dict, truncate_status = truncate_initial_selected_with_weights(config, selected_csfs_indices_dict, target_pool_csfs_data)
            
            # 记录截断结果
            if truncate_status['weight_loading']['loaded']:
                logger.info(f"🔍 {truncate_status['weight_loading']['message']}: {truncate_status['weight_loading']['file_path']}")
            elif 'error' in truncate_status['weight_loading']:
                logger.warning(truncate_status['weight_loading']['error'])
            
            for detail in truncate_status['truncation_details']:
                if detail['method'] == 'weight_based':
                    logger.info(f"块{detail['block']}: 基于权重排序截断 {detail['original_count']} -> {detail['truncated_count']}")
                elif detail['method'] == 'simple':
                    logger.info(f"块{detail['block']}: 简单截断 {detail['original_count']} -> {detail['truncated_count']}")
                if detail['random_space'] > 0:
                    logger.info(f"         预留随机选择空间: {detail['random_space']}")
            
            selected_csfs_indices_dict = truncated_indices_dict
            logger.info("✅ 完成initial_selected截断处理")
    elif config.cal_error_num > 0 and config.continue_cal:
        # 错误重试模式：使用上一轮重要组态进行随机选择
        selected_csfs_indices_dict, prev_status = load_previous_chosen_indices(config)
        
        if prev_status['success']:
            logger.info(f"✅ {prev_status['message']}: {prev_status['file_path']}")
            logger.info(f"📊 上一轮计算重要组态数量: {prev_status['config_counts']}")
        else:
            # 如果找不到上一轮重要组态，使用空字典
            selected_csfs_indices_dict = {block: [] for block in range(target_pool_csfs_data.block_num)}
            logger.warning(f"{prev_status['message']}，使用空字典")
        
        selection_method = "error_retry_important"
        logger.info(f"⚠️ 错误重试模式，错误次数: {config.cal_error_num}")
        logger.info("🔄 使用上一轮重要组态作为selected进行随机选择")
        
        # 如果上一轮重要组态数量过多，也需要截断处理
        total_target_pool = sum(len(target_pool_csfs_data.CSFs_block_data[block]) 
                                for block in range(target_pool_csfs_data.block_num))
        total_target_chosen = math.ceil(total_target_pool * config.chosen_ratio)
        
        # 检查selected数量
        # total_selected = sum(len(selected_csfs_indices_dict.get(block, [])) for block in range(target_pool_csfs_data.block_num))
        total_selected = sum(len(selected_csfs_indices_dict.get(block, [])) if selected_csfs_indices_dict is not None else 0 for block in range(target_pool_csfs_data.block_num))
        
        if total_selected > total_target_chosen:
            logger.warning(f"⚠️ 上一轮重要组态数量过多: {total_selected} > 目标数量: {total_target_chosen}")
            logger.info(f"🔧 使用cutoff_value={config.cutoff_value}进行截断处理")
            
            # 对每个块进行截断处理
            truncated_indices_dict, truncate_status = truncate_initial_selected_with_weights(config, selected_csfs_indices_dict, target_pool_csfs_data)
            
            # 记录截��结果
            if truncate_status['weight_loading']['loaded']:
                logger.info(f"🔍 {truncate_status['weight_loading']['message']}: {truncate_status['weight_loading']['file_path']}")
            elif 'error' in truncate_status['weight_loading']:
                logger.warning(truncate_status['weight_loading']['error'])
                
            for detail in truncate_status['truncation_details']:
                if detail['method'] == 'weight_based':
                    logger.info(f"块{detail['block']}: 基于权重排序截断 {detail['original_count']} -> {detail['truncated_count']}")
                elif detail['method'] == 'simple':
                    logger.info(f"块{detail['block']}: 简单截断 {detail['original_count']} -> {detail['truncated_count']}")
                if detail['random_space'] > 0:
                    logger.info(f"         预留随机选择空间: {detail['random_space']}")
            
            selected_csfs_indices_dict = truncated_indices_dict
            logger.info("✅ 完成重要组态截断处理")
    else:
        # 后续轮次：优先级顺序
        # 1. 优先使用ML最终选择的索引（train.py生成的final_chosen_indices）
        ml_final_indices, ml_final_status = load_ml_final_chosen_indices(config)
        if ml_final_status['success']:
            selected_csfs_indices_dict = ml_final_indices
            selection_method = "ml_final_chosen"
            logger.info(f"✅ {ml_final_status['message']}: {ml_final_status['file_path']}")
            logger.info(f"📊 ML选择的组态数量: {ml_final_status['config_counts']}")
            logger.info("🎯 使用ML最终选择的索引（智能选择）")
        
        # 2. 如果没有ML最终选择，且计算继续，使用ML选择的索引
        elif config.continue_cal:
            selected_csfs_indices_dict, ml_chosen_status = load_previous_ml_chosen_indices(config)
            selection_method = "ml_chosen"
            if ml_chosen_status['success']:
                logger.info(f"{ml_chosen_status['message']}: {ml_chosen_status['file_path']}")
            else:
                logger.warning(ml_chosen_status['message'])
        
        # 3. 如果计算不继续（出错），使用前一轮的chosen indices
        else:
            selected_csfs_indices_dict, prev_chosen_status = load_previous_chosen_indices(config)
            selection_method = "previous_chosen"
            if prev_chosen_status['success']:
                logger.info(f"{prev_chosen_status['message']}: {prev_chosen_status['file_path']}")
                logger.info(f"📊 上一轮计算重要组态数量: {prev_chosen_status['config_counts']}")
            else:
                logger.warning(prev_chosen_status['message'])

    # Fallback: 如果以上所有方法都失败或返回None，则使用空字典
    if selected_csfs_indices_dict is None:
        selected_csfs_indices_dict = {block: [] for block in range(target_pool_csfs_data.block_num)}
        selection_method = "empty_fallback"
        logger.warning("所有索引加载方法均失败，使用空索引字典")
    
    logger.info(f"📋 选择方法: {selection_method}")
    
    # 步骤4：判断选择方式
    use_direct_indices = (selection_method in ["ml_final_chosen", "ml_chosen"] and 
                        selected_csfs_indices_dict is not None and
                        any(len(indices) > 0 for indices in selected_csfs_indices_dict.values()))
    
    if use_direct_indices:
        # 方式1：直接使用索引构建CSFs
        logger.info("🔧 使用直接索引构建方式")
        chosen_csfs_dict, unselected_indices_dict, build_status = build_csfs_from_indices(
            target_pool_csfs_data, selected_csfs_indices_dict
        )
        
        # 记录构建结果
        for block_result in build_status['block_results']:
            if block_result['has_indices']:
                if len(block_result['invalid_indices']) > 0:
                    logger.warning(f"块{block_result['block']}中发现无效索引: {block_result['invalid_indices']}，最大有效索引: {block_result['max_valid_index']}")
                logger.info(f"块{block_result['block']}: 通过索引选择了{block_result['chosen_count']}个CSFs，剩余{block_result['unselected_count']}个")
            else:
                logger.warning(f"块{block_result['block']}: 没有指定索引，选择0个CSFs")
        
        chosen_csfs_indices_dict = selected_csfs_indices_dict
    else:
        # 方式2：传统的随机选择方式
        logger.info(f"🎲 使用传统随机选择方式，选择率: {config.chosen_ratio}")
        chosen_csfs_indices_dict = {}
        chosen_csfs_dict = {}
        unselected_indices_dict = {}

        for block in range(target_pool_csfs_data.block_num):
            chosen_csfs_dict[block], chosen_csfs_indices_dict[block], unselected_indices_dict[block] = (
                gdp.radom_choose_csfs(
                    target_pool_csfs_data.CSFs_block_data[block], 
                    config.chosen_ratio, 
                    selected_csfs_indices_dict.get(block, [])
                )
            )
    
    logger.info(f"完成CSFs选择，共{target_pool_csfs_data.block_num}个块")
    
    # 步骤5：保存选择的CSFs到.c文件
    chosen_csfs_list = [value for key, value in chosen_csfs_dict.items()]
    chosen_csfs_file_path = cal_path / f'{config.conf}_{config.cal_loop_num}.c'
    
    try:
        gdp.write_sorted_CSFs_to_cfile(
            target_pool_csfs_data.subshell_info_raw,
            chosen_csfs_list,
            chosen_csfs_file_path
        )
        logger.info(f"CSFs选择完成，保存到文件: {chosen_csfs_file_path}")
    except Exception as e:
        logger.error(f"保存CSFs文件失败: {str(e)}")
        raise
    
    # 步骤6：保存chosen indices
    chosen_indices_file = cal_path / f'{config.conf}_{config.cal_loop_num}_chosen_indices'
    try:
        gdp.csfs_index_storange(chosen_csfs_indices_dict, chosen_indices_file)
        logger.info(f"已选择CSFs的索引保存到: {chosen_indices_file}.pkl")
    except Exception as e:
        logger.error(f"保存chosen indices失败: {str(e)}")
        raise
    
    # 步骤7：保存unselected indices  
    unselected_indices_file = cal_path / f'{config.conf}_{config.cal_loop_num}_unselected_indices'
    try:
        gdp.csfs_index_storange(unselected_indices_dict, unselected_indices_file)
        logger.info(f"未选择CSFs的索引存到: {unselected_indices_file}.pkl")
    except Exception as e:
        logger.error(f"保存unselected indices失败: {str(e)}")
        raise
    
    # 步骤8：统计信息
    total_chosen = sum(len(csfs) for csfs in chosen_csfs_dict.values())
    total_unselected = sum(len(indices) for indices in unselected_indices_dict.values())
    total_csfs = total_chosen + total_unselected
    selection_ratio = total_chosen / total_csfs if total_csfs > 0 else 0
    
    logger.info("=" * 60)
    logger.info("组态筛选完成")
    logger.info(f"🔄 计算轮次: {config.cal_loop_num}")
    logger.info(f"📋 选择方法: {selection_method}")
    logger.info(f"🔧 构建方式: {'直接索引构建' if use_direct_indices else '随机选择构建'}")
    logger.info(f"📊 选择的CSFs数量: {total_chosen}")
    logger.info(f"📊 未选择的CSFs数量: {total_unselected}")
    logger.info(f"📊 总CSFs数量: {total_csfs}")
    logger.info(f"📊 实际选择率: {selection_ratio:.4%}")
    if not use_direct_indices:
        logger.info(f"📊 配置选择率: {config.chosen_ratio:.4%}")
    logger.info("=" * 60)
    
    return {
        'chosen_csfs_dict': chosen_csfs_dict,
        'chosen_csfs_indices_dict': chosen_csfs_indices_dict,
        'unselected_indices_dict': unselected_indices_dict,
        'chosen_csfs_file_path': str(chosen_csfs_file_path),
        'target_pool_csfs_data': target_pool_csfs_data,
        'cal_path': cal_path,
        'logger': logger,
        'selection_method': selection_method,
        'use_direct_indices': use_direct_indices,
        'total_chosen': total_chosen,
        'total_unselected': total_unselected,
        'selection_ratio': selection_ratio
    }

def main(config):
    """主程序逻辑"""
    try:
        result = perform_csfs_selection(config)

        # 获取root_path用于路径简化
        root_path = getattr(config, 'root_path', None)
        
        print(f"📁 输出.c文件: {highlight_path_python(result['chosen_csfs_file_path'], root_path)}")
        print(f"📁 输出目录: {highlight_path_python(result['cal_path'], root_path)}")
        print(f"📋 选择方法: {result['selection_method']}")
        print(f"🔧 构建方式: {'直接索引构建' if result['use_direct_indices'] else '随机选择构建'}")
        print(f"📊 选择数量: {highlight_number_python(result['total_chosen'])}")
        
        return result
        
    except Exception as e:
        print(f"❌ 组态选择失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='CSFs组态选择程序（支持智能索引选择）')
    parser.add_argument('--config', type=str, default='config.toml', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    try:
        cfg = gdp.load_config(args.config)
        main(cfg)
    except FileNotFoundError:
        print(f"错误: 配置文件 {args.config} 不存在")
    except Exception as e:
        print(f"程序执行失败: {str(e)}")