#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :machine_learning_initialization.py
@date :2025/06/09 15:13:58
@author :YenochQin (秦毅)
'''

import logging
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

from ..data_IO import GraspFileLoad, load_csfs_binary, load_descriptors, csfs_index_load, save_descriptors, load_descriptors_with_multi_block
from ..processing.ASF_data_collection import LevelsEnergyData
from ..CSFs_choosing import batch_asfs_mix_square_above_threshold
from ..utils.data_modules import MixCoefficientData

def setup_logging(config):
    """配置日志系统"""
    log_dir = config.root_path / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "machine_learning_training.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_directories(config):
    """创建必要的目录结构"""
    
    directories = ["models", "descripotors", "test_data", "roc_curves", "results"]
    
    for directory in directories:
        (config.root_path / directory).mkdir(parents=True, exist_ok=True)
    
    return '目录创建成功'

def initialize_iteration_results_csv(config, logger=None):
    """
    初始化迭代结果CSV文件的表头
    
    Args:
        config: 配置对象
        logger: 日志记录器
    """
    results_file = Path(config.root_path) / 'results' / 'iteration_results.csv'
    
    # 如果文件已存在，不重新创建表头
    if results_file.exists():
        if logger:
            logger.info(f"迭代结果文件已存在: {results_file}")
        return
    
    # 创建目录
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入表头
    headers = [
        'training_time', 'eval_time', 'execution_time', 'total_time',
        'test_f1', 'test_roc_auc', 'test_accuracy', 'test_precision', 'test_recall',
        'Es_term', 'import_count', 'stay_count', 'MLsampling_ratio', 'chosen_count', 'weight',
        'train_f1', 'train_roc_auc', 'train_accuracy', 'train_precision', 'train_recall'
    ]
    
    with open(results_file, mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    
    if logger:
        logger.info(f"初始化迭代结果CSV文件: {results_file}")

def validate_initial_files(config, logger) -> None:
    """验证初始文件的存在和有效性"""
    # 验证初始CSFs文件
    target_pool_file_path = config.root_path / config.target_pool_file
    try:
        if not target_pool_file_path.is_file():
            logger.error(f"初始CSFs文件无效或不存在: {target_pool_file_path}")
            raise FileNotFoundError(f"初始CSFs文件无效或不存在: {target_pool_file_path}")
        logger.info(f"成功加载初始CSFs文件: {target_pool_file_path}")
    except PermissionError as e:
        logger.error(f"无权限访问CSFs文件: {target_pool_file_path}")
        raise
    except Exception as e:
        logger.error(f"加载CSFs文件时发生未知错误: {str(e)}")
        raise


def load_data_files(config, logger) -> tuple:
    """加载数据文件"""
    # config.yaml文件读取时已经处理好root_path和config.scf_cal_path路径
    
    # 加载能级文件
    energy_level_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.level'
    energy_level_file_load = LevelsEnergyData.from_filepath(str(energy_level_file_path), 'LEVEL')
    energy_level_data_pd = energy_level_file_load.energy_level_2_pd()
    logger.info(f"加载能级数据: {energy_level_file_path}")
    
    # 加载rmix文件
    cal_method = getattr(config, 'cal_method', 'rmcdhf')  # 默认使用rmcdhf方法
    
    if cal_method == 'rci':
        rmix_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.cm'
    elif cal_method == 'rmcdhf':
        rmix_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.m'
    else:
        raise ValueError(f"不支持的计算方法: {cal_method}，请在config.toml中设置cal_method为'rci'或'rmcdhf'")
    
    rmix_file_load = GraspFileLoad.from_filepath(str(rmix_file_path), 'mix')
    rmix_file_data = rmix_file_load.data_file_process()
    logger.info(f"加载 *.m 文件数据: {rmix_file_path}")
    
    # 加载初始 CSFs 文件
    target_pool_file_path = config.root_path / f'{config.conf}'
    target_pool_binary_file_path = target_pool_file_path.with_suffix('.pkl.gz')
    target_pool_csfs_data = load_csfs_binary(target_pool_binary_file_path)
    logger.info(f"加载初始 CSFs 文件: {target_pool_binary_file_path}")
    
    # 加载初始 CSFs 描述符文件
    result = load_descriptors_with_multi_block(target_pool_file_path, 'npy')
    if result is None:
        raise FileNotFoundError(f"无法加载初始 CSFs 描述符文件: {target_pool_file_path}")
    raw_csfs_descriptors, raw_csfs_indices = result
    logger.info(f"加载初始 CSFs 描述符文件: {target_pool_file_path}")
    
    # 加载本轮计算CSFs文件
    cal_csfs_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.c'
    cal_csfs_file_laod = GraspFileLoad.from_filepath(str(cal_csfs_file_path), 'CSFs')
    cal_csfs_data = cal_csfs_file_laod.data_file_process()
    logger.info(f"加载本轮计算 CSFs 文件: {cal_csfs_file_path}")
    
    # 加载本轮选择的CSFs的索引文件
    caled_csfs_indices_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}_chosen_indices.pkl'
    caled_csfs_indices_dict = csfs_index_load(caled_csfs_indices_file_path)
    logger.info(f"加载本轮选择的 CSFs 的索引文件: {caled_csfs_indices_file_path}")
    
    # 加载本轮未选择的CSFs的索引文件
    unselected_csfs_indices_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}_unselected_indices.pkl'
    unselected_csfs_indices_dict = csfs_index_load(unselected_csfs_indices_file_path)
    logger.info(f"加载本轮选择的 CSFs 的索引文件: {unselected_csfs_indices_file_path}")

    return energy_level_data_pd, rmix_file_data, target_pool_csfs_data, raw_csfs_descriptors, cal_csfs_data, caled_csfs_indices_dict, unselected_csfs_indices_dict

def check_configuration_coupling(config, energy_level_data_pd, logger):
    """检查组态耦合是否正确"""
    cal_configuration_list = energy_level_data_pd['configuration'].tolist()
    
    # 检查每个光谱项是否有且仅有一次出现，并记录位置
    spectral_term_positions = []
    all_found_once = True
    
    for term in config.spetral_term:
        count = cal_configuration_list.count(term)
        if count == 1:
            position = cal_configuration_list.index(term)
            spectral_term_positions.append(position)
            logger.info(f"光谱项 '{term}' 在位置 {position} 找到")
        elif count == 0:
            logger.error(f"光谱项 '{term}' 未找到")
            all_found_once = False
        else:
            logger.error(f"光谱项 '{term}' 出现 {count} 次，应该有且仅有一次")
            all_found_once = False
    
    if all_found_once:
        logger.info(f"cal_loop {config.cal_loop_num} 组态耦合正确，位置索引: {spectral_term_positions}")
        return True, spectral_term_positions
    else:
        logger.error(f"cal_loop {config.cal_loop_num} 组态耦合错误")
        return False, []


def generate_chosen_csfs_descriptors(
                                    config, 
                                    chosen_csfs_indices_dict: Dict, 
                                    raw_csfs_descriptors: np.ndarray, 
                                    rmix_file_data: MixCoefficientData, asfs_position: List[int], 
                                    logger, 
                                    include_wrong_level_negatives: bool = False) -> np.ndarray:
    """
    生成用于机器学习训练的CSFs描述符数据
    
    Args:
        config: 配置对象
        chosen_csfs_indices_dict: 选中的CSFs索引字典
        raw_csfs_descriptors: 原始CSFs描述符数组
        rmix_file_data: 混合系数数据
        asfs_position: 正确能级位置索引列表
        logger: 日志记录器
        include_wrong_level_negatives: 是否包含错误能级的组态作为负样本
        
    Returns:
        np.ndarray: 包含描述符和标签的训练数据
    """
    
    # 验证字典并安全获取block 0的索引
    if not chosen_csfs_indices_dict:
        raise ValueError("chosen_csfs_indices_dict为空，无法获取选中的CSFs索引")
    
    if 0 not in chosen_csfs_indices_dict:
        raise KeyError(f"chosen_csfs_indices_dict中缺少键0，可用键: {list(chosen_csfs_indices_dict.keys())}")
    
    selected_indices = np.array(chosen_csfs_indices_dict[0])
    selected_csfs_descriptors = raw_csfs_descriptors[selected_indices]
    
    # 获取所有能级的混合系数数据
    cal_mix_coeffs = rmix_file_data.mix_coefficient_List[0]  # shape: (n_levels, n_csfs)
    
    ## 方案1：仅使用正确能级位置的数据（原有方案）
    if not include_wrong_level_negatives:
        # 获取正确能级位置的混合系数平方和
        correct_mix_coeff_squared_sum = np.sum(cal_mix_coeffs[asfs_position]**2, axis=0)
        
        # 生成标签（只基于正确能级位置的混合系数）
        csf_mix_coeff_descriptors = correct_mix_coeff_squared_sum >= np.float64(config.cutoff_value)
        
        logger.info(f"使用原有方案（仅正确能级）")
        logger.info(f"正确能级位置: {asfs_position}")
        logger.info(f"本轮计算CSFs数量: {len(selected_csfs_descriptors)}")
        logger.info(f"超过阈值的CSFs数量: {np.sum(csf_mix_coeff_descriptors)}")
        logger.info(f"正样本比例: {np.sum(csf_mix_coeff_descriptors)/len(csf_mix_coeff_descriptors):.4f}")
        
        caled_csfs_descriptors = np.column_stack([selected_csfs_descriptors, csf_mix_coeff_descriptors])
    
    ## 方案2：增强方案 - 包含错误能级的组态作为负样本
    else:
        # 重要：cal_mix_coeffs的shape是(n_levels, cal_csfs_num)
        # 其中cal_csfs_num是本轮计算的CSFs数量
        
        # 获取正确能级位置的混合系数平方和
        correct_mix_coeff_squared_sum = np.sum(cal_mix_coeffs[asfs_position]**2, axis=0)
        
        # 获取错误能级位置的索引
        wrong_level_indices = list(set(range(cal_mix_coeffs.shape[0])) - set(asfs_position))
        
        # 获取错误能级位置的混合系数平方和
        if len(wrong_level_indices) > 0:
            wrong_mix_coeff_squared_sum = np.sum(cal_mix_coeffs[wrong_level_indices]**2, axis=0)
        else:
            wrong_mix_coeff_squared_sum = np.zeros(cal_mix_coeffs.shape[1])
        
        # 生成增强标签
        cutoff_value = np.float64(config.cutoff_value)
        
        # 正样本：在正确能级位置有较高混合系数
        positive_mask = correct_mix_coeff_squared_sum >= cutoff_value
        
        # 负样本包括：
        # 1. 在正确能级位置混合系数较低的CSFs
        # 2. 在错误能级位置有较高混合系数但在正确能级位置较低的CSFs（这些是"坏"组态）
        negative_mask_low_correct = correct_mix_coeff_squared_sum < cutoff_value
        negative_mask_high_wrong = (wrong_mix_coeff_squared_sum >= cutoff_value * 100) & (correct_mix_coeff_squared_sum < cutoff_value)
        
        # 最终标签：正样本为True，负样本为False
        csf_mix_coeff_descriptors = positive_mask
        
        # 统计信息
        n_positive = np.sum(positive_mask)
        n_negative_low_correct = np.sum(negative_mask_low_correct & ~negative_mask_high_wrong)
        n_negative_high_wrong = np.sum(negative_mask_high_wrong)
        n_total = len(selected_csfs_descriptors)
        
        logger.info(f"使用增强方案（包含错误能级负样本）")
        logger.info(f"正确能级位置: {asfs_position}")
        logger.info(f"错误能级位置: {wrong_level_indices}")
        logger.info(f"本轮计算CSFs数量: {n_total}")
        logger.info(f"正样本数量: {n_positive} (在正确能级位置混合系数 ≥ {cutoff_value})")
        logger.info(f"负样本数量: {n_total - n_positive}")
        logger.info(f"  - 正确能级位置低混合系数: {n_negative_low_correct}")
        logger.info(f"  - 错误能级位置高混合系数: {n_negative_high_wrong}")
        logger.info(f"正样本比例: {n_positive/n_total:.4f}")
        logger.info(f"错误能级高混合系数比例: {n_negative_high_wrong/n_total:.4f}")
        
        # 如果有错误能级的高混合系数组态，说明这些是"坏"组态
        if n_negative_high_wrong > 0:
            logger.info(f"发现 {n_negative_high_wrong} 个在错误能级有高混合系数的组态，这些将作为负样本帮助模型学习识别错误组态")
        
        caled_csfs_descriptors = np.column_stack([selected_csfs_descriptors, csf_mix_coeff_descriptors])
    
    # 保存描述符文件
    cal_path = config.root_path / f'{config.conf}_{config.cal_loop_num}'
    save_descriptors(caled_csfs_descriptors, f'{cal_path}/{config.conf}_{config.cal_loop_num}', 'npy')
    logger.info(f"保存本轮选择的 CSFs 的描述符文件: {cal_path}/{config.conf}_{config.cal_loop_num}.npy")

    return caled_csfs_descriptors



def get_stay_descriptors(raw_csfs_descriptors: np.ndarray, chosen_csfs_indices_dict: Dict[int, List[int]]) -> np.ndarray:
    """
    找出不在chosen_csfs_indices_dict索引中的描述符
    
    Args:
        raw_csfs_descriptors: 原始CSFs描述符数组
        chosen_csfs_indices_dict: 已选择的CSFs索引字典，格式为{block_index: [indices]}
        
    Returns:
        np.ndarray: 不在chosen_csfs_indices_dict中的描述符数组
    """
    # 验证字典并安全获取所有已选择的索引
    if not chosen_csfs_indices_dict:
        raise ValueError("chosen_csfs_indices_dict为空，无法获取选中的CSFs索引")
    
    if 0 not in chosen_csfs_indices_dict:
        raise KeyError(f"chosen_csfs_indices_dict中缺少键0，可用键: {list(chosen_csfs_indices_dict.keys())}")
    
    chosen_indices = set(chosen_csfs_indices_dict[0])
    
    # 获取所有可能的索引
    all_indices = set(range(len(raw_csfs_descriptors)))
    
    # 找出不在chosen_indices中的索引
    stay_indices = list(all_indices - chosen_indices)
    
    # 返回对应的描述符
    return raw_csfs_descriptors[stay_indices]
