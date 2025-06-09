#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :machine_learning_initialization.py
@date :2025/06/09 15:13:58
@author :YenochQin (秦毅)
'''
import os

import logging
from pathlib import Path
import csv

from .data_IO import GraspFileLoad, load_csfs_binary, load_descriptors, csfs_index_load
from ASF_data_collection import LevelsEnergyData

def setup_logging(config):
    """配置日志系统"""
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/machine_learning_training.log"),
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

def initialize_results_file(config, logger):
    """初始化结果CSV文件"""
    result_csv_path = config.root_path / 'results' / 'results.csv'
    try:
        if not result_csv_path.exists():
            with result_csv_path.open(mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    'training_time', 'eval_time', 'abinitio_time', 'all_time',
                    'f1', 'roc_auc', 'accuracy', 'precision', 'recall',
                    'Es', 'abimport_csfnum', 'MLimport_csfnum', 'MLsampling_ratio', 'next_itr_num',
                    'weight', 'f1_train', 'roc_auc_train', 'accuracy_train', 'precision_train', 'recall_train'
                ])
    except IOError as e:
        logger.error(f"无法创建结果文件 {result_csv_path}: {str(e)}")
        raise

def validate_initial_files(config, logger):
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
    
    return target_pool_file_path


def load_data_files(config):
    """加载数据文件"""
    # config.yaml文件读取时已经处理好root_path和config.scf_cal_path路径
    
    # 加载能级文件
    energy_level_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.level'
    energy_level_file_load = LevelsEnergyData.from_filepath(str(energy_level_file_path), 'LEVEL')
    energy_level_data_pd = energy_level_file_load.energy_level_2_pd()
    
    # 加载rmix文件
    rmix_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.m'
    rmix_file_load = GraspFileLoad.from_filepath(str(rmix_file_path), 'mix')
    rmix_file_data = rmix_file_load.data_file_process()
    
    # 加载原始CSFs文件
    target_pool_file_path = config.root_path / f'{config.conf}'
    target_pool_csfs_data = load_csfs_binary(target_pool_file_path)
    
    # 加载原始CSFs描述符文件
    raw_csfs_descriptors = load_descriptors(target_pool_file_path)
    
    # 加载本轮计算CSFs文件
    cal_csfs_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.c'
    cal_csfs_file_laod = GraspFileLoad.from_filepath(str(cal_csfs_file_path), 'CSFs')
    cal_csfs_data = cal_csfs_file_laod.data_file_process()
    
    # 加载本轮选择的CSFs的索引文件
    caled_csfs_indices_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}_chosen_indices.pkl'
    caled_csfs_indices_dict = csfs_index_load(caled_csfs_indices_file_path)
    
    return energy_level_data_pd, rmix_file_data, target_pool_csfs_data, raw_csfs_descriptors, cal_csfs_data, caled_csfs_indices_dict

def check_configuration_coupling(config, energy_level_data_pd, logger):
    """检查组态耦合是否正确"""
    cal_configuration_set = set(energy_level_data_pd['configuration'])
    
    if set(config.spetral_term).issubset(cal_configuration_set):
        logger.info(f"cal_loop {config.cal_loop_num} 组态耦合正确")
        return True
    else:
        logger.error(f"cal_loop {config.cal_loop_num} 组态耦合错误")
        return False


