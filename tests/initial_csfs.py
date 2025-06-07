#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :initial_csfs.py
@date :2025/05/25 13:52:19
@author :YenochQin (秦毅)
'''
import yaml
import argparse
import logging
from types import SimpleNamespace
import os
from pathlib import Path
import csv
import sys
import math
import numpy as np
import pandas as pd
import time
import joblib
import json 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
sys.path.append('D:\\PythonProjects\\GraspDataProcessing\\src')
import graspdataprocessing as gdp

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)

def update_config(config_path, updates):
    """更新YAML配置文件
    
    Args:
        config_path: 配置文件路径
        updates: 要更新的键值对字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新配置值
    config.update(updates)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

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

def main(config):
    """主程序逻辑"""
    logger = setup_logging(config)
    logger.info("CSFs选择程序启动")
    logger.info(f'计算循环次数: {config.cal_loop_num}')

    # 使用 pathlib 创建目录
    root_path = Path(config.root_path)
    
    cal_path = root_path.joinpath(f'{config.conf}_{config.cal_loop_num}')
    
    # 确保输出目录存在
    cal_path.mkdir(exist_ok=True)

    logger.info(f"初始比例: {config.initial_ratio}")
    logger.info(f"光谱项: {config.spetral_term}")
    
    target_pool_file_path = root_path.joinpath(config.target_pool_file)
    try:
        if not target_pool_file_path.is_file():  # 检查是否是有效文件
            logger.error(f"初始CSFs文件无效或不存在: {target_pool_file_path}")
            raise FileNotFoundError(f"初始CSFs文件无效或不存在: {target_pool_file_path}")
        logger.info(f"成功加载初始CSFs文件: {target_pool_file_path}")
    except PermissionError as e:
        logger.error(f"无权限访问CSFs文件: {target_pool_file_path}")
        raise
    except Exception as e:
        logger.error(f"加载CSFs文件时发生未知错误: {str(e)}")
        raise
    
    if config.cal_loop_num == 1:

        target_pool_csfs_load = gdp.GraspFileLoad.from_filepath(target_pool_file_path, file_type='CSF')
        target_pool_csfs_data = target_pool_csfs_load.data_file_process()
        
        logger.info(f"初始CSFs文件{config.target_pool_file} CSFs 读取成功")
        
        descriptors_array, labels_array = gdp.batch_process_csfs_with_multi_block(target_pool_csfs_data, label_type='sequential')
        logger.info(f"初始CSFs文件{config.target_pool_file} CSFs 描述符计算成功")

        target_pool_path = root_path.joinpath(config.conf)

        gdp.save_descriptors_with_multi_block(descriptors_array, labels_array, target_pool_path, 'npy')
        logger.info(f"初始CSFs文件{config.target_pool_file} CSFs 描述符保存成功")

        gdp.save_csfs_binary(target_pool_csfs_data, target_pool_path)
        logger.info(f"初始CSFs文件{config.target_pool_file} CSFs 保存成功")
        
        if config.selected_csfs_file:
            gdp.precompute_large_hash(target_pool_csfs_data.CSFs_block_data, target_pool_path.with_suffix('.pkl'))
            logger.info(f"初始CSFs文件{config.target_pool_file} CSFs 哈希校验文件保存成功")
            
            target_pool_csfs_hash_file = target_pool_path.with_suffix('.pkl')
            
            selected_csfs_file_path = root_path.joinpath(config.selected_csfs_file)
            selected_csfs_load = gdp.GraspFileLoad.from_filepath(selected_csfs_file_path, file_type='CSF')
            selected_csfs_data = selected_csfs_load.data_file_process()
            logger.info(f"已选择CSFs文件{config.target_pool_file} CSFs 读取成功")
            
            selected_csfs_indices_dict = gdp.maping_two_csfs_indices(
                selected_csfs_data.CSFs_block_data, 
                target_pool_csfs_hash_file
            )
            logger.info(f"已选择CSFs文件{config.target_pool_file} CSFs 索引映射成功")
            
        else:
            selected_csfs_indices_dict = {block: [] for block in range(target_pool_csfs_data.block_num)}
    else:
        target_pool_csfs_data = gdp.load_csfs_binary(target_pool_file_path)
        logger.info(f"初始CSFs文件{target_pool_file_path} CSFs 读取成功")
        previous_cal_path = root_path.joinpath(f'{config.conf}_{config.cal_loop_num-1}')
        selected_csfs_indices_dict = gdp.csfs_index_load(previous_cal_path.joinpath(f'{config.conf}_{config.cal_loop_num-1}_chosen_indices'))

        

    chosen_csfs_indices_dict = {}
    chosen_csfs_dict = {}
    unselected_indices_dict = {}

    for block in range(target_pool_csfs_data.block_num):
        # 计算每个块的初始采样数量
        chosen_csfs_dict[block], chosen_csfs_indices_dict[block], unselected_indices_dict[block] = gdp.radom_choose_csfs(target_pool_csfs_data.CSFs_block_data[block], config.initial_ratio, selected_csfs_indices_dict[block])
    
    chosen_csfs_list = [value for key, value in chosen_csfs_dict.items()] ## 这里是临时的chosen_csfs_dict[0]
    ## CSFs_file_info: List, sorted_CSFs_data_list: List, output_file: str
    gdp.write_sorted_CSFs_to_cfile(
                                    target_pool_csfs_data.subshell_info_raw,
                                    chosen_csfs_list,
                                    cal_path.joinpath(f'{config.conf}_{config.cal_loop_num}.c')
    )
    logger.info(f"CSFs选择完成，保存到文件{config.conf}_{config.cal_loop_num}.c")
    
    gdp.csfs_index_storange(
                            chosen_csfs_indices_dict,
                            cal_path.joinpath(f'{config.conf}_{config.cal_loop_num}_chosen_indices')
    )
    logger.info(f"已选择CSFs的索引保存到文件{config.conf}_{config.cal_loop_num}_chosen_indices.pkl")
    
    gdp.csfs_index_storange(
                            unselected_indices_dict,
                            cal_path.joinpath(f'{config.conf}_{config.cal_loop_num}_unselected_indices')
    )
    logger.info(f"未选择CSFs的索引保存到文件{config.conf}_{config.cal_loop_num}_unselected_indices.pkl")
    
    logger.info("组态筛选完成")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='组态选择程序')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    args = parser.parse_args()

    # 加载配置
    try:
        cfg = load_config(args.config)
        main(cfg)
    except FileNotFoundError as e:
        if args.config in str(e):
            print(f"错误: 配置文件 {args.config} 不存在")
        else:
            print(f"错误: 文件未找到 - {str(e)}")
    except yaml.YAMLError as e:
        print(f"错误: 配置文件解析失败 - {str(e)}")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()