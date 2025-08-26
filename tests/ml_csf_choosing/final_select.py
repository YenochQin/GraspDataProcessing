#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :final_select.py
@date :2025/08/26 15:22:57
@author :YenochQin (秦毅)
'''
import argparse
import logging
from typing import Dict, Tuple, List
from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

import graspdataprocessing as gdp


def final_csfs_select(config, logger):
    
    # 加载rmix文件
    # 根据计算轮次确定文件后缀
    if config.cal_method == 'rmcdhf':
        rmix_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.m'
    elif config.cal_method == 'rci':
        rmix_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.cm'
    else:
        raise ValueError(f"不支持的计算方法: {config.cal_method}")
    
    rmix_file_load = gdp.GraspFileLoad.from_filepath(str(rmix_file_path), 'mix')
    rmix_file_data = rmix_file_load.data_file_process()
    logger.info(f"加载 mix coefficient 数据文件: {rmix_file_path}")
    
    # 加载本轮计算CSFs文件
    cal_csfs_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.c'
    cal_csfs_file_laod = gdp.GraspFileLoad.from_filepath(str(cal_csfs_file_path), 'CSFs')
    cal_csfs_data = cal_csfs_file_laod.data_file_process()
    logger.info(f"加载本轮计算 CSFs 文件: {cal_csfs_file_path}")
    
    csfs_final_coupling_J_collection = gdp.batch_blocks_CSFs_final_coupling_J_collection(cal_csfs_data.CSFs_block_data, rmix_file_data.mix_coefficient_list)
    
    levels_import_csfs_indices = gdp.batch_asfs_mix_square_above_threshold(rmix_file_data, config.cutoff_value)
    import_csfs_indices = gdp.merge_csfs_indices_lists_by_block_key(levels_import_csfs_indices)

    selected_csfs = []
    for block_key, indices in import_csfs_indices.items():
        block_selected_csfs = []
        for idx in indices:
            # 获取对应组态，这里假设组态列表是按block组织的
            csf = gdp.CSFs_block_get_CSF(cal_csfs_data.CSFs_block_data[block_key], (block_key,))
            block_selected_csfs.append(csf)
            
        selected_csfs.append(block_selected_csfs)

    chosen_middle_J_index = {}
    for block, value in csfs_final_coupling_J_collection.items():
        for level in range(rmix_file_load.block_energy_count_list[block]):
            max_ci_key = max(value, key=lambda k: value[k]['sum_ci'][level])
            print(f"    Chosen middle J for block {block}, level {level}: {max_ci_key}")
            print(f"    Max CI: {value[max_ci_key]['sum_ci'][level]}")
            chosen_middle_J_index[(block, level)] = value[max_ci_key]['indices']

    merged_dict = defaultdict(list)
    for key, indices_list in chosen_middle_J_index.items():
        block = key[0]
        merged_dict[block].extend(indices_list)

    final_selected_csfs_indices = {}
    for (block, chosen_index), (block1, middle_j_index) in zip(import_csfs_indices.items(), merged_dict.items()):
        print(f"Block {block}")
        if block == block1:
            final_selected_csfs_indices[block] = gdp.union_lists_with_order(chosen_index, middle_j_index)

    sorted_csfs_data = []
    for block, csfs_indices in final_selected_csfs_indices.items():
        block_csfs_data =[]
        for index in csfs_indices:
            # print(index)
            # print(cal_csfs_data.CSFs_block_length[block])
            if not index > cal_csfs_data.CSFs_block_length[block]:
                block_csfs_data.extend(cal_csfs_data.CSFs_block_data[block][index])
        sorted_csfs_data.append(block_csfs_data)
        
    sorted_csfs_file_path = config.scf_cal_path / f"{config.conf}_{config.cal_loop_num}_mJ-1_im{config.cutoff_value:.0e}.c"

    new_csfs_data = gdp.write_sorted_CSFs_to_cfile(
                        cal_csfs_data.subshell_info_raw, 
                        sorted_csfs_data, 
                        output_file=sorted_csfs_file_path)

    return new_csfs_data

def main(config):
    """主程序逻辑"""
    logger = gdp.setup_logging(config)
    logger.info("Final CSFs selection")

    try:
        result = final_csfs_select(config, logger)

        
    except Exception as e:
        print(f"❌ Target Pool CSFs 数据预处理失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Target Pool CSFs 数据预处理程序')
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