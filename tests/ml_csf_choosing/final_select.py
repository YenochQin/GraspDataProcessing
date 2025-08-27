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
    logger.info(f"加载 mix coefficient 数据文件: {rmix_file_path=}")
    
    # 加载本轮计算CSFs文件
    cal_csfs_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.c'
    cal_csfs_file_laod = gdp.GraspFileLoad.from_filepath(str(cal_csfs_file_path), 'CSFs')
    cal_csfs_data = cal_csfs_file_laod.data_file_process()
    logger.info(f"加载本轮计算 CSFs 文件: {cal_csfs_file_path=}")

    energy_level_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.level'
    energy_level_file_load = gdp.LevelsEnergyData.from_filepath(str(energy_level_file_path), 'LEVEL')
    energy_level_data_pd = energy_level_file_load.energy_level_2_pd()
    logger.info(f"加载能级数据: {energy_level_file_path=}")
    # 检查组态耦合
    cal_result, asfs_position = gdp.check_configuration_coupling(config, energy_level_data_pd, logger)
    
    csfs_final_coupling_J_collection = gdp.single_block_batch_asfs_CSFs_final_coupling_J_collection(
                                                    cal_csfs_data.CSFs_block_data[0], 
                                                    rmix_file_data.mix_coefficient_List[0], 
                                                    asfs_position
                                                    )
    
    levels_import_csfs_indices = gdp.batch_asfs_mix_square_above_threshold(
                                            asfs_mix_data=rmix_file_data, 
                                            threshold=config.cutoff_value, 
                                            asfs_position=asfs_position
                                            )
    block=0 ## temporary use
    logger.info(f" important csfs (mix coeff > {config.cutoff_value=}) num: {len(levels_import_csfs_indices[block])}")

    chosen_middle_J_index = {}
    for level in asfs_position:
        max_ci_key = max(csfs_final_coupling_J_collection, key=lambda k: csfs_final_coupling_J_collection[k]['sum_ci'][level])
        logger.info(f"    Chosen middle J for level {level=}: {max_ci_key=}")
        logger.info(f"    Max CI: {csfs_final_coupling_J_collection[max_ci_key]['sum_ci'][level]}")
        chosen_middle_J_index[(block, level)] = np.array(csfs_final_coupling_J_collection[max_ci_key]['indices'])
        
    asfs_chosen_middle_J_indices = {}
    asfs_chosen_middle_J_indices[block] = [
                chosen_middle_J_index[(block, lvl)]             # 这里也可以先 .ravel() 如果数组是多维
                for lvl in asfs_position
                if (block, lvl) in chosen_middle_J_index
                ]
    logger.info(f" select asfs middle J chosen indices length: {len(asfs_chosen_middle_J_indices[block])}")
    
    final_selected_csfs_indices = {}
    final_selected_csfs_indices[block] = np.union1d(levels_import_csfs_indices[block], asfs_chosen_middle_J_indices[block])

    selected_csfs = []
    for block in range(cal_csfs_data.block_num):
        block_csfs = gdp.CSFs_block_get_CSF(cal_csfs_data.CSFs_block_data[block], final_selected_csfs_indices[block])
        logger.info(f" {block=} selected csfs num: {len(block_csfs)=}")
        selected_csfs.append(block_csfs)
        
    sorted_csfs_file_path = config.scf_cal_path / f"{config.conf}_{config.cal_loop_num}_mJ-1_im{config.cutoff_value:.0e}.c"

    new_csfs_data = gdp.write_sorted_CSFs_to_cfile(
                        cal_csfs_data.subshell_info_raw, 
                        selected_csfs, 
                        output_file=sorted_csfs_file_path)
    logger.info(f" csfs file writed to path: {sorted_csfs_file_path=}")

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