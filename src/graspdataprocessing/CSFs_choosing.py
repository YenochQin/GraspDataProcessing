#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :CSFs_choosing.py
@date :2024/08/02 20:38:24
@author :YenochQin (秦毅)
'''
import re
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from tqdm import tqdm

from .CSFs_compress_extract import *


#######################################################################




def level_mix_data_abs_above_threshold(level_mix_data_array: np.ndarray, threshold=0.1):
    """
    筛选并排序混合数据中绝对值超过阈值的元素
    
    参数：
        mix_data_array: 输入的混合数据数组
        threshold: 阈值，默认0.1
        
    返回：
        按绝对值降序排列的字典，键为索引数组（转换为元组），值为对应元素值
    """
    # 生成布尔掩码标识绝对值超过阈值的元素
    abs_above_threshold_mask = np.abs(level_mix_data_array) > threshold

    # 提取超过阈值的实际值
    values_above_threshold = level_mix_data_array[abs_above_threshold_mask]

    # 获取满足条件的元素索引（返回二维坐标数组）
    indices_where_abs_above_threshold = np.argwhere(abs_above_threshold_mask)

    # 将索引数组转换为元组作为键，值与索引配对组合
    result_dict = {tuple(index): value for index, value in zip(indices_where_abs_above_threshold, values_above_threshold)}
    
    # 按元素绝对值降序排序
    sorted_result = dict(sorted(result_dict.items(), key=lambda x: abs(x[1]), reverse=True))
    
    return sorted_result

def csf_mix_data_abs_above_threshold(level_mix_data: Dict, threshold=0.1):

    for block in range(level_mix_data['CSFs_blocks_num']):
        block_level_num = len(level_mix_data['mix_coefficient_list'][block])
    
        for i in range(block_level_num):
            temp_coeff = level_mix_data_abs_above_threshold(level_mix_data['mix_coefficient_list'][block][i], threshold=0.1)
            level_mix_data[f'block{block}_No{i}'] = temp_coeff
            
    return level_mix_data

def level_mix_above_threshold_coupling_info(mix_data_above_threshold: Dict, csf_data_list: List):
    csf_list = []
    csf_mid_coupling_info = []
    csf_coupling_info = []

        
    for key, value in mix_data_above_threshold.items():
        temp_csf_data = CSFs_block_get_CSF(csf_data_list, key)
        csf_list.append(temp_csf_data[0])
        temp_csf_data_dict = CSF_item_2_dict(temp_csf_data)
        
        csf_mid_coupling_info.append(temp_csf_data_dict['temp_coupled_j'])
        csf_coupling_info.append(temp_csf_data_dict['final_coupled_j_parity'])

    return csf_list, csf_mid_coupling_info, csf_coupling_info












#######################################################################

def CSFs_block_get_CSF(CSFs_block: List, CSf_index: Tuple) -> List:
    """
    根据CSF的索引获取对应的CSF

    参数：
        CSFs_block: 包含CSF的列表
        CSf_index: 要获取的CSF的索引，元组形式

    返回：
        对应的CSF，如果索引无效则返回None
    """
    # 检查索引是否有效
    if CSf_index[0] < 0 and len(CSFs_block)%3 == 0:
        raise ValueError("CSF index must be non-negative, and CSFs_block length must be a multiple of 3.")
    
    return CSFs_block[CSf_index[0]*3:CSf_index[0]*3+3]





#######################################################################


# def main():
#     if len(sys.argv) != 2:
#         print("用法: python test.py <文件名>")
#         sys.exit(1)
        
#     csf_file = sys.argv[1]
#     load_csf_data = []
#     with open(csf_file, 'r') as load_csf_file:
#         load_csf_data = load_csf_file.readlines()


#     csf_list = load_csf_data[5:]

#     csf_block_list = split_by_asterisk(csf_list)
#     print(len(csf_block_list[0]), len(csf_block_list[1]), len(csf_block_list[2]))
    
#     csf_random_list = []
#     for block in csf_block_list:
#         csf_random_list.append(shuffle_three_line_groups(block))
        
#     with open(f'random.c', 'w') as write_csf_file:
#         write_csf_file.write(''.join(load_csf_data[:5]))
#         for index, block in enumerate(csf_random_list):
#             write_csf_file.write(''.join(block))
#             if index != len(csf_random_list) - 1:
#                 write_csf_file.write(' *\n')

# if __name__ == "__main__":
#     main()