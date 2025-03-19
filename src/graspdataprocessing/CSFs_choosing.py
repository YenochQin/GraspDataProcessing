#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :CSFs_choosing.py
@date :2024/08/02 20:38:24
@author :YenochQin (秦毅)
'''

import numpy as np
import pandas as pd
# from pathlib import Path
from tqdm import tqdm
from .data_IO import GraspFileLoad
import re
from typing import Dict, Tuple, List

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
    abs_above_threshold_mask = np.abs(mix_data_array) > threshold

    # 提取超过阈值的实际值
    values_above_threshold = mix_data_array[abs_above_threshold_mask]

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

def level_mix_above_threshold_coupling_info(mix_data_above_threshold_list: List, csf_data_list: List):
    
    csf_coupling_info = []

    mix_data_dim = mix_data_above_threshold_list.shape[0]
    
    for i in mix_data_dim:
        












#######################################################################





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