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
from collections import Counter
from dataclasses import dataclass

from tqdm import tqdm

from .CSFs_compress_extract import *

@dataclass(frozen=True)
class MixCoefficientData:
    CSFs_blocks_num: int
    block_index_list: list
    block_CSFs_nums: list
    block_energy_count_list: list 
    j_value_location_list: list
    parity_list: list
    block_levels_index_list: list
    block_energy_list: list
    block_level_energy_list: list
    # mix_coefficient_list shape is [CSFs_blocks_num*array([block_energy_count_list[i]*[ncfblk_list[i]]])]
    mix_coefficient_list: list

#######################################################################
def single_asf_mix_square_above_threshold(asf_mix_data_array: np.ndarray, threshold=0.1):
    """
    筛选并排序混合数据中绝对值超过阈值的元素
    
    参数：
        mix_data_array: 输入的混合数据数组
        threshold: 阈值，默认0.1
        
    返回：
        按绝对值降序排列的字典，键为索引数组（转换为元组），值为对应元素值
    """
    # 生成布尔掩码标识绝对值超过阈值的元素
    square_above_threshold_mask = np.square(asf_mix_data_array) > threshold

    # 提取超过阈值的实际值
    values_above_threshold = asf_mix_data_array[square_above_threshold_mask]

    # 获取满足条件的元素索引（返回二维坐标数组）
    indices_where_square_above_threshold = np.argwhere(square_above_threshold_mask)

    # 将索引数组转换为元组作为键，值与索引配对组合
    result_dict = {tuple(index): value for index, value in zip(indices_where_square_above_threshold, values_above_threshold)}
    
    # 按元素绝对值降序排序
    sorted_result = dict(sorted(result_dict.items(), key=lambda x: abs(x[1]), reverse=True))
    
    return sorted_result

def batch_asfs_mix_square_above_threshold(asfs_mix_data: MixCoefficientData, threshold=0.1):
    
    csf_mix_square_data_above_threshold = {}
    for block in range(asfs_mix_data.CSFs_blocks_num):
        block_level_num = len(asfs_mix_data.mix_coefficient_list[block])
    
        for i in range(block_level_num):
            temp_coeff = single_asf_mix_square_above_threshold(asfs_mix_data.mix_coefficient_list[block][i], threshold)
            csf_mix_square_data_above_threshold[f'block{block}_level{i}'] = temp_coeff
            
    return csf_mix_square_data_above_threshold

def asf_mix_square_above_threshold_coupling_info(mix_square_data_above_threshold: Dict, csf_data_list: List):
    csf_list = []
    csf_mid_coupling_info = []
    csf_coupling_info = []

        
    for key, value in mix_square_data_above_threshold.items():
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

def CSF_final_coupling_J_collection(block_CSFs: List, coupling_level: int = -1):
    """
    从CSF块中提取最终J值集合

    参数：
        CSFs_block: 包含CSF的列表

    返回：
        最终J值集合，按元素出现次数从大到小排序
    """
    CSFs_coupling_info_list = [tuple(block_CSFs[i+2].lstrip().split()) for i in range(0, len(block_CSFs)-2, 3)]
    CSFs_choosed_coupling_info = [CSF_coupling_info[-coupling_level:] if len(CSF_coupling_info) >= coupling_level else CSF_coupling_info for CSF_coupling_info in CSFs_coupling_info_list]
    
    result = {}
    if coupling_level == -1:
        coupling_J_counts = Counter(CSFs_coupling_info_list)
        for element in coupling_J_counts:
            result[element] = {
                'count': coupling_J_counts[element],
                'indices': []
            }

        for index, element in enumerate(CSFs_coupling_info_list):
            result[element]['indices'].append(index)
    else:
        coupling_J_counts = Counter(CSFs_choosed_coupling_info)
        for element in coupling_J_counts:
            result[element] = {
                'count': coupling_J_counts[element],
                'indices': []
            }

        for index, element in enumerate(CSFs_choosed_coupling_info):
            result[tuple(element)]['indices'].append(index)
    # 按count值从大到小排序
    sorted_result = dict(sorted(result.items(), key=lambda x: x[1]['count'], reverse=True))
    
    for element, info in sorted_result.items():
        print(f"元素 {element} 出现次数: {info['count']}, 索引: {info['indices']}")
        
    return sorted_result

#######################################################################
def CSFs_sort_by_mix_coefficient(CSFs_block: List, mix_coefficient: np.array, threshold: float = None):
    """
    根据混合系数对CSF块进行排序，并可选择返回截断值对应的索引

    参数：
        CSFs_block: 包含CSF的列表
        mix_coefficient: 混合系数
        threshold: 可选，截断阈值

    返回：
        如果threshold为None: 返回排序后的CSF块
        如果threshold不为None: 返回元组(排序后的CSF块, 截断值对应的原始索引列表)
    """
    # 检查输入参数的有效性
    if not isinstance(mix_coefficient, np.ndarray):
        raise ValueError("mix_coefficient must be a numpy array")
    if len(CSFs_block) == 0 or len(mix_coefficient) == 0:
        return [] if threshold is None else ([], [])
    if len(CSFs_block) % 3 != 0:
        raise ValueError("CSFs_block length must be a multiple of 3.")
    if len(CSFs_block) // 3 != len(mix_coefficient):
        raise ValueError("mix_coefficient length must match number of CSFs")
    
    # 将CSF块分成每三个元素一组
    csf_groups = [CSFs_block[i:i+3] for i in range(0, len(CSFs_block), 3)]
    
    # 使用numpy的argsort进行排序（更高效）
    sorted_indices = np.argsort(-np.square(mix_coefficient))
    
    # 构建排序后的CSF块
    sorted_csf_block = [item for i in sorted_indices for item in csf_groups[i]]
    
    # 根据threshold参数决定返回值
    if threshold is not None:
        # 找出绝对值大于阈值的系数的原始索引
        threshold_indices = [i for i in sorted_indices if abs(mix_coefficient[i]) > threshold]
        return sorted_csf_block, threshold_indices
    return sorted_csf_block




#######################################################################

# def main()
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