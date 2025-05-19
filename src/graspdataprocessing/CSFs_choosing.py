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
from collections import Counter, defaultdict
from dataclasses import dataclass

from tqdm import tqdm

from .CSFs_compress_extract import *
from .data_modules import MixCoefficientData

'''
    csfs data dictionary:
    {   
        'CSFs_block_data': List[
                                blocks[
                                       block_csfs[CSF_item]
                                       ]
                                ],
        'CSFs_block_j_value',
        'CSFs_block_length': List[length of each block],
        'parity',
        'subshell_info_raw'
    }
    
    rmix data dictionary:
    {
        'block_CSFs_nums': List[length of each block],
        'block_energy_count_list': List[levels of each block],
        'block_energy_list': List[energy of each block],
        'block_index_list': List[index of each block],
        'block_level_energy_list': List[
                                        block[level energy]
                                    ],
        'block_levels_index_list': List[
                                        block[level index]
                                    ],
        'j_value_location_list': List[location of j value],
        'mix_coefficient_list': List[
                                    block numpy.ndarray[
                                                        level numpy.ndarray[mix coefficient]
                                    ]
                                ],
        'parity_list': List[parity of each block],
    }
'''

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
    
    csfs_mix_square_data_above_threshold = {}
    for block in range(asfs_mix_data.CSFs_blocks_num):
        block_level_num = len(asfs_mix_data.mix_coefficient_list[block])
    
        for level in range(block_level_num):
            temp_coeff = single_asf_mix_square_above_threshold(asfs_mix_data.mix_coefficient_list[block][level], threshold)
            csfs_mix_square_data_above_threshold[(block, level)] = temp_coeff
            
    return csfs_mix_square_data_above_threshold

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
    if CSf_index[0] < 0 and len(CSFs_block) < 0:
        raise ValueError("CSF index and CSFs_block length must be non-negative.")
    
    return CSFs_block[CSf_index[0]]

#######################################################################

def single_block_csfs_final_coupling_J_collection(block_csfs: List, coupling_level: int = -1) -> Dict:
    """
    从CSF块中提取耦合J值集合

    参数：
        CSFs_asf: 包含CSF的列表

    返回：
        耦合J值集合和对应索引
    """
    CSFs_coupling_info_list = [tuple(block_csfs[i][2].lstrip().split()) for i in range(0, len(block_csfs))]
    CSFs_choosed_coupling_info = [CSF_coupling_info[-coupling_level:] if len(CSF_coupling_info) >= coupling_level else CSF_coupling_info for CSF_coupling_info in CSFs_coupling_info_list]
    
    coupling_J_collection = {}
    if coupling_level == -1:
        coupling_J_counts = Counter(CSFs_coupling_info_list)
        for element in coupling_J_counts:
            coupling_J_collection[element] = {
                'count': coupling_J_counts[element],
                'indices': []
            }

        for index, element in enumerate(CSFs_coupling_info_list):
            coupling_J_collection[element]['indices'].append(index)
    else:
        coupling_J_counts = Counter(CSFs_choosed_coupling_info)
        for element in coupling_J_counts:
            coupling_J_collection[element] = {
                'count': coupling_J_counts[element],
                'indices': []
            }

        for index, element in enumerate(CSFs_choosed_coupling_info):
            coupling_J_collection[tuple(element)]['indices'].append(index)

    return coupling_J_collection

def batch_blocks_csfs_final_coupling_J_collection(blocks_csfs_list: List[List], coupling_level: int = -1) -> Dict:
    blocks_coupling_J_collection = {}
    for block, block_csfs in enumerate(blocks_csfs_list):
        print(f"第{block+1}个block包含{len(block_csfs)}个csf")
        block_coupling_J_collection = single_block_csfs_final_coupling_J_collection(block_csfs, coupling_level)
        blocks_coupling_J_collection[block] = block_coupling_J_collection
    return blocks_coupling_J_collection

def single_asf_csfs_final_coupling_J_mix_coefficient_sum(block_csfs_coupling_J_collection_dict: Dict, mix_coefficient_list: List) -> Dict:
    
    for index, element in enumerate(block_csfs_coupling_J_collection_dict):
        print(f"元素 {element} 出现次数: {block_csfs_coupling_J_collection_dict[element]['count']}")
        sum_ci = 0
        for ci in block_csfs_coupling_J_collection_dict[element]['indices']:
            sum_ci += mix_coefficient_list[ci]**2
        print(f"元素 {element} 对应的索引之和: {sum_ci}")
        block_csfs_coupling_J_collection_dict[element]['sum_ci'] = sum_ci
        print(f"元素 {element} 对应的索引之和: {block_csfs_coupling_J_collection_dict[element]['sum_ci']}")
        
        
    # 按sum_ci值从大到小排序
    # sorted_block_csfs_coupling_J_collection_dict = dict(sorted(block_csfs_coupling_J_collection_dict.items(), key=lambda x: x[1]['sum_ci'], reverse=True))

    # return sorted_block_csfs_coupling_J_collection_dict
    return block_csfs_coupling_J_collection_dict

def single_block_batch_asfs_CSFs_final_coupling_J_collection(block_CSFs: List, block_asfs_mix_coefficient_list: List, coupling_level: int = -1) -> Dict:
    # 获取初始耦合信息
    base_coupling_dict = single_block_csfs_final_coupling_J_collection(block_CSFs, coupling_level)
    
    # block_asfs_coupling_J_sum_ci = {}
    
    for index, element in enumerate(base_coupling_dict):
        base_coupling_dict[element]['sum_ci'] = []
        for asf_index, asf_mix_coefficient in enumerate(block_asfs_mix_coefficient_list):
            sum_ci = 0
            for csf_index in base_coupling_dict[element]['indices']:
                sum_ci += asf_mix_coefficient[csf_index]**2

            base_coupling_dict[element]['sum_ci'].append(sum_ci)

    # return block_asfs_coupling_J_sum_ci
    return base_coupling_dict

def batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum(blocks_CSFs_list: List, blocks_asfs_mix_coefficient_list: List, coupling_level: int = -1) -> Dict:
    blocks_asfs_coupling_J_sum_ci = {}
    for block, (block_csfs, block_asfs_mix) in enumerate(zip(blocks_CSFs_list, blocks_asfs_mix_coefficient_list)):
        print(f"第{block+1}个block包含{len(block_asfs_mix)}个asf")
        if any(len(asf_mix) != len(block_csfs)  for asf_mix in block_asfs_mix):
            raise ValueError("block_CSFs和block_asfs_mix_coefficient长度不匹配")
        
        block_asfs_coupling_J_collection = single_block_batch_asfs_CSFs_final_coupling_J_collection(block_csfs, block_asfs_mix, coupling_level)
        
        blocks_asfs_coupling_J_sum_ci[block] = block_asfs_coupling_J_collection

        
    return blocks_asfs_coupling_J_sum_ci
    
#######################################################################
def block_csfs_coupling_J_chosen(block_asfs_coupling_J_sum_ci: Dict) -> List:
    """
    Selects the most significant CSF indices for each ASF based on coupling J values.
    
    Args:
        block_asfs_coupling_J_sum_ci: Dictionary containing coupling information for multiple ASFs,
        where each ASF has a dictionary of coupling configurations
        with their sum of squared mix coefficients ('sum_ci')
    
    Returns:
        List of lists containing indices of CSFs with the strongest coupling for each ASF
    """
    chosen_csfs_indices = []
    for key, inner_dict in block_asfs_coupling_J_sum_ci.items():
        if inner_dict:

            first_key, first_value = next(iter(inner_dict.items()))
            chosen_csfs_indices.append(first_value['indices'])

    return chosen_csfs_indices

def union_lists_with_order(*lists):
    """
    计算多个列表的并集，保留元素首次出现的顺序。

    参数:
        *lists: 任意数量的列表

    返回:
        包含所有列表元素并去重，且保留元素首次出现顺序的列表
    """
    # 使用 dict.fromkeys 保留元素顺序并去重
    all_elements = []
    for lst in lists:
        all_elements.extend(lst)
    return list(dict.fromkeys(all_elements))

def merge_multiple_dicts_with_ordered_union(*dicts: Dict[Tuple[int], List]) -> dict:
    """
    合并多个字典中相同键对应的列表，保持顺序并去重
    
    参数:
        *dicts: 可变数量的字典参数，每个字典的值应为列表
        
    返回:
        合并后的字典，包含所有输入字典中相同键的合并列表
    """
    if not dicts:
        return {}
    
    # 获取所有字典共有的键
    common_keys = set(dicts[0].keys())
    for d in dicts[1:]:
        common_keys.intersection_update(d.keys())
    
    merged_result = {}
    for key in common_keys:
        print(f"Processing block {key}")
        # 收集所有字典中该键对应的列表
        lists_to_merge = [d[key] for d in dicts]
        # 使用union_lists_with_order合并列表
        merged_result[key] = gdp.union_lists_with_order(*lists_to_merge)
    
    return merged_result

#######################################################################

def merge_csfs_indices_lists_by_block_key(chosen_csfs_indices: Dict[Tuple[int, int], List]) -> Dict[int, List]:
    """
    将字典中第一个数字相同的键对应的列表合并
    
    参数：
        chosen_csfs_indices: 原始字典，键为(block, level)元组，值为CSF索引列表
        
    返回：
        按第一个数字(block)分组的字典，值为合并后的CSF索引列表（保持顺序并去重）
    """
    merged_dict = defaultdict(list)
    
    for key, indices_list in chosen_csfs_indices.items():
        first_num = key[0]  # 提取元组的第一个数字
        merged_dict[first_num].extend(indices_list)  # 合并列表
    
    return {
        group_key: union_lists_with_order(*group_lists) 
        for group_key, group_lists in merged_dict.items()
    }


def CSFs_sort_by_mix_coefficient(CSFs_block: List, *mix_coefficients: np.ndarray, threshold: float = None):
    """
    根据多个混合系数的对应元素和来对CSF块进行排序，并可选择返回截断值对应的索引

    参数：
        CSFs_block: 包含CSF的列表
        *mix_coefficients: 一个或多个混合系数数组
        对于同一个block拥有多个asfs的情况，现将asfs的系数进行求和，再进行排序
        threshold: 可选，截断阈值

    返回：
        如果threshold为None: 返回排序后的CSF块
        如果threshold不为None: 返回元组(排序后的CSF块, 截断值对应的原始索引列表)
    """
    # 检查输入参数的有效性
    if len(CSFs_block) == 0 or len(mix_coefficients) == 0:
        raise ValueError("CSFs_block和mix_coefficients不能为空")
    
    # 检查所有系数数组长度一致
    coeff_lengths = {len(coeff) for coeff in mix_coefficients}
    if len(coeff_lengths) > 1:
        raise ValueError("所有mix_coefficients数组长度必须相同")
    if len(CSFs_block) != next(iter(coeff_lengths)):
        raise ValueError("mix_coefficients长度必须与CSFs_block匹配")

    # 计算所有系数数组的对应元素和
    combined_coeff = np.sum([np.square(coeff) for coeff in mix_coefficients], axis=0)
    
    # 使用numpy的argsort进行排序（降序）
    sorted_indices = np.argsort(-combined_coeff)
    
    # 构建排序后的CSF块
    sorted_csf_block = [item for i in sorted_indices for item in CSFs_block[i]]
    
    # 根据threshold参数决定返回值
    if threshold is not None:
        # 找出组合系数大于阈值的原始索引
        threshold_indices = [i for i in sorted_indices if combined_coeff[i] > threshold**2]
        return sorted_csf_block, threshold_indices
        
    return sorted_csf_block


def radom_choose_csfs(block_csfs_list: List, choose_csfs_num: int, start_index: int = 0) -> Tuple[List, List[int]]:
    """
    从CSF列表中随机选择指定数量的CSF及其索引

    参数：
        block_csfs_list: 包含CSF的列表
        choose_csfs_num: 要选择的CSF数量
        start_index: 起始索引(默认为0)

    返回：
        元组(包含随机选择的CSF列表, 对应的原始索引列表)
    """
    block_csfs_len = len(block_csfs_list)
    if start_index > block_csfs_len:
        raise ValueError("start_index不能大于block_csfs_list的长度")
    if start_index + choose_csfs_num > block_csfs_len:
        raise ValueError("选择的CSF数量超过可用范围")
    
    # 生成随机索引
    random_indices = np.random.choice(
        range(start_index, block_csfs_len),
        size=choose_csfs_num,
        replace=False
    )
    
    # 获取对应的CSF和索引
    chosen_csfs = [CSFs_block_get_CSF(block_csfs_list, (idx,)) for idx in random_indices]
    
    return chosen_csfs, random_indices.tolist()




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