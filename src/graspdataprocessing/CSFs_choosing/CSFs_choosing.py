#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :CSFs_choosing.py
@date :2024/08/02 20:38:24
@author :YenochQin (秦毅)
'''
import re
import random
from typing import Dict, Tuple, List, TYPE_CHECKING
from collections import Counter, defaultdict
import math
import numpy as np
import pickle
from tqdm import tqdm

# if TYPE_CHECKING:
#     from ..data_IO.processing_data_load import load_large_hash
#     from ..utils.data_modules import MixCoefficientData
from ..data_IO.processing_data_load import load_large_hash
from ..utils.tool_function import *
from ..utils.data_modules import MixCoefficientData
from .CSFs_compress_extract import CSF_item_2_dict

'''
    csfs data dictionary:
    {   
        'CSFs_block_data': List[
                                blocks[
                                       block_csfs[CSF_item[csf_1], CSF_item[csf_2], ...]]
                                       ]
                                ],
        'CSFs_block_j_value',
        'CSFs_block_length': List[length of each block],
        'parity',
        'subshell_info_raw'
    }
    
    rmix data dictionary:
    {
        'block_num': List[length of each block],
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
        'mix_coefficient_List': List[
                                    block numpy.ndarray[
                                                        level numpy.ndarray[mix coefficient]
                                    ]
                                ],
        'parity_list': List[parity of each block],
    }
'''

#######################################################################
def single_asf_mix_square_above_threshold(asf_mix_data_array: np.ndarray, threshold: float = 0.1) -> List[Tuple[int]]:
    """
    获取一维数组中平方值超过阈值的元素的索引，并按绝对值降序排序
    
    Args:
        asf_mix_data_array: 一维输入数组
        threshold: 阈值(平方值比较)，默认0.1
        
    Returns:
        包含索引的元组列表，如[(index1,), (index2,), ...]，按绝对值降序排列
        如果数组为空或没有元素超过阈值，返回空列表
    """
    # 检查输入是否为一维数组
    if asf_mix_data_array.ndim != 1:
        raise ValueError("输入数组必须是一维的")
    
    # 找出平方值超过阈值的索引
    indices = np.where(np.square(asf_mix_data_array) > threshold)[0]
    
    # 如果没有满足条件的元素，返回空列表
    if len(indices) == 0:
        return []
    
    # 获取对应的值
    values = asf_mix_data_array[indices]
    
    # 按绝对值降序排序索引
    sorted_indices = indices[np.argsort(-np.abs(values))]
    
    # 返回索引元组列表
    return [(idx,) for idx in sorted_indices]

def batch_asfs_mix_square_above_threshold(asfs_mix_data: MixCoefficientData, threshold: float = 0.1) -> Dict[int, np.ndarray]:
    """
    批量处理多个块的混合系数数据，找出每个块中所有层级中超过阈值的系数索引
    
    Args:
        asfs_mix_data: 包含以下属性的对象:
            - block_num: 块的总数
            - mix_coefficient_List: 按块组织的系数列表(每个块包含多个层级的一维数组)
        threshold: 阈值(平方值比较)，默认0.1
        
    Returns:
        字典，键是block编号，值是该块中所有超过阈值的系数索引(已去重的排序数组)
        如果没有满足条件的索引，对应的值为空数组
    """
    result = {}
    
    for block in range(asfs_mix_data.block_num):
        # 检查块是否存在或数据是否有效
        if (block >= len(asfs_mix_data.mix_coefficient_List) or 
            asfs_mix_data.mix_coefficient_List[block] is None):
            result[block] = np.array([], dtype=np.int64)
            continue
            
        # 使用集合来存储索引以实现去重
        unique_indices = set()
        
        for level_data in asfs_mix_data.mix_coefficient_List[block]:
            if level_data is None:
                continue
                
            # 获取该层级的重要系数索引并添加到集合中
            level_indices = single_asf_mix_square_above_threshold(level_data, threshold)
            unique_indices.update(idx[0] for idx in level_indices)  # 解包单元素元组
        
        # 转换为排序后的numpy数组
        if unique_indices:
            result[block] = np.array(sorted(unique_indices), dtype=np.int64)
        else:
            result[block] = np.array([], dtype=np.int64)
        
    return result



#### batch_blocks_mix_square_above_threshold 重复了，以后不再使用
def batch_blocks_mix_square_above_threshold(asfs_mix_data: MixCoefficientData, threshold=0.1):
    
    block_csfs_mix_square_data_above_threshold = {}
    for block in range(asfs_mix_data.block_num):
        block_level_num = len(asfs_mix_data.mix_coefficient_List[block])
    
        temp_block_coeff = []
        for level in range(block_level_num):
            temp_coeff = single_asf_mix_square_above_threshold(asfs_mix_data.mix_coefficient_List[block][level], threshold)
            # block_csfs_mix_square_data_above_threshold[(block, level)] = temp_coeff
            temp_block_coeff.extend(temp_coeff)
            
        unique_indices = union_lists_with_order(temp_block_coeff)
        
        block_csfs_mix_square_data_above_threshold[(block,)] = unique_indices

    return block_csfs_mix_square_data_above_threshold

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

def single_asf_csfs_final_coupling_J_mix_coefficient_sum(block_csfs_coupling_J_collection_dict: Dict, mix_coefficient_List: List) -> Dict:
    
    for index, element in enumerate(block_csfs_coupling_J_collection_dict):
        print(f"元素 {element} 出现次数: {block_csfs_coupling_J_collection_dict[element]['count']}")
        sum_ci = 0
        for ci in block_csfs_coupling_J_collection_dict[element]['indices']:
            sum_ci += mix_coefficient_List[ci]**2
        print(f"元素 {element} 对应的索引之和: {sum_ci}")
        block_csfs_coupling_J_collection_dict[element]['sum_ci'] = sum_ci
        print(f"元素 {element} 对应的索引之和: {block_csfs_coupling_J_collection_dict[element]['sum_ci']}")
        
        
    # 按sum_ci值从大到小排序
    # sorted_block_csfs_coupling_J_collection_dict = dict(sorted(block_csfs_coupling_J_collection_dict.items(), key=lambda x: x[1]['sum_ci'], reverse=True))

    # return sorted_block_csfs_coupling_J_collection_dict
    return block_csfs_coupling_J_collection_dict

def single_block_batch_asfs_CSFs_final_coupling_J_collection(block_CSFs: List, block_asfs_mix_coefficient_List: List, coupling_level: int = -1) -> Dict:
    # 获取初始耦合信息
    base_coupling_dict = single_block_csfs_final_coupling_J_collection(block_CSFs, coupling_level)
    
    # block_asfs_coupling_J_sum_ci = {}
    
    for index, element in enumerate(base_coupling_dict):
        base_coupling_dict[element]['sum_ci'] = []
        for asf_index, asf_mix_coefficient in enumerate(block_asfs_mix_coefficient_List):
            sum_ci = 0
            for csf_index in base_coupling_dict[element]['indices']:
                sum_ci += asf_mix_coefficient[csf_index]**2

            base_coupling_dict[element]['sum_ci'].append(sum_ci)

    # return block_asfs_coupling_J_sum_ci
    return base_coupling_dict

def batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum(blocks_CSFs_list: List, blocks_asfs_mix_coefficient_List: List, coupling_level: int = -1) -> Dict:
    blocks_asfs_coupling_J_sum_ci = {}
    for block, (block_csfs, block_asfs_mix) in enumerate(zip(blocks_CSFs_list, blocks_asfs_mix_coefficient_List)):
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
        merged_result[key] = union_lists_with_order(*lists_to_merge)
    
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


def CSFs_sort_by_mix_coefficient(CSFs_block: List, *mix_coefficients: np.ndarray, threshold = None):
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


#######################################################################
# random select csfs from block_csfs_list
#######################################################################

def generate_unique_random_numbers(max_num: int, count: int) -> list:
    """
    生成指定数量不重复的随机正整数
    
    参数:
        max_num: 随机数的最大值(包含)
        count: 需要生成的随机数数量
        
    返回:
        包含不重复随机数的列表，按升序排列
    """
    if count <= 0:
        return []
    
    if max_num <= 0:
        raise ValueError("max_num必须大于0")
        
    if count > max_num:
        raise ValueError(f"无法生成{count}个不重复的1-{max_num}之间的数字")
    
    # 使用sample方法确保不重复
    numbers = random.sample(range(1, max_num + 1), count)
    numbers.sort()  # 排序结果
    
    return numbers

def radom_choose_csfs(block_csfs_list: List, ratio_CSFs_select_num: float, selected_csfs_indices: List = []):
    """
    优化版的大规模CSF随机选择函数
    
    优化点：
    1. 使用numpy加速数组操作
    2. 减少中间变量创建
    3. 优化索引计算逻辑
    """
    block_csfs_num = len(block_csfs_list)
    selected_csfs_num = len(selected_csfs_indices)
    
    # 计算需要选择的总数
    total_needed = math.ceil(block_csfs_num * ratio_CSFs_select_num)
    
    # 计算还需要补充的数量
    choose_csfs_num = max(0, total_needed - selected_csfs_num)
    
    # 使用numpy数组加速操作
    all_indices = np.arange(block_csfs_num)
    
    if selected_csfs_num > 0:
        # 使用numpy的set操作
        selected_set = set(selected_csfs_indices)
        unselected_mask = ~np.isin(all_indices, list(selected_set))
        unselected_indices = all_indices[unselected_mask]
        
        if choose_csfs_num > 0:
            # 使用numpy的随机选择
            random_indices = np.random.choice(unselected_indices, size=choose_csfs_num, replace=False)
            chosen_csfs_indices = np.concatenate([selected_csfs_indices, random_indices])
        else:
            chosen_csfs_indices = np.array(selected_csfs_indices)
    else:
        # 直接随机选择
        chosen_csfs_indices = np.random.choice(all_indices, size=total_needed, replace=False)
    
    # 获取未选择的索引
    unselected_indices = np.setdiff1d(all_indices, chosen_csfs_indices)
    
    # 使用列表推导式获取CSF数据
    chosen_csfs = [block_csfs_list[idx] for idx in chosen_csfs_indices]
    
    return chosen_csfs, chosen_csfs_indices, unselected_indices




#######################################################################

def process_block(args):
    """Helper function for parallel processing moved to global scope"""
    block_idx, small_data, large_data = args
    # 预生成large数据的哈希映射
    large_map = {
        ''.join(item for sublist in large_csf for item in sublist): idx
        for idx, large_csf in enumerate(large_data[block_idx])
    }
    
    return [
        large_map[''.join(item for sublist in small_csf for item in sublist)]
        for small_csf in small_data[block_idx]
        if ''.join(item for sublist in small_csf for item in sublist) in large_map
    ]

def maping_two_csfs_indices(
    small_as_csfs_data: List[List[List[str]]],
    large_hash_file: str = "large_data_hash.pkl"
) -> Dict[int, List[int]]:
    """
    将 small_as_csfs_data 映射到预计算的 large_hash
    
    返回:
        {small_block_idx: [matched_large_indices]}
    """
    large_hash = load_large_hash(large_hash_file)  # Dict[int, Dict[str, int]]
    
    results = []
    for small_block_idx, small_block in enumerate(small_as_csfs_data):
        matched_indices = []
        for small_csf in small_block:
            csf_str = ''.join(item for sublist in small_csf for item in sublist)
            # 在所有 large block 中查找匹配
            for large_block_idx, block_map in large_hash.items():
                if csf_str in block_map:
                    matched_indices.append((large_block_idx, block_map[csf_str]))
        
        # 仅保留匹配的 large_csf 全局索引（按需调整）
        results.append([idx for (_, idx) in matched_indices])
    
    return {block_idx: indices for block_idx, indices in enumerate(results)}

