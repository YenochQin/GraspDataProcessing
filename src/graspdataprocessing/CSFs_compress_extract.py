#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :CSFs_compress_extract.py
@date :2025/03/10 16:02:06
@author :YenochQin (秦毅)
'''
import random
import re
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from tqdm import tqdm

from .tool_function import *


#######################################################################
# CSFs source data compress to a simplified form
#######################################################################


def subshell_charged_state(subshell_CSF: str) -> Dict[str, str]:
    """
    解析轨道电荷状态，返回包含主量子数、轨道名称和电荷数的字典。
    """
    temp_subshell_state = re.findall(r'([0-9]*)([s,p,d,f,g][\s,-])\( (\d+)\)', subshell_CSF)[0]
    main_quantum_num = temp_subshell_state[0]
    subshell_name = temp_subshell_state[1]
    subshell_charged_num = int(temp_subshell_state[2])
    return {
        'subshell_main_quantum_num': main_quantum_num,
        'subshell_name': subshell_name,
        'subshell_charged_num': subshell_charged_num
    }

def if_subshell_full_charged(subshell_name: str, subshell_charged_num: int) -> bool:

    full_charged = {
        "s ": 2,
        "p-": 2,
        "p ": 4,
        "d-": 4,
        "d ": 6,
        "f-": 6,
        "f ": 8,
        "g-": 8,
        "g ": 10,
    }
    return full_charged.get(subshell_name, 0) == subshell_charged_num

def CSF_subshell_split(CSFs_configuration_raw: str) -> Dict[str, int]:

    subshells_charged = re.split(r'(\d*\w[\s|-]\(\s\d*\))', CSFs_configuration_raw)
    print(subshells_charged)
    
    subshells_charged = [item for item in subshells_charged if item.strip()]
    
    subshell_unfully_charged = {}
    subshell_fully_charged = {}
    
    csf_electron_num = 0
    for subshell in subshells_charged:
        temp_subshell_charged_state = subshell_charged_state(subshell)

        temp_quantum_num = temp_subshell_charged_state['subshell_main_quantum_num']
        temp_subshell = temp_subshell_charged_state['subshell_name']
        temp_charged_num = temp_subshell_charged_state['subshell_charged_num']
        csf_electron_num += temp_charged_num
        if if_subshell_full_charged(temp_subshell, temp_charged_num):
            print(f"{temp_quantum_num}{temp_subshell}({temp_charged_num}) is fully charged.")
            subshell_fully_charged[temp_quantum_num + temp_subshell] = temp_charged_num
        else:
            subshell_unfully_charged[temp_quantum_num + temp_subshell] = temp_charged_num
    
    return {
        'unfully_charged_subshell': subshell_unfully_charged,
        'fully_charged_subshell': subshell_fully_charged
        }

#######################################################################

def CSF_subshell_compress(CSF_configuration_raw: str):
    '''
    compress CSF subshell from
      5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)  4f-( 1)  4f ( 6)  5d ( 1)
    to
    '5|-1|2;4|2|4;4|-3|6;5|1|2;5|-2|4;6|-1|2;4|3|1;4|-4|6;5|-3|1;'
    '''
    subshells_charged = re.split(r'(\d*\w[\s|-]\(\s\d*\))', CSF_configuration_raw)
    subshells_charged = [item for item in subshells_charged if item.strip()]
    
    print(subshells_charged)
    compressed_CSF = ''
    
    for subshell in subshells_charged:
        temp_subshell_charged_state = subshell_charged_state(subshell)
        temp_subshell_kappa = str_subshell_2_kappa(temp_subshell_charged_state['subshell_name'])
        compressed_CSF += f'{temp_subshell_charged_state['subshell_main_quantum_num']}|{temp_subshell_kappa}|{temp_subshell_charged_state['subshell_charged_num']};'

    return compressed_CSF

def CSF_compress(CSF_raw: List):
    '''
    compress CSF
    '''
    if len(CSF_raw) != 3:
        raise ValueError("CSF_raw need to be 3 line")

    return CSF_subshell_compress(CSF_raw[0]) + CSF_raw[1:]

def CSF_subshell_extract(simplified_str):
    '''
    extract CSF subshell from
    '5|-1|2;4|2|4;4|-3|6;5|1|2;5|-2|4;6|-1|2;4|3|1;4|-4|6;5|-3|1;'
    to 
      5s ( 2)  4d-( 4)  4d ( 6)  5p-( 2)  5p ( 4)  6s ( 2)  4f-( 1)  4f ( 6)  5d ( 1)
    '''
    reverse_kappa = {
        -1: "s ",
        1: "p-",
        -2: "p ",
        2: "d-",
        -3: "d ",
        3: "f-",
        -4: "f ",
        4: "g-",
        -5: "g ",
        5: "h-",
        -6: "h ",
        6: "i-",
        -7: "i "
    }
    
    items = simplified_str.split(';')
    restored = []
    
    for item in items:
        try:
            n, kappa, value = item.split('|')
            orbital_key = reverse_kappa.get(int(kappa), "")
            restored.append(f"{n}{orbital_key}({value})")
        except:
            continue
    
    return '  '.join(restored)

#######################################################################

def CSF_J(csf_3rd_line: str):
    '''
    extract J from CSF 3rd line
    '''
    J_parity = re.search(r'(\d+)([+-])\n', csf_3rd_line)
    
    if J_parity:
        J_value = J_parity.group(1)
        parity = J_parity.group(2)
        return J_value, parity


def CSF_info_2_dict(CSF_item_list: List[str]) -> Dict:

    # 解析 subshell 信息
    CSF_info_dict = CSF_subshell_split(CSF_item_list[0])
    
    # 添加 temp_coupled_j 和 final_coupled_j_parity
    CSF_info_dict.update({
        'temp_coupled_j': CSF_item_list[1],
        'final_coupled_j_parity': CSF_item_list[2],
    })
    
    # 解析 final_coupled_j_parity 中的 J 和 parity
    j_p = CSF_item_list[2].split()[-1]  # 提取 J 和 parity 部分
    CSF_info_dict['parity'] = j_p[-1]   # parity 是最后一个字符
    CSF_info_dict['J'] = j_p[:-1]       # J 是 parity 之前的部分
    
    return CSF_info_dict

def CSF_item_2_dict(CSF_item_list: List[str]) -> Dict:

    CSF_item_dict = {}
    
    CSF_item_dict.update({
        'subshell_raw': CSF_item_list[0],
        'temp_coupled_j': CSF_item_list[1],
        'final_coupled_j_parity': CSF_item_list[2],
    })

    j_p = CSF_item_list[2].split()[-1]  # 提取 J 和 parity 部分
    CSF_item_dict['parity'] = j_p[-1]   # parity 是最后一个字符
    CSF_item_dict['J'] = j_p[:-1]       # J 是 parity 之前的部分
    
    return CSF_item_dict


def get_CSFs_file_info(CSFs_file_data: List) -> Dict:
    """
    Process CSF file data and extract structured information.
    
    Args:
        CSFs_file_data: Raw CSF data list containing subshell info and CSFs entries
    
    Returns:
        Dictionary containing:
        - raw_subshell_info: Original header lines
        - parsed subshell parameters (n, orbitals, etc.)
        - star_indices: Positions of CSF separators
        - CSFs_j_value: Collected J-values from CSFs
    """
    # Extract first 4 lines containing subshell information
    subshell_info = CSFs_file_data[0:4]
    
    CSFs_file_info = {}
    CSFs_file_info['raw_subshell_info'] = subshell_info
    
    # Process subshell info pairs (parameter name + values)
    for i in range(0, len(subshell_info), 2):
        key = subshell_info[i].rstrip(':')      # Remove colon from key
        value = subshell_info[i + 1].split()    # Split values into list
        CSFs_file_info[key] = value

    # Find all CSF separators ('*') in the data
    star_indices = []
    for index, value in enumerate(CSFs_file_data):
        if '*' in value:
            star_indices.append(index)
    CSFs_file_info['star_indices'] = star_indices

    # Collect J-values preceding each separator and the final value
    CSFs_j_value = []
    CSFs_block_parity = []
    prev_index = 5
    CSFs_file_info['CSFs_block_data'] = []  # 初始化 CSFs_block_data 列表

    for index in star_indices:
        temp_j_value, temp_parity = CSF_J(CSFs_file_data[index - 1])
        CSFs_j_value.append(temp_j_value)
        CSFs_block_parity.append(temp_parity)
        # 处理每个块的数据，而不是一次性存储所有块
        block_data = CSFs_file_data[prev_index:index]
        if len(block_data) % 3 != 0:
            raise ValueError("CSFs_list length must be a multiple of 3")
        CSFs_file_info['CSFs_block_data'].append(block_data)  # 添加当前块的数据
        prev_index = index + 1

    temp_j_value, temp_parity = CSF_J(CSFs_file_data[-1])
    CSFs_j_value.append(temp_j_value)
    CSFs_block_parity.append(temp_parity)
    CSFs_file_info['CSFs_j_value'] = CSFs_j_value

    CSFs_parity = set(CSFs_block_parity)
    if len(CSFs_parity) == 1:
        CSFs_file_info['parity'] = list(CSFs_parity)[0]

    # 处理最后一个块的数据
    last_block_data = CSFs_file_data[prev_index:]
    if len(last_block_data) % 3 != 0:
        raise ValueError("CSFs_list length must be a multiple of 3")
    CSFs_file_info['CSFs_block_data'].append(last_block_data)  # 添加最后一个块的数据

    return CSFs_file_info

#######################################################################



def split_by_asterisk(lines):
    """
    将列表按单独一个星号行分割为二维列表
    
    :param lines: readlines 读取的列表
    :param keep_empty: 是否保留空块（默认过滤）
    :return: 二维列表，例如 [[块1行], [块2行], ...]
    """
    result = []
    current_chunk = []
    
    for line in lines:
        # 严格匹配：仅当行内容为单个星号（含换行符）
        if line.strip() == "*":
            result.append(current_chunk)
            current_chunk = []
        else:
            current_chunk.append(line)  # 可选：去除换行符
    
    # 添加最后一个块
    result.append(current_chunk)
    
    return result


def shuffle_three_line_groups(lst):
    """
    将列表按每三行一组随机打乱顺序
    示例输入格式：
    ['行1', '行2', '行3', '行4', '行5', '行6', ...]
    输出格式：
    ['行4', '行5', '行6', '行1', '行2', '行3', ...]
    """
    # 检查是否能被3整除
    if len(lst) % 3 != 0:
        raise ValueError("列表长度必须是3的倍数")

    # 将列表分成三元组
    groups = [lst[i:i+3] for i in range(0, len(lst), 3)]
    
    # 打乱组顺序
    random.shuffle(groups)
    
    # 重新展开为平铺列表
    shuffled = [line for group in groups for line in group]
    
    return shuffled