#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :CSFs_compress_extract.py
@date :2025/03/10 16:02:06
@author :YenochQin (秦毅)
'''


import numpy as np
import pandas as pd
# from pathlib import Path
from tqdm import tqdm
from .data_IO import GraspFileLoad
import re
from typing import Dict, Tuple, List


################################################################
# CSFs source data compress to a simplified form
################################################################

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

def CSF_subshell_split(CSF: str) -> Dict[str, int]:

    subshells_charged = re.split(r'(\d*\w[\s|-]\(\s\d*\))', CSF)
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

def simplify_orbital_string(original_str):
    kappa_value = {
        "s ": -1,
        "p-": 1,
        "p ": -2,
        "d-": 2,
        "d ": -3,
        "f-": 3,
        "f ": -4,
        "g-": 4,
        "g ": -5,
        "h-": 5,
        "h ": -6,
        "i-": 6,
        "i ": -7
    }
    
    # 改进正则：使用更严格的分隔符匹配
    pattern = re.compile(r'(\d+)([a-z][ -]?)\s*\(\s*(\d+)\s*\)')
    simplified = []
    
    for match in re.finditer(pattern, original_str):
        n, orbital, value = match.groups()
        # 标准化轨道符号格式
        orbital = orbital.replace(' ', '').rstrip('-') + ('-' if '-' in orbital else ' ')
        if orbital not in kappa_value:
            continue
        simplified.append(f"{int(n)}|{kappa_value[orbital]}|{value}")
    
    return ' '.join(simplified)

def desimplify_orbital_string(simplified_str):
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
    
    items = simplified_str.split()
    restored = []
    
    for item in items:
        try:
            n, kappa, value = item.split('|')
            orbital_key = reverse_kappa.get(int(kappa), "")
            restored.append(f"{n}{orbital_key.strip()} ({value})")
        except:
            continue
    
    return '  '.join(restored)

def subshells_J_value_parser(subshells_J_value: str, subshell_unfully_charged: Dict[str, int]) -> Dict[str, str]:

    subshells_J_value_list = re.findall(r'\S+', subshells_J_value)
    return {key: value for key, value in zip(subshell_unfully_charged.keys(), subshells_J_value_list)}


def CSF_item_2_dict(CSF_item_list: List[str]) -> Dict:

    # 解析 subshell 信息
    CSF_item_dict = CSF_subshell_split(CSF_item_list[0])
    
    # 添加 temp_coupled_j 和 final_coupled_j_parity
    CSF_item_dict.update({
        'temp_coupled_j': CSF_item_list[1],
        'final_coupled_j_parity': CSF_item_list[2],
    })
    
    # 解析 final_coupled_j_parity 中的 J 和 parity
    j_p = CSF_item_list[2].split()[-1]  # 提取 J 和 parity 部分
    CSF_item_dict['parity'] = j_p[-1]   # parity 是最后一个字符
    CSF_item_dict['J'] = j_p[:-1]       # J 是 parity 之前的部分
    
    return CSF_item_dict


def get_CSFs_file_info(CSFs_file_data: List) -> Dict:
    
    subshell_info = CSFs_file_data[0:4]
    
    CSFs_file_info = {}
    CSFs_file_info['raw_subshell_info'] = subshell_info
    for i in range(0, len(subshell_info), 2):
        key = subshell_info[i].rstrip(':')      # 去掉键中的冒号
        value = subshell_info[i + 1].split()
        CSFs_file_info[key] = value

    star_indices = [index for index, value in enumerate(CSFs_file_data) if value == '*']
    CSFs_file_info['star_indices'] = star_indices
    
    
    CSFs_j_value = []
    for index in star_indices:
        CSFs_j_value.append(CSFs_file_data[index-1])
        
    CSFs_j_value.append(CSFs_file_data[-1])
    CSFs_file_info['CSFs_j_value'] = CSFs_j_value

    return CSFs_file_info