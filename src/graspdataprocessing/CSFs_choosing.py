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

def count_prim_pool(flnm_full, flnm_head):
    with open(flnm_full, "r") as f_full:
        with open(flnm_head, "r") as f_head:
            for _ in range(0, 5):
                f_head.readline()
                f_full.readline()

            csfs_prim_num = 0
            while True:
                ln = f_head.readline()
                if not ln:
                    break
                else:
                    f_head.readline()
                    f_head.readline()
                    # 这里不再跳过 flnm_full 中的对应行
                    csfs_prim_num += 1

        csfs_pool_num = 0
        while True:
            ln = f_full.readline()
            if not ln:
                break
            else:
                f_full.readline()
                f_full.readline()
                csfs_pool_num += 1

    return csfs_prim_num, csfs_pool_num

# 读取文件并删除每行的第一个字符
def remove_first_char_from_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # 删除每行的第一个字符，并写入到新文件
            outfile.write(line[1:])

def extract_confinfo_part_ind(filename,cmin):
    """
    从指定文件中逐行提取编号、CI系数和组态。
    返回一个包含编号、CI系数和组态的列表。
    """
    pattern = r"^\s*(\d+)\s+([+-]?\d*\.?\d+)\s*$"
    curr_index_ci_conf = []
    current_block = []
    current_ci = 0
    current_index = 0

    with open(filename, 'r') as file:
        for line in file:
            match = re.match(pattern, line)

            if match:
                # 如果在一个组态块中，保存符合条件的块
                if current_block and current_ci != 0:
                    curr_index_ci_conf.append((current_index, current_ci, "".join(current_block)))
                
                # 开始新的组态块（不包括匹配行），将 current_index 转换为 int
                current_index = int(match.group(1))
                current_ci = float(match.group(2))
                current_block = []  # 清空组态块
            elif current_block or line.strip():  # 处理组态行
                current_block.append(line)  # 保留原始行内容
    
    # 处理最后一个组态块
    if current_block and current_ci != 0:
        curr_index_ci_conf.append((current_index, current_ci, "".join(current_block)))
        
    # 根据CI系数的平方进行筛选
    part_ind = [int(item[0]) for item in curr_index_ci_conf if item[1] ** 2 >= cmin]
    part_ind = np.array(part_ind)  # 转为numpy数组

    return curr_index_ci_conf, part_ind

def generate_onoff(basis_size, csfs_prim_num, part_ind):
    onoff = np.zeros(basis_size, dtype=bool)
    onoff[part_ind-1] = True
    onoff[csfs_prim_num:] = True
    mark_train = onoff.copy()
    mark_apply = ~onoff
    true_count = np.count_nonzero(onoff)
    return onoff, mark_train, mark_apply, true_count

def generate_import_onoff(csfs_prim_num, part_ind):
    onoff = np.zeros(csfs_prim_num, dtype=bool)
    onoff[part_ind-1] = True
    mark_train = onoff.copy()
    mark_apply = ~onoff
    return onoff, mark_train, mark_apply

def write_atcomp_input(curr_grasp_inp, full_grasp_inp, basis_size, onoff):
    with open(curr_grasp_inp, "w") as f_curr:
        with open(full_grasp_inp, "r") as f_full:
            # 先写入full_grasp_inp文件的前5行
            for _ in range(5):
                ln = f_full.readline()
                f_curr.write(ln)
            
            # 继续按照onoff[csfs_ind]的信息写入
            for csfs_ind in range(basis_size):
                ln1 = f_full.readline()
                ln2 = f_full.readline()
                ln3 = f_full.readline()

                if onoff[csfs_ind]:
                    f_curr.write(ln1)
                    f_curr.write(ln2)
                    f_curr.write(ln3)
                    
    return None