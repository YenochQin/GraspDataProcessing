#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :tool_function.py
@date :2024/05/07 11:11:09
@author :YenochQin (秦毅)
'''

import numpy as np
import pandas as pd

from tqdm import tqdm
from .data_modules import CSFs
######################################################################

'''
print energy levels function
'''
def level_print_title(Rydberg = 109737.31568508):
    print(
f"""
    Energy levels for ...
Rydberg constant is  {Rydberg}

---------------------------------------------
 No Pos  J  Parity   Energy Total    Levels
                      (a.u.)         (cm^-1)
---------------------------------------------
"""
    )


def level_J_value(j_index: int) -> str:
    j_value_list = ['0', '1/2', '1', '3/2', '2', '5/2', '3', '7/2', '4', '9/2', '5', '11/2', '6', '13/2', '7', '15/2', '8', '17/2', '9', '19/2', '10', '21/2', '11', '23/2', '12', '25/2', '13', '27/2', '14', '29/2', '15', '31/2', '16', '33/2', '17', '35/2', '18', '37/2', '19', '39/2', '20', '41/2', '21', '43/2', '22']

    return j_value_list[j_index-1]

def level_parity(parity_index: int) -> str:
    parity_list = ['+', '-']

    return parity_list[parity_index-1]

def energy_au_cm(energy_au: float, Rydberg = 109737.31568508) -> float:

    return energy_au * Rydberg * 2


######################################################################

def align_2d_list_columns(two_dimensional_list):
    """
    根据最长列的长度，对二维列表中的所有列进行补0对齐。

    参数:
    two_dimensional_list: 一个二维列表（列表的列表），其中每列的长度可能不同。

    返回:
    一个新的二维列表，其中所有列的长度都与最长列对齐。
    """
    # 获取最长列的长度
    max_length = max(len(column) for column in two_dimensional_list)
    
    # 对每列进行处理，确保长度与最长列对齐
    aligned_list = []
    for column in two_dimensional_list:
        # 计算当前列需要补充的0的数量
        fill_count = max_length - len(column)
        # 进行补0操作并添加到结果列表中
        aligned_column = np.pad(column, (0, fill_count), mode='constant', constant_values=0)
        aligned_list.append(aligned_column)
    
    return aligned_list

######################################################################

def int_nl_2_str_nl(n: int, kappa: int) -> str:
    r'''
    $j = l + 1/2, \kappa = -(l+1)$
    $j = l - 1/2, \kappa = +l $
    '''
    l_list = ['s', 'p', 'd', 'f', 'g', 'h', 'i']
    str_nl = ''
    if kappa > 0:
        l = kappa
        str_nl = f"{n}{l_list[l]}-"
    elif kappa < 0:
        l = -kappa - 1
        str_nl = f"{n}{l_list[l]} "
    else:
        print("error: kappa should not be zero")
    
    return str_nl


def str_subshell_2_kappa(str_subshell: str) -> int:
    r'''
    $j = l + 1/2, \kappa = -(l+1)$
    $j = l - 1/2, \kappa = +l $
    '''
    kappa_value = {
        "s ": -1,
        "p-":  1,
        "p ": -2,
        "d-":  2,
        "d ": -3,
        "f-":  3,
        "f ": -4,
        "g-":  4,
        "g ": -5,
        "h-":  5,
        "h ": -6,
        "i-":  6,
        "i ": -7,
        "j-":  7,
        "j ": -8,
        "k-":  8,
        "k ": -9
    }

    return kappa_value.get(str_subshell, 0)

######################################################################

def doubleJ_to_J(doubleJ):
    if doubleJ % 2 == 0:
        return f"{int(doubleJ/2)}"
    else:
        return f"{doubleJ}/2"

######################################################################

def lsj_transition_data_level_location(transition_data_df : pd.DataFrame, level_df : pd.DataFrame, level_file_parameters : dict) -> pd.DataFrame:
    
    transition_data_df['Upper_level_location'] = 0
    transition_data_df['Upper_level_location'] = transition_data_df['Upper_level_location'].astype('int')
    transition_data_df['Lower_level_location'] = 0
    transition_data_df['Lower_level_location'] = transition_data_df['Lower_level_location'].astype('int')
    level_paramenter = level_file_parameters.get('level_parameter')
    level_as = level_file_parameters.get('this_as')
    
    level_conf_column = f"Configuration_{level_paramenter}{level_as}raw"
    
    def get_level_index(level_J : str, level_conf : str, level_df : pd.DataFrame) -> int:
        
        level_index = level_df[(level_df['J'] == level_J) & (level_df[f'{level_conf_column}'] == level_conf)].index.values[0]
        
        return level_index

    for index, row in transition_data_df.iterrows():
        print(row["Upper_J"], row["Upper_configuration"])
        temp_upper_index = get_level_index(row['Upper_J'], row['Upper_configuration'], level_df)
        
        print(row['Lower_J'], row["Lower_configuration"])
        temp_lower_index = get_level_index(row['Lower_J'], row['Lower_configuration'], level_df)

        transition_data_df.loc[index, 'Upper_level_location'] = temp_upper_index + 1
        transition_data_df.loc[index, 'Lower_level_location'] = temp_lower_index + 1
    return transition_data_df


######################################################################

def transition_data_level_location(transition_data_df : pd.DataFrame, level_df : pd.DataFrame) -> pd.DataFrame:
    
    transition_data_df['Upper_level_location'] = 0
    transition_data_df['Upper_level_location'] = transition_data_df['Upper_level_location'].astype('int')
    transition_data_df['Lower_level_location'] = 0
    transition_data_df['Lower_level_location'] = transition_data_df['Lower_level_location'].astype('int')

    for index, row in tqdm(transition_data_df.iterrows()):

        # print(row["Upper_loc"], row["Upper_J"], row["Upper_parity"])
        temp_upper_index = level_df[(level_df["Pos"] == row["Upper_loc"]) & (level_df["J"] == row["Upper_J"]) & (level_df["Parity"] == row["Upper_parity"])].index.values[0]
        # print(row["Lower_loc"], row["Lower_J"], row["Lower_parity"])
        temp_lower_index = level_df[(level_df["Pos"] == row["Lower_loc"]) & (level_df["J"] == row["Lower_J"]) & (level_df["Parity"] == row["Lower_parity"])].index.values[0]
        transition_data_df.loc[index, 'Upper_level_location'] = temp_upper_index + 1
        transition_data_df.loc[index, 'Lower_level_location'] = temp_lower_index + 1

    return transition_data_df

######################################################################

def transition_dT_cal(transition_rate_B, transition_rate_C):
    
    transition_dT = abs(transition_rate_B - transition_rate_C) / max(transition_rate_B, transition_rate_C)
    
    return transition_dT

######################################################################

# Function to read Fortran-style binary records (assume 4-byte record marker)
def read_fortran_record(file, dtype, count=1):
    # Read the record length (4 bytes before the data)
    record_len_before = np.fromfile(file, dtype=np.int32, count=1)[0]
    
    # Print for debugging
    # print(f"Record length before: {record_len_before}")
    
    # Read the actual data
    data = np.fromfile(file, dtype=dtype, count=count)
    
    # Read the record length (4 bytes after the data)
    record_len_after = np.fromfile(file, dtype=np.int32, count=1)[0]
    
    # Print for debugging
    # print(f"Record length after: {record_len_after}")
    
    # Verify that the record lengths match
    if record_len_before != record_len_after:
        raise ValueError(f"Record length mismatch: {record_len_before} != {record_len_after}")
    
    return data


######################################################################

def chunk_string(s: str, n: int) -> list[str]:
    """将字符串分割成固定长度的块"""
    return [s[i:i+n] for i in range(0, len(s), n)]

######################################################################

def get_csfs_file_peel_subshells(csfs_file_data: CSFs) -> list[str]:
    # 从CSFs文件数据中获取最外层电子壳层信息
    peel_subshells = csfs_file_data.subshell_info_raw[-1].rstrip('\n')
    
    # 将长字符串按每5个字符一组分割成列表
    return chunk_string(peel_subshells, 5)

######################################################################
