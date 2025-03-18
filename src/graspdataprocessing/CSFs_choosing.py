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



def mix_data_abs_above_threshold(mix_data_array: np.ndarray, threshold=0.1):
    """
    筛选并排序混合数据中绝对值超过阈值的元素
    
    参数：
        mix_data_array: 输入的混合数据数组
        threshold: 阈值，默认0.1
        
    返回：
        按绝对值降序排列的元组列表，每个元组包含：
        (元素值, 对应索引数组)
    """
    # 生成布尔掩码标识绝对值超过阈值的元素
    abs_above_threshold_mask = np.abs(mix_data_array) > threshold

    # 提取超过阈值的实际值
    values_above_threshold = mix_data_array[abs_above_threshold_mask]

    # 获取满足条件的元素索引（返回二维坐标数组）
    indices_where_abs_above_threshold = np.argwhere(abs_above_threshold_mask)

    # 将值和索引配对组合
    result_pairs = list(zip(values_above_threshold, indices_where_abs_above_threshold))
    
    # 按元素绝对值降序排序
    sorted_result = sorted(result_pairs, key=lambda x: abs(x[0]), reverse=True)
    
    return sorted_result

# def csf_mix_above_threshold_coupling_info(mix_data_above_threshold_list: List, csf_data_list: List):
    
#     csf_coupling_info = []
    
#     for i in mix_data_above_threshold_list:












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