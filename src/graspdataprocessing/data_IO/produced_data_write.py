#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :produced_data_write.py
@date :2025/06/16 16:15:12
@author :YenochQin (秦毅)
'''

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from types import SimpleNamespace
from dataclasses import dataclass
import gzip
import pickle
import tomllib

import numpy as np
import pandas as pd


from ..utils.tool_function import *
from ..utils.data_modules import *
from ..CSFs_choosing.CSFs_choosing import *
from ..CSFs_choosing.CSFs_compress_extract import *

# TODO not good enough
def write_sorted_CSFs_to_cfile(CSFs_file_info: List, sorted_CSFs_data_list: List, output_file: str):
    """
    将排序后的CSFs数据写入到指定的输出文件中。

    Args:
        CSFs_file_info (List): CSFs文件的头部信息(CSF(s):行上面的信息)
        sorted_CSFs_data (List): 排序后的CSFs数据列表
            sorted_CSFs_data[block
                                [CSFs
                                    [CSFS_1]
                                    [CSFS_2]
                                    ...
                                    [CSFS_n]
                                ]
            ]
        output_file (str): 输出文件的路径。
    """
    if len(CSFs_file_info) != 4:
        raise ValueError('CSFs file header info error!')
    blocks_num = len(sorted_CSFs_data_list)
    with open(output_file, 'w') as file:
        for line in CSFs_file_info:
            file.write(line)  
        
        file.write('CSF(s):\n')
        for index, block in enumerate(sorted_CSFs_data_list):
            if index!= blocks_num-1:
                for csf in block:
                    for line in csf:
                        file.write(line)
                file.write(' *\n')
            else:
                for csf in block:
                    for line in csf:
                        file.write(line)

#######################################################################

def save_csf_metadata(
                        csf_obj: CSFs, 
                        filepath: Union[str, Path]
                        ):
    """保存CSFs元数据（排除CSFs_block_data）到pickle文件"""
    # 转换为Path对象
    filepath = Path(filepath)
    
    metadata = {
        'subshell_info_raw': csf_obj.subshell_info_raw,
        'CSFs_block_j_value': csf_obj.CSFs_block_j_value,
        'parity': csf_obj.parity,
        'CSFs_block_length': csf_obj.CSFs_block_length,
        'block_num': csf_obj.block_num
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(metadata, f)

#######################################################################

def save_csfs_binary(csf_obj: CSFs, filepath: Union[str, Path]):
    filepath = Path(filepath)
    
    # 元数据存储
    metadata = {
        'subshell_info_raw': csf_obj.subshell_info_raw,
        'CSFs_block_j_value': csf_obj.CSFs_block_j_value,
        'parity': csf_obj.parity,
        'CSFs_block_length': np.asarray(csf_obj.CSFs_block_length),
        'block_num': csf_obj.block_num,
        'data_type': 'nested_string'  # 标记特殊数据结构
    }
    
    # 专用压缩存储
    with gzip.open(filepath.with_suffix('.pkl.gz'), 'wb') as f:
        pickle.dump({
            'metadata': metadata,
            'block_data': csf_obj.CSFs_block_data  # 直接存储原始结构
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

#######################################################################

def continue_calculate(
                        save_path: Union[str, Path], 
                        continue_calculate: bool
                        ):
    save_path = Path(save_path)
    
    with open(save_path/'run.input', 'rw') as file:
        file.write(continue_calculate)
        
    return f'Continue calculate is set to {continue_calculate}'

def update_config(config_path, updates):
    """更新TOML配置文件
    
    Args:
        config_path: 配置文件路径
        updates: 要更新的键值对字典
    """
    # 使用标准库 tomllib 读取TOML文件
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    # 更新配置值
    config.update(updates)
    
    # 写入配置文件（标准库 tomllib 不支持写入，使用 tomli-w）
    try:
        import tomli_w
        with open(config_path, 'wb') as f:
            tomli_w.dump(config, f)
    except ImportError:
        raise ImportError(
            "需要安装 tomli-w 库来写入TOML文件。请运行：\n"
            "pip install tomli-w"
        )

#######################################################################

def csfs_index_storange(blocks_csfs_index: Dict, save_file_path):
    """
    将CSFs索引存储到指定的文件中。
    Args:
        blocks_csfs_index (Dict): 包含CSFs索引的字典。
        save_file_path: 存储文件的路径（字符串或Path对象）。
    """
    # 转换为Path对象并检查是否有扩展名
    file_path = Path(save_file_path)
    if not file_path.suffix:
        file_path = file_path.with_suffix('.pkl')
    
    with open(file_path, 'wb') as f:
        pickle.dump(blocks_csfs_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return f'CSFs index has been stored to {file_path}'

#######################################################################
def precompute_large_hash(
                            large_data: List[List[List[str]]], 
                            save_path: Union[str, Path] = "large_data_hash.pkl"
                            ):
    """
    预计算 large_data 的哈希映射（双层字典结构）
    
    返回:
        {block_idx: {csf_str: csf_index}}
    """
    # 转换为Path对象
    save_path = Path(save_path)
    
    # 计算总的CSF数量以提供更有意义的进度信息
    total_csfs = sum(len(block_data) for block_data in large_data)
    print(f"开始计算哈希映射 - 总计 {total_csfs} 个CSFs, {len(large_data)} 个blocks")
    
    large_hash = {}
    processed_csfs = 0
    
    # 使用tqdm显示CSF级别的进度
    with tqdm(total=total_csfs, desc="计算哈希映射", unit="CSFs") as pbar:
        for block_idx, block_data in enumerate(large_data):
            # 显示当前处理的block信息
            pbar.set_postfix({
                'block': f"{block_idx+1}/{len(large_data)}", 
                'block_size': len(block_data)
            })
            
            # 为当前block构建哈希映射
            block_hash = {}
            for idx, csf in enumerate(block_data):
                csf_str = ''.join(item for sublist in csf for item in sublist)
                block_hash[csf_str] = idx
                processed_csfs += 1
                pbar.update(1)
            
            large_hash[block_idx] = block_hash
    
    print(f"哈希映射计算完成 - 处理了 {processed_csfs} 个CSFs")
    
    with open(save_path, "wb") as f:
        pickle.dump(large_hash, f)
    
    return f'hash file has written in file {save_path}'

#######################################################################

def save_descriptors(
                        descriptors: np.ndarray, 
                        save_path: Union[str, Path], 
                        file_format: str = 'npy'
                        ):
    """
    保存描述符数组
    
    Args:
        descriptors (np.ndarray): 描述符数组
        save_path (Union[str, Path]): 保存路径（不含扩展名）
        file_format (str): 保存格式 ('npy', 'csv', 'pkl')
    
    Example:
        >>> descriptors = batch_process_csfs_to_descriptors(csfs_data)
        >>> save_descriptors(descriptors, 'output/csf_descriptors', 'csv')
        >>> save_descriptors(descriptors, Path('output/csf_descriptors'), 'npy')
    """
    
    # 转换为Path对象
    save_path = Path(save_path)
    
    if file_format.lower() == 'npy':
        file_path = save_path.parent / f"{save_path.name}_descriptors.npy"
        np.save(file_path, descriptors)
        print(f"Descriptors saved to: {file_path}")
        
    elif file_format.lower() == 'csv':
        file_path = save_path.parent / f"{save_path.name}_descriptors.csv"
        df = pd.DataFrame(descriptors)
        df.to_csv(file_path, index=False)
        print(f"Descriptors saved to: {file_path}")
        
    elif file_format.lower() == 'pkl':
        file_path = save_path.parent / f"{save_path.name}_descriptors.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(descriptors, f)
        print(f"Descriptors saved to: {file_path}")
        
    else:
        raise ValueError("file_format must be 'npy', 'csv', or 'pkl'")


def save_descriptors_with_multi_block(
                                        descriptors: np.ndarray, 
                                        labels: np.ndarray, 
                                        save_path: Union[str, Path], 
                                        file_format: str = 'npy'
                                        ):
    """
    保存带标签的描述符数组
    
    Args:
        descriptors (np.ndarray): 描述符数组
        labels (np.ndarray): 标签数组
        save_path (Union[str, Path]): 保存路径（不含扩展名）
        file_format (str): 保存格式 ('csv', 'npy', 'pkl')
    
    Example:
        >>> X, y = batch_process_csfs_with_block_indices(csfs_data)
        >>> save_descriptors_with_block_indices(X, y, 'ml_data/features', 'csv')
        >>> save_descriptors_with_block_indices(X, y, Path('ml_data/features'), 'npy')
    """
    
    # 转换为Path对象
    save_path = Path(save_path)
    
    if file_format.lower() == 'csv':
        # CSV格式：将标签作为最后一列
        file_path = save_path.parent / f"{save_path.name}_descriptors_block_indices.csv"
        df = pd.DataFrame(descriptors)
        df['label'] = labels
        df.to_csv(file_path, index=False)
        print(f"Descriptors with labels saved to: {file_path}")
        
    elif file_format.lower() == 'npy':
        # NPY格式：分别保存数据和标签
        data_path = save_path.parent / f"{save_path.name}_descriptors.npy"
        labels_path = save_path.parent / f"{save_path.name}_descriptors_block_indices.npy"
        np.save(data_path, descriptors)
        np.save(labels_path, labels)
        print(f"Descriptors saved to: {data_path}")
        print(f"Labels saved to: {labels_path}")
        
    elif file_format.lower() == 'pkl':
        # PKL格式：保存为字典
        file_path = save_path.parent / f"{save_path.name}_descriptors_block_indices.pkl"
        data_dict = {
            'descriptors': descriptors,
            'labels': labels
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"Descriptors and labels saved to: {file_path}")
        
    else:
        raise ValueError("file_format must be 'csv', 'npy', or 'pkl'")
