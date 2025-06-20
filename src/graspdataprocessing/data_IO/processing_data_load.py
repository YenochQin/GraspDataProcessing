#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :processing_data_load.py
@date :2025/06/16 16:30:36
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



def load_csf_metadata(
                        filepath: Union[str, Path]
                        ) -> dict:
    # 转换为Path对象
    filepath = Path(filepath)
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    

#######################################################################

def load_csfs_binary(filepath: Union[str, Path]) -> CSFs:
    filepath = Path(filepath)
    
    # 检查文件路径是否已经有正确的后缀
    if not str(filepath).endswith('.pkl.gz'):
        filepath = filepath.with_suffix('.pkl.gz')
    
    with gzip.open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return CSFs(
        subshell_info_raw=data['metadata']['subshell_info_raw'],
        CSFs_block_j_value=data['metadata']['CSFs_block_j_value'],
        parity=data['metadata']['parity'],
        CSFs_block_data=data['block_data'],  # 原始嵌套结构
        CSFs_block_length=data['metadata']['CSFs_block_length'],
        block_num=data['metadata']['block_num']
    )

#######################################################################

def csfs_index_load(load_csfs_index_file_path):
    
    # 转换为Path对象便于处理
    file_path = Path(load_csfs_index_file_path)
    
    # 检查路径是否有扩展名
    if not file_path.suffix:
        # 没有扩展名时，按优先级尝试不同格式
        for ext in ['.pkl', '.msgpack']:
            full_path = file_path.with_suffix(ext)
            if full_path.exists():
                file_path = full_path
                break
        else:
            # 如果都不存在，默认使用.pkl扩展名（会在下面报错）
            file_path = file_path.with_suffix('.pkl')
    
    # 根据文件扩展名选择加载方式
    if file_path.suffix == '.msgpack':
        # 向后兼容：加载旧的msgpack格式文件
        import msgpack
        with open(file_path, 'rb') as f:
            blocks_csfs_index = msgpack.load(f, strict_map_key=False)
    else:
        # 默认使用pickle格式
        with open(file_path, 'rb') as f:
            blocks_csfs_index = pickle.load(f)
        
    return blocks_csfs_index

#######################################################################

def load_large_hash(
                        file_path: Union[str, Path]
                        ) -> Dict[int, Dict[str, int]]:
    """从文件加载预计算的哈希映射"""
    # 转换为Path对象
    file_path = Path(file_path)
    
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
#######################################################################

def load_config(
                    config_path: Union[str, Path]
                    ):
    """加载TOML配置文件并进行类型转换和数据处理"""
    # 转换为Path对象
    config_path = Path(config_path)
    
    # 使用标准库 tomllib 读取TOML文件
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    # 类型转换和数据处理
    config = _process_config_data(config)
    
    return SimpleNamespace(**config)

def _process_config_data(config):
    """处理配置数据，进行类型转换和验证"""
    # 浮点数转换
    config['cutoff_value'] = float(config['cutoff_value'])
    config['initial_ratio'] = float(config['initial_ratio'])
    config['expansion_ratio'] = float(config['expansion_ratio'])
    
    # 整数转换
    config['cal_loop_num'] = int(config['cal_loop_num'])
    config['difference'] = int(config['difference'])
    
    # 路径转换
    config['root_path'] = Path(config['root_path'])
    config['scf_cal_path'] = config['root_path'] / f"{config['conf']}_{config['cal_loop_num']}"
    
    # 模型参数处理
    if 'model_params' in config:
        model_params = config['model_params']
        
        # 转换模型参数中的整数
        if 'n_estimators' in model_params:
            model_params['n_estimators'] = int(model_params['n_estimators'])
        if 'random_state' in model_params:
            model_params['random_state'] = int(model_params['random_state'])
            
        # 处理class_weight字典，确保键为整数
        if 'class_weight' in model_params and isinstance(model_params['class_weight'], dict):
            class_weight = {}
            for k, v in model_params['class_weight'].items():
                class_weight[int(k)] = float(v)
            model_params['class_weight'] = class_weight
    
    # 数据验证
    _validate_config_data(config)
    
    return config

def _validate_config_data(config):
    """验证配置数据的有效性"""
    # 验证必需字段
    required_fields = ['atom', 'conf', 'cal_loop_num', 'cutoff_value', 'initial_ratio']
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"配置文件缺少必需字段: {missing_fields}")
    
    # 验证数值范围
    if 'cutoff_value' in config:
        if config['cutoff_value'] <= 0:
            raise ValueError("cutoff_value 必须大于 0")
    
    if 'initial_ratio' in config:
        if not (0 < config['initial_ratio'] <= 1):
            raise ValueError("initial_ratio 必须在 (0, 1] 范围内")
    
    if 'expansion_ratio' in config:
        if config['expansion_ratio'] < 1:
            raise ValueError("expansion_ratio 必须大于等于 1")
    
    # 验证光谱项列表
    if 'spetral_term' in config:
        if not isinstance(config['spetral_term'], list) or len(config['spetral_term']) == 0:
            raise ValueError("spetral_term 必须是非空列表")
    
    print(f"配置验证通过: cutoff_value={config.get('cutoff_value')}, initial_ratio={config.get('initial_ratio')}")


#######################################################################

def load_descriptors(
                        load_path: Union[str, Path], 
                        file_format: Optional[str] = None
                        ) -> Optional[np.ndarray]:
    """
    加载描述符数组
    
    Args:
        load_path (Union[str, Path]): 加载路径（可含或不含扩展名）
        file_format (Optional[str]): 文件格式，如果为None则从文件扩展名自动推断
    
    Returns:
        Optional[np.ndarray]: 描述符数组，加载失败返回None
    
    Example:
        >>> descriptors = load_descriptors('output/csf_descriptors.npy')
        >>> descriptors = load_descriptors(Path('output/csf_descriptors.npy'))
        >>> descriptors = load_descriptors('output/csf_descriptors', 'csv')
    """
    
    # 转换为Path对象
    load_path = Path(load_path)
    
    # 自动推断文件格式
    if file_format is None:
        if load_path.suffix == '.npy':
            file_format = 'npy'
            load_path = load_path.with_suffix('')  # 移除扩展名
        elif load_path.suffix == '.csv':
            file_format = 'csv'
            load_path = load_path.with_suffix('')
        elif load_path.suffix == '.pkl':
            file_format = 'pkl'
            load_path = load_path.with_suffix('')
        else:
            # 尝试自动检测（使用新的文件名格式）
            if (load_path.parent / f"{load_path.name}_descriptors.npy").exists():
                file_format = 'npy'
            elif (load_path.parent / f"{load_path.name}_descriptors.csv").exists():
                file_format = 'csv'
            elif (load_path.parent / f"{load_path.name}_descriptors.pkl").exists():
                file_format = 'pkl'
            else:
                print(f"Error: Cannot find file with path: {load_path}")
                return None
    
    try:
        if file_format.lower() == 'npy':
            file_path = load_path.parent / f"{load_path.name}_descriptors.npy"
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return None
            descriptors = np.load(file_path)
            print(f"Descriptors loaded from: {file_path}")
            return descriptors
            
        elif file_format.lower() == 'csv':
            file_path = load_path.parent / f"{load_path.name}_descriptors.csv"
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return None
            df = pd.read_csv(file_path)
            descriptors = df.values
            print(f"Descriptors loaded from: {file_path}")
            return descriptors
            
        elif file_format.lower() == 'pkl':
            file_path = load_path.parent / f"{load_path.name}_descriptors.pkl"
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return None
            with open(file_path, 'rb') as f:
                descriptors = pickle.load(f)
            print(f"Descriptors loaded from: {file_path}")
            return descriptors
            
        else:
            print(f"Error: Unsupported file format: {file_format}")
            return None
            
    except Exception as e:
        print(f"Error loading descriptors: {str(e)}")
        return None

def load_descriptors_with_multi_block(
                                        load_path: Union[str, Path], 
                                        file_format: Optional[str] = None
                                        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    加载带标签的描述符数组
    
    Args:
        load_path (Union[str, Path]): 加载路径（不含扩展名）
        file_format (Optional[str]): 文件格式，如果为None则自动推断
    
    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: (描述符数组, 标签数组)，加载失败返回None
    
    Example:
        >>> descriptors, labels = load_descriptors_with_block_indices('ml_data/features')
        >>> descriptors, labels = load_descriptors_with_block_indices(Path('ml_data/features'))
        >>> descriptors, labels = load_descriptors_with_block_indices('ml_data/features', 'csv')
    """
    
    # 转换为Path对象
    load_path = Path(load_path)
    
    # 自动推断文件格式
    if file_format is None:
        if (load_path.parent / f"{load_path.name}_descriptors_block_indices.csv").exists():
            file_format = 'csv'
        elif (load_path.parent / f"{load_path.name}_descriptors.npy").exists() and (load_path.parent / f"{load_path.name}_descriptors_block_indices.npy").exists():
            file_format = 'npy'
        elif (load_path.parent / f"{load_path.name}_descriptors_block_indices.pkl").exists():
            file_format = 'pkl'
        else:
            print(f"Error: Cannot find files with path: {load_path}")
            return None
    
    try:
        if file_format.lower() == 'csv':
            file_path = load_path.parent / f"{load_path.name}_descriptors_block_indices.csv"
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return None
                
            df = pd.read_csv(file_path)
            # 最后一列是标签，其余是描述符
            descriptors = df.iloc[:, :-1].to_numpy()
            labels = df.iloc[:, -1].to_numpy()
            print(f"Descriptors and labels loaded from: {file_path}")
            return descriptors, labels
            
        elif file_format.lower() == 'npy':
            data_path = load_path.parent / f"{load_path.name}_descriptors.npy"
            labels_path = load_path.parent / f"{load_path.name}_descriptors_block_indices.npy"
            
            if not data_path.exists():
                print(f"Error: Data file not found: {data_path}")
                return None
            if not labels_path.exists():
                print(f"Error: Labels file not found: {labels_path}")
                return None
                
            descriptors = np.load(data_path)
            labels = np.load(labels_path)
            print(f"Descriptors loaded from: {data_path}")
            print(f"Labels loaded from: {labels_path}")
            return descriptors, labels
            
        elif file_format.lower() == 'pkl':
            file_path = load_path.parent / f"{load_path.name}_descriptors_block_indices.pkl"
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return None
                
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)
                
            if 'descriptors' not in data_dict or 'labels' not in data_dict:
                print(f"Error: Invalid data format in {file_path}")
                return None
                
            descriptors = data_dict['descriptors']
            labels = data_dict['labels']
            print(f"Descriptors and labels loaded from: {file_path}")
            return descriptors, labels
            
        else:
            print(f"Error: Unsupported file format: {file_format}")
            return None
            
    except Exception as e:
        print(f"Error loading descriptors with labels: {str(e)}")
        return None
