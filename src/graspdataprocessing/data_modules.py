#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :data_modules.py
@date :2025/04/09 17:04:00
@author :YenochQin (秦毅)
'''

from dataclasses import dataclass
import numpy as np
from typing import Union, List


@dataclass(frozen=True)
class MixCoefficientData:
    block_num: int
    block_index_List: List
    block_CSFs_nums: List
    block_energy_count_List: List 
    level_J_value_List: List
    parity_List: List
    block_levels_index_List: List
    block_energy_List: List
    block_level_energy_List: List
    mix_coefficient_List: List
    level_List: List

@dataclass
class CSFs:
    subshell_info_raw: List[str]
    CSFs_block_j_value: List[str]
    parity: str
    CSFs_block_data: List
    CSFs_block_length: Union[List[int], np.ndarray]  # 兼容列表或ndarray
    block_num: int

    @classmethod
    def from_dict(cls, data: dict) -> 'CSFs':
        """从字典创建CSFs实例（自动处理NumPy数组转换）"""
        return cls(
            subshell_info_raw=data.get('subshell_info_raw', []),
            CSFs_block_j_value=data.get('CSFs_block_j_value', []),
            parity=data.get('parity', ''),
            CSFs_block_data=data.get('CSFs_block_data', []),
            CSFs_block_length=np.array(data['CSFs_block_length']) 
                if isinstance(data.get('CSFs_block_length', []), List) 
                else data.get('CSFs_block_length', np.array([])),
            block_num=data.get('block_num', 0)
        )
