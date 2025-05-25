#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :data_modules.py
@date :2025/04/09 17:04:00
@author :YenochQin (秦毅)
'''

from dataclasses import dataclass

@dataclass(frozen=True)
class MixCoefficientData:
    block_num: int
    block_index_list: list
    block_CSFs_nums: list
    block_energy_count_list: list 
    j_value_location_list: list
    parity_list: list
    block_levels_index_list: list
    block_energy_list: list
    block_level_energy_list: list
    mix_coefficient_list: list
    level_list: list
    
@dataclass
class CSFs:
    subshell_info_raw: list
    CSFs_block_j_value: list
    parity: str
    CSFs_block_data: list
    CSFs_block_length: list
    block_num: int