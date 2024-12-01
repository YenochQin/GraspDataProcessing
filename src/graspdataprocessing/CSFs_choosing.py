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
import re
from tqdm import tqdm
from .data_IO import GraspFileLoad





def orbital_charged_state(orbital_CSF: str):
    orbital_state = re.findall(r'([0-9]*)([s,p,d,f,g][\s,-])\( (\d+)\)', orbital_CSF)[0]
    main_quantum_num = orbital_state[0]
    orbital_name = orbital_state[1]
    orbital_charged_num = int(orbital_state[2])
    
    return main_quantum_num, orbital_name, orbital_charged_num


def if_orbital_full_charged(orbital_name: str, orbital_charged_num: int):
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
    if full_charged[orbital_name] == orbital_charged_num:
        return True
    else:
        return False

def CSF_orbital_split(CSF: str):
    orbitals_charged = re.split(r'\s*(?=\d\w[\s*,-]\(.*?\))', CSF)
    orbitals_charged = [item for item in orbitals_charged if item!= '']
    # print(orbitals_charged)
    orbital_unfully_charged = {}
    orbital_fully_charged = {}

    for orbital in orbitals_charged:
        # print(orbital)
        temp_quantum_num, temp_orbital, temp_charged_num = orbital_charged_state(orbital)
        
        if if_orbital_full_charged(temp_orbital, temp_charged_num):
            print(f"{temp_quantum_num}{temp_orbital}({temp_charged_num}) is fully charged.")
            orbital_fully_charged.update({temp_quantum_num + temp_orbital : temp_charged_num})
        else:
            orbital_unfully_charged.update({temp_quantum_num + temp_orbital : temp_charged_num})

    return orbital_unfully_charged, orbital_fully_charged

def orbitals_J_value_parser(orbitals_J_value: str, orbital_unfully_charged: dict):
    orbitals_J_value_list = re.findall(r'\S+', orbitals_J_value)
    orbital_unfully_charged_j = {key: value for key, value in zip(orbital_unfully_charged.keys(), orbitals_J_value_list)}
    return orbital_unfully_charged_j
