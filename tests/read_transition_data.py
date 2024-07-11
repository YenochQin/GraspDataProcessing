#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :read_transition_data.py
@date :2024/05/08 14:26:22
@author :YenochQin (秦毅)
'''

import sys

# 添加包的路径到 sys.path
sys.path.append('D:\\Python\\grasp_use\\graspdataprocessing')
import graspdataprocessing as gdt


level_test = {"atom": "NdGalike", 
              "file_dir": "D:\\Python\\grasp_use\\graspdataprocessing\\graspdataprocessing\\data", 
              "file_name": "", 
              "parameters": "cv3pCIas", 
              "this_as": 0, 
              "cut_off_subshell": "3d", 
              "file_type": "TRANSITION"}


asdf = gdt.TransitionDataCollection(level_test)

tttt = asdf.transition_data2dataframe()