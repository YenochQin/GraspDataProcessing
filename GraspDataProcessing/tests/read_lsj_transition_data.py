#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :read_lsj_transition_data.py
@date :2024/05/08 14:26:22
@author :YenochQin (秦毅)
'''

import sys

# 添加包的路径到 sys.path
# sys.path.append('D:\\Python\\grasp_use\\GraspDataTransfer')
sys.path.append("C:\\Users\\yenoch\\CodeFiles\\PythonCodes")
import graspdatatransfer as gdt


# test = {"atom": "NdGalike", 
#               "file_dir": "D:\\Python\\grasp_use\\GraspDataTransfer\\graspdatatransfer\\data", 
#               "file_name": "", 
#               "parameters": "cv3pCIas", 
#               "this_as": 0, 
#               "cut_off_subshell": "3d", 
#               "file_type": "lsjTRANSITION"}
test = {"atom": "NdGalike", 
              "file_dir": "C:\\Users\\yenoch\\CodeFiles\\PythonCodes\\graspdatatransfer\\data\\trans", 
              "file_name": "", 
              "parameters": "cv3pCIas", 
              "this_as": 0, 
              "cut_off_subshell": "3d", 
              "file_type": "lsjTRANSITION"}


transition_data_collection = gdt.LSJTransitionCollection(test)

data_pd = transition_data_collection.transition_data2dataframe()

print(data_pd)