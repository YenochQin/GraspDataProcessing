#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :read_level.py
@date :2024/05/07 15:26:20
@author :YenochQin (秦毅)
'''

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 添加包的路径到 sys.path
sys.path.append('D:\\Python\\grasp_use\\GraspDataTransfer')
import graspdatatransfer as gdt


level_test = {"atom": "NdGalike", 
              "file_dir": "D:\\Python\\grasp_use\\GraspDataTransfer\\graspdatatransfer\\data", 
              "file_name": "", 
              "parameters": "cv3pas", 
              "this_as": 0, 
              "cut_off_subshell": "3d", 
              "file_type": "level"}

as_list = [0,1, 2, 3, 4, 5]

lllldata = gdt.mcdhf_energy_data_collection(level_test, as_list)
print(lllldata)


data_parameter1 = {"atom": "NdGalike", 
              "file_dir": "D:\\Python\\grasp_use\\GraspDataTransfer\\graspdatatransfer\\data", 
              "file_name": "NdGalikecv3pCIas5", 
              "parameters": "cv3pCIas", 
              "this_as": 5, 
              "cut_off_subshell": "3d", 
              "file_type": "level"}
lllldata = gdt.ci_energy_data_collection(lllldata, data_parameter1)


data_parameter2 = {"atom": "NdGalike", 
              "file_dir": "D:\\Python\\grasp_use\\GraspDataTransfer\\graspdatatransfer\\data", 
              "file_name": "NdGalikecv3pCIas5", 
              "parameters": "cv3pCIas", 
              "this_as": 5, 
              "cut_off_subshell": "3d", 
              "file_type": "lsj"}
# lllldata = gdt.LevelsASFComposition(lllldata, data_parameter2)

lllldata = gdt.LevelsASFComposition(lllldata, data_parameter2).level_comp_of_asf()

lllldata.to_excel("test.xlsx")
print(lllldata)