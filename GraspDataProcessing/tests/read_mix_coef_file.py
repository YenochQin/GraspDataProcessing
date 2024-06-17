#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :read_mix_coef_file.py
@date :2024/05/07 14:22:17
@author :YenochQin (秦毅)
'''
import sys

# 添加包的路径到 sys.path
sys.path.append('D:\\Python\\grasp_use\\GraspDataTransfer')
import graspdatatransfer as gdt

plottest = {
    "atom": "Ni", 
    "dir": "D:\\Python\\grasp_use\\GraspDataTransfer\\data\\cv\\cv3", 
    "file": "4f2as5.cm", 
    "parameters": "cv3",
    "f_type": "mix"
    }

test = gdt.GraspFileLoad(plottest)
test.data_file_process()