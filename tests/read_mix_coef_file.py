#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :read_mix_coef_file.py
@date :2024/05/07 14:22:17
@author :YenochQin (秦毅)
'''
import graspdataprocessing as gdp

plottest = {
    "atom": "NdGalike", 
    "file_dir": "D:\\PythonProjects\\graspdatatransfer\\data", 
    "file_name": "oddcv3pCIas5.cm", 
    "level_parameter": "cv3pCI",
    "file_type": "mix"
    }

test = gdp.GraspFileLoad(plottest)
test.data_file_process()