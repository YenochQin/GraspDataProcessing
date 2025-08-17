#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :read_lsj_transition_data.py
@date :2024/05/08 14:26:22
@author :YenochQin (秦毅)
'''

import graspdataprocessing as gdp


test = {"atom": "NdGalike", 
              "file_dir": "D:\\PythonProjects\\graspdatatransfer\\data\\trans", 
              "file_name": "", 
              "level_parameter": "cv3pCIas", 
              "this_as": 0, 
              "cut_off_subshell": "3d", 
              "file_type": "lsjTRANSITION"}


transition_data_collection = gdp.LSJTransitionDataCollection(test)

data_pd = transition_data_collection.transition_data2dataframe()

print(data_pd)