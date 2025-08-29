#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :read_transition_data.py
@date :2024/05/08 14:26:22
@author :YenochQin (秦毅)
'''
import numpy as np
import pandas as pd
import graspdataprocessing as gdt


level_test = {"atom": "NdGalike", 
              "file_dir": "D:\\PythonProjects\\graspdatatransfer\\data\\trans", 
              "file_name": "", 
              "level_parameter": "cv3pCIas", 
              "this_as": 0, 
              "cut_off_subshell": "3d", 
              "file_type": "TRANSITION"}


asdf = gdt.TransitionDataCollection(level_test)

tttt = asdf.transition_data2dataframe()

print(tttt)