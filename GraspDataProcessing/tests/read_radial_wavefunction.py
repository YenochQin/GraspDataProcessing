#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :read_radial_wavefunction.py
@date :2024/05/07 14:48:07
@author :YenochQin (秦毅)
'''


import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 添加包的路径到 sys.path
sys.path.append('D:\\Python\\grasp_use\\GraspDataTransfer')
import graspdatatransfer as gdt



''''
test1 or test2 
choose one to get radial wavefunction data
'''

test1 = {"atom": "Ni", 
              "file_dir": 'D:\\Python\\grasp_use\\GraspDataTransfer\\graspdatatransfer\\data\\Ni_even4.w', 
              "file_name": "D:\\Python\\grasp_use\\GraspDataTransfer\\graspdatatransfer\\data\\Ni_even4.w", 
              "parameters": "_even4", 
              "max_r": 1,
              "f_type": "wavefunction"}

test2 = {"atom": "NdV", 
              "file_dir": 'D:\\Python\\grasp_use\\GraspDataTransfer\\graspdatatransfer\\data', 
              "file_name": "D:\\Python\\grasp_use\\GraspDataTransfer\\graspdatatransfer\\data\\tttt123.plot", 
              "parameters": "cv1", 
              "max_r": 1,
              "f_type": "plot"}

data = gdt.GraspFileLoad(test1)
data = gdt.GraspFileLoad(test2)

test = data.data_file_process()

plt.axhline(y=0.0, color='black', lw=0.75, ls='--')
plt.plot(test['r(a.u.)'][1:], test['P(1s)'][1:] ,label='P(1s)')
plt.plot(test['r(a.u.)'][1:], test['P(2s)'][1:],label = 'P(2s)')
plt.plot(test['r(a.u.)'][1:], test['P(2p)'][1:],label='P(2p)')
plt.plot(test['r(a.u.)'][1:], test['P(2p-)'][1:],label='P(2p-)')
plt.plot(test['r(a.u.)'][1:], test['P(3s)'][1:],label='P(3s)')
plt.plot(test['r(a.u.)'][1:], test['P(3p)'][1:],label='P(3p)')
plt.plot(test['r(a.u.)'][1:], test['P(3p-)'][1:],label='P(3p-)')
plt.plot(test['r(a.u.)'][1:], test['P(3d)'][1:],label='P(3d)')
plt.plot(test['r(a.u.)'][1:], test['P(3d-)'][1:],label='P(3d-)')
plt.xlim([0,plottest['max_r']])
plt.legend()
# plt.savefig(f"{atom}{parameters}.pdf", bbox_inches='tight')
plt.show()