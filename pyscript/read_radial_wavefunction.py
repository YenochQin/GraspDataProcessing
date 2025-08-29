#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :read_radial_wavefunction.py
@date :2024/05/07 14:48:07
@author :YenochQin (秦毅)
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import graspkit as gk



''''
test1 or test2 
choose one to get radial wavefunction data
'''

test1 = {"atom": "Ni", 
              "file_dir": "D:\\PythonProjects\\graspdatatransfer\\data",
              "file_name": "Ni_even4.w", 
              "parameters": "_even4", 
              "max_r": 1,
              "file_type": "wavefunction"}

data = gk.GraspFileLoad(test1)


test = data.data_file_process()

plt.axhline(y=0.0, color='black', lw=0.75, ls='--')
plt.plot(test['r(a.u)'][1:], test['P(1s )'][1:])
plt.plot(test['r(a.u)'][1:], test['P(2s )'][1:],label ='P(2s)')
plt.plot(test['r(a.u)'][1:], test['P(2p )'][1:],label='P(2p)')
plt.plot(test['r(a.u)'][1:], test['P(2p-)'][1:],label='P(2p-)')
plt.plot(test['r(a.u)'][1:], test['P(3s )'][1:],label='P(3s)')
plt.plot(test['r(a.u)'][1:], test['P(3p )'][1:],label='P(3p)')
plt.plot(test['r(a.u)'][1:], test['P(3p-)'][1:],label='P(3p-)')
plt.plot(test['r(a.u)'][1:], test['P(3d )'][1:],label='P(3d)')  
plt.plot(test['r(a.u)'][1:], test['P(3d-)'][1:],label='P(3d-)')
# plt.xlim([0, test['r(a.u)']])
plt.legend()
# plt.savefig(f"{atom}{parameters}.pdf", bbox_inches='tight')
plt.show()