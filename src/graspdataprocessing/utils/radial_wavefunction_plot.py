#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :radial_wavefunction_plot.py
@date :2024/04/09 23:02:43
@author :YenochQin (秦毅)
'''

import numpy as np
import pandas as pd

from ..data_IO.grasp_raw_data_load import GraspFileLoad
from matplotlib import pyplot as plt

class RadialWavefunctionPlot:
    def __init__(self, data_file_parameter):
        self.data_file_parameter = data_file_parameter
        self.max_r = data_file_parameter.get("max_r", 1)
      
        
    def radial_wavefunction_data_load(self):
        self.raw_radial_wavefunction_data = GraspFileLoad(self.data_file_parameter).data_identification()
        
        if "PLOT" in self.f_type.upper():
            self.temp_data = []
            for line in self.raw_radial_wavefunction_data:
                self.temp_data.append(line.split())
            
            self.temp_data[1:] = [[float(x) for x in row] for row in self.temp_data[1:]]
            self.dataframe_radial_wavefunction = pd.DataFrame(self.temp_data[1:], columns=self.temp_data[0])
            
        elif "RADIAL" in self.f_type.upper():
            self.dataframe_radial_wavefunction = self.raw_radial_wavefunction_data

        
        return self.dataframe_radial_wavefunction
        
    def figure_data_set(self):
        self.orbital_set = self.data_file_parameter.get("orbital_set", "all")
        self.dataframe_radial_wavefunction = self.radial_wavefunction_data_load(self)
        raw_data_orbitals = self.dataframe_radial_wavefunction.columns.tolist()
        figure_orbitals = []
        if self.orbital_set == "all":
            self.dataframe_radial_wavefunction = self.dataframe_radial_wavefunction
        elif "P" in self.orbital_set.upper():
            self.dataframe_radial_wavefunction = self.dataframe_radial_wavefunction[raw_data_orbitals[1:]]
        elif "Q" in self.orbital_set.upper():
            self.dataframe_radial_wavefunction = self.dataframe_radial_wavefunction[raw_data_orbitals[2:]]

       
        
    # def draw_P_radial_wavefunction(self):
    #     self.radial_wavefunction_data()
    #     plt.plot(self.radial_wavefunction_data["r(a.u.)"], self.radial_wavefunction_data["P"])
    #     plt.xlabel("r (Bohr)")
    #     plt.ylabel("P (a.u.)")
        
        
    # def draw_Q_radial_wavefunction(self):
def orbital_set_extract(raw_data_orbitals, orbital_sets):
    extracted_orbitals = []
    if orbital_sets == "all":
        extracted_orbitals = raw_data_orbitals
    elif "P" in orbital_sets.upper():
        extracted_orbitals = [orb for orb in raw_data_orbitals if "P" in orb]
    elif "Q" in orbital_sets.upper():
        extracted_orbitals = [orb for orb in raw_data_orbitals if "Q" in orb]
    elif isinstance(orbital_sets, list):
        if isinstance(input_value[0], str):
            extracted_orbitals = [orb for orb in raw_data_orbitals if "P" in orb]