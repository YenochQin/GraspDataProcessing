##!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :level_data_collection.py
@date :2023/04/28 11:08:58
@author :YenochQin (秦毅)

@version 1.0: Object Oriented Programming modified from level_data_collection.py
'''

import numpy as np
import pandas as pd
# from pathlib import Path
import re
from tqdm import tqdm
from .data_IO import GraspFileLoad, EnergyFile2csv

#######################################################################
class ConfigurationFormat:
    """ 
    Here is the explanation for the code above:
    1. The class "ConfigurationFormat" is used to format the electron configuration string.
        1.1 subshell_format(): format the subshell string;
        1.2 subshell_skip(): skip the subshells before the given start subshell;
        1.3 conf_format(): format the electron configuration;
        1.4 ls_coupling_format(): format the ls coupling string.
    2. The function "subshell_format()" is used to format the subshell string. 
        2.1 subshell: the subshell string;
        2.2 format_subshell: the formatted subshell string;
        2.3 format_subshell_ls: the formatted subshell in the L-S coupling state;
        2.4 ele_num: the number of electrons in the subshell.
    3. The function "subshell_skip()" is used to skip the subshells before the given start subshell.
    4. The function "conf_format()" is used to format the electron configuration.
        4.1 skipped_formatted_conf: the formatted electron configuration string;
        4.2 conf_skipped_list: the skipped electron configuration list;
    5. The function "ls_coupling_format()" is used to format the ls coupling string.
    """
    
    def __init__(self, temp_configuration:str, cut_off_subshell:str=""):
        self.temp_configuration = re.sub(r'\n', '', temp_configuration)
        self.cut_off_subshell = cut_off_subshell
        self.temp_conf_list = self.temp_configuration.split(".")
        
    def subshell_format(self):
        self.format_subshell = ""
        self.format_subshell_ls = ""
        self.ele_num = "0"
        if "(" and ")" in self.subshell:
            self.ele_num = re.findall(r"[(](.*?)[)]", self.subshell)
            self.format_subshell = f"{self.subshell[0:2]}^{{{self.ele_num[0]}}}"
            if re.findall(r"[)]([0-9][A-Z][0-9]?)[_]", self.subshell):
                self.temp_subshell_ls = re.findall(r"[)]([0-9][A-Z][0-9]?)[_]", self.subshell)
                self.format_subshell_ls = f"(^{self.temp_subshell_ls[0][0]}_{self.temp_subshell_ls[0][-1]}\\text{{{self.temp_subshell_ls[0][1]}}})"
        elif "_" in self.subshell:
            self.ele_num = "1"
            self.format_subshell = f"{self.subshell[0:2]}{self.format_subshell_ls}"
            self.temp_subshell_ls = re.findall(r"[_]([0-9][A-Z]?)", self.subshell)
            self.format_subshell_ls = f"(^{self.temp_subshell_ls[0][0]}\\text{{{self.temp_subshell_ls[0][1]}}})"
        return self.format_subshell, self.format_subshell_ls, self.ele_num
    
    def subshell_skip(self):
        for self.temp_subshell in self.temp_conf_list:
            if self.cut_off_subshell in self.temp_subshell:
                self.start_subshell_index = self.temp_conf_list.index(self.temp_subshell) + 1
                break
            else:
                self.start_subshell_index = 0
        self.temp_conf_list = self.temp_conf_list[self.start_subshell_index:]
        return self.temp_conf_list
    
    def conf_format(self):
        self.skipped_formatted_conf = ""
        self.conf_skipped_list = ConfigurationFormat.subshell_skip(self)
        self.conf_skipped_unformat = ".".join(self.conf_skipped_list)
        for subshell in self.conf_skipped_list:
            self.subshell = subshell
            self.subshell_info = ConfigurationFormat.subshell_format(self)
            if subshell != self.conf_skipped_list[-1]:
                self.skipped_formatted_conf = self.skipped_formatted_conf + self.subshell_info[0] + "\\," + self.subshell_info[1] + "\\;"
            elif subshell == self.conf_skipped_list[-1] and self.subshell_info[2] != "1":
                self.skipped_formatted_conf = self.skipped_formatted_conf + self.subshell_info[0] + "\\," + self.subshell_info[1] + "\\;"
            else:
                self.skipped_formatted_conf = self.skipped_formatted_conf + self.subshell_info[0] + "\\;"
        return self.skipped_formatted_conf, self.conf_skipped_unformat
    
    def ls_coupling_format(self):
        self.conf_ls_format = ""
        self.temp_conf_ls = self.temp_conf_list[-1]
        self.temp_conf_ls_index = self.temp_conf_ls.rfind('_')
        self.temp_conf_ls = self.temp_conf_ls[self.temp_conf_ls_index+1:]
        self.conf_ls_format = f"\\;^{self.temp_conf_ls[0:-1]}\\text{{{self.temp_conf_ls[-1]}}}"
        return self.conf_ls_format

#######################################################################

class LevelsEnergyData:
    '''
    This class is used to read the energy data from the grasp output file and format the data.
    '''
    def __init__(self, data_file_info):
        self.data_file_info = data_file_info
        self.cut_off_subshell =  data_file_info.get("cut_off_subshell", "")
        # self.f_type = "energy"
        self.data_file_info["f_type"] = "energy"
        self.level_parameter = data_file_info.get("level_parameter")
        self.atom = data_file_info.get("atom")
        self.this_as = data_file_info.get("this_as")
        self.level_read_df = pd.DataFrame(columns=['No', 'Pos', 'J', 'Parity', f'Energy_Total_{self.level_parameter}{self.this_as}', f'E_as{self.this_as}', 'Splitting', f'Configuration_{self.level_parameter}{self.this_as}raw'])
        
        self.file_dir = data_file_info.get("file_dir")
        
        self.file_name = f"{self.atom}{self.level_parameter}{self.this_as}"
        
        
    def energy_file2dataframe(self):
        
        self.raw_data_file = GraspFileLoad(self.data_file_info)
        
        self.raw_data2csv = EnergyFile2csv(self.data_file_info)
        
        self.saved_csv_file_path = self.raw_data2csv.energy_file2csv()
        
        self.level_read_df = pd.read_csv(f"{self.saved_csv_file_path}", sep=r'\s+', names=['No', 'Pos', 'J', 'Parity', f'Energy_Total_{self.level_parameter}{self.this_as}', f'E_as{self.this_as}', 'Splitting', f'Configuration_{self.level_parameter}{self.this_as}raw'], dtype=str)
        
        return self.level_read_df
        
    def energy_data_formate(self):
        self.energy_file2dataframe()
        if not self.level_read_df[f'Configuration_{self.level_parameter}{self.this_as}raw'].isnull().all():
            self.level_read_df[f'Configuration_{self.level_parameter}{self.this_as}'] = self.level_read_df[f'Configuration_{self.level_parameter}{self.this_as}raw'].apply(lambda x: ConfigurationFormat(x, self.cut_off_subshell).conf_format()[0])
            self.level_read_df[f'Configuration_LSJ_{self.level_parameter}_as{self.this_as}'] = self.level_read_df[f'Configuration_{self.level_parameter}{self.this_as}raw'].apply(lambda x: ConfigurationFormat(x, self.cut_off_subshell).ls_coupling_format()) + "_{" + self.level_read_df['J'] +"}"
            self.level_read_df[f'ASF_LSJ_as{self.this_as}'] = self.level_read_df[f'Configuration_{self.level_parameter}{self.this_as}'] + self.level_read_df[f'Configuration_LSJ_{self.level_parameter}_as{self.this_as}']
        # self.level_read_df[[f'E_as{self.this_as}', 'Splitting']].fillna(0, inplace=True)
        self.level_read_df[['No', f'Energy_Total_{self.level_parameter}{self.this_as}', f'E_as{self.this_as}']] = self.level_read_df[['No', f'Energy_Total_{self.level_parameter}{self.this_as}', f'E_as{self.this_as}']].apply(pd.to_numeric)
        
        return self.level_read_df

#######################################################################

def mcdhf_energy_data_collection(data_file_info, a_s_list):
    """
    This function is used to merge the energy data from different a_s.
    """
    data_file_info["this_as"] = a_s_list[0]
    energy_data = LevelsEnergyData(data_file_info).energy_data_formate()
    for a_s in a_s_list[1:]:

        data_file_info["file"] = f"{data_file_info['atom']}{data_file_info['level_parameter']}{a_s}"
        data_file_info["this_as"] = a_s
        temp_level_as = LevelsEnergyData(data_file_info).energy_data_formate()
        energy_data = pd.merge(energy_data, temp_level_as, how='outer', on=["Pos", "J", "Parity"], suffixes=("", str(a_s)))
        energy_data[f'dE{a_s}'] = energy_data[f'E_as{a_s}'] - energy_data[f'E_as{a_s-1}']
        energy_data[f'dE{a_s}per'] = energy_data[f'dE{a_s}']/energy_data[f'E_as{a_s-1}']
    energy_data = energy_data.dropna(how = 'all', axis=1)
    energy_data = energy_data.fillna(0)
    return energy_data


def ci_energy_data_collection(energy_data: None, data_file_info: dict):
    """
    This function is used to collect single energy levels data or merge energy levels data from ci calculation.
    """
    if energy_data is None:
        energy_data = LevelsEnergyData(data_file_info).energy_data_formate()
    else:
        temp_ci_level_as = LevelsEnergyData(data_file_info).energy_data_formate()
        energy_data = pd.merge(energy_data, temp_ci_level_as, how='outer', on=["Pos", "J", "Parity"],suffixes=("", str(data_file_info["level_parameter"])+str(data_file_info["this_as"])))
        energy_data = energy_data.dropna(how = 'all', axis=1)
        energy_data = energy_data.fillna(0.0)
        energy_data = energy_data.sort_values(by=f'No{data_file_info["level_parameter"]}{data_file_info["this_as"]}', ascending=True)
    return energy_data

#######################################################################

# Add level's composition of ASF

class LevelsASFComposition:
    def __init__(self, energy_data_df: pd.DataFrame, data_file_info: dict, min_comp: float=0.03):
        self.energy_data_df = energy_data_df
        self.data_file_info = data_file_info
        self.min_comp = min_comp
        self.data_file_info["f_type"] = "lsj_lbl"
        self.data_file_load = GraspFileLoad(self.data_file_info)
        self.lsj_lbl_data, self.level_loc_lbl = self.data_file_load.data_file_process()

    def level_composition_unit_format(self):
        self.cut_off_subshell = self.data_file_info["cut_off_subshell"]
        self.temp_lsj_unit_info_list = self.temp_lsj_unit_information
        # print(self.temp_lsj_unit_info_list)
        self.temp_lsj_unit_coefficient = np.float64(self.temp_lsj_unit_info_list[0])
        self.temp_lsj_unit_w = np.float64(self.temp_lsj_unit_info_list[1]).round(3)
        self.temp_lsj_unit_conf = self.temp_lsj_unit_info_list[2]
        self.temp_lsj_unit_format = ConfigurationFormat(self.temp_lsj_unit_conf, self.cut_off_subshell)
        self.temp_lsj_unit_format_conf = self.temp_lsj_unit_format.conf_format()[0]
        self.temp_lsj_unit_format_conf_ls = self.temp_lsj_unit_format.ls_coupling_format()

        self.temp_comp_unit_format = '$' + str(self.temp_lsj_unit_w) + '\\;' + self.temp_lsj_unit_format_conf + '\\,' + self.temp_lsj_unit_format_conf_ls + '$ + '
        self.temp_comp_unit_format = f'${str(self.temp_lsj_unit_w)}\\;{self.temp_lsj_unit_format_conf}\\,{self.temp_lsj_unit_format_conf_ls}$ +'
        
        return self.temp_comp_unit_format, self.temp_lsj_unit_coefficient, self.temp_lsj_unit_w
        
    # def level_composition_formate(self, self.temp_lsj_information):
    def level_composition_formate(self):
        self.temp_level_asf_comp = ''
        for self.temp_lsj_unit in self.temp_lsj_information:
            self.temp_lsj_unit_information = self.temp_lsj_unit.split()
            if len(self.temp_lsj_unit_information) == 3:
                self.temp_comp_unit_format = LevelsASFComposition.level_composition_unit_format(self)[0]
                self.temp_level_asf_comp = self.temp_level_asf_comp + self.temp_comp_unit_format
            else:
                continue
            self.temp_level_asf_comp = re.sub(r"\$ \+ \$", " + ", self.temp_level_asf_comp)
            self.temp_level_asf_comp = self.temp_level_asf_comp.strip(' +')

        return self.temp_level_asf_comp
        
    
    def asf_comp_locate(self):
        
        self.temp_level_asf_comp_loc = self.temp_level_asf.split()
        self.temp_level_asf_dataframe_loc = self.energy_data_df.loc[(self.energy_data_df['Pos'] == self.temp_level_asf_comp_loc[0]) & (self.energy_data_df['J'] == self.temp_level_asf_comp_loc[1]) & (self.energy_data_df['Parity']== self.temp_level_asf_comp_loc[2])].index[0]
        
        return self.temp_level_asf_dataframe_loc
    
    def level_comp_of_asf(self):

        for self.temp_level_loc in self.level_loc_lbl:
            if self.temp_level_loc != self.level_loc_lbl[-1]:
                self.temp_level_lsj_info = self.lsj_lbl_data[self.temp_level_loc+1: self.level_loc_lbl[self.level_loc_lbl.index(self.temp_level_loc)+1]]
            else:
                self.temp_level_lsj_info = self.lsj_lbl_data[self.temp_level_loc+1:]
            self.temp_level_asf = self.lsj_lbl_data[self.temp_level_loc]
            self.temp_level_df_loc = LevelsASFComposition.asf_comp_locate(self)
            self.temp_lsj_information = self.temp_level_lsj_info
            self.temp_level_asf_comp = LevelsASFComposition.level_composition_formate(self)
            self.energy_data_df.loc[self.temp_level_df_loc, f'Comp_of_asf_{self.data_file_info["level_parameter"]}{self.data_file_info["this_as"]}'] = self.temp_level_asf_comp
        
        return self.energy_data_df
            
    
#######################################################################


def asf_radial_wavefunction_collection(data_file_info: dict) -> pd.DataFrame:
    '''
    read asf radial wavefunction data,
    no matter the format of the data file is binary file or plot file,
    return the data as a DataFrame.
    '''
    data_file_load = GraspFileLoad(data_file_info)
    asf_radial_wavefunction_data = data_file_load.data_file_process()

    return asf_radial_wavefunction_data

#######################################################################


class RadialElectrondensityFunction:
    
    def __init__(self, data_file_info: dict):
        self.data_file_info = data_file_info
        self.data_file_load = GraspFileLoad(self.data_file_info)
        self.radial_electron_density_data = self.data_file_load.data_file_process()
        self.radial_electron_density_data_df = pd.DataFrame(columns=['r', 'D(r)', 'rho(r)'])
        
    def density_data_collection(self):
        block_index = []
        for i in range(len(self.radial_electron_density_data)):
            if re.match(r'(\d)+\s+([0-9,/])+\s+([+,-])', self.radial_electron_density_data[i]):
                print(self.radial_electron_density_data[i])
                block_index.append(i)
        block_index.append(len(self.radial_electron_density_data))
                
        for i in range(len(block_index)-1):
            print(self.radial_electron_density_data[block_index[i]])
            block_group = self.radial_electron_density_data[block_index[i]]
            temp_block = self.radial_electron_density_data[block_index[i]+1:block_index[i+1]]
            temp_block = [i.replace('D', 'e').split() for i in temp_block]
            
            print(temp_block)
            temp_block_pd = pd.DataFrame(temp_block, columns=['r', 'D(r)', 'rho(r)'])
            temp_block_pd["Group"] = block_group
            self.radial_electron_density_data_df = pd.concat([self.radial_electron_density_data_df, temp_block_pd], ignore_index=True)
        
        return self.radial_electron_density_data_df
