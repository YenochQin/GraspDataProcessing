#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :transition_data_collection.py
@date :2024/01/15 15:15:57
@author :YenochQin (秦毅)

@version 1.0: Object Oriented Programming modified from transition_data2csv.py
'''


import numpy as np
import pandas as pd
import re
from tqdm import tqdm

from .data_IO import GraspFileLoad
from .tool_function import transition_dT_cal, doubleJ_to_J

class TransitionDataCollection:
    def __init__(self, data_file_info: dict):
        self.data_file_info = data_file_info
        self.file_type = data_file_info.get('file_type')

        self.transition_data_list = GraspFileLoad(self.data_file_info).data_file_process()

    def transition_type_check(self, transition_type_line):
        transition_type_match = re.match(r'([A-Za-z]*) 2\*\*\( ([0-9])\)-pole transitions', transition_type_line)

        if not transition_type_match:
            raise ValueError("Transition type line not detected.")
        else:
            return transition_type_match.group(1)[0]+transition_type_match.group(2)

    def trans_data_line_location(self):
        self.trans_data_line_index = []
        transition_type_line_index = [i for i in range(len(self.transition_data_list)) if re.match(r'([A-Za-z]*) 2\*\*\( ([0-9])\)-pole transitions', self.transition_data_list[i])]
        print(transition_type_line_index)

        self.transition_data_type_line_index = [line 
                                                  for line in transition_type_line_index
                                                    if re.match(r"f.*f.*[M,C,B]", self.transition_data_list[line+5])]

        for i in self.transition_data_type_line_index:
            if i == self.transition_data_type_line_index[-1] and i == transition_type_line_index[-1]:
                self.trans_data_line_index.append(i+5)
                self.trans_data_line_index.append(len(self.transition_data_list)-2)
            else:
                temp_index = transition_type_line_index.index(i)
                self.trans_data_line_index.append(i+5)
                self.trans_data_line_index.append(transition_type_line_index[temp_index+1]-2)



        return self.trans_data_line_index, self.transition_data_type_line_index
    
    def transition_data2dataframe(self):

        self.transition_data_pd = pd.DataFrame()
        columns_orders = [
                        'Upper_file',
                        'Upper_loc',
                        'Upper_J',
                        'Upper_parity',
                        'Lower_file',
                        'Lower_loc',
                        'Lower_J',
                        'transition_type',
                        'Lower_parity',
                        'energy_level_difference',
                        'wavelength_vac',
                        'transition_rate_C',
                        'oscillator_strength_C',
                        'line_strength_C',
                        'transition_rate_B',
                        'oscillator_strength_B',
                        'line_strength_B',
                        'transition_rate_M',
                        'oscillator_strength_M',
                        'line_strength_M',
                        'transition_dT'
                            ]

        self.trans_data_line_location()
        for i in tqdm(range(len(self.transition_data_type_line_index))):
            temp_transition_type = TransitionDataCollection.transition_type_check(self, self.transition_data_list[self.transition_data_type_line_index[i]])
            if "E" in temp_transition_type:
                trans_data_block_step = 2
            elif "M" in temp_transition_type:
                trans_data_block_step = 1
            
            for line in tqdm(range(self.trans_data_line_index[2*i], self.trans_data_line_index[2*i+1], trans_data_block_step), leave=False):
                if self.transition_data_list[line] == '':
                    break
                temp_transition_data_dict = TransitionDataBlock(self.transition_data_list[line:line+trans_data_block_step], temp_transition_type).transition_data_block2dict()
                # print(temp_transition_data_dict)

                if temp_transition_data_dict.get('line_strength_C') == 0 or temp_transition_data_dict.get('line_strength_B') == 0 or temp_transition_data_dict.get('line_strength_M') == 0:
                    continue
                temp_transition_data_df = pd.DataFrame([temp_transition_data_dict])
                self.transition_data_pd = pd.concat([self.transition_data_pd, temp_transition_data_df], ignore_index=True, join='outer')
            
        self.transition_data_pd = self.transition_data_pd.reindex(columns=columns_orders)
        
        self.transition_data_pd['transistion_type'] = pd.Categorical(self.transition_data_pd['transition_type'], categories=['E1', 'E2', 'E3', 'M1', 'M2'], ordered=True)
        
        self.transition_data_pd.sort_values('transition_type', inplace=True)
        
        self.transition_data_pd['transition_type'] = self.transition_data_pd['transition_type'].astype(str)
        temp_E_transition_data_pd = self.transition_data_pd[self.transition_data_pd['transition_type'].str.contains('E')]
        
        for index, row in temp_E_transition_data_pd.iterrows():
            # print(row)
            rate_C = row['transition_rate_C']
            rate_B = row['transition_rate_B']
            transition_dT = transition_dT_cal(rate_C, rate_B)
            self.transition_data_pd.at[index, 'transition_dT'] = transition_dT

        return self.transition_data_pd

class LSJTransitionDataCollection:
    def __init__(self, data_file_info: dict, debug=False):
        self.data_file_info = data_file_info
        self.file_type = data_file_info.get('file_type')
        self.transition_data_list = GraspFileLoad(self.data_file_info).data_file_process()
        self.debug = debug

    def transition_block_index(self):
        self.transition_block_index_list = []
        temp_data_index = []
        for i in range(len(self.transition_data_list)):
            
            if self.transition_data_list[i].strip() and re.match(r'[0-9]*-[0-9]*\.[0-9]*.*', self.transition_data_list[i].strip()):
                if self.debug:
                    print(self.transition_data_list[i])
                temp_data_index.append(i)


        temp_data_index_len = len(temp_data_index)
        for i in range(0,temp_data_index_len,2):
            # print(temp_data_index[i])

            if i + 2 < temp_data_index_len and temp_data_index[i+2]-temp_data_index[i]<= 7:
                self.transition_block_index_list.append(temp_data_index[i])
                temp_block = self.transition_data_list[temp_data_index[i]:temp_data_index[i+2]]
                data_block = [x for x in temp_block if x.strip() and x!= "Transition between files:"]
                self.transition_block_index_list.append(temp_data_index[i] + len(data_block))
            else:
                self.transition_block_index_list.append(temp_data_index[i])
                temp_block = self.transition_data_list[temp_data_index[i]:temp_data_index[i]+5]
                data_block = [x for x in temp_block if x.strip() and x!= "Transition between files:"]
                self.transition_block_index_list.append(temp_data_index[i] + len(data_block))

        return self.transition_block_index_list

    
    def transition_data2dataframe(self):

        self.transition_data_pd = pd.DataFrame()
        columns_orders = [
                        'Upper_J',
                        'Upper_energy',
                        'Upper_configuration',
                        'Lower_J',
                        'Lower_energy',
                        'Lower_configuration',
                        'transition_type',
                        'energy_level_difference',
                        'wavelength_vac',
                        'wavelength_air',
                        'line_strength_B',
                        'oscillator_strength_B',
                        'transition_rate_B',
                        'line_strength_C',
                        'oscillator_strength_C',
                        'transition_rate_C',
                        'line_strength_M',
                        'oscillator_strength_M',
                        'transition_rate_M',
                        'transition_dT'
                        ]
        self.transition_block_index()
        for i in tqdm(range(0,len(self.transition_block_index_list),2)):
            temp_transition_data_block = self.transition_data_list[self.transition_block_index_list[i]:self.transition_block_index_list[i+1]]
            temp_transition_data_dict = LSJTransitionDataBlock(temp_transition_data_block).transition_data_block2dict()
            if temp_transition_data_dict.get('line_strength_C') == 0 or temp_transition_data_dict.get('line_strength_B') == 0 or temp_transition_data_dict.get('line_strength_M') == 0:
                    continue
            temp_transition_data_df = pd.DataFrame([temp_transition_data_dict])

            self.transition_data_pd = pd.concat([self.transition_data_pd, temp_transition_data_df], ignore_index=True, join='outer')
        self.transition_data_pd = self.transition_data_pd.reindex(columns=columns_orders)
        
        self.transition_data_pd['transistion_type'] = pd.Categorical(self.transition_data_pd['transition_type'], categories=['E1', 'E2', 'E3', 'M1', 'M2'], ordered=True)
        
        self.transition_data_pd.sort_values('transition_type', inplace=True)
        
        return self.transition_data_pd


class LSJTransitionDataBlock:
    '''
    Transition Data block in LSJ form like below:

    electronic *-pole transitions:
    2J&LevelEnergy  Configuration(Upper)
    2J&LevelEnergy  Configuration(Lower)
    energy_level_difference CM-1        wavelength ANGS(VAC)        wavelength ANGS(AIR)
    transition_type  line_strength_B =  0.00000D+00   oscillator_strength_B =  0.00000D+00   transition_rate_B =  0.00000D+00
              line_strength_C   oscillator_strength_C   transition_rate_C


    magnetic *-pole transitions:
    2J&LevelEnergy  Configuration(Upper)
    2J&LevelEnergy  Configuration(Lower)
    energy_level_difference CM-1        wavelength ANGS(VAC)        wavelength ANGS(AIR)
    transition_type  line_strength =  0.00000D+00   oscillator_strength =  0.00000D+00   transition_rate =  0.00000D+00
    '''
    def __init__(self, data_block, debug=False):
        self.data_block = data_block
        self.debug = debug
        if self.debug:
            print(self.data_block)
        self.block_trasnsition_data_dict = {}
        
    def get_transition_level_info(self, level_line):
        level_info = re.split(r'\s+', level_line)
        level_configuration = level_info[1]
        j_energy = level_info[0].split('-')
        double_j = j_energy[0]
        # level_energy_str = np.float64('-' + j_energy[1])
        level_energy = np.float64('-' + j_energy[1])
        # if int(level_energy_str[-1]) >5:
        #     level_energy = np.float64(level_energy_str).round(7)
        
        # elif int(level_energy_str[-1]) == 5 and int(level_energy_str[-2]) == 0:
        #     level_energy = np.float64(level_energy_str).round(7) - 0.0000001
        # else:
        #     level_energy = np.float64(level_energy_str[:-1])

        return [double_j, level_energy, level_configuration]
    
    def get_transition_wavelength(self, wavelength_data_line):
        wavelength_info = re.split(r'\s+', wavelength_data_line)
        self.energy_level = np.float64(wavelength_info[0])
        self.wavelength_vac = np.float64(wavelength_info[2])
        self.wavelength_air = np.float64(wavelength_info[4])

        return self.energy_level, self.wavelength_vac, self.wavelength_air
    
    def get_transition_properties(self, properties_lines):
        self.block_trasnsition_properties_dict = {}
        properties_line_1 = properties_lines[0]
        properties_info_1 = re.split(r'\s+', properties_line_1)
        properties_info_1 = [i.replace('D', 'e') for i in properties_info_1]
        # print(properties_info_1)
        if 'E' in properties_info_1[0]:
            self.block_trasnsition_properties_dict["transition_type"] = properties_info_1[0]

            self.block_trasnsition_properties_dict["line_strength_B"] = np.float64(properties_info_1[3])
            self.block_trasnsition_properties_dict["oscillator_strength_B"] = np.float64(properties_info_1[6])
            self.block_trasnsition_properties_dict["transition_rate_B"] = np.float64(properties_info_1[9])
            self.block_trasnsition_properties_dict["transition_dT"] = np.float64(properties_info_1[12])
            properties_line_2 = properties_lines[1]
            properties_info_2 = re.split(r'\s+', properties_line_2)
            properties_info_2 = [i.replace('D', 'e') for i in properties_info_2]
            self.block_trasnsition_properties_dict["line_strength_C"] = np.float64(properties_info_2[0])
            self.block_trasnsition_properties_dict["oscillator_strength_C"] = np.float64(properties_info_2[1])
            self.block_trasnsition_properties_dict["transition_rate_C"] = np.float64(properties_info_2[2])
            
        elif 'M' in properties_info_1[0]:
            self.block_trasnsition_properties_dict["transition_type"] = properties_info_1[0]
            self.block_trasnsition_properties_dict["line_strength_M"] = np.float64(properties_info_1[3])
            self.block_trasnsition_properties_dict["oscillator_strength_M"] = np.float64(properties_info_1[6])
            self.block_trasnsition_properties_dict["transition_rate_M"] = np.float64(properties_info_1[9])
            
        else:
            return None

        return self.block_trasnsition_properties_dict
    
    def transition_data_block2dict(self):
        upper_level_info = self.data_block[0]

        lower_level_info = self.data_block[1]
        energy_level_difference_info = self.data_block[2]
        properties_info = self.data_block[3:]

        upper_level = self.get_transition_level_info(upper_level_info)
        upper_double_j = upper_level[0]
        self.block_trasnsition_data_dict["Upper_J"] = doubleJ_to_J(int(upper_double_j))
        self.block_trasnsition_data_dict["Upper_energy"] = upper_level[1]
        self.block_trasnsition_data_dict["Upper_configuration"] = upper_level[2]

        lower_level = self.get_transition_level_info(lower_level_info)
        lower_double_j = lower_level[0]
        self.block_trasnsition_data_dict["Lower_J"] = doubleJ_to_J(int(lower_double_j))
        self.block_trasnsition_data_dict["Lower_energy"] = lower_level[1]
        self.block_trasnsition_data_dict["Lower_configuration"] = lower_level[2]
        
        self.block_trasnsition_data_dict["energy_level_difference"] = self.get_transition_wavelength(energy_level_difference_info)[0]
        self.block_trasnsition_data_dict["wavelength_vac"] = self.get_transition_wavelength(energy_level_difference_info)[1]
        self.block_trasnsition_data_dict["wavelength_air"] = self.get_transition_wavelength(energy_level_difference_info)[2]

        self.get_transition_properties(properties_info)

        self.block_trasnsition_data_dict.update(self.block_trasnsition_properties_dict)

        return self.block_trasnsition_data_dict
    

class TransitionDataBlock:
    '''
    Transition Data block in regular form like below:

    'Upper       Lower',
    'Lev  J P   Lev  J P       E (Kays)         A (s-1)          gf            S',
    'f2  1  1/2 +  f1  1  1/2 -      energy_level_difference C  transition_rate_C  oscillator_strength_C  line_strength_C',
    'B  transition_rate_B  oscillator_strength_B  line_strength_B',

    'f2  1  3/2 +  f1  1  1/2 -      energy_level_difference M  transition_rate_M  oscillator_strength_M  line_strength_M'
    '''
    def __init__(self, data_block, transition_type, debug=False):
        '''
        Note this class need transition_type to distinguish between Electronic and Magnetic *-poles transition
        '''
        self.data_block = data_block
        self.debug = debug
        if self.debug:
            print(self.data_block)
        self.data_block_list = [i.replace('D', 'e').split() for i in self.data_block]
        
        self.block_trasnsition_data_dict = {}
        self.block_trasnsition_data_dict["transition_type"] = transition_type

    def get_transition_level_info(self):
        level_info = self.data_block_list[0][:8]
        self.block_trasnsition_data_dict["Upper_file"] = level_info[0]
        self.block_trasnsition_data_dict["Upper_loc"] = level_info[1]
        self.block_trasnsition_data_dict["Upper_J"] = level_info[2]
        self.block_trasnsition_data_dict["Upper_parity"] = level_info[3]

        self.block_trasnsition_data_dict["Lower_file"] = level_info[4]
        self.block_trasnsition_data_dict["Lower_loc"] = level_info[5]
        self.block_trasnsition_data_dict["Lower_J"] = level_info[6]
        self.block_trasnsition_data_dict["Lower_parity"] = level_info[7]

        self.block_trasnsition_data_dict["energy_level_difference"] = round(np.float64(self.data_block_list[0][8]), 7)

        wavelength_vac = 10 ** 8 / self.block_trasnsition_data_dict["energy_level_difference"]

        self.block_trasnsition_data_dict["wavelength_vac"] = wavelength_vac

        return self.block_trasnsition_data_dict
    
    def get_transition_properties(self):
        self.block_trasnsition_properties_dict = {}

        if "C" in self.data_block_list[0]:

            self.block_trasnsition_properties_dict["transition_rate_C"] = np.float64(self.data_block_list[0][10])
            self.block_trasnsition_properties_dict["oscillator_strength_C"] = np.float64(self.data_block_list[0][11])
            self.block_trasnsition_properties_dict["line_strength_C"] = np.float64(self.data_block_list[0][12])

            if "B" in self.data_block_list[1]:

                self.block_trasnsition_properties_dict["transition_rate_B"] = np.float64(self.data_block_list[1][1])
                self.block_trasnsition_properties_dict["oscillator_strength_B"] = np.float64(self.data_block_list[1][2])
                self.block_trasnsition_properties_dict["line_strength_B"] = np.float64(self.data_block_list[1][3])

        if "M" in self.data_block_list[0]:

            self.block_trasnsition_properties_dict["transition_rate_M"] = np.float64(self.data_block_list[0][10])
            self.block_trasnsition_properties_dict["oscillator_strength_M"] = np.float64(self.data_block_list[0][11])
            self.block_trasnsition_properties_dict["line_strength_M"] = np.float64(self.data_block_list[0][12])
        
        return self.block_trasnsition_properties_dict
    
    def transition_data_block2dict(self):
        self.get_transition_level_info()

        self.get_transition_properties()

        self.block_trasnsition_data_dict.update(self.block_trasnsition_properties_dict)

        return self.block_trasnsition_data_dict




def data_process(transition_df, level_df, data_file_info, Branching_Fraction=0.0001):
    """[summary]: Find the index of lower and upper levels in level_DataFrame. Add the results into transition_df and sort the Lower_index from smallest to biggest, so do the Upper_index.

    Args:
        transition_df ([type: DataFrame]): [description] : columns_orders = ['transition_type',
                                                        'Upper_file',
                                                        'Upper_loc',
                                                        'Upper_J',
                                                        'Upper_parity',
                                                        'Lower_file',
                                                        'Lower_loc',
                                                        'Lower_J',
                                                        'Lower_parity',
                                                        'energy_level_difference',
                                                        'wavelength_vac',
                                                        'transition_rate_C',
                                                        'oscillator_strength_C',
                                                        'line_strength_C',
                                                        'transition_rate_B',
                                                        'oscillator_strength_B',
                                                        'line_strength_B',
                                                        'transition_rate_M',
                                                        'oscillator_strength_M',
                                                        'line_strength_M']

    Returns:
        [type : DataFrame]: [description: return transition_df]
    """
    data_parameter = data_file_info.get('level_parameter')
    file_type = data_file_info.get('file_type')
    a_s = data_file_info.get('this_as')
    
    if file_type.upper() == 'TRANSITION':
        
        level_index_dict = {(row['Pos'], row['J'], row['Parity']): row[f'No{data_parameter}{a_s}'] for index, row in level_df.iterrows()}
        transition_df['Upper_index'] = transition_df.apply(lambda row: level_index_dict.get((row['Upper_loc'], row['Upper_J'], row['Upper_parity'])), axis=1)
        transition_df['Lower_index'] = transition_df.apply(lambda row: level_index_dict.get((row['Lower_loc'], row['Lower_J'], row['Lower_parity'])), axis=1)

    elif file_type.upper() == 'TRANSITION_LSJ':
        
        level_index_dict = {(row['J'], row[f'Energy_Total_{data_parameter}{a_s}'], row[f'Configuration_{data_parameter}{a_s}raw']): row[f'No{data_parameter}{a_s}'] for index, row in level_df.iterrows()}
        transition_df['Upper_index'] = transition_df.apply(lambda row: level_index_dict.get((row['Upper_J'], row['Upper_configuration'], row['Upper_energy'])), axis=1)
        transition_df['Lower_index'] = transition_df.apply(lambda row: level_index_dict.get((row['Lower_J'], row['Lower_configuration'], row['Lower_energy'])), axis=1)

    else:
        raise ValueError("file_type doesn't match, it should be 'TRANSITION' or 'TRANSITION_LSJ'")

    transition_df.sort_values(by=['Lower_index', 'Upper_index'], ascending=True, inplace=True)

    transition_df.A_B_to_A_C = transition_df.transition_rate_B / transition_df.transition_rate_C

    level_df['lifetime_B'] = np.float64(0)
    level_df['lifetime_C'] = np.float64(0)
    
    transition_df['sum_A_B'] = np.float64(0)
    transition_df['sum_A_C'] = np.float64(0)

    for lno in tqdm(range(2, len(level_df)+1, 1)):
        temp_sum_A_B = transition_df.loc[(transition_df['Upper_index'] == lno), 'transition_rate_B'].sum()
        temp_sum_A_C = transition_df.loc[(transition_df['Upper_index'] == lno), 'transition_rate_C'].sum()
        # print(temp_sum_A_B,temp_sum_A_C)
        if temp_sum_A_B != 0:
            transition_df.loc[(transition_df['Upper_index'] == lno), 'sum_A_B'] = temp_sum_A_B
            temp_lifetime_B = 1 / temp_sum_A_B
            level_df.loc[lno-1, 'lifetime_B'] = temp_lifetime_B
        else:
            continue
        if temp_sum_A_C != 0:
            transition_df.loc[(transition_df['Upper_index'] == lno), 'sum_A_C'] = temp_sum_A_C
            temp_lifetime_C = 1 / temp_sum_A_C
            level_df.loc[lno-1, 'lifetime_C'] = temp_lifetime_C
        else:
            continue
    transition_df['branching_fraction'] = np.float64(0)
    for trannum in range(len(transition_df)):
        transition_df.loc[trannum, 'branching_fraction'] = transition_df.loc[trannum, 'transition_rate_C']/transition_df.loc[trannum, 'sum_A_B']
        if transition_df.loc[trannum, 'branching_fraction'] <= Branching_Fraction:
            transition_df.drop(trannum, axis=0, inplace=True)
    
    return transition_df, level_df