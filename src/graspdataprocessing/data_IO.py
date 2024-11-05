#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :dataload.py
@date :2023/03/21 11:36:06
@author :YenochQin (秦毅)
'''
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm
from .tool_function import *
import struct
# import csv
# the class "GraspFileLoad" is used to load the data file
class GraspFileLoad:
    # the initialization function of the class "GraspFileLoad"

    def __init__(self, data_file_info: dict):

        self.atom = data_file_info.get("atom", "")
        # data_file_dir -> data files directory
        # data_file_path -> data file absolute path
        self.file_dir = data_file_info.get("file_dir")
        if not Path(self.file_dir).exists():
            # Handle the non-existent directory appropriately
            raise ValueError("Directory does not exist")
            
        else:
            self.data_file_dir = Path(self.file_dir)
            print(f"load data files from {self.data_file_dir}")

        # Add similar checks and error handling for other keys in data_file_info
        # Use get method to avoid KeyErrors
        self.file_type = data_file_info.get("file_type")  
        self.level_parameter = data_file_info.get("level_parameter")
        self.this_as = data_file_info.get("this_as")
        self.file_name = data_file_info.get("file_name")
        if Path(self.file_name).is_file():
            self.data_file_path = Path(self.file_name)

        elif isinstance(self.file_name, str) and self.file_name:
            self.data_file_path = self.data_file_dir.joinpath(self.file_name)
        else:
            self.data_file_path = None
        self.file_keyword = {
                            # "TRANSITION": f"*{self.level_parameter}*.*{self.level_parameter}*.*t",
                            "TRANSITION": f"*.*.*t",
                             "TRANSITION_LSJ": f"*{self.level_parameter}*.*{self.level_parameter}*.*t.lsj", 
                             "LSJCOMPOSITION": f"*{self.level_parameter}*{self.this_as}.lsj.lbl", 
                             "PLOT": f"*{self.level_parameter}*.plot", 
                             "BINARY_RADIAL_WAVEFUNCTIONS": f"*{self.level_parameter}*.w", 
                             "MIX_COEFFICIENT": f"*{self.level_parameter}*m",
                             "DENSITY": f"*{self.this_as}.cd"}
        # set the data file path as the attribute of the class
        

    # the function "file_read" has two input arguments: the file name and the number of columns
    # the function "file_read" returns the data array
    def file_read(self):
        # open the data file
        # print(f"load data file {self.data_file_path}")
        with open(self.load_file_path, 'r') as temp_load_file:
            # load data file
            self.load_file_data = temp_load_file.readlines()
        self.load_file_data = [line.strip() for line in self.load_file_data]
        print(f"file {self.load_file_path} loaded")
        return self.load_file_data
    
    def files_read(self):
        self.load_files_data = []
        for temp_file_path in self.data_path_list:
            self.load_file_path = temp_file_path
            temp_load_file_data = GraspFileLoad.file_read(self)
            temp_load_file_data.append('')
            self.load_files_data.extend(temp_load_file_data)
        return self.load_files_data
    

    
    def radial_wavefunction_binary_file_read(self):
        self.nn_list = []
        self.laky_list = []
        self.energy_list = []
        self.npts_list = []
        self.a0_list = []
        self.pg_list = []
        self.qg_list = []  
        self.rg_list = []
        
        with open(self.load_file_path, 'rb') as binary_file:
            
            temp_int = binary_file.read(4)
            
            title_bin = binary_file.read(6)                 # read (3) title*6
            title = struct.unpack('6s', title_bin)
            print(title[0])
            temp_int = binary_file.read(4)
            
            while True:
                
                temp_int = binary_file.read(4)  
                if not temp_int:  # temp_int为空则表明已经到达文件末尾
                    print("已经到达文件末尾")
                    break

                # read (3, end=20) nn, laky, energy, npts
                nn_bin = binary_file.read(4)
                nn = struct.unpack('i', nn_bin)
                print(nn)
                self.nn_list.append(nn[0])

                laky_bin = binary_file.read(4)
                laky = struct.unpack('i', laky_bin)
                print(laky)
                self.laky_list.append(laky[0])
        
                energy_bin = binary_file.read(8)
                energy = struct.unpack('d', energy_bin)
                print(energy)
                self.energy_list.append(energy[0])

                npts_bin = binary_file.read(4)
                npts = struct.unpack('i', npts_bin)
                print(npts)
                self.npts_list.append(npts[0])
                
                
                temp_int = binary_file.read(4)
                temp_int = binary_file.read(4)      # read (3) a0, (pg(j,i), j=1, npts), (qg(j,i), j=1, npts)
                
                
                a0_bin = binary_file.read(8)
                a0 = struct.unpack('d', a0_bin)
                self.a0_list.append(a0[0])
                print(a0)

                pg_bin = binary_file.read(8*npts[0])
                pg = struct.unpack('d'*npts[0], pg_bin)
                pg_array = np.array(pg)
                self.pg_list.append(pg_array)

                qg_bin = binary_file.read(8*npts[0])
                qg = struct.unpack('d'*npts[0], qg_bin)
                qg_array = np.array(qg)
                self.qg_list.append(qg_array)
                
                temp_int = binary_file.read(4)
                temp_int = binary_file.read(4)      # read (3) (rg(j,i), j=1, npts)

                rg_bin = binary_file.read(8*npts[0])
                rg = struct.unpack('d'*npts[0], rg_bin)
                rg_array = np.array(rg)
                print(rg[0])
                self.rg_list.append(rg_array)
                
                temp_int = binary_file.read(4)
                # temp_int = binary_file.read(4)
                
                
        return self.nn_list, self.laky_list, self.energy_list, self.npts_list, self.a0_list, self.pg_list, self.qg_list, self.rg_list
    

    def mix_coefficient_file_read(self):

        self.index_block_list = []
        self.ncfblk_list = []
        self.block_energy_count_list = []
        self.j_value_location_list = []
        self.parity_list = []
        self.ivec_list = []
        self.block_energy_list = []
        self.block_level_energy_list = []
        self.mix_coefficient_list = []

        with open(self.load_file_path, 'rb') as binary_file:


            g92mix = read_fortran_record(binary_file, dtype=np.dtype('S6')).tobytes().decode('utf-8').strip()

            print(f"g92mix: {g92mix}")  # Debugging print
            
            if g92mix != 'G92MIX':
                raise ValueError('Not a mixing coefficient file!')

            # READ (nfmix) nelec, ncftot, nw, nvectot, nvecsiz, nblock
            header_data = read_fortran_record(binary_file, dtype=np.int32, count=6)
            # nelec -> num_electron, ncftot -> total_num_configuration, nw -> NW, ncmin -> ncmin, nvecsiz -> nvecsiz, nblock -> num_block
            self.num_electron, self.total_num_configuration, self.NW, self.ncmin, self.nvecsiz, self.num_block = header_data

            print("title=", g92mix)
            print(f" nblock = {self.num_block},       ncftot =   {self.total_num_configuration},          nw =  {self.NW},            nelec =   {self.num_electron}")
            for jblock in tqdm(range(1, self.num_block+1)):
                print('cycle jblock =',jblock)
                
                # Read block data: nb, ncfblk, nevblk, iatjp, iaspa
                block_data = read_fortran_record(binary_file, dtype=np.int32, count=5)
                nb, ncfblk, nevblk, j_value_location, parity = block_data
                print(f' Block no. = {nb}, 2J+1 = {j_value_location}, Parity = {parity}, No. of eigenvalues = {nevblk}, No. of CSFs = {ncfblk}')
                # nb -> index_block, ncfblk -> ncfblk, nevblk -> block_energy_count, j_value_location -> iatjp, parity -> iaspa
                self.index_block_list.append(nb)
                self.ncfblk_list.append(ncfblk)
                self.block_energy_count_list.append(nevblk)
                self.j_value_location_list.append(j_value_location)
                self.parity_list.append(parity)
                if jblock != nb:
                    raise ValueError('jblock != nb')

                ivec = read_fortran_record(binary_file, dtype=np.int32, count=nevblk)

                ivec_array = np.array(ivec)
                self.ivec_list.append(ivec_array)

                
                # READ (3) eav, (eval(i+ncountState), i = 1, nevblk)
                eva_evals = read_fortran_record(binary_file, dtype=np.float64, count=nevblk+1)
                eva = eva_evals[0]
                evals = eva_evals[1:]

                self.block_energy_list.append(eva)
                self.block_level_energy_list.append(evals)

                
                # READ (3) (evec, i = 1, ncfblk*nevblk)
                evecsblock = read_fortran_record(binary_file, dtype=np.float64, count=nevblk * ncfblk)
                evecs = evecsblock.reshape(nevblk, ncfblk)

                self.mix_coefficient_list.append(evecs)

                # temp_int = binary_file.read(4)  

        return self.index_block_list, self.ncfblk_list, self.block_energy_count_list, self.j_value_location_list, self.parity_list, self.ivec_list, self.block_energy_list, self.block_level_energy_list, self.mix_coefficient_list



    def grasp_data_file_location(self):
        if self.data_file_dir.rglob(f"{self.file_keyword[self.file_type]}"):
            print(f"{self.file_keyword[self.file_type]} data file is found")
            self.temp_path_list = list(Path(self.data_file_dir).rglob(f"{self.file_keyword[self.file_type]}"))
            self.raw_file_path = self.temp_path_list[0].parent
            
        self.grasp_data_file_path_list = list(self.raw_file_path.glob(f"{self.file_keyword[self.file_type]}"))

        return self.grasp_data_file_path_list
    

    def data_file_process(self):
        '''
        The data_file_process method in the GraspFileLoad class is designed to identify the data type of the input file and process it accordingly. 
        '''
        
        # set the data type as the attribute of the class
        if "ENERGY" in self.file_type.upper() or "LEVEL" in self.file_type.upper():
            # if self.file_name:
            #     self.load_file_path = Path(self.raw_data_file_dir).joinpath(self.file_name)
            # else:
            #     self.file_name = f"{self.atom}{self.level_parameter}{self.this_as}"
            #     self.load_file_path = Path(self.raw_data_file_dir).joinpath(self.file_name)
            if self.data_file_path:
                self.load_file_path = self.data_file_path
            else:
                self.file_name = f"{self.atom}{self.level_parameter}{self.this_as}"
                self.load_file_path = Path(self.data_file_dir).joinpath(self.file_name)
            # set "ENERGY" or "LEVEL" as the key of "file_type"
            self.file_type = "ENERGY"
            self.level_data = []
            self.level_data = GraspFileLoad.file_read(self)
            print("data file type: level data")
            return self.level_data
        
        elif "LSJ" in self.file_type.upper() and "TRAN" not in self.file_type.upper():
            self.file_type = "LSJCOMPOSITION"
            self.lsj_lbl_data = []
            self.level_loc_lbl = []
            self.file_name = ""
            self.data_path_list = GraspFileLoad.grasp_data_file_location(self)
            print(self.data_path_list)
            self.lsj_lbl_data = GraspFileLoad.files_read(self)
            
            for self.index_num in range(0,len(self.lsj_lbl_data)):
                # print(lsj_lbl[line])
                if re.search(r'\d+.\d+%', self.lsj_lbl_data[self.index_num]) :
                    self.level_loc_lbl.append(self.index_num)
            print("data file type: level jj2lsj data")
            return self.lsj_lbl_data, self.level_loc_lbl
        
        elif "TRANS" in self.file_type.upper() and "LSJ" not in self.file_type.upper():
            self.file_type = "TRANSITION"
            self.transition_data = []
            self.data_path_list = GraspFileLoad.grasp_data_file_location(self)
            self.transition_data = GraspFileLoad.files_read(self)
            print("data file type: transition_data")
            self.transition_data.append('')
            return self.transition_data
        
        elif "TRANS" in self.file_type.upper() and "LSJ" in self.file_type.upper():
            self.file_type = "TRANSITION_LSJ"
            self.transition_data = []
            self.data_path_list = GraspFileLoad.grasp_data_file_location(self)
            self.transition_lsj_data = GraspFileLoad.files_read(self)
            print("data file type: transition_LSJ_data")
            return self.transition_lsj_data
        
        elif "PLOT" in self.file_type.upper():
            self.file_type = "PLOT"
            temp_plot_data = []
            
            if self.data_file_path:
                self.load_file_path = self.data_file_path
            else:
                self.file_name = f"{self.atom}{self.level_parameter}{self.this_as}.plot"
                self.load_file_path = Path(self.data_file_dir).joinpath(self.file_name)
            temp_plot_data = GraspFileLoad.file_read(self)
            print("data file type: plot_data")
            self.plot_data = [line.split() for line in temp_plot_data[1:]]
            self.radial_wavefunction_data = pd.DataFrame(self.plot_data, columns=temp_plot_data[0].split())

            return self.radial_wavefunction_data
        
        elif "WAVEFUNCTION" in self.file_type.upper():
            self.file_type = "BINARY_RADIAL_WAVEFUNCTIONS"

            self.radial_wavefunction_data = pd.DataFrame({})

            if self.data_file_path:
                self.load_file_path = self.data_file_path
            else:
                self.file_name = f"{self.atom}{self.level_parameter}{self.this_as}.w"
                self.load_file_path = Path(self.data_file_dir).joinpath(self.file_name)

            GraspFileLoad.radial_wavefunction_binary_file_read(self)
            print("data file type: radial_wavefunction_data")
            rg_list_len = [len(self.rg_list[i]) for i in range(len(self.rg_list))]
            self.max_rg_index = rg_list_len.index(max(rg_list_len))
            self.radial_wavefunction_data[f'r(a.u)'] = self.rg_list[self.max_rg_index]
            pg_aligned_list = align_2d_list_columns(self.pg_list)
            qg_aligned_list = align_2d_list_columns(self.qg_list)
            
            for n in range(len(self.nn_list)):
                str_nl = int_nl_2_str_nl(self.nn_list[n], self.laky_list[n])
                self.radial_wavefunction_data[f'P({str_nl})'] = pg_aligned_list[n]
                self.radial_wavefunction_data[f'Q({str_nl})'] = qg_aligned_list[n]
                
            return self.radial_wavefunction_data
        
        elif "MIX" in self.file_type.upper() or "COEF" in self.file_type.upper():
            self.file_type = "MIX_COEFFICIENT"
            self.mix_coefficient_dict = {}

            if self.data_file_path:
                self.load_file_path = self.data_file_path
            else:
                self.file_name = f"{self.atom}{self.level_parameter}{self.this_as}*m"
                self.load_file_path = Path(self.data_file_dir).joinpath(self.file_name)

            GraspFileLoad.mix_coefficient_file_read(self)
            print("data file type: mix_coefficient_data")
            
            level_print_title()
            temp_pos = []
            temp_J = []
            temp_parity = []
            temp_energy = []
            for jblock in range(self.num_block):
                for pos in self.ivec_list[jblock-1]:
                    temp_pos.append(pos)
                    temp_J.append(level_J_value(self.j_value_location_list[jblock-1]))
                    temp_parity.append(level_parity(self.parity_list[jblock-1]))
                    temp_energy.append(self.block_energy_list[jblock-1]+self.block_level_energy_list[jblock-1][pos-1])
                    
            
            level_index = np.argsort(temp_energy)
            
            for i in range(len(level_index)):
                if i == 0:
                    print(f"{i+1:3}{temp_pos[level_index[i]]:3}{temp_J[level_index[i]]:>4}   {temp_parity[level_index[i]]:1}    {temp_energy[level_index[i]]:14.7f}{0.0000000:12.2f}")
                else:
                    print(f"{i+1:3}{temp_pos[level_index[i]]:3}{temp_J[level_index[i]]:>4}   {temp_parity[level_index[i]]:1}    {temp_energy[level_index[i]]:14.7f}{energy_au_cm(temp_energy[level_index[i]]-temp_energy[level_index[0]]):12.2f}")
                
            return self.index_block_list, self.ncfblk_list, self.block_energy_count_list, self.j_value_location_list, self.parity_list, self.ivec_list, self.block_energy_list, self.block_level_energy_list, self.mix_coefficient_list
        
        elif "DENSITY" in self.file_type.upper():
            self.file_type = "DENSITY"
            temp_density_data = []

            self.radial_wavefunction_data = pd.DataFrame({})

            if self.data_file_path:
                self.load_file_path = self.data_file_path
            else:
                self.file_name = f"*{self.this_as}.cd"
                self.load_file_path = Path(self.data_file_dir).joinpath(self.file_name)
                
            temp_density_data = GraspFileLoad.file_read(self)
            
            return temp_density_data

        else:
            return 0



#######################################################################
class EnergyFile2csv():

    def __init__(self, data_file_info: dict):
        self.data_file_info = data_file_info
        self.atom = data_file_info.get("atom")
        self.file_dir = data_file_info.get("file_dir")
        if not Path(self.file_dir).exists():
            # Handle the non-existent directory appropriately
            raise ValueError("Directory does not exist")
            
        else:
            self.data_file_dir = Path(self.file_dir)

        self.file_name = data_file_info.get("file_name")
        if Path(self.file_name).is_file():
            self.data_file_path = Path(self.file_name)

        elif isinstance(self.file_name, str) and self.file_name:
            self.data_file_path = self.data_file_dir.joinpath(self.file_name)
        else:
            self.data_file_path = None
        self.file_type = data_file_info.get("file_type")
        self.level_parameter = data_file_info.get("level_parameter")
        self.this_as = data_file_info.get("this_as")
        # self.load_file_path = Path(self.raw_data_file_dir, self.file)
        # self.store_file_path = Path(self.raw_data_file_dir)
        self.store_file_path = Path(self.data_file_dir).joinpath("csv_file")
        if not self.store_file_path.exists():
            self.store_file_path.mkdir()

    def energy_file2csv(self):
        # load the data file
        self.load_level_data = GraspFileLoad.data_file_process(self)
        for i in range(len(self.load_level_data)):
            if 'No Pos  J ' in self.load_level_data[i]:
                self.skip_line = i + 3
                break
        
        self.temp_level_data = self.load_level_data[self.skip_line:]
        self.save_level_data = []
        for line in self.temp_level_data:
            if '-----' in line:
                break
            else:
                self.save_level_data.append(line)
        # set the data file path
        self.saved_csv_file_path = self.store_file_path.joinpath(f"{self.file_name}_level.csv")
        # open the csv file
        with open(self.saved_csv_file_path, 'w', newline='') as csv_file:
            # write the data file into the csv file
            # writer = csv.writer(csv_file)
            for item in self.save_level_data:
                # writer.writerow(item)
                csv_file.write(item+'\n')
        print(f"energy levels data has been written into {self.store_file_path}\\{self.file_name}_level.csv csv file")
        return self.saved_csv_file_path



