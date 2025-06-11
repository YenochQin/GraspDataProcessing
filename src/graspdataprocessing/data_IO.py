#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :dataload.py
@date :2023/03/21 11:36:06
@author :YenochQin (秦毅)
'''

import re
import csv 

from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from types import SimpleNamespace
from dataclasses import dataclass
import gzip
import pickle
import tomllib

import numpy as np
import pandas as pd
import struct
from tqdm import tqdm

from .tool_function import *
from .CSFs_compress_extract import *
# from .CSFs_choosing import *
from .data_modules import *


class GraspFileLoad:
    # the initialization function of the class "GraspFileLoad"
    @classmethod
    def from_filepath(cls, filepath, file_type=None):
        """从文件路径直接创建实例的类方法"""
        file_dir = str(Path(filepath).parent)
        file_name = Path(filepath).name
        config = {
            "atom": "",
            "file_dir": file_dir,
            "file_name": file_name,
            "level_parameter": "",
            "this_as": 0,
            "file_type": file_type
        }
        return cls(config)

    def __init__(self, data_file_info: Dict):
        """初始化文件加载器
        
        Args:
            data_file_info: 包含文件配置信息的字典，需要包含以下键：
                - atom: 原子标识
                - file_dir: 数据文件目录路径
                - file_type: 文件类型标识
                - level_parameter: 能级参数
                - this_as: this active space No.
                - file_name: 具体文件名（可选）
        """
        # 原子系统标识初始化
        self.atom = data_file_info.get("atom", "")  # 默认空字符串
        
        # 处理文件目录路径
        self.file_dir = data_file_info.get("file_dir", "")
        if self.file_dir and not Path(self.file_dir).exists():
            raise ValueError("数据目录不存在")  # 严格校验目录有效性
        else:
            self.data_file_dir = Path(self.file_dir) if self.file_dir else Path(".")

        # 加载文件配置参数
        self.file_type = data_file_info.get("file_type", "")    # 文件类型标识，默认空字符串
        self.level_parameter = data_file_info.get("level_parameter", "")  # 能级参数
        self.this_as = data_file_info.get("this_as", 0)        # AS过程标识
        
        # 处理文件路径逻辑
        self.file_name = data_file_info.get("file_name", "")
        if Path(self.file_dir).is_file():
            self.data_file_path = Path(self.file_dir)  # 直接使用完整路径
            print(f"Data file {self.data_file_path} loaded.")
            
        elif isinstance(self.file_name, str) and self.file_name:
            self.data_file_path = self.data_file_dir / self.file_name  # 路径拼接
            print(f"Data file {self.data_file_path} loaded.")
        else:
            self.data_file_path = None  # 无明确文件名时需要后续处理

        # 文件类型匹配规则配置
        self.file_keyword = {
            "TRANSITION": "*.*.*t",  # 匹配 transition 文件的通配符
            "TRANSITION_LSJ": f"*{self.level_parameter}*.*{self.level_parameter}*.*t.lsj",
            "LSJCOMPOSITION": f"*{self.level_parameter}*{self.this_as}.lsj.lbl",
            "PLOT": f"*{self.level_parameter}*.plot",  # 绘图数据文件匹配
            "BINARY_RADIAL_WAVEFUNCTIONS": f"*{self.level_parameter}*.w",  # 径向波函数二进制文件
            "MIX_COEFFICIENT": f"*{self.level_parameter}*{self.this_as}.m",  # 混合系数文件
            "CI_MIX_COEFFICIENT": f"*{self.level_parameter}*{self.this_as}.cm",  # 混合系数文件
            "DENSITY": f"*{self.level_parameter}*{self.this_as}.cd",  # 密度文件
            "Configuration_state_functions": f"*{self.level_parameter}*{self.this_as}.c",  # CSF配置文件
            "LEVEL": f"*{self.level_parameter}*{self.this_as}.level",  # 能级文件

        }

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

            g92rwf = read_fortran_record(binary_file, dtype=np.dtype('S6')).tobytes().decode('utf-8').strip()
            if g92rwf != 'G92RWF':
                raise ValueError('Not a radial wavefunction file!')

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
            if g92mix != 'G92MIX':
                raise ValueError('Not a mixing coefficient file!')
            print(f"g92mix: {g92mix}")  # Debugging print

            # READ (nfmix) nelec, ncftot, nw, nvectot, nvecsiz, nblock
            header_data = read_fortran_record(binary_file, dtype=np.int32, count=6)
            # nelec -> num_electron, ncftot -> total_num_configuration, nw -> NW, ncmin -> ncmin, nvecsiz -> nvecsiz, nblock -> num_block
            self.num_electron, self.total_num_configuration, self.NW, self.ncmin, self.nvecsiz, self.num_block = header_data

            print(f" nblock = {self.num_block},       ncftot =   {self.total_num_configuration},          nw =  {self.NW},            nelec =   {self.num_electron}")
            for jblock in tqdm(range(1, self.num_block+1)):
                print('cycle jblock =',jblock)

                # Read block data: nb, ncfblk, nevblk, iatjp, iaspa
                block_data = read_fortran_record(binary_file, dtype=np.int32, count=5)
                nb, ncfblk, nevblk, j_value_location, parity = block_data
                print(f' Block no. = {nb}, 2J+1 = {j_value_location}, Parity = {parity}, No. of eigenvalues = {nevblk}, No. of CSFs = {ncfblk}')
                # nb -> index_block, ncfblk -> ncfblk, nevblk -> block_energy_count, j_value_location -> iatjp, parity -> iaspa
                self.index_block_list.append(nb - 1)  # use python index method not fortran index method

                self.ncfblk_list.append(ncfblk)
                self.block_energy_count_list.append(nevblk)
                self.j_value_location_list.append(j_value_location)
                self.parity_list.append(parity)
                if jblock != nb:
                    raise ValueError('jblock != nb')

                ivec = read_fortran_record(binary_file, dtype=np.int32, count=nevblk)

                ivec_array = np.array(ivec) - 1  # use python index method not fortran index method
                
                
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

                if ncfblk != len(evecs[0]):
                    raise ValueError('''ncfblk: number of configuration functions in block
            len(evecs[0]): number of level eigenvalues in block
            ncfblk should equal len(evecs[0])''')

                self.mix_coefficient_list.append(evecs)

        return self.num_block, self.index_block_list, self.ncfblk_list, self.block_energy_count_list, self.j_value_location_list, self.parity_list, self.ivec_list, self.block_energy_list, self.block_level_energy_list, self.mix_coefficient_list
    
    def csfs_file_read(self):

        self.subshell_info_raw = []
        self.CSFs_block_j_value = []
        self.parity = ''
        self.CSFs_block_data = []
        self.CSFs_block_length = []
        csfs_file_data = []
        with open(self.load_file_path, 'r') as csfs_file:
            for line in csfs_file:
                csfs_file_data.append(line)
        
        self.subshell_info_raw = csfs_file_data[0:4]
        
        star_indices = []
        CSFs_block_parity = []
        
        for index, value in enumerate(csfs_file_data):
            if '*' in value:
                star_indices.append(index)

        prev_index = 5
        for index in star_indices:
            temp_j_value, temp_parity = csf_J(csfs_file_data[index - 1])
            self.CSFs_block_j_value.append(temp_j_value)
            CSFs_block_parity.append(temp_parity)
            # 处理每个块的数据，而不是一次性存储所有块
            block_data = csfs_file_data[prev_index:index]
            if len(block_data) % 3 != 0:
                raise ValueError("CSFs_list length must be a multiple of 3")
            
            # 将CSF块分成每三个元素一组
            block_csfs = [block_data[i:i+3] for i in range(0, len(block_data), 3)]
            self.CSFs_block_length.append(len(block_csfs))
            self.CSFs_block_data.append(block_csfs)  # 添加当前块的数据
            prev_index = index + 1
            
        temp_j_value, temp_parity = csf_J(csfs_file_data[-1])
        self.CSFs_block_j_value.append(temp_j_value)
        CSFs_block_parity.append(temp_parity)

        CSFs_parity = set(CSFs_block_parity)
        if len(CSFs_parity) == 1:
            self.parity = list(CSFs_parity)[0]

        # 处理最后一个块的数据
        last_block_data = csfs_file_data[prev_index:]
        if len(last_block_data) % 3 != 0:
            raise ValueError("CSFs_list length must be a multiple of 3")
        
        # 将CSF块分成每三个元素一组
        block_csfs = [last_block_data[i:i+3] for i in range(0, len(last_block_data), 3)]
        self.CSFs_block_length.append(len(block_csfs))        
        self.CSFs_block_data.append(block_csfs)  # 添加最后一个块的数据
        self.CSFs_block_length = np.array(self.CSFs_block_length)
        
        return self.subshell_info_raw, self.CSFs_block_j_value, self.parity, self.CSFs_block_data, self.CSFs_block_length


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
            if "CI" in self.file_type.upper():
                self.file_type = "CI_MIX_COEFFICIENT"
                print("data file type: ci mix_coefficient_data")

            else:
                self.file_type = "MIX_COEFFICIENT"
                print("data file type: rmcdhf mix_coefficient_data")

            self.mix_coefficient_dict = {}

            if self.data_file_path:
                self.load_file_path = self.data_file_path
            else:
                self.file_name = f"{self.atom}{self.level_parameter}{self.this_as}*m"
                self.load_file_path = Path(self.data_file_dir).joinpath(self.file_name)

            GraspFileLoad.mix_coefficient_file_read(self)

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
            level_energy_list = []

            for i in range(len(level_index)):
                if i == 0:
                    print(f"{i+1:3}{temp_pos[level_index[i]]:3}{temp_J[level_index[i]]:>4}   {temp_parity[level_index[i]]:1}    {temp_energy[level_index[i]]:14.7f}{0.0000000:12.2f}")
                    level_energy_list.append(temp_energy[level_index[i]])
                else:
                    print(f"{i+1:3}{temp_pos[level_index[i]]:3}{temp_J[level_index[i]]:>4}   {temp_parity[level_index[i]]:1}    {temp_energy[level_index[i]]:14.7f}{energy_au_cm(temp_energy[level_index[i]]-temp_energy[level_index[0]]):12.2f}")
                    level_energy_list.append(temp_energy[level_index[i]])

            # set mix file data as a class
            self.mix_file_data = MixCoefficientData(
                block_num=self.num_block,
                block_index_List=self.index_block_list,  # 注意大小写和命名一致性
                block_CSFs_nums=self.ncfblk_list,
                block_energy_count_List=self.block_energy_count_list,
                level_J_value_List=temp_J,
                parity_List=self.parity_list,
                block_levels_index_List=self.ivec_list,
                block_energy_List=self.block_energy_list,
                block_level_energy_List=self.block_level_energy_list,
                mix_coefficient_List=self.mix_coefficient_list,
                level_List=level_energy_list
            )
            return self.mix_file_data

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

        elif "CSF" in self.file_type.upper():
            self.file_type = "Configuration_state_functions"

            if self.data_file_path:
                self.load_file_path = self.data_file_path
            else:
                self.file_name = f"*{self.this_as}.c"
                self.load_file_path = Path(self.data_file_dir).joinpath(self.file_name)

            # GraspFileLoad.file_read(self) module cannot be utilized here, as the parsing of CSFs files serves exclusively for CSF refinement purposes, and preservation of trailing newline characters is mandatory.
            GraspFileLoad.csfs_file_read(self)
            
            self.block_num = len(self.CSFs_block_length)
            
            self.csfs_file_data = CSFs(
                subshell_info_raw = self.subshell_info_raw,
                CSFs_block_j_value = self.CSFs_block_j_value,
                parity = self.parity,
                CSFs_block_data = self.CSFs_block_data,
                CSFs_block_length = self.CSFs_block_length,
                block_num = self.block_num
            )

            return self.csfs_file_data

        else:
            return 0

#######################################################################
class EnergyFile2csv(GraspFileLoad):
    @classmethod
    def from_filepath(cls, filepath, file_type=None, *, store_csv_path: str=''):
        """从文件路径直接创建实例的类方法"""
        file_dir = str(Path(filepath).parent)
        file_name = Path(filepath).name
        config = {
            "atom": "",
            "file_dir": file_dir,
            "file_name": file_name,
            "level_parameter": "",
            "this_as": 0,
            "file_type": file_type or "ENERGY",
            "store_csv_path": store_csv_path
        }
        return cls(config)
    def __init__(self, data_file_info: Dict):
        super().__init__(data_file_info)
        store_csv_path = data_file_info.get("store_csv_path")
        if store_csv_path:
            self.store_file_path = Path(store_csv_path)
        else:
            self.store_file_path = Path(self.data_file_dir).joinpath("level_csv_file")
        if not self.store_file_path.exists():
            self.store_file_path.mkdir()

    def energy_file2csv(self):
        # load the data file
        self.load_level_data = GraspFileLoad.data_file_process(self)
        if not isinstance(self.load_level_data, list):
            raise ValueError(f"Expected list data for energy file, got {type(self.load_level_data)}")
        for i, line in enumerate(self.load_level_data):  # 使用enumerate获取行号
            if 'No Pos  J ' in line:
                self.skip_line = i + 3
                break
        
        self.temp_level_data = self.load_level_data[self.skip_line:]
        self.save_level_data = []
        for line in self.temp_level_data:
            if '-----' in line:
                break
            else:
                self.save_level_data.append(line.split())
        # set the data file path
        self.saved_csv_file_path = self.store_file_path.joinpath(f"{self.file_name}_level.csv")
        # open the csv file
        
        with open(self.saved_csv_file_path, 'w', newline='') as csv_file:
            # write the data file into the csv file
            writer = csv.writer(csv_file)
            writer.writerow(['No', 'Pos', 'J', 'Parity', 'EnergyTotal', 'EnergyLevel', 'splitting', 'configuration'])
            writer.writerows(self.save_level_data)
            # for item in self.save_level_data:
            #     # writer.writerow(item)
            #     csv_file.write(item+'\n')
        print(f"energy levels data has been written into {self.store_file_path}/{self.file_name}_level.csv csv file")
        return self.saved_csv_file_path

#######################################################################
# TODO not good enough
def write_sorted_CSFs_to_cfile(CSFs_file_info: List, sorted_CSFs_data_list: List, output_file: str):
    """
    将排序后的CSFs数据写入到指定的输出文件中。

    Args:
        CSFs_file_info (List): CSFs文件的头部信息(CSF(s):行上面的信息)
        sorted_CSFs_data (List): 排序后的CSFs数据列表
            sorted_CSFs_data[block
                                [CSFs
                                    [CSFS_1]
                                    [CSFS_2]
                                    ...
                                    [CSFS_n]
                                ]
            ]
        output_file (str): 输出文件的路径。
    """
    if len(CSFs_file_info) != 4:
        raise ValueError('CSFs file header info error!')
    blocks_num = len(sorted_CSFs_data_list)
    with open(output_file, 'w') as file:
        for line in CSFs_file_info:
            file.write(line)  
        
        file.write('CSF(s):\n')
        for index, block in enumerate(sorted_CSFs_data_list):
            if index!= blocks_num-1:
                for csf in block:
                    for line in csf:
                        file.write(line)
                file.write(' *\n')
            else:
                for csf in block:
                    for line in csf:
                        file.write(line)
                
def save_csf_metadata(
                        csf_obj: CSFs, 
                        filepath: Union[str, Path]
                        ):
    """保存CSFs元数据（排除CSFs_block_data）到pickle文件"""
    # 转换为Path对象
    filepath = Path(filepath)
    
    metadata = {
        'subshell_info_raw': csf_obj.subshell_info_raw,
        'CSFs_block_j_value': csf_obj.CSFs_block_j_value,
        'parity': csf_obj.parity,
        'CSFs_block_length': csf_obj.CSFs_block_length,
        'block_num': csf_obj.block_num
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(metadata, f)
        
def load_csf_metadata(
                        filepath: Union[str, Path]
                        ) -> dict:
    # 转换为Path对象
    filepath = Path(filepath)
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_csfs_binary(csf_obj: CSFs, filepath: Union[str, Path]):
    filepath = Path(filepath)
    
    # 元数据存储
    metadata = {
        'subshell_info_raw': csf_obj.subshell_info_raw,
        'CSFs_block_j_value': csf_obj.CSFs_block_j_value,
        'parity': csf_obj.parity,
        'CSFs_block_length': np.asarray(csf_obj.CSFs_block_length),
        'block_num': csf_obj.block_num,
        'data_type': 'nested_string'  # 标记特殊数据结构
    }
    
    # 专用压缩存储
    with gzip.open(filepath.with_suffix('.pkl.gz'), 'wb') as f:
        pickle.dump({
            'metadata': metadata,
            'block_data': csf_obj.CSFs_block_data  # 直接存储原始结构
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    
def load_csfs_binary(filepath: Union[str, Path]) -> CSFs:
    filepath = Path(filepath)
    
    with gzip.open(filepath.with_suffix('.pkl.gz'), 'rb') as f:
        data = pickle.load(f)
    
    return CSFs(
        subshell_info_raw=data['metadata']['subshell_info_raw'],
        CSFs_block_j_value=data['metadata']['CSFs_block_j_value'],
        parity=data['metadata']['parity'],
        CSFs_block_data=data['block_data'],  # 原始嵌套结构
        CSFs_block_length=data['metadata']['CSFs_block_length'],
        block_num=data['metadata']['block_num']
    )

#######################################################################

def level_data_compare(levels_file_1: List, levels_file_2: List):
    
    level_data_1 = []
    level_data_2 = []
    skip_line = 0
    for i, line in enumerate(levels_file_1):  # 使用enumerate获取行号
        if 'No Pos  J'in line:
            skip_line = i + 3
            break
    level_data_1 = levels_file_1[skip_line:]
    for i, line in enumerate(levels_file_2):  # 使用enumerate获取行号
        if 'No Pos  J'in line:
            skip_line = i + 3
            break
    level_data_2 = levels_file_2[skip_line:]
    
    if len(level_data_1)!= len(level_data_2):
        raise ValueError('The number of levels is not equal!')
    
    for i, (line1, line2) in enumerate(zip(level_data_1, level_data_2)):
        if not line1 or not line2:  # 跳过空行
            continue
        if line1.split()[-1] != line2.split()[-1]:
            raise ValueError(f'Configuration state functions differ at line {i+1}')
        
    return True


#######################################################################

def continue_calculate(
                        save_path: Union[str, Path], 
                        continue_calculate: bool
                        ):
    save_path = Path(save_path)
    
    with open(save_path/'run.input', 'rw') as file:
        file.write(continue_calculate)
        
    return f'Continue calculate is set to {continue_calculate}'


#######################################################################

def csfs_index_storange(blocks_csfs_index: Dict, save_file_path):
    """
    将CSFs索引存储到指定的文件中。
    Args:
        blocks_csfs_index (Dict): 包含CSFs索引的字典。
        save_file_path: 存储文件的路径（字符串或Path对象）。
    """
    # 转换为Path对象并检查是否有扩展名
    file_path = Path(save_file_path)
    if not file_path.suffix:
        file_path = file_path.with_suffix('.pkl')
    
    with open(file_path, 'wb') as f:
        pickle.dump(blocks_csfs_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return f'CSFs index has been stored to {file_path}'

def csfs_index_load(load_csfs_index_file_path):
    
    # 转换为Path对象便于处理
    file_path = Path(load_csfs_index_file_path)
    
    # 检查路径是否有扩展名
    if not file_path.suffix:
        # 没有扩展名时，按优先级尝试不同格式
        for ext in ['.pkl', '.msgpack']:
            full_path = file_path.with_suffix(ext)
            if full_path.exists():
                file_path = full_path
                break
        else:
            # 如果都不存在，默认使用.pkl扩展名（会在下面报错）
            file_path = file_path.with_suffix('.pkl')
    
    # 根据文件扩展名选择加载方式
    if file_path.suffix == '.msgpack':
        # 向后兼容：加载旧的msgpack格式文件
        import msgpack
        with open(file_path, 'rb') as f:
            blocks_csfs_index = msgpack.load(f, strict_map_key=False)
    else:
        # 默认使用pickle格式
        with open(file_path, 'rb') as f:
            blocks_csfs_index = pickle.load(f)
        
    return blocks_csfs_index

#######################################################################
def precompute_large_hash(
                            large_data: List[List[List[str]]], 
                            save_path: Union[str, Path] = "large_data_hash.pkl"
                            ):
    """
    预计算 large_data 的哈希映射（双层字典结构）
    
    返回:
        {block_idx: {csf_str: csf_index}}
    """
    # 转换为Path对象
    save_path = Path(save_path)
    
    large_hash = {}
    for block_idx, block_data in tqdm(enumerate(large_data), total=len(large_data), desc="计算哈希映射"):
        large_hash[block_idx] = {
            ''.join(item for sublist in csf for item in sublist): idx
            for idx, csf in enumerate(block_data)
        }
    
    with open(save_path, "wb") as f:
        pickle.dump(large_hash, f)
    
    return f'hash file has written in file {save_path}'

def load_large_hash(
                        file_path: Union[str, Path]
                        ) -> Dict[int, Dict[str, int]]:
    """从文件加载预计算的哈希映射"""
    # 转换为Path对象
    file_path = Path(file_path)
    
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
#######################################################################

def save_descriptors(
                        descriptors: np.ndarray, 
                        save_path: Union[str, Path], 
                        file_format: str = 'npy'
                        ):
    """
    保存描述符数组
    
    Args:
        descriptors (np.ndarray): 描述符数组
        save_path (Union[str, Path]): 保存路径（不含扩展名）
        file_format (str): 保存格式 ('npy', 'csv', 'pkl')
    
    Example:
        >>> descriptors = batch_process_csfs_to_descriptors(csfs_data)
        >>> save_descriptors(descriptors, 'output/csf_descriptors', 'csv')
        >>> save_descriptors(descriptors, Path('output/csf_descriptors'), 'npy')
    """
    
    # 转换为Path对象
    save_path = Path(save_path)
    
    if file_format.lower() == 'npy':
        file_path = save_path.parent / f"{save_path.name}_descriptors.npy"
        np.save(file_path, descriptors)
        print(f"Descriptors saved to: {file_path}")
        
    elif file_format.lower() == 'csv':
        file_path = save_path.parent / f"{save_path.name}_descriptors.csv"
        df = pd.DataFrame(descriptors)
        df.to_csv(file_path, index=False)
        print(f"Descriptors saved to: {file_path}")
        
    elif file_format.lower() == 'pkl':
        file_path = save_path.parent / f"{save_path.name}_descriptors.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(descriptors, f)
        print(f"Descriptors saved to: {file_path}")
        
    else:
        raise ValueError("file_format must be 'npy', 'csv', or 'pkl'")


def save_descriptors_with_multi_block(
                                        descriptors: np.ndarray, 
                                        labels: np.ndarray, 
                                        save_path: Union[str, Path], 
                                        file_format: str = 'npy'
                                        ):
    """
    保存带标签的描述符数组
    
    Args:
        descriptors (np.ndarray): 描述符数组
        labels (np.ndarray): 标签数组
        save_path (Union[str, Path]): 保存路径（不含扩展名）
        file_format (str): 保存格式 ('csv', 'npy', 'pkl')
    
    Example:
        >>> X, y = batch_process_csfs_with_block_indices(csfs_data)
        >>> save_descriptors_with_block_indices(X, y, 'ml_data/features', 'csv')
        >>> save_descriptors_with_block_indices(X, y, Path('ml_data/features'), 'npy')
    """
    
    # 转换为Path对象
    save_path = Path(save_path)
    
    if file_format.lower() == 'csv':
        # CSV格式：将标签作为最后一列
        file_path = save_path.parent / f"{save_path.name}_descriptors_block_indices.csv"
        df = pd.DataFrame(descriptors)
        df['label'] = labels
        df.to_csv(file_path, index=False)
        print(f"Descriptors with labels saved to: {file_path}")
        
    elif file_format.lower() == 'npy':
        # NPY格式：分别保存数据和标签
        data_path = save_path.parent / f"{save_path.name}_descriptors.npy"
        labels_path = save_path.parent / f"{save_path.name}_descriptors_block_indices.npy"
        np.save(data_path, descriptors)
        np.save(labels_path, labels)
        print(f"Descriptors saved to: {data_path}")
        print(f"Labels saved to: {labels_path}")
        
    elif file_format.lower() == 'pkl':
        # PKL格式：保存为字典
        file_path = save_path.parent / f"{save_path.name}_descriptors_block_indices.pkl"
        data_dict = {
            'descriptors': descriptors,
            'labels': labels
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"Descriptors and labels saved to: {file_path}")
        
    else:
        raise ValueError("file_format must be 'csv', 'npy', or 'pkl'")

def load_descriptors(
                        load_path: Union[str, Path], 
                        file_format: Optional[str] = None
                        ) -> Optional[np.ndarray]:
    """
    加载描述符数组
    
    Args:
        load_path (Union[str, Path]): 加载路径（可含或不含扩展名）
        file_format (Optional[str]): 文件格式，如果为None则从文件扩展名自动推断
    
    Returns:
        Optional[np.ndarray]: 描述符数组，加载失败返回None
    
    Example:
        >>> descriptors = load_descriptors('output/csf_descriptors.npy')
        >>> descriptors = load_descriptors(Path('output/csf_descriptors.npy'))
        >>> descriptors = load_descriptors('output/csf_descriptors', 'csv')
    """
    
    # 转换为Path对象
    load_path = Path(load_path)
    
    # 自动推断文件格式
    if file_format is None:
        if load_path.suffix == '.npy':
            file_format = 'npy'
            load_path = load_path.with_suffix('')  # 移除扩展名
        elif load_path.suffix == '.csv':
            file_format = 'csv'
            load_path = load_path.with_suffix('')
        elif load_path.suffix == '.pkl':
            file_format = 'pkl'
            load_path = load_path.with_suffix('')
        else:
            # 尝试自动检测（使用新的文件名格式）
            if (load_path.parent / f"{load_path.name}_descriptors.npy").exists():
                file_format = 'npy'
            elif (load_path.parent / f"{load_path.name}_descriptors.csv").exists():
                file_format = 'csv'
            elif (load_path.parent / f"{load_path.name}_descriptors.pkl").exists():
                file_format = 'pkl'
            else:
                print(f"Error: Cannot find file with path: {load_path}")
                return None
    
    try:
        if file_format.lower() == 'npy':
            file_path = load_path.parent / f"{load_path.name}_descriptors.npy"
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return None
            descriptors = np.load(file_path)
            print(f"Descriptors loaded from: {file_path}")
            return descriptors
            
        elif file_format.lower() == 'csv':
            file_path = load_path.parent / f"{load_path.name}_descriptors.csv"
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return None
            df = pd.read_csv(file_path)
            descriptors = df.values
            print(f"Descriptors loaded from: {file_path}")
            return descriptors
            
        elif file_format.lower() == 'pkl':
            file_path = load_path.parent / f"{load_path.name}_descriptors.pkl"
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return None
            with open(file_path, 'rb') as f:
                descriptors = pickle.load(f)
            print(f"Descriptors loaded from: {file_path}")
            return descriptors
            
        else:
            print(f"Error: Unsupported file format: {file_format}")
            return None
            
    except Exception as e:
        print(f"Error loading descriptors: {str(e)}")
        return None


def load_descriptors_with_multi_block(
                                        load_path: Union[str, Path], 
                                        file_format: Optional[str] = None
                                        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    加载带标签的描述符数组
    
    Args:
        load_path (Union[str, Path]): 加载路径（不含扩展名）
        file_format (Optional[str]): 文件格式，如果为None则自动推断
    
    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: (描述符数组, 标签数组)，加载失败返回None
    
    Example:
        >>> descriptors, labels = load_descriptors_with_block_indices('ml_data/features')
        >>> descriptors, labels = load_descriptors_with_block_indices(Path('ml_data/features'))
        >>> descriptors, labels = load_descriptors_with_block_indices('ml_data/features', 'csv')
    """
    
    # 转换为Path对象
    load_path = Path(load_path)
    
    # 自动推断文件格式
    if file_format is None:
        if (load_path.parent / f"{load_path.name}_descriptors_block_indices.csv").exists():
            file_format = 'csv'
        elif (load_path.parent / f"{load_path.name}_descriptors.npy").exists() and (load_path.parent / f"{load_path.name}_descriptors_block_indices.npy").exists():
            file_format = 'npy'
        elif (load_path.parent / f"{load_path.name}_descriptors_block_indices.pkl").exists():
            file_format = 'pkl'
        else:
            print(f"Error: Cannot find files with path: {load_path}")
            return None
    
    try:
        if file_format.lower() == 'csv':
            file_path = load_path.parent / f"{load_path.name}_descriptors_block_indices.csv"
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return None
                
            df = pd.read_csv(file_path)
            # 最后一列是标签，其余是描述符
            descriptors = df.iloc[:, :-1].to_numpy()
            labels = df.iloc[:, -1].to_numpy()
            print(f"Descriptors and labels loaded from: {file_path}")
            return descriptors, labels
            
        elif file_format.lower() == 'npy':
            data_path = load_path.parent / f"{load_path.name}_descriptors.npy"
            labels_path = load_path.parent / f"{load_path.name}_descriptors_block_indices.npy"
            
            if not data_path.exists():
                print(f"Error: Data file not found: {data_path}")
                return None
            if not labels_path.exists():
                print(f"Error: Labels file not found: {labels_path}")
                return None
                
            descriptors = np.load(data_path)
            labels = np.load(labels_path)
            print(f"Descriptors loaded from: {data_path}")
            print(f"Labels loaded from: {labels_path}")
            return descriptors, labels
            
        elif file_format.lower() == 'pkl':
            file_path = load_path.parent / f"{load_path.name}_descriptors_block_indices.pkl"
            if not file_path.exists():
                print(f"Error: File not found: {file_path}")
                return None
                
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)
                
            if 'descriptors' not in data_dict or 'labels' not in data_dict:
                print(f"Error: Invalid data format in {file_path}")
                return None
                
            descriptors = data_dict['descriptors']
            labels = data_dict['labels']
            print(f"Descriptors and labels loaded from: {file_path}")
            return descriptors, labels
            
        else:
            print(f"Error: Unsupported file format: {file_format}")
            return None
            
    except Exception as e:
        print(f"Error loading descriptors with labels: {str(e)}")
        return None

def load_config(
                    config_path: Union[str, Path]
                    ):
    """加载TOML配置文件并进行类型转换和数据处理"""
    # 转换为Path对象
    config_path = Path(config_path)
    
    # 使用标准库 tomllib 读取TOML文件
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    # 类型转换和数据处理
    config = _process_config_data(config)
    
    return SimpleNamespace(**config)

def _process_config_data(config):
    """处理配置数据，进行类型转换和验证"""
    # 浮点数转换
    config['cutoff_value'] = float(config['cutoff_value'])
    config['initial_ratio'] = float(config['initial_ratio'])
    config['expansion_ratio'] = float(config['expansion_ratio'])
    
    # 整数转换
    config['cal_loop_num'] = int(config['cal_loop_num'])
    config['difference'] = int(config['difference'])
    
    # 路径转换
    config['root_path'] = Path(config['root_path'])
    config['scf_cal_path'] = config['root_path'] / f'{config['conf']}_{config['cal_loop_num']}'
    
    # 模型参数处理
    if 'model_params' in config:
        model_params = config['model_params']
        
        # 转换模型参数中的整数
        if 'n_estimators' in model_params:
            model_params['n_estimators'] = int(model_params['n_estimators'])
        if 'random_state' in model_params:
            model_params['random_state'] = int(model_params['random_state'])
            
        # 处理class_weight字典，确保键为整数
        if 'class_weight' in model_params and isinstance(model_params['class_weight'], dict):
            class_weight = {}
            for k, v in model_params['class_weight'].items():
                class_weight[int(k)] = float(v)
            model_params['class_weight'] = class_weight
    
    # 数据验证
    _validate_config_data(config)
    
    return config

def _validate_config_data(config):
    """验证配置数据的有效性"""
    # 验证必需字段
    required_fields = ['atom', 'conf', 'cal_loop_num', 'cutoff_value', 'initial_ratio']
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"配置文件缺少必需字段: {missing_fields}")
    
    # 验证数值范围
    if 'cutoff_value' in config:
        if config['cutoff_value'] <= 0:
            raise ValueError("cutoff_value 必须大于 0")
    
    if 'initial_ratio' in config:
        if not (0 < config['initial_ratio'] <= 1):
            raise ValueError("initial_ratio 必须在 (0, 1] 范围内")
    
    if 'expansion_ratio' in config:
        if config['expansion_ratio'] < 1:
            raise ValueError("expansion_ratio 必须大于等于 1")
    
    # 验证光谱项列表
    if 'spetral_term' in config:
        if not isinstance(config['spetral_term'], list) or len(config['spetral_term']) == 0:
            raise ValueError("spetral_term 必须是非空列表")
    
    print(f"配置验证通过: cutoff_value={config.get('cutoff_value')}, initial_ratio={config.get('initial_ratio')}")
    
def update_config(config_path, updates):
    """更新TOML配置文件
    
    Args:
        config_path: 配置文件路径
        updates: 要更新的键值对字典
    """
    # 使用标准库 tomllib 读取TOML文件
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    # 更新配置值
    config.update(updates)
    
    # 写入配置文件（标准库 tomllib 不支持写入，使用 tomli-w）
    try:
        import tomli_w
        with open(config_path, 'wb') as f:
            tomli_w.dump(config, f)
    except ImportError:
        raise ImportError(
            "需要安装 tomli-w 库来写入TOML文件。请运行：\n"
            "pip install tomli-w"
        )