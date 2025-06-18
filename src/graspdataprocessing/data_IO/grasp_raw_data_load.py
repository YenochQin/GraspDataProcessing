#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :grasp_raw_data_load.py
@date :2025/06/16 16:05:36
@author :YenochQin (秦毅)
'''


import re
import csv
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from types import SimpleNamespace
from dataclasses import dataclass
import numpy as np
import pandas as pd
import struct
from tqdm import tqdm

from ..utils.tool_function import *
from ..utils.data_modules import *
from ..CSFs_choosing.CSFs_compress_extract import *

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
        self.load_level_data = GraspFileLoad.data_file_process(self)

    def energy_file2csv(self):
        # load the data file
        
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
