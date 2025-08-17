#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :config_class_generator.py
@date :2025/08/17 16:27:41
@author :YenochQin (秦毅)
'''

import rtoml
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class CPUConfig:
    pytorch_threads: int


@dataclass
class StepControl:
    enable_step_control: bool
    target_loop: int
    start_step: str
    end_step: str
    skip_completed_steps: bool


@dataclass
class MLConfig:
    descriptors_with_subshell_info: bool
    high_prob_percentile: int
    overfitting_threshold: float
    underfitting_threshold: float
    include_wrong_level_negatives: bool


@dataclass
class ModelParams:
    n_estimators: int
    random_state: int
    class_weight: Dict[str, int]


@dataclass
class Config:
    atom: str
    conf: str
    spectral_term: List[str]
    continue_cal: bool
    cal_loop_num: int
    cal_error_num: int
    backward_loop_needed: bool
    target_backward_loop: int
    difference: int
    cutoff_value: float
    chosen_ratio: float
    expansion_ratio: int
    target_pool_file: str
    root_path: str
    selected_csfs_file: str
    selected_csfs_mix_file: str
    energy_std_threshold: float
    csfs_num_relative_std_threshold: float
    cal_method: str
    loop1_rwfn_file: str
    active_space: str
    cal_levels: str
    tasks_per_node: int
    mpi_tmp_path: str
    atomic_number: int
    mass_number: int
    atomic_mass: float
    nuclear_spin: int
    nuclear_dipole: int
    nuclear_quadrupole: int
    cpu_config: CPUConfig
    step_control: StepControl
    ml_config: MLConfig
    model_params: ModelParams


def load_config(config_path: str) -> Config:
    """从TOML文件加载配置并返回Config实例"""
    with open(config_path, 'r', encoding='utf-8') as f:
        data = rtoml.load(f)
    
    # 创建嵌套配置对象
    cpu_config = CPUConfig(**data['cpu_config'])
    step_control = StepControl(**data['step_control'])
    ml_config = MLConfig(**data['ml_config'])
    
    # 处理model_params
    model_params_data = data['model_params']
    class_weight = model_params_data['class_weight']
    model_params = ModelParams(
        n_estimators=model_params_data['n_estimators'],
        random_state=model_params_data['random_state'],
        class_weight={str(k): v for k, v in class_weight.items()}
    )
    
    # 创建主配置对象
    # 创建主配置对象
    nested_configs = {
        'cpu_config': cpu_config,
        'step_control': step_control,
        'ml_config': ml_config,
        'model_params': model_params
    }
    
    # 获取Config类的所有字段
    config_fields = [f.name for f in Config.__dataclass_fields__.values()]
    
    # 构建参数字典
    config_kwargs = {}
    for field_name in config_fields:
        if field_name in nested_configs:
            config_kwargs[field_name] = nested_configs[field_name]
        else:
            config_kwargs[field_name] = data[field_name]
    
    config = Config(**config_kwargs)
    
    return config


def save_config(config: Config, config_path: str) -> None:
    """将Config实例保存到TOML文件"""
    # 将dataclass转换为dict
    config_dict = {}
    
    # 处理基本字段
    for key, value in config.__dict__.items():
        if key not in ['cpu_config', 'step_control', 'ml_config', 'model_params']:
            config_dict[key] = value
    
    # 处理嵌套配置
    config_dict['cpu_config'] = config.cpu_config.__dict__
    config_dict['step_control'] = config.step_control.__dict__
    config_dict['ml_config'] = config.ml_config.__dict__
    
    # 处理model_params
    model_params_dict = config.model_params.__dict__.copy()
    class_weight = model_params_dict.pop('class_weight')
    model_params_dict['class_weight'] = class_weight
    config_dict['model_params'] = model_params_dict
    
    # 保存到文件
    with open(config_path, 'w', encoding='utf-8') as f:
        rtoml.dump(config_dict, f)


# 使用示例
if __name__ == "__main__":
    config_path = "/Users/yiqin/Documents/PythonProjects/GraspDataProcessing-script/scripts/config.toml"
    
    # 加载配置
    config = load_config(config_path)
    print("加载的配置:")
    print(f"原子: {config.atom}")
    print(f"配置: {config.conf}")
    print(f"CPU线程数: {config.cpu_config.pytorch_threads}")
    print(f"模型参数 - n_estimators: {config.model_params.n_estimators}")
    print(f"模型参数 - class_weight: {config.model_params.class_weight}")
    
    # 修改配置
    config.cpu_config.pytorch_threads = 64
    config.model_params.n_estimators = 2000
    
    # 保存配置
    save_config(config, config_path.replace('.toml', '_modified.toml'))
    print("\n配置已保存到 modified.toml")