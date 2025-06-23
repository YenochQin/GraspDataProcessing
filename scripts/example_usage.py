#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :example_usage.py
@date :2025/01/27
@author :YenochQin (秦毅)
@description: 脚本生成器使用示例
'''

import sys
from pathlib import Path

# 添加脚本目录到路径
sys.path.append(str(Path(__file__).parent))

from generate_calculation_scripts import CalculationScriptGenerator


def example_gdI_calculation():
    """示例：生成GdI原子的计算脚本"""
    
    print("🔧 生成GdI原子计算脚本...")
    
    # 创建生成器
    generator = CalculationScriptGenerator("./gdI_example")
    
    # 配置参数
    config_params = {
        'atom': 'GdI',
        'conf': 'cv4odd1as3_odd1',
        'spetral_term': ['5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(7)8S0_8S.5d_7D'],
        'root_path': '/home/workstation3/caldata/GdI/cvodd1/as3_odd1',
        'target_pool_file': 'cv4odd1as3_odd1.c',
        'selected_csfs_file': 'mJ-1-90chosenas3_odd1.c',
        'selected_csfs_mix_file': 'mJ-1-90chosenas3_odd1.cm',
        'chosen_ratio': 0.09,
        'cutoff_value': 1e-09,
        'cal_loop_num': 12,
        'generation_time': '2025-01-27'
    }
    
    # 脚本参数
    script_params = {
        'job_name': 'GdIoddImlcias3_odd1',
        'tasks_per_node': 46,
        'partition': 'work3',
        'atom': 'GdI',
        'conf': 'cv4odd1as3_odd1',
        'loop1_rwfn_file': 'mJ-1-90chosenas3_odd1.w'
    }
    
    # 生成文件
    files = generator.generate_all_files(config_params, script_params, "GdI原子计算项目")
    
    print("✅ GdI计算脚本生成完成！")
    return files


def example_custom_atom():
    """示例：生成自定义原子的计算脚本"""
    
    print("🔧 生成自定义原子计算脚本...")
    
    # 创建生成器
    generator = CalculationScriptGenerator("./custom_atom_example")
    
    # 配置参数 - 以Ce为例
    config_params = {
        'atom': 'Ce',
        'conf': 'cv4odd1as2_odd1',
        'spetral_term': ['5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(2)3H_3H.5d_4D'],
        'root_path': '/home/workstation3/caldata/Ce/cvodd1/as2_odd1',
        'target_pool_file': 'cv4odd1as2_odd1.c',
        'selected_csfs_file': 'mJ-1-90chosenas2_odd1.c',
        'selected_csfs_mix_file': 'mJ-1-90chosenas2_odd1.cm',
        'chosen_ratio': 0.1,
        'cutoff_value': 1e-08,
        'cal_loop_num': 10,
        'generation_time': '2025-01-27'
    }
    
    # 脚本参数
    script_params = {
        'job_name': 'CeoddImlcias2_odd1',
        'tasks_per_node': 32,
        'partition': 'work2',
        'atom': 'Ce',
        'conf': 'cv4odd1as2_odd1',
        'loop1_rwfn_file': 'mJ-1-90chosenas2_odd1.w'
    }
    
    # 生成文件
    files = generator.generate_all_files(config_params, script_params, "Ce原子计算项目")
    
    print("✅ Ce计算脚本生成完成！")
    return files


def example_batch_generation():
    """示例：批量生成多个原子的计算脚本"""
    
    print("🔧 批量生成多个原子计算脚本...")
    
    # 定义多个原子的配置
    atoms_config = [
        {
            'atom': 'GdI',
            'conf': 'cv4odd1as3_odd1',
            'spectral_term': '5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(7)8S0_8S.5d_7D',
            'root_path': '/home/workstation3/caldata/GdI/cvodd1/as3_odd1',
            'job_name': 'GdIoddImlcias3_odd1',
            'tasks_per_node': 46
        },
        {
            'atom': 'Ce',
            'conf': 'cv4odd1as2_odd1', 
            'spectral_term': '5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(2)3H_3H.5d_4D',
            'root_path': '/home/workstation3/caldata/Ce/cvodd1/as2_odd1',
            'job_name': 'CeoddImlcias2_odd1',
            'tasks_per_node': 32
        },
        {
            'atom': 'Pr',
            'conf': 'cv4odd1as2_odd1',
            'spectral_term': '5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(3)4I_4I.5d_5D',
            'root_path': '/home/workstation3/caldata/Pr/cvodd1/as2_odd1',
            'job_name': 'ProddImlcias2_odd1',
            'tasks_per_node': 32
        }
    ]
    
    all_files = {}
    
    for atom_config in atoms_config:
        atom = atom_config['atom']
        print(f"🔧 生成{atom}原子计算脚本...")
        
        # 创建生成器
        output_dir = f"./batch_example/{atom.lower()}"
        generator = CalculationScriptGenerator(output_dir)
        
        # 配置参数
        config_params = {
            'atom': atom_config['atom'],
            'conf': atom_config['conf'],
            'spetral_term': [atom_config['spectral_term']],
            'root_path': atom_config['root_path'],
            'target_pool_file': f"{atom_config['conf']}.c",
            'selected_csfs_file': f"mJ-1-90chosen{atom_config['conf']}.c",
            'selected_csfs_mix_file': f"mJ-1-90chosen{atom_config['conf']}.cm",
            'chosen_ratio': 0.09,
            'cutoff_value': 1e-09,
            'cal_loop_num': 12,
            'generation_time': '2025-01-27'
        }
        
        # 脚本参数
        script_params = {
            'job_name': atom_config['job_name'],
            'tasks_per_node': atom_config['tasks_per_node'],
            'partition': 'work3',
            'atom': atom_config['atom'],
            'conf': atom_config['conf'],
            'loop1_rwfn_file': f"mJ-1-90chosen{atom_config['conf']}.w"
        }
        
        # 生成文件
        files = generator.generate_all_files(
            config_params, 
            script_params, 
            f"{atom}原子计算项目"
        )
        
        all_files[atom] = files
        print(f"✅ {atom}计算脚本生成完成！")
    
    print(f"\n🎉 批量生成完成！共生成{len(atoms_config)}个原子的计算脚本")
    return all_files


def main():
    """主函数 - 运行示例"""
    
    print("🚀 GRASP计算脚本生成器示例")
    print("=" * 50)
    
    # 示例1：GdI原子
    print("\n📋 示例1：GdI原子计算脚本")
    gdI_files = example_gdI_calculation()
    
    # 示例2：自定义原子
    print("\n📋 示例2：Ce原子计算脚本")
    ce_files = example_custom_atom()
    
    # 示例3：批量生成
    print("\n📋 示例3：批量生成多个原子")
    batch_files = example_batch_generation()
    
    print("\n" + "=" * 50)
    print("🎯 所有示例运行完成！")
    print("\n📁 生成的文件目录:")
    print("  - ./gdI_example/")
    print("  - ./custom_atom_example/")
    print("  - ./batch_example/")
    
    print("\n💡 使用建议:")
    print("1. 根据实际需要修改配置文件中的路径和参数")
    print("2. 确保GRASP环境和Python环境正确配置")
    print("3. 在运行前检查所有输入文件是否存在")
    print("4. 根据计算资源调整SLURM参数")


if __name__ == "__main__":
    main() 