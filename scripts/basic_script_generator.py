#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :basic_script_generator.py
@date :2025/01/27
@author :YenochQin (秦毅)
@description: 基础版GRASP计算脚本生成器
'''

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


class BasicScriptGenerator:
    """基础版GRASP计算脚本生成器"""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_config_toml(self, config_params: Dict[str, Any]) -> str:
        """生成config.toml配置文件"""
        
        atom = config_params.get('atom', '')
        conf = config_params.get('conf', '')
        spectral_term = config_params.get('spectral_term', '')
        root_path = config_params.get('root_path', '')
        chosen_ratio = config_params.get('chosen_ratio', 0.09)
        cutoff_value = config_params.get('cutoff_value', 1e-09)
        selected_csfs = config_params.get('selected_csfs', '')
        
        config_content = f"""# GRASP计算配置文件

atom = "{atom}"
conf = "{conf}"
spetral_term = [
    "{spectral_term}",
]
continue_cal = true
cal_loop_num = 1
cal_error_num = 0
cal_method = "rci"
difference = 0
cutoff_value = {cutoff_value}
chosen_ratio = {chosen_ratio}
expansion_ratio = 2
target_pool_file = "{conf}.c"
root_path = "{root_path}"
selected_csfs_file = "{selected_csfs}.c"
selected_csfs_mix_file = "{selected_csfs}.cm"

# 收敛性检查参数
std_threshold = 1e-6  # 标准差阈值
relative_std_threshold = 1e-4  # 相对标准差阈值

[model_params]
n_estimators = 1000
random_state = 42

[model_params.class_weight]
0 = 1
1 = 3
"""
        
        return config_content
    
    def generate_calculation_script(self, script_params: Dict[str, Any]) -> str:
        """生成GRASP计算脚本"""
        
        job_name = script_params.get('job_name', 'GdIoddImlcias3_odd1')
        tasks_per_node = script_params.get('tasks_per_node', 46)
        partition = script_params.get('partition', 'work3')
        atom = script_params.get('atom', 'GdI')
        conf = script_params.get('conf', 'cv4odd1as3_odd1')
        loop1_rwfn_file = script_params.get('loop1_rwfn_file', f'mJ-1-90chosen{conf}.w')
        
        script_content = f"""#!/bin/zsh
#SBATCH -J {job_name}
#SBATCH -N 1
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH -p {partition}
#SBATCH --output={job_name}.log
. /usr/share/Modules/init/zsh
###########################################
# mpi run CPU core
processor={tasks_per_node}
###########################################
## module load
module load mpi/openmpi-x86_64-gcc
module load openblas/0.3.28-gcc-11.4.1
module load grasp/grasp_openblas
###########################################
# 加载 Conda 环境
source /home/workstation3/AppFiles/miniconda3/etc/profile.d/conda.sh || {{
    echo "❌ 加载 Conda 失败！"
    exit 1
}}
conda activate grasp-env || {{
    echo "❌ 激活环境失败！"
    exit 1
}}
###########################################
echo "True" > run.input
###########################################
## configuration
atom={atom}
conf="{conf}"
varied="3"
loop1_rwfn_file="{loop1_rwfn_file}"
rwfnestimate_file=${{conf}}_1.w
###########################################
# 检查 Python 路径
which python
python --version
###########################################
while true
do
###########################################
loop=$(csfs_ml_choosing_config_load.py get cal_loop_num)
if [ $loop -eq 1 ]; then
    echo "================初始化必要csfs文件数据================"
    python initial_csfs.py 
fi
###########################################
cal_status=$(csfs_ml_choosing_config_load.py get continue_cal)
if [[ "$cal_status" == "false" ]]; then
    echo "================计算终止================"
    break
fi
###########################################
## 组态选择处理
echo "================执行组态选择================"
python choosing_csfs.py 
if [ $? -ne 0 ]; then
    echo "组态选择失败!"
    exit 1
fi
###########################################
## grasp calculation routine

cd ${{conf}}_${{loop}}
mkdisks ${{processor}} caltmp
### 1. rcsf
cp ${{conf}}_${{loop}}.c rcsf.inp
cp ../isodata .

if [ $loop -eq 1 ]; then
echo "================第一次循环，使用${{loop1_rwfn_file}}================"
cp ../${{loop1_rwfn_file}} ${{conf}}.w
else
echo "================第${{loop}}次循环，使用${{rwfnestimate_file}}================"
cp ../${{rwfnestimate_file}} ${{conf}}.w
fi

### 2. rangular
mpirun -np ${{processor}} rangular_mpi <<EOF
y
EOF

### 3.rwfnestimate
rwfnestimate << EOF
y
1
${{conf}}.w
*
2
*
3
*
EOF

### 4. rmcdhf
if [ $loop -eq 1 ]; then
orbital_params="10s,9p,8d,7f,6g"
else
orbital_params=""
fi

mpirun -np ${{processor}} rmcdhf_mem_mpi <<EOF
y
1-4
5
${{orbital_params}}

100
EOF

### 5. rsave
rsave ${{conf}}_${{loop}}

if [ $loop -eq 1 ]; then
cp ${{conf}}_${{loop}}.w ..
fi

### 6. jj2lsj rmcdhf
jj2lsj << EOF
${{conf}}_${{loop}}
n
y
y
EOF

### 7. generate energy levels data file
rlevels ${{conf}}_${{loop}}.m > ${{conf}}_${{loop}}.level
cd ..

## 机器学习训练
echo "================执行机器学习训练================"
python train.py 
if [ $? -ne 0 ]; then
    echo "机器学习训练失败!"
    exit 1
fi

done
"""
        
        return script_content

    def generate_all_files(self, config_params: Dict[str, Any], 
                          script_params: Dict[str, Any],
                          project_name: str = "GRASP计算项目") -> Dict[str, str]:
        """生成所有文件"""
        
        files = {}
        
        # 生成配置文件
        config_content = self.generate_config_toml(config_params)
        config_path = self.output_dir / "config.toml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        files['config.toml'] = str(config_path)
        
        # 生成计算脚本
        script_content = self.generate_calculation_script(script_params)
        script_path = self.output_dir / "csfs_choosing_SCF_cal_ml_choosing.sh"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        # 设置执行权限
        os.chmod(script_path, 0o755)
        files['csfs_choosing_SCF_cal_ml_choosing.sh'] = str(script_path)
        
        return files


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成GRASP计算脚本和配置文件')
    parser.add_argument('--output-dir', '-o', default='.', help='输出目录')
    parser.add_argument('--atom', default='test', help='原子符号')
    parser.add_argument('--conf', default='test1', help='组态名称')
    parser.add_argument('--spectral-term', default='5', help='光谱项')
    parser.add_argument('--root-path', default='./', help='根路径')
    parser.add_argument('--job-name', default='test', help='作业名称')
    parser.add_argument('--tasks-per-node', type=int, default=46, help='每节点任务数')
    parser.add_argument('--partition', default='work3', help='SLURM分区')
    parser.add_argument('--project-name', default='ml_csfs_choosing_grasp_cal', help='项目名称')
    
    args = parser.parse_args()
    
    # 配置参数
    config_params = {
        'atom': args.atom,
        'conf': args.conf,
        'spectral_term': args.spectral_term,
        'root_path': args.root_path,
    }
    
    # 脚本参数
    script_params = {
        'job_name': args.job_name,
        'tasks_per_node': args.tasks_per_node,
        'partition': args.partition,
        'atom': args.atom,
        'conf': args.conf
    }
    
    # 生成文件
    generator = BasicScriptGenerator(args.output_dir)
    files = generator.generate_all_files(config_params, script_params, args.project_name)
    
    print("✅ 文件生成完成！")
    print("\n生成的文件:")
    for filename, filepath in files.items():
        print(f"  📄 {filename}: {filepath}")
    
    print(f"\n📁 输出目录: {args.output_dir}")
    print("\n🚀 下一步操作:")
    print("1. 检查并修改配置文件 config.toml")
    print("2. 运行初始化脚本: python initial_csfs.py")
    print("3. 提交计算作业: sbatch csfs_choosing_SCF_cal_ml_choosing.sh")


if __name__ == "__main__":
    main() 