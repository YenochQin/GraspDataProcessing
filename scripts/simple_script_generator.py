#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :simple_script_generator.py
@date :2025/01/27
@author :YenochQin (秦毅)
@description: 简化版GRASP计算脚本生成器
'''

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


class SimpleScriptGenerator:
    """简化版GRASP计算脚本生成器"""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_config_toml(self, config_params: Dict[str, Any]) -> str:
        """生成config.toml配置文件"""
        
        atom = config_params.get('atom', 'GdI')
        conf = config_params.get('conf', 'cv4odd1as3_odd1')
        spectral_term = config_params.get('spectral_term', '5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(7)8S0_8S.5d_7D')
        root_path = config_params.get('root_path', '/home/workstation3/caldata/GdI/cvodd1/as3_odd1')
        chosen_ratio = config_params.get('chosen_ratio', 0.09)
        cutoff_value = config_params.get('cutoff_value', 1e-09)
        cal_loop_num = config_params.get('cal_loop_num', 12)
        
        config_content = f"""# GRASP计算配置文件
# 生成时间: 2025-01-27

atom = "{atom}"
conf = "{conf}"
spetral_term = [
    "{spectral_term}",
]
continue_cal = true
cal_loop_num = {cal_loop_num}
cal_error_num = 0
cal_method = "rci"
difference = 0
cutoff_value = {cutoff_value}
chosen_ratio = {chosen_ratio}
expansion_ratio = 2
target_pool_file = "{conf}.c"
root_path = "{root_path}"
selected_csfs_file = "mJ-1-90chosen{conf}.c"
selected_csfs_mix_file = "mJ-1-90chosen{conf}.cm"

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
${{loop1_rwfn_file}}
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
    
    def generate_initial_csfs_script(self, config_params: Dict[str, Any]) -> str:
        """生成初始化CSFs的脚本"""
        
        conf = config_params.get('conf', 'cv4odd1as3_odd1')
        
        script_content = f"""#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :initial_csfs.py
@date :2025/01/27
@author :YenochQin (秦毅)
@description: 初始化CSFs数据预处理
'''

import sys
from pathlib import Path

sys.path.append('/home/workstation3/AppFiles/GraspDataProcessing/src')
try:
    import graspdataprocessing as gdp
except ImportError:
    print("错误: 无法导入 graspdataprocessing 模块")
    sys.exit(1)

def main():
    """初始化CSFs数据预处理"""
    try:
        # 加载配置
        config = gdp.load_config('config.toml')
        logger = gdp.setup_logging(config)
        
        logger.info("开始初始化CSFs数据预处理")
        logger.info(f"原子: {{config.atom}}")
        logger.info(f"组态: {{config.conf}}")
        
        # 创建必要的目录
        gdp.setup_directories(config)
        
        # 加载目标池CSFs数据
        target_pool_file = Path(config.root_path) / config.target_pool_file
        if not target_pool_file.exists():
            logger.error(f"目标池文件不存在: {{target_pool_file}}")
            raise FileNotFoundError(f"目标池文件不存在: {{target_pool_file}}")
        
        # 读取CSFs数据
        target_pool_csfs_data = gdp.read_CSFs_from_cfile(target_pool_file)
        logger.info(f"成功读取目标池CSFs数据，共{{target_pool_csfs_data.total_csfs_num}}个CSFs")
        
        # 保存为二进制格式
        binary_file = Path(config.root_path) / f"{{config.conf}}.pkl"
        gdp.save_csfs_binary(target_pool_csfs_data, binary_file)
        logger.info(f"CSFs数据已保存为二进制格式: {{binary_file}}")
        
        # 生成CSFs描述符
        descriptors = gdp.generate_csfs_descriptors(target_pool_csfs_data)
        descriptors_file = Path(config.root_path) / f"{{config.conf}}_descriptors.npy"
        gdp.save_descriptors(descriptors, descriptors_file)
        logger.info(f"CSFs描述符已保存: {{descriptors_file}}")
        
        # 如果有选定的CSFs文件，处理选定数据
        selected_csfs_file = Path(config.root_path) / config.selected_csfs_file
        if selected_csfs_file.exists():
            selected_csfs_data = gdp.read_CSFs_from_cfile(selected_csfs_file)
            selected_indices = gdp.find_csfs_indices(target_pool_csfs_data, selected_csfs_data)
            selected_indices_file = Path(config.root_path) / f"{{config.conf}}_selected_indices.pkl"
            gdp.csfs_index_storange(selected_indices, selected_indices_file)
            logger.info(f"选定CSFs索引已保存: {{selected_indices_file}}")
        else:
            logger.warning(f"选定CSFs文件不存在: {{selected_csfs_file}}")
        
        logger.info("CSFs数据预处理完成")
        
    except Exception as e:
        print(f"初始化失败: {{str(e)}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
        
        return script_content
    
    def generate_requirements_txt(self) -> str:
        """生成requirements.txt文件"""
        return """numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
imbalanced-learn>=0.8.0
joblib>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
tabulate>=0.8.9
toml>=0.10.0
msgpack>=1.0.0
"""
    
    def generate_readme_md(self, project_name: str = "GRASP计算项目") -> str:
        """生成README.md文件"""
        return f"""# {project_name}

## 项目描述
这是一个基于GRASP代码的原子结构计算项目，使用机器学习方法进行组态选择。

## 文件结构
```
.
├── config.toml              # 配置文件
├── csfs_choosing_SCF_cal_ml_choosing.sh  # 主计算脚本
├── initial_csfs.py          # CSFs初始化脚本
├── choosing_csfs.py         # 组态选择脚本
├── train.py                 # 机器学习训练脚本
├── requirements.txt         # Python依赖
└── README.md               # 项目说明
```

## 使用方法

### 1. 环境准备
```bash
# 激活conda环境
conda activate grasp-env

# 安装依赖
pip install -r requirements.txt
```

### 2. 初始化数据
```bash
python initial_csfs.py
```

### 3. 运行计算
```bash
# 使用SLURM提交作业
sbatch csfs_choosing_SCF_cal_ml_choosing.sh

# 或直接运行
bash csfs_choosing_SCF_cal_ml_choosing.sh
```

## 配置说明

### config.toml 主要参数
- `atom`: 原子符号
- `conf`: 组态名称
- `spetral_term`: 光谱项
- `cal_loop_num`: 计算循环次数
- `chosen_ratio`: 初始选择比例
- `cutoff_value`: 截断值

### 机器学习参数
- `n_estimators`: 随机森林树的数量
- `class_weight`: 类别权重设置

## 注意事项
1. 确保GRASP环境正确配置
2. 检查所有输入文件路径
3. 监控计算收敛性
4. 定期备份计算结果

## 故障排除
- 如果遇到模块导入错误，检查Python路径设置
- 如果计算不收敛，调整`cutoff_value`和`chosen_ratio`参数
- 如果内存不足，减少`tasks_per_node`数量
"""
    
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
        
        # 生成初始化脚本
        init_script_content = self.generate_initial_csfs_script(config_params)
        init_script_path = self.output_dir / "initial_csfs.py"
        with open(init_script_path, 'w', encoding='utf-8') as f:
            f.write(init_script_content)
        os.chmod(init_script_path, 0o755)
        files['initial_csfs.py'] = str(init_script_path)
        
        # 生成requirements.txt
        req_content = self.generate_requirements_txt()
        req_path = self.output_dir / "requirements.txt"
        with open(req_path, 'w', encoding='utf-8') as f:
            f.write(req_content)
        files['requirements.txt'] = str(req_path)
        
        # 生成README.md
        readme_content = self.generate_readme_md(project_name)
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        files['README.md'] = str(readme_path)
        
        return files


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成GRASP计算脚本和配置文件')
    parser.add_argument('--output-dir', '-o', default='.', help='输出目录')
    parser.add_argument('--atom', default='GdI', help='原子符号')
    parser.add_argument('--conf', default='cv4odd1as3_odd1', help='组态名称')
    parser.add_argument('--spectral-term', default='5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(7)8S0_8S.5d_7D', help='光谱项')
    parser.add_argument('--root-path', default='/home/workstation3/caldata/GdI/cvodd1/as3_odd1', help='根路径')
    parser.add_argument('--job-name', default='GdIoddImlcias3_odd1', help='作业名称')
    parser.add_argument('--tasks-per-node', type=int, default=46, help='每节点任务数')
    parser.add_argument('--partition', default='work3', help='SLURM分区')
    parser.add_argument('--project-name', default='GRASP计算项目', help='项目名称')
    
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
    generator = SimpleScriptGenerator(args.output_dir)
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