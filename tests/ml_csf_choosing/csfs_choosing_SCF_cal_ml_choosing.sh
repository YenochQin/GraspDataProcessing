#!/bin/zsh
#SBATCH -J GdIoddImlcias4_odd2
#SBATCH -N 1
#SBATCH --ntasks-per-node=46
#SBATCH -p work3
#SBATCH --output=%j_%x.log
#SBATCH --error=%j_%x.log
. /usr/share/Modules/init/zsh

# 添加时间戳函数
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_with_timestamp "========== 开始执行 sbatch 脚本 =========="
log_with_timestamp "作业名: ${SLURM_JOB_NAME:-未设置}"
log_with_timestamp "作业编号: ${SLURM_JOB_ID:-未设置}"

###########################################
# mpi run CPU core
processor=46
###########################################
## module load
log_with_timestamp "加载必要的模块..."
module load mpi/openmpi-x86_64-gcc
module load openblas/0.3.28-gcc-11.4.1
module load grasp/grasp_openblas
###########################################
# ⚠️ 关键修改：确保正确加载 Conda（zsh 需要手动初始化）
log_with_timestamp "初始化 Conda 环境..."
source /home/workstation3/AppFiles/miniconda3/etc/profile.d/conda.sh  || {
    log_with_timestamp "❌ 加载 Conda 失败！请检查路径是否正确。"
    exit 1
}
# conda init zsh  # 如果未初始化过 Conda for zsh，运行一次（本地终端）
conda activate grasp-env || {
    log_with_timestamp "❌ 激活环境失败！请确认环境名是否正确。"
    exit 1
}
log_with_timestamp "✅ Conda 环境激活成功"
###########################################
## configuration
atom=GdI
conf="cv4odd1as4_odd2"
varied="3"
loop1_rwfn_file="mJ-1-90chosenas3_odd2.w"
rwfnestimate_file=${conf}_1.w
Active_space="10s,9p,8d,7f,6g"
log_with_timestamp "配置参数: atom=$atom, conf=$conf, processor=$processor"
###########################################
# 检查 Python 路径
log_with_timestamp "检查 Python 环境..."
which python
python --version
###########################################
while true
do
###########################################
log_with_timestamp "获取循环计数..."
loop=$(csfs_ml_choosing_config_load.py get cal_loop_num 2>&1)
log_with_timestamp "当前循环: $loop"

if [ $loop -eq 1 ]; then
    # 初始化必要csfs文件数据
    log_with_timestamp "================初始化必要csfs文件数据================"
    # python initial_csfs.py 
fi
###########################################
log_with_timestamp "检查计算状态..."
cal_status=$(csfs_ml_choosing_config_load.py get continue_cal 2>&1)
log_with_timestamp "计算状态: $cal_status"

if [[ "$cal_status" == "false" ]]; then
    log_with_timestamp "================计算终止================"
    break
fi
###########################################
## 组态选择处理
log_with_timestamp "================执行组态选择================"
python choosing_csfs.py 2>&1
if [ $? -ne 0 ]; then
    log_with_timestamp "❌ 组态选择失败!"
    exit 1
fi
log_with_timestamp "✅ 组态选择完成"
###########################################
## grasp calculation routine

log_with_timestamp "进入计算目录: ${conf}_${loop}"
cd ${conf}_${loop}

log_with_timestamp "创建磁盘空间..."
mkdisks ${processor} caltmp 2>&1

### rcsf
log_with_timestamp "准备 rcsf 输入文件..."
cp ${conf}_${loop}.c rcsf.inp # rmcdhf
cp ../isodata .

if [ $loop -eq 1 ]; then
    log_with_timestamp "================第一次循环，使用${loop1_rwfn_file}================"
    cp ../${loop1_rwfn_file} ${conf}.w
    orbital_params=${Active_space}

else
    log_with_timestamp "================第${loop}次循环，使用${rwfnestimate_file}================"
    cp ../${rwfnestimate_file} ${conf}.w
    orbital_params=""
fi

### rangular
log_with_timestamp "执行 rangular_mpi..."
mpirun -np ${processor} rangular_mpi 2>&1 <<EOF
y
EOF
log_with_timestamp "✅ rangular_mpi 完成"

### rwfnestimate
log_with_timestamp "执行 rwfnestimate (第一次循环)..."
rwfnestimate 2>&1 << EOF
y
1
${conf}.w
*
2
*
3
*
EOF
log_with_timestamp "✅ rwfnestimate 完成"

### rmcdhf
log_with_timestamp "执行 rmcdhf_mem_mpi..."
mpirun -np ${processor} rmcdhf_mem_mpi 2>&1 <<EOF
y
1-4
5
${orbital_params}

100
EOF
log_with_timestamp "✅ rmcdhf_mem_mpi 完成"

### rsave
log_with_timestamp "执行 rsave..."
rsave ${conf}_${loop} 2>&1
log_with_timestamp "✅ rsave 完成"

if [ $loop -eq 1 ]; then
    log_with_timestamp "复制波函数文件..."
    cp ${conf}_${loop}.w ..
fi

# # rci
# log_with_timestamp "执行 rci_mpi..."
# mpirun -np ${processor} rci_mpi 2>&1 <<EOF
# y
# ${conf}_${loop}
# y
# y
# 1.d-6
# y
# n
# n
# y
# 5
# 1-4
# EOF

### jj2lsj rmcdhf
log_with_timestamp "执行 jj2lsj (rmcdhf)..."
jj2lsj 2>&1 << EOF
${conf}_${loop}
n
y
y
EOF
log_with_timestamp "✅ jj2lsj 完成"

# ### jj2lsj rci
# log_with_timestamp "执行 jj2lsj (rci)..."
# jj2lsj 2>&1 << EOF
# ${conf}_${loop}
# y
# y
# y
# EOF

### generate energy levels data file
log_with_timestamp "生成能级数据文件..."
rlevels ${conf}_${loop}.m > ${conf}_${loop}.level 2>&1 # rmcdhf
# rlevels ${conf}_${loop}.cm > ${conf}_${loop}.level 2>&1 # rci
log_with_timestamp "✅ 能级数据文件生成完成"

log_with_timestamp "返回上级目录..."
cd ..

## 机器学习训练
log_with_timestamp "================执行机器学习训练================"
python train.py 2>&1
if [ $? -ne 0 ]; then
    log_with_timestamp "❌ 机器学习训练失败!"
    exit 1
fi
log_with_timestamp "✅ 机器学习训练完成"

log_with_timestamp "循环 $loop 完成，准备下一次迭代..."
done

log_with_timestamp "========== sbatch 脚本执行完成 =========="