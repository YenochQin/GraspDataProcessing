#!/bin/zsh
#SBATCH -J GdIoddImlcias3_odd4
#SBATCH -N 1
#SBATCH --ntasks-per-node=46
#SBATCH -p work3
#SBATCH --output=as3oddImlci.log
. /usr/share/Modules/init/zsh
###########################################
# mpi run CPU core
processor=46
###########################################
## module load
module load mpi/openmpi-x86_64-gcc   
module load openblas/0.3.28-gcc-11.4.1
module load grasp/grasp_openblas 
###########################################
# ⚠️ 关键修改：确保正确加载 Conda（zsh 需要手动初始化）
source /home/workstation3/AppFiles/miniconda3/etc/profile.d/conda.sh  || {
    echo "❌ 加载 Conda 失败！请检查路径是否正确。"
    exit 1
}
# conda init zsh  # 如果未初始化过 Conda for zsh，运行一次（本地终端）
conda activate grasp-env || {
    echo "❌ 激活环境失败！请确认环境名是否正确。"
    exit 1
}
###########################################
echo "True" > run.input
###########################################
## configuration
atom=GdI
conf="cv4odd1as3_odd4"
varied="3"
rwfnestimate_file="mJ-1-90chosenas3_odd4.w"

###########################################
# 检查 Python 路径
which python
python --version
###########################################
# 初始化必要csfs文件数据
# python initial_csfs.py 
###########################################
loop=$(toml get cal_loop_num)
while true
do
###########################################
# 检查文件是否存在
if [[ ! -f "run.input" ]]; then
    echo "================等待输入文件...================"
    break
fi

# 读取文件内容
run_status=$(head -n 1 run.input | tr -d '[:space:]')
rm -f run.input  # ⚠️ 关键修复：处理完立即删除文件

# 检查退出条件
if [[ "$run_status" == "False" ]]; then
    echo "================计算终止================"
    break
fi

# 主程序逻辑
echo "执行任务..."
# 控制循环速度
sleep 1
###########################################

## 组态选择处理
echo "================执行组态选择================"
python choosing_csfs.py 
if [ $? -ne 0 ]; then
    echo "组态选择失败!"
    exit 1
fi

## grasp calculation routine
# mkdir ${conf}_${loop}
cd ${conf}_${loop}
mkdisks ${processor} caltmp
### 1. rcsf
# cp ${conf}_${loop}.c rcsf.inp # rmcdhf
cp ../isodata .
cp ../${rwfnestimate_file} ${conf}_${loop}.w

# ### 2. rangular
# mpirun -np ${processor} rangular_mpi <<EOF
# y
# EOF

# ### 3.rwfnestimate
# rwfnestimate << EOF
# y
# 1
# ${rwfnestimate_file}
# *
# 2
# *
# 3
# *
# EOF

# ### 4. rmcdhf
# mpirun -np ${processor} rmcdhf_mem_mpi <<EOF
# y
# 1-3
# 5


# 100
# EOF

# ### 5. rsave
# rsave ${conf}_${loop}

# 5. rci
mpirun -np ${processor} rci_mpi <<EOF
y
${conf}_${loop}
y
y
1.d-6
y
n
n
y
5
1-4
EOF

# ### 6. jj2lsj rmcdhf
# jj2lsj << EOF
# ${conf}_${loop}
# n
# y
# y
# EOF
### 6. jj2lsj rci
jj2lsj << EOF
${conf}_${loop}
y
y
y
EOF
### 7. generate energy levels data file
rlevels ${conf}_${loop}.cm > ${conf}_${loop}.level
cd ..

## 机器学习训练
echo "================执行机器学习训练================"
python train.py 
if [ $? -ne 0 ]; then
    echo "机器学习训练失败!"
    exit 1
fi

done