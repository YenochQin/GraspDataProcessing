#!/bin/zsh
#SBATCH -J GdIoddImlcias4_odd2
#SBATCH -N 1
#SBATCH --ntasks-per-node=46
#SBATCH -p work3
#SBATCH --output=as4_odd2GdIoddImlci.log
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
conf="cv4odd1as4_odd2"
varied="3"
loop1_rwfn_file="mJ-1-90chosenas3_odd2.w"
rwfnestimate_file=${conf}_1.w
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
    # 初始化必要csfs文件数据
    echo "================初始化必要csfs文件数据================"
    # python initial_csfs.py 
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

cd ${conf}_${loop}
mkdisks ${processor} caltmp
### 1. rcsf
cp ${conf}_${loop}.c rcsf.inp # rmcdhf
cp ../isodata .

if [ $loop -eq 1 ]; then
echo "================第一次循环，使用${loop1_rwfn_file}================"
cp ../${loop1_rwfn_file} ${conf}.w
else
echo "================第${loop}次循环，使用${rwfnestimate_file}================"
cp ../${rwfnestimate_file} ${conf}.w
fi

### 2. rangular
mpirun -np ${processor} rangular_mpi <<EOF
y
EOF

if [ $loop -eq 1 ]; then
### 3.rwfnestimate
rwfnestimate << EOF
y
1
${loop1_rwfn_file}
*
2
*
3
*
EOF

### 4. rmcdhf - 首次循环使用特定轨道参数
orbital_params="10s,9p,8d,7f,6g"
else
### 3.rwfnestimate (与上面相同)
rwfnestimate << EOF
y
1
${loop1_rwfn_file}
*
2
*
3
*
EOF

### 4. rmcdhf - 后续循环不使用轨道参数
orbital_params=""
fi

### 4. rmcdhf (统一执行)
mpirun -np ${processor} rmcdhf_mem_mpi <<EOF
y
1-4
5
${orbital_params}

100
EOF
### 5. rsave
rsave ${conf}_${loop}

if [ $loop -eq 1 ]; then
cp ${conf}_${loop}.w ..
fi
# # 5. rci
# mpirun -np ${processor} rci_mpi <<EOF
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

### 6. jj2lsj rmcdhf
jj2lsj << EOF
${conf}_${loop}
n
y
y
EOF
# ### 6. jj2lsj rci
# jj2lsj << EOF
# ${conf}_${loop}
# y
# y
# y
# EOF
### 7. generate energy levels data file
rlevels ${conf}_${loop}.m > ${conf}_${loop}.level # rmcdhf
# rlevels ${conf}_${loop}.cm > ${conf}_${loop}.level # rci
cd ..

## 机器学习训练
echo "================执行机器学习训练================"
python train.py 
if [ $? -ne 0 ]; then
    echo "机器学习训练失败!"
    exit 1
fi

done