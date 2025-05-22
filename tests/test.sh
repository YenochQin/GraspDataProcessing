#!/bin/zsh
#SBATCH -J GdIoddImlcias3
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -p batch
#SBATCH --output=as3oddImlci.log
. /usr/share/Modules/init/zsh
###########################################

processor=32
mkdisks ${processor} caltmp

###########################################
rm run.input
echo "True" > run.input
###########################################

## module load
module load mpi/openmpi-x86_64
###########################################

## configuration
conf="GdIoddImlcias3"
varied="3"
rwfnestimate_file="rwfnestimate.inp"

###########################################
loop=1
while true
do
###########################################
    # 检查文件是否存在
    if [[ ! -f "run.input" ]]; then
        echo "================等待输入文件...================"
        continue
    fi

    # 读取文件内容
    run_status=$(head -n 1 run.input | tr -d '[:space:]')

    # 检查退出条件
    if [[ "$run_status" == "False" ]]; then
        echo "================计算终止================"
        break
    elif [[ "$run_status" == "True" ]]; then
        echo "================下一步计算================"
        continue
    fi

    # 主程序逻辑
    echo "执行任务..."
    # 控制循环速度
    sleep 1
###########################################

## grasp calculation routine
mkdir ${conf}_${loop}
cd ${conf}_${loop}
### 1. rcsf
cp ../${conf}_${loop}.c rcsf.inp

### 2. rangular
mpirun -np ${processor} rangular_mpi <<EOF
y
EOF

### 3.rwfnestimate
rwfnestimate << EOF
y
1
${rwfnestimate_file}
*
2
*
3
*
EOF

### 4. rmcdhf
mpirun -np ${processor} rmcdhf_mem_mpi <<EOF
y
1
1-2
1-2
1-2
1-2
1
5


100
EOF

### 5. rsave
rsave ${conf}_${loop}

### 6. jj2lsj
jj2lsj << EOF
${conf}_${loop}
n
y
y
EOF

### 7. generate energy levels data file
rlevels ${conf}_${loop}.m > ${conf}_${loop}.level


done