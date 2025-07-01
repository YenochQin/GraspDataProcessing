# GRASP计算脚本生成器

## 用户输入参数

|  | 参数  | 参数意义 |
| :------------ |:---------------| :----- |
| 1. | atom | 计算原子 |
| 2. | conf | 目标原子的组态名（电子关联+奇偶宇称+活性空间_能级数） |
| 3. | spetral_term | 需要计算出的能级 |
| 4. | target_pool_file | 需要被选择优化的CSFs组态 |
| 5. | root_path | 计算target_pool_file的根目录 |
| 6. | selected_csfs_file | 初筛的CSFs组态 |
| 7. | conda_path | conda的安装位置 |
| 8. | Active_space | 活性空间轨道 |
| 9. | partition | sbatch计算分区(使用sinfo获得) |
| 10. | cal_levels | 计算的能级 |

## 合成参数
|  | 参数  | 参数意义 |
| :------------ |:---------------| :----- |
| 1. | job_name | sbatch -J = atom + conf |
| 2. | loop1_rwfn_file | 第一轮计算的轨道镜像波函数*.w = target_pool_file + .w |
| 3. | rwfnestimate_file | 后续计算使用的轨道镜像波函数*.w = conf + _1.w |

