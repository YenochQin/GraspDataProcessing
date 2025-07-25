#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
使用示例：测试能级图绘制程序
'''

import sys
from pathlib import Path

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from plot_energy_levels import EnergyLevelPlotter

# 示例用法
if __name__ == "__main__":
    # 文件路径列表（你可以根据实际情况修改）
    file_paths = [
        str(current_dir / "cv6odd1_j5as5_19.level_level.csv"),
        # 添加更多文件路径...
    ]
    
    # 创建绘图器
    plotter = EnergyLevelPlotter()
    
    # 你可以修改目标组态列表
    plotter.target_configurations = [
        "5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(7)8S0_8S.5d_9D",
        "5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(7)8S0_8S.5d_7D",
        "5s(2).4d(10)1S0_1S.5p(6).6s_2S.4f(7)8S0_9S.5d(2)3F2_1F"
    ]
    
    print("正在加载数据...")
    data_dict = plotter.load_data(file_paths)
    
    if data_dict:
        print("正在处理数据...")
        processed_data = plotter.filter_and_sort_data(data_dict)
        
        print("正在创建图形...")
        fig = plotter.create_comparison_plot(
            processed_data, 
            output_path=str(current_dir / "energy_levels_comparison.png")
        )
        
        # 显示图形
        import matplotlib.pyplot as plt
        plt.show()
        
        print("完成!")
    else:
        print("未找到有效的数据文件")