#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :plot_energy_levels.py
@date :2025/07/25 
@author :YenochQin (秦毅)
@description: 绘制多组能级对比图，支持垂直阶梯图和水平对比图
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import argparse
from typing import List, Dict, Tuple, Optional

# 尝试导入项目模块
try:
    sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
    import graspkit as gk
    from graspkit.data_IO.fig_settings import configure_matplotlib_for_publication
    use_project_style = True
except ImportError:
    print("Warning: 无法导入项目模块，使用默认matplotlib样式")
    use_project_style = False

class EnergyLevelPlotter:
    """能级图绘制器"""
    
    def __init__(self):
        # 配置matplotlib样式
        if use_project_style:
            configure_matplotlib_for_publication()
        else:
            plt.style.use('default')
        
        # 需要标注的组态列表（在代码中直接定义）
        self.target_configurations = [
            "5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(7)8S0_8S.5d_9D",
            "5s(2).4d(10)1S0_1S.5p(6).6s(2).4f(7)8S0_8S.5d_7D",
            "5s(2).4d(10)1S0_1S.5p(6).6s_2S.4f(7)8S0_9S.5d(2)3F2_1F"
        ]
        
        # 颜色方案
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def load_data(self, file_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """
        加载多个CSV文件的能级数据
        
        Args:
            file_paths: CSV文件路径列表
            
        Returns:
            Dict[str, pd.DataFrame]: 文件名到DataFrame的映射
        """
        data_dict = {}
        
        for file_path in file_paths:
            try:
                path_obj = Path(file_path)
                if not path_obj.exists():
                    print(f"Warning: 文件不存在: {file_path}")
                    continue
                
                df = pd.read_csv(file_path)
                
                # 验证必要的列
                required_columns = ['J', 'Parity', 'EnergyLevel', 'configuration']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"Warning: 文件 {file_path} 缺少列: {missing_columns}")
                    continue
                
                # 使用文件名（不含扩展名）作为标识
                file_key = path_obj.stem
                data_dict[file_key] = df
                print(f"成功加载数据: {file_key} ({len(df)} 个能级)")
                
            except Exception as e:
                print(f"Error: 加载文件 {file_path} 失败: {e}")
                continue
        
        return data_dict
    
    def filter_and_sort_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        过滤和排序数据
        
        Args:
            data_dict: 原始数据字典
            
        Returns:
            Dict[str, pd.DataFrame]: 处理后的数据字典
        """
        processed_dict = {}
        
        for file_key, df in data_dict.items():
            # 按能级升序排序
            df_sorted = df.sort_values('EnergyLevel').copy()
            
            # 添加标注标记
            df_sorted['should_label'] = df_sorted['configuration'].isin(self.target_configurations)
            
            processed_dict[file_key] = df_sorted
            
        return processed_dict
    
    def plot_vertical_step_diagram(self, ax, data_dict: Dict[str, pd.DataFrame], 
                                  x_positions: Dict[str, float]):
        """
        绘制垂直阶梯图
        
        Args:
            ax: matplotlib轴对象
            data_dict: 数据字典
            x_positions: 每组数据的x位置
        """
        max_energy = 0
        
        for i, (file_key, df) in enumerate(data_dict.items()):
            x_pos = x_positions[file_key]
            color = self.colors[i % len(self.colors)]
            
            # 绘制能级线
            for _, row in df.iterrows():
                energy = row['EnergyLevel']
                j_value = row['J']
                parity = row['Parity']
                
                # 绘制水平线
                line_width = 2.0 if row['should_label'] else 1.0
                alpha = 1.0 if row['should_label'] else 0.7
                
                ax.hlines(energy, x_pos - 0.3, x_pos + 0.3, 
                         colors=color, linewidth=line_width, alpha=alpha)
                
                # 标注J值和宇称
                ax.text(x_pos + 0.35, energy, f"{j_value}{parity}", 
                       fontsize=8, va='center', color=color)
                
                # 如果需要标注组态
                if row['should_label']:
                    # 简化组态标注（只显示主要部分）
                    config_short = self._simplify_configuration(row['configuration'])
                    ax.text(x_pos - 0.35, energy, config_short, 
                           fontsize=7, va='center', ha='right', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                max_energy = max(max_energy, energy)
        
        # 设置轴标签和标题
        ax.set_xlabel('Data Groups')
        ax.set_ylabel('Energy Level (cm⁻¹)')
        ax.set_title('Vertical Energy Level Step Diagram')
        
        # 设置x轴刻度
        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels(list(x_positions.keys()), rotation=45, ha='right')
        
        # 设置y轴范围
        ax.set_ylim(-max_energy * 0.05, max_energy * 1.05)
        
    def plot_horizontal_comparison(self, ax, data_dict: Dict[str, pd.DataFrame]):
        """
        绘制水平对比图（按J值对齐）
        
        Args:
            ax: matplotlib轴对象
            data_dict: 数据字典
        """
        # 收集所有J值
        all_j_values = set()
        for df in data_dict.values():
            all_j_values.update(df['J'].unique())
        all_j_values = sorted(all_j_values)
        
        # 为每个J值分配y位置
        j_positions = {j: i for i, j in enumerate(all_j_values)}
        
        max_energy = 0
        
        for i, (file_key, df) in enumerate(data_dict.items()):
            color = self.colors[i % len(self.colors)]
            
            # 按J值分组
            for j_value in all_j_values:
                j_levels = df[df['J'] == j_value]
                if len(j_levels) == 0:
                    continue
                
                y_base = j_positions[j_value]
                
                # 在同一J值内，按能量排序
                j_levels_sorted = j_levels.sort_values('EnergyLevel')
                
                for level_idx, (_, row) in enumerate(j_levels_sorted.iterrows()):
                    energy = row['EnergyLevel']
                    
                    # y位置稍作偏移以避免重叠
                    y_pos = y_base + (level_idx - len(j_levels_sorted)/2 + 0.5) * 0.1
                    
                    # 绘制垂直线
                    line_width = 2.0 if row['should_label'] else 1.0
                    alpha = 1.0 if row['should_label'] else 0.7
                    
                    ax.vlines(energy, y_pos - 0.3, y_pos + 0.3, 
                             colors=color, linewidth=line_width, alpha=alpha)
                    
                    # 如果需要标注
                    if row['should_label']:
                        parity = row['Parity']
                        ax.text(energy, y_pos + 0.35, f"{j_value}{parity}", 
                               fontsize=8, ha='center', color=color, rotation=90)
                    
                    max_energy = max(max_energy, energy)
        
        # 设置轴标签和标题
        ax.set_ylabel('J Values')
        ax.set_xlabel('Energy Level (cm⁻¹)')
        ax.set_title('Horizontal Energy Level Comparison (by J value)')
        
        # 设置y轴刻度
        ax.set_yticks(list(j_positions.values()))
        ax.set_yticklabels([f"J = {j}" for j in all_j_values])
        
        # 设置x轴范围
        ax.set_xlim(-max_energy * 0.05, max_energy * 1.05)
        
    def _simplify_configuration(self, config: str) -> str:
        """
        简化组态字符串用于显示
        
        Args:
            config: 完整组态字符串
            
        Returns:
            str: 简化的组态字符串
        """
        # 简单的简化策略：取最后一部分
        if '.' in config:
            parts = config.split('.')
            return parts[-1]
        return config[:20] + "..." if len(config) > 20 else config
    
    def create_comparison_plot(self, data_dict: Dict[str, pd.DataFrame], 
                              output_path: Optional[str] = None) -> plt.Figure:
        """
        创建完整的对比图
        
        Args:
            data_dict: 数据字典
            output_path: 输出文件路径（可选）
            
        Returns:
            plt.Figure: 图形对象
        """
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 为垂直图分配x位置
        x_positions = {file_key: i for i, file_key in enumerate(data_dict.keys())}
        
        # 绘制垂直阶梯图
        self.plot_vertical_step_diagram(ax1, data_dict, x_positions)
        
        # 绘制水平对比图
        self.plot_horizontal_comparison(ax2, data_dict)
        
        # 添加图例
        legend_elements = []
        for i, file_key in enumerate(data_dict.keys()):
            color = self.colors[i % len(self.colors)]
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=file_key))
        
        fig.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=len(data_dict), fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        # 保存图形
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存到: {output_path}")
        
        return fig

def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='绘制多组能级对比图')
    parser.add_argument('files', nargs='+', help='CSV文件路径列表')
    parser.add_argument('-o', '--output', type=str, help='输出图形文件路径')
    parser.add_argument('--show', action='store_true', help='显示图形')
    
    args = parser.parse_args()
    
    # 创建绘图器
    plotter = EnergyLevelPlotter()
    
    # 加载数据
    print("正在加载数据...")
    data_dict = plotter.load_data(args.files)
    
    if not data_dict:
        print("Error: 没有成功加载任何数据文件")
        return
    
    # 处理数据
    print("正在处理数据...")
    processed_data = plotter.filter_and_sort_data(data_dict)
    
    # 创建图形
    print("正在创建图形...")
    fig = plotter.create_comparison_plot(processed_data, args.output)
    
    # 显示图形
    if args.show:
        plt.show()
    
    print("完成!")

if __name__ == "__main__":
    main()