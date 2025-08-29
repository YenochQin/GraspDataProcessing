"""
图表样式设置模块

提供统一的图表样式配置，适用于科学发表。
现在使用英文标签，无需中文字体支持。
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

def configure_matplotlib_for_publication():
    """
    配置matplotlib用于科学发表的图表样式
    
    Returns:
        bool: 配置是否成功
    """
    try:
        # 高质量图形设置
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.1
        
        # 字体设置（英文）
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
        
        # 线条和标记设置
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['lines.markersize'] = 6
        plt.rcParams['patch.linewidth'] = 0.5
        
        # 坐标轴设置
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linewidth'] = 0.5
        
        # 负号显示修复
        plt.rcParams['axes.unicode_minus'] = False
        
        # 图例设置
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.framealpha'] = 0.9
        plt.rcParams['legend.fancybox'] = True
        plt.rcParams['legend.shadow'] = False
        
        # 刻度设置
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.major.size'] = 3
        plt.rcParams['ytick.major.size'] = 3
        plt.rcParams['xtick.minor.size'] = 1.5
        plt.rcParams['ytick.minor.size'] = 1.5
        
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to configure matplotlib: {e}")
        return False

def disable_font_warnings():
    """
    禁用matplotlib字体相关警告
    作为备用方案，在字体配置失败时使用
    """
    try:
        # 禁用字体警告
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        warnings.filterwarnings('ignore', message='.*font.*')
        warnings.filterwarnings('ignore', message='.*Glyph.*missing.*')
        
        # 设置matplotlib日志级别
        import logging
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.ERROR)
        
        return True
        
    except Exception as e:
        print(f"Warning: Could not disable font warnings: {e}")
        return False

# 自动配置matplotlib（当模块被导入时）
if __name__ != "__main__":
    success = configure_matplotlib_for_publication()
    if not success:
        disable_font_warnings()
