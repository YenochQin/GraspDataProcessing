#! python3
# UTF-8
# Qin yi
# GRASP levels comparision figure and chart(latex-form)
# Version 1.0.0(11/1/2021)

import numpy as np
import level_compare as lc
from graspdatatransfer.fig_settings import set_size
import matplotlib.pyplot as plt


width = 426.79135

# load level data
level = lc.level_storage(atom, dir, parityas, max_as)
level = lc.level_diff(level, max_as)

# levels difference fig   
fig, ax = plt.subplots(1, 1, figsize=set_size(width))

ax = plt.gca()
ax.tick_params(direction='in')
ax.yaxis.grid(linestyle="--", alpha=0.15)
props = {'title': atom, 'xlabel': 'Level Number','ylabel':r'dE(cm$^{-1})$'} # 坐标系-坐标轴-刻度-标签
ax.set(**props)


style_dict = {
    'define':dict(s=15, facecolors='none')
}

marker_list = ['s', '^', 'o', 'h', '*']
for a_s in range(max_as):
    ax.scatter(level['No'], level[f'dE{a_s + 1}'].abs(), marker=marker_list[a_s], **style_dict['define'])

ax.legend((r'dE$_1$',r'dE$_2$',r'dE$_3$',r'dE$_4$'),frameon=True, loc='upper left',handlelength=2) # 图例

fig.savefig(f'{atom}.pdf', format='pdf', bbox_inches='tight')

