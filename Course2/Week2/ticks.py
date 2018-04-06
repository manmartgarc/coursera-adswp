# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 19:36:42 2018

@author: manma
"""

#%%

import matplotlib.pyplot as plt
import numpy as np

plt.figure()

languages =['Python', 'SQL', 'Java', 'C++', 'JavaScript']
pos = np.arange(len(languages))
popularity = [56, 39, 34, 34, 29]

bars = plt.bar(pos, popularity, align='center', 
               color='lightslategrey', linewidth=0)
bars[0].set_color('#42003F')


plt.xticks(pos, languages, alpha=0.5)
plt.tick_params(top=False,
                bottom=False,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=True)
plt.gca().set_frame_on(False)

rects = bars.patches

for bar in bars:
    plt.gca().text(bar.get_x() + bar.get_width()/2,
            bar.get_height() - 5, 
            str(int(bar.get_height())) + '%',
            ha='center', color='w', fontsize=11)


plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow',
          alpha=0.8)

#TODO: remove all the ticks (both axes), and tick labels on the Y axis

plt.show()