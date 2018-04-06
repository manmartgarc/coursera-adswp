import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm

np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                    index=[1992,1993,1994,1995])

df = df.T

means = df.describe().loc['mean']
stds = df.describe().loc['std']
stes = stds / len(df) ** 0.5


fig = plt.figure(frameon=False)
ax = fig.add_subplot(1,1,1)
bars = ax.bar(means.index, means, yerr=stes)

threshold = 42000


# %%

colormap = mcol.LinearSegmentedColormap.from_list("blue_white_red",
                                                  ['#032f3c', "w", "#FFDB58"])
cm_mappable = cm.ScalarMappable(cmap=colormap)
cm_mappable.set_array([])

parameters = []
for bar, ste in zip(bars, stes):
    height = bar.get_height()
    low_bound = height - ste
    high_bound = height + ste
    parameter = (high_bound - threshold) / (high_bound - low_bound)
    if parameter > 1:
        parameter = 1
    elif parameter < 0:
        parameter = 0
    parameters.append(parameter)    
    
cm_mappable.to_rgba([0,1], alpha=0.8)

bars = ax.bar(means.index,
              means,
              yerr=stes,
              color=cm_mappable.to_rgba(parameters),
              alpha=0.8)

# Threshold Bar
ax.axhline(threshold, color='gray', alpha=0.8)

plt.colorbar(cm_mappable)

plt.xticks(means.index, alpha=0.8)
plt.yticks(alpha=0.8)
