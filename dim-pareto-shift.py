import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from tqdm import tqdm
from paretoset import paretoset

import model_functions as fmodel

# dims = np.array(range(8,64, 8))
vlBs = np.array([128, 256, 512])/8 
kls = np.array([1,2,4])
dims = np.array(range(8,64,16)) + 1
fig, axs = plt.subplots(figsize=(8,6))
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(vmin=dims.min(), vmax=dims.max())

for dim in dims:
    perf_model = fmodel.init_pm(
                        vlB=vlBs,
                        kl=kls,
                        width_datapath=np.array([1/2, 1]),
                        M=np.array([dim]),
                        N=np.array([dim]),
                        K=np.array([dim])
                        )
    perf_model = perf_model[perf_model['ops_cycle'] >= 1/3e-2]
    perf_model = perf_model[perf_model['opu_area'] <= 3e6]
    pm_pareto = perf_model[['ops_cycle','opu_area']]
    mask = paretoset(pm_pareto, sense=['max', 'min'])
    opc = perf_model.loc[mask,'ops_cycle']
    opu_area = perf_model.loc[mask,'opu_area']
    
    # Sort by area from smallest to largest
    sorted_indices = np.argsort(opu_area)
    opc_sorted = opc.iloc[sorted_indices]
    opu_area_sorted = opu_area.iloc[sorted_indices]
    
    axs.plot(1/opc_sorted, opu_area_sorted, color=cmap(norm(dim)))
 
axs.legend([f'vl={dim}' for dim in dims])
axs.set_xlabel('Throughput [cycles per operation]')
axs.set_ylabel('Area [um^2]')

# Save the plot to PNG file
plt.savefig('dim_pareto_shift.png', dpi=300, bbox_inches='tight')
