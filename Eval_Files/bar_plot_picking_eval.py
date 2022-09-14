"""
Plots bar graph for the Picking task in CausalRep Transfer.
Refer to paper.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Values from bar graphs in the ./....Picking_Generalize folders.
intervene_full_int_fs = [0.97, 0.97, 0.41, 0.91, 0.22, 0.90, 0.01, 0.24, 0.01, 0.0, 0.31, 0.02]
CF_intervene_full_int_fs = [0.98, 0.98, 0.43, 0.96, 0.22, 0.94, 0.0, 0.19, 0.0, 0.01, 0.27, 0.04]

bar_width = 0.25
bar_inter = np.arange(len(intervene_full_int_fs))
bar_CF_inter = [x + bar_width for x in bar_inter]

plt.bar(bar_inter, intervene_full_int_fs, width=bar_width, label='Intervene')
plt.bar(bar_CF_inter, CF_intervene_full_int_fs, width=bar_width, label='transfer Causal_rep + Intervene')

plt.title('Evaluation performance for picking task')
plt.xlabel('Evaluation protocols')
plt.ylabel('Full integrated fractional success')
plt.xticks([r + bar_width for r in range(len(bar_inter))],
            ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11'])
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', fancybox=True)
plt.tight_layout()
plt.savefig('./Picking_Eval_bar.png')
