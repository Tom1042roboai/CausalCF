"""
Plots bar graph for the Pushing task in Component Testing.
Refer to paper.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Values from bar graphs in the ./....SameTask_Generalize folders.
no_intervene_full_int_fs = [0.98, 0.98, 0.98, 0.92, 0.58, 0.17, 0.02, 0.1, 0.03, 0.03, 0.31, 0.09]
intervene_full_int_fs = [0.97, 0.97, 0.97, 0.76, 0.52, 0.94, 0.11, 0.49, 0.16, 0.16, 0.48, 0.15]
CF_intervene_full_int_fs = [0.98, 0.98, 0.97, 0.96, 0.48, 0.98, 0.19, 0.44, 0.17, 0.15, 0.58, 0.34]
CF_intervene_iter_full_int_fs = [0.98, 0.98, 0.98, 0.67, 0.54, 0.98, 0.22, 0.76, 0.28, 0.2, 0.67, 0.35]

bar_width = 0.15
bar_no_inter = np.arange(len(no_intervene_full_int_fs))
bar_inter = [x + bar_width for x in bar_no_inter]
bar_CF_inter = [x + bar_width for x in bar_inter]
bar_CF_inter_iter = [x + bar_width for x in bar_CF_inter]

plt.bar(bar_no_inter, no_intervene_full_int_fs, width=bar_width, label='no_intervene')
plt.bar(bar_inter, intervene_full_int_fs, width=bar_width, label='Intervene')
plt.bar(bar_CF_inter, CF_intervene_full_int_fs, width=bar_width, label='Counterfactual + Intervene')
plt.bar(bar_CF_inter_iter, CF_intervene_iter_full_int_fs, width=bar_width, label='CausalCF (iter)')

plt.title('Evaluation performance for pushing task')
plt.xlabel('Evaluation protocols')
plt.ylabel('Full integrated fractional success')
plt.xticks([r + bar_width for r in range(len(bar_no_inter))],
            ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11'])
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', fancybox=True)
plt.tight_layout()
plt.savefig('./Pushing_Eval_bar.png')
