"""
Plots training performance for the Picking task in Component Testing.
Refer to paper.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# To change to curr dir of python script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

file_path_Inter = "./0_Intervene_monitor_pick.csv"
file_path_CFRL = "./0_CFRL_monitor_push_pick.csv"

df_Intervene = pd.read_csv(file_path_Inter, skiprows=2, names=['Reward', 'Len_of_eps', 'Time_elapsed', 'Frac_success'])
df_CFRL = pd.read_csv(file_path_CFRL, skiprows=2, names=['Reward', 'Len_of_eps', 'Time_elapsed', 'Frac_success'])
# Print first 5 rows of df
# print(df_noIntervene.Frac_success.head)
np_timesteps_Inter = np.arange(834, 7000000, 834)
np_timesteps_CFRL = np.arange(834, 7000000, 834)
# print(len(np_timesteps_noInter))
#print(np_timesteps_noInter[:5])
ind_timesteps_Inter = len(np_timesteps_Inter) # len = 8393 episodes
ind_timesteps_CFRL = len(np_timesteps_CFRL) # len = 8393 episodes

# Get only that portion of timesteps up till a little pass 7000 000
df_Intervene = df_Intervene[:ind_timesteps_Inter]
df_Intervene['Timesteps'] = np_timesteps_Inter
df_CFRL = df_CFRL[:ind_timesteps_CFRL]
df_CFRL['Timesteps'] = np_timesteps_CFRL
# print(df_noIntervene.head)

# Get mean of fractional success for every 100 episodes or 83 400 time steps, because graph have rapid changes.
np_frac_success_Inter = np.array(df_Intervene.Frac_success)
np_frac_success_CFRL = np.array(df_CFRL.Frac_success)
lower_index = 0
upper_index = 100
np_mean_100eps_FS_Inter = np.zeros(84)
np_mean_100eps_FS_CFRL = np.zeros(84)
for i in range(84):
    if upper_index < 8393:
        np_mean_100eps_FS_Inter[i] = np.mean(np_frac_success_Inter[lower_index:upper_index])
        np_mean_100eps_FS_CFRL[i] = np.mean(np_frac_success_CFRL[lower_index:upper_index])
    else:
        np_mean_100eps_FS_Inter[i] = np.mean(np_frac_success_Inter[lower_index:])
        np_mean_100eps_FS_CFRL[i] = np.mean(np_frac_success_CFRL[lower_index:])

    lower_index += 100
    upper_index += 100

np_timesteps_mean_FS = np.arange(83400, 7089000, 83400)
# print(np_mean_100eps_FS_Inter[:5])
# print(len(np_mean_100eps_FS_Inter))
# print(len(np_timesteps_mean_FS))

# Plot graph
plt.plot(np_timesteps_mean_FS, np_mean_100eps_FS_Inter)
plt.plot(np_timesteps_mean_FS, np_mean_100eps_FS_CFRL)
plt.title('Picking (train)', fontsize=10)
plt.xlabel('Time steps')
plt.ylabel('Mean fractional success (100 eps)')
plt.legend(['Intervene', 'transfer Causal_rep + Intervene'], bbox_to_anchor=(0.5, -0.15), loc='upper center', fancybox=True)
plt.tight_layout()
plt.savefig('./Picking_training.png')