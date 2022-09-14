"""
Plots training performance of CF_model during Counterfactual training.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# To change to curr dir of python script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

file_log_train = "./log_train_iter_cleaned.txt"

df_log_train = pd.read_csv(file_log_train, sep=":", skiprows=1, names=['Metric', 'Epoch_num', 'mse_value'])
# Print first 5 rows of df
#print(df_log_train.head)

# Plot graph
plt.plot(df_log_train.Epoch_num, df_log_train.mse_value)
plt.title('CF_model (iter) train on Pushing', fontsize=10)
plt.xlabel('Epoch number')
plt.ylabel('Mean of mse_3d')
plt.legend(['Mean_mse_3d_epochs'], bbox_to_anchor=(0.5, -0.15), loc='upper center', fancybox=True)
plt.tight_layout()
plt.savefig('./CF_model_iter_Push_train.png')