# AICS_2022_CausalCF
Causal Counterfactuals for Improving the Robustness of Reinforcement Learning

## Installation
Recommended to use a virtual environment. Compatibility is best on Linux. If using Windows OS can install WSL2.
1. Clone the github repo and cd into folder:
```bash
git clone https://github.com/Tom1042roboai/AICS_2022_CausalCF
cd AICS_2022_CausalCF
```
2. create virtual environment using conda:  
```bash
conda create --name causalCF python=3.7.4
```
3. Activate environment and install libraries:
```bash
conda activate causalCF
pip install -r requirements.txt
```
4. Install Pytorch CUDA libraries:
- Pytorch 1.8.0 was used for CausalCF.
- Search the version of cudatoolkit that is compatible with your GPU. (e.g. RTX 3080 uses sm_86 and will need CUDA>=11.1)
- Go to the Pytorch website and look for the corresponding version and install using the commands on the website: https://pytorch.org/get-started/previous-versions/

## Changes to CausalWorld Files
CausalCF modified relevant CausalWorld files to enable the use and update of a causal representation. The modified files are in the "AICS_2022_CausalCF/Changes_CausalWorld" folder. The corresponding original CausalWorld files have to be replaced with these files to run the experiments. The specific modifications made to the original files are described in the code files itself.
1. Locate the CausalWorld library installed in your virtual environment, the path will be something like this: (Replace "userName" with your own system's one)
```bash
cd /home/userName/anaconda3/envs/causalCF/lib/python3.7/site-packages/causal_world
```
2. Move the modified causalworld.py file over from the path where AICS_2022_CausalCF was stored:
```bash
mv pathToFolder/AICS_2022_CausalCF/Changes_CausalWorld/causalworld.py ./envs
```
3. Move the task files over from the path where AICS_2022_CausalCF was stored:
```bash
mv pathToFolder/AICS_2022_CausalCF/Changes_CausalWorld/* ./task_generators
```

## Setup Folders
Need to setup the folders for storing the models and causal representations for the different solutions in the experiment.
1. Create new folders where the models and causal representations will be stored for the experiments:
- CausalCF_iter.py
- Component_3_Counterfactual_Intervene.py
- Component_2_Intervene.py
- Component_1_no_Intervene.py
- transfer_Causal_rep_Intervene_train.py
2. Change the variables "model_save_path" and if applicable "CausalRep_path" to the path of the folder you created.

## Run CausalCF experiments:
To run any of the experiment files just e.g.:
```bash
python CausalCF_iter.py
```
When the experiments are complete, you may check the "log_dir" folder for all the training results. The csv file stores the results for Agent training and "log_train.txt" stores the results for Counterfactual training. You can also check the models and the causal representations saved in the folder(s) you created.

## Run CausalCF evaluation
Before running the evaluation files, remember to change the variables for model paths and CausalRep paths. Run the evaluation files like the experiment files.

