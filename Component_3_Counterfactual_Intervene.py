"""
CausalCF solution
This solution does not have the additional component of ...
where the RL agent can iterate between counterfactual and ...
agent training.

"""

from CF_model import CFNet
from torch import optim
from torch.utils.data import DataLoader
import torch
import numpy as np
# import argparse
import ipdb
import os
from tqdm import *
from random import choice
import torch.nn.functional as F
import time
import tensorflow as tf
# from dataloaders.utils import *
from causal_world.task_generators import generate_task
from causal_world.envs.causalworld import CausalWorld
from causal_world.actors.pushing_policy import PushingActorPolicy
from causal_world.actors.stacking2_policy import Stacking2ActorPolicy
from causal_world.intervention_actors import GoalInterventionActorPolicy
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper
from stable_baselines import PPO2, SAC
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common.callbacks import CheckpointCallback

def Calc_num_objects(env_obs):
    """
    Calculates number of objects in observation.
    Used for generating the new representation to be processed by CF_model.

        Parameters:
            env_obs: structured observation from CausalWorld
        
        Returns:
            num_of_objects: number of objects in the observation
    """
    len_obs = env_obs.shape[0]
    # print("length of obs: ", len_obs)
    num_of_objects = int(len_obs/28) - 1
    # print("num of objects: ", num_of_objects)
    return num_of_objects

def Convert_input_shape(stack_obs, timesteps, num_of_objects):
    """
    Converts the temporally stacked structured observations to a new...
    representation that can be processed by CF_model.
    It allows the CF_model to be adaptable to a varying amount of ...
    timesteps and objects.

        Parameters:
            stack_obs: temporally stacked structured observations
            timesteps: number of timesteps to perform counterfactuals for
            num_of_objects: number of objects in the observation
        
        Returns:
            desired_input_obs: new representation for CF_model
    """
    # 28 len vector for T,R1,R2,R3 in struct obs of CausalWorld and ...
    # Another 28 for each object (object features, partial goal features) ...
    # for struct obs
    desired_input_obs = np.empty([timesteps, num_of_objects, 28 + 28])
    lens_obs = stack_obs.shape[1]

    for t in range(timesteps):
        for k in range(num_of_objects):
            # get T, R1, R2, R3 which is same for all objects
            desired_input_obs[t, k, :28] = stack_obs[t, :28]
            # Get features for each object 17 dim
            obj_index_lower = 28 + (k * 17)
            obj_index_upper = obj_index_lower + 17
            desired_input_obs[t, k, 28:45] = stack_obs[t, obj_index_lower:obj_index_upper]
            # Get partial goal feature of each object 11 dim, concat after the object features
            part_goal_ind_low = (28 + (num_of_objects*17)) + (k*11)
            part_goal_ind_up = part_goal_ind_low + 11
            if part_goal_ind_up >= lens_obs:
                # print("convert input shape part_goal_ind_low: ", part_goal_ind_low)
                desired_input_obs[t, k, 45:] = stack_obs[t, part_goal_ind_low:]
            else:
                desired_input_obs[t, k, 45:] = stack_obs[t, part_goal_ind_low:part_goal_ind_up]


    return desired_input_obs

def Calc_loss(pred_obs_d, actual_obs_d):
    """
    Function calculates loss for CF_model prediction.

        Parameters:
            pred_obs_d: Counterfactual prediction by CF_model for every timestep
            actual_obs_d: Actual observation in CausalWorld environment for every timestep

        Returns:
            mse_3d: The mean squared error between actual observation and prediction
    """
    total_num_features = pred_obs_d.shape[1] * pred_obs_d.shape[2] * pred_obs_d.shape[3]
    mse_3d = torch.sum(((pred_obs_d - actual_obs_d) ** 2).mean(-1)) / total_num_features  # (B,T-1,K,56)
    return mse_3d

def Pretrain_CF_Model(device= None, env=None, RL_agent=None, initial_obs=None, num_of_objects=1, CF_model=None, print_freq=5, list_mse_3d=None, num_timesteps=30, max_iter=40):
    """
    Function pretrains the CF_Model using an existing RL baseline provided by CausalWorld.

        Parameters:
            device: Using either cpu or gpu option
            env: Created CausalWorld environment for running simulation
            RL_agent: Pretrained RL baseline provided by CausalWorld
            initial_obs: Initial observation that the training data will be generated from, using the RL agent
            num_of_objects: Number of objects dealt with in the task
            CF_model: Supervised model to be trained
            print_freq: How frequent the loss should be printed (in iterations)
            list_mse_3d: List for storing the mse_3d loss
            num_timesteps: Number of timesteps to perform Counterfactuals for into the future
            max_iter: Maximum number of iterations to train CF_model for

        Returns:
            list_mse_3d: Loss values for CF_model during Pretrain Stage (to be written to text file)
            CF_causal_rep: Trained causal representation from CF_model
    """
    optimizer = optim.Adam(CF_model.parameters(), lr=1e-3)

    for curr_iter in range(max_iter):
        start_iter_time = time.time()
        stack_input_obs = np.empty(28 + (num_of_objects * 28))

        # Obtain sequence of observations to train CF model
        action = RL_agent.act(initial_obs)
        # print("agent action: ", action)
        for i in range(num_timesteps):
            next_obs, _, _, _ = env.step(action)
            action = RL_agent.act(next_obs)
            if i == 0:
                stack_input_obs = np.expand_dims(next_obs, axis=0)
            elif i > 0:
                stack_input_obs = np.concatenate((stack_input_obs, np.expand_dims(next_obs, axis=0)), axis=0)
        # print("stack input obs: ", stack_input_obs)

        # Get observations ab for CF model
        desired_input_obs = Convert_input_shape(stack_input_obs, num_timesteps, num_of_objects)
        tensor_input_obs_ab = torch.from_numpy(desired_input_obs).to(device)
        # print("tensor input obs ab: ", tensor_input_obs_ab.shape)

        # Get observations c for CF model
        goal_intervention_dict = env.sample_new_goal()
        success_signal, intervene_obs_c = env.do_intervention(goal_intervention_dict)
        print("Goal Intervention for obs c success signal", success_signal)
        action = RL_agent.act(intervene_obs_c)
        intervene_obs_c = np.expand_dims(intervene_obs_c, axis=0)
        desired_input_obs = Convert_input_shape(intervene_obs_c, 1, num_of_objects)
        tensor_input_obs_c = torch.from_numpy(desired_input_obs).to(device)
        # print("tensor input obs c: ", tensor_input_obs_c.shape)

        # Send to CF model
        CF_model_out, CF_model_stab, CF_causal_rep = CF_model.forward(tensor_input_obs_ab, tensor_input_obs_c)

        # Testing the Calc_loss function
        stack_obs_cd = np.empty(28 + (num_of_objects * 28))
        # print("env sample action: ", env.action_space.sample()) # To check true shape of action
        # print("agent action: ", action)
        for i in range(num_timesteps-1):
            next_obs, _, _, _ = env.step(action)
            action = RL_agent.act(next_obs)
            # print("agent action: ", action)
            if i == 0:
                stack_obs_cd = np.expand_dims(next_obs, axis=0)
            elif i > 0:
                stack_obs_cd = np.concatenate((stack_obs_cd, np.expand_dims(next_obs, axis=0)), axis=0)
        # print("stack obs cd shape: ", stack_obs_cd.shape)
        desired_input_obs = Convert_input_shape(stack_obs_cd, num_timesteps-1, num_of_objects)
        tensor_actual_obs_d = torch.from_numpy(desired_input_obs)
        # print("tensor actual obs d: ", tensor_actual_obs_d.shape)
        actual_obs_d = tensor_actual_obs_d.unsqueeze(0).to(device)
        # print("actual obs d: ", actual_obs_d.shape)

        # loss
        mse_3d = Calc_loss(CF_model_out, actual_obs_d)

        # backprop
        optimizer.zero_grad()
        mse_3d.backward()
        optimizer.step()

        goal_intervention_dict = env.sample_new_goal()
        success_signal, obs = env.do_intervention(goal_intervention_dict)
        print("Goal Intervention success signal", success_signal)

        end_iter_time = time.time()
        iter_time = end_iter_time - start_iter_time

        print("Current iter: ", curr_iter)
        print("Max iter: ", max_iter)
        print(f"time for this iter: {iter_time:.3f}")
        if curr_iter % print_freq == 0:
            print(f"curr iter mse loss: {mse_3d:.6f}")
            list_mse_3d.append(mse_3d.item())
            print(f"Mean of mse_3d over {print_freq} iters: {np.mean(list_mse_3d):.6f}")

        initial_obs = env.reset()

    return list_mse_3d, CF_causal_rep

def main():
    # init gpu for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs_loader = {'batch_size': 32}
    if device.type == 'cuda':
        kwargs_loader.update({'num_workers': 10, 'pin_memory': True})

    # init RL environment and model
    task = generate_task(task_generator_id="pushing", 
                        variables_space='space_a',
                        dense_reward_weights=np.array([750,250,0]))
    env = CausalWorld(task=task, skip_frame = 3, enable_visualization=False)
    pretrained_RL_agent = PushingActorPolicy()

    # Init and calc some values
    obs = env.reset()
    num_of_objects = Calc_num_objects(obs)
    CF_model = CFNet(num_objects=num_of_objects).to(device)

    # Pre-training of CF model with pretrained RL agent
    print_freq = 5
    log_file_train = "./log_dir/log_train.txt"
    list_mse_3d = []
    num_timesteps = 30
    max_iter = 40
    num_epochs_mass = 3
    num_epochs_goal = 5
    epoch_counter = 0

    CF_model.train()
    for _ in range(num_epochs_mass):
        mass = np.random.uniform(0.015, 0.045, [1,]) # space a
        success_signal, obs = env.do_intervention({'tool_block': {'mass': mass}})
        print("Mass Intervention for CF env success signal", success_signal)
        for _ in range(num_epochs_goal):
            goal_intervention_dict = env.sample_new_goal()
            success_signal, obs = env.do_intervention(goal_intervention_dict)
            print("Goal Intervention for CF env success signal", success_signal)
            list_mse_3d, CF_causal_rep = Pretrain_CF_Model(device=device, env=env, RL_agent=pretrained_RL_agent, initial_obs=obs, 
                                                            num_of_objects=num_of_objects, CF_model=CF_model, print_freq=print_freq,
                                                            list_mse_3d=list_mse_3d, num_timesteps=num_timesteps, max_iter=max_iter)

            epoch_counter += 1
            with open(log_file_train, "+a") as f:
                f.write(f"Mean of mse_3d over {epoch_counter} epoch: {np.mean(list_mse_3d):.6f} \n")

    # TRY CONCAT CAUSAL REP TO OBS FOR TRAIN RL AGENT
    numpy_causal_rep = CF_causal_rep.detach().cpu().numpy()
    numpy_causal_rep = np.squeeze(numpy_causal_rep, axis=0)
    #print("numpy_causal_rep shape: ", numpy_causal_rep.shape)
    # Save CausalRep to use in evalution
    CausalRep_path = "./trained_CFRL_model"
    CausalRep_name = "CausalRep_" + str(1)
    np.save(os.path.join(CausalRep_path, CausalRep_name), numpy_causal_rep)
    # For passing to environment
    min_val_causal_rep = np.amin(numpy_causal_rep)
    max_val_causal_rep = np.amax(numpy_causal_rep)
    shape_causal_rep = numpy_causal_rep.shape
    #print(f"low of CRep= {min_val_causal_rep} and high of CRep= {max_val_causal_rep}")

    # Delete the env that has been created first: can also use del(env)
    env.close()


    # Train Causal RL model
    log_relative_path = "./log_dir"
    model_save_path = "./trained_CFRL_model"
    file_prefix = "0_CFRL"

    task = generate_task(task_generator_id="pushing", variables_space='space_a',
                            dense_reward_weights=np.array([750,250,0]), mode=1, causal_rep=numpy_causal_rep,
                            low_cRep=min_val_causal_rep, high_cRep=max_val_causal_rep,
                            shape_cRep=shape_causal_rep)
    env = CausalWorld(task=task, skip_frame = 3, enable_visualization=False)
    # use curriculum learning to make interventions on goal during model.learn
    # actives = (episode_start, episode_end, episode_periodicity, time_step_for_intervention)
    # episode end just has to be bigger than the number of training episodes otherwise intervention stops
    env = CurriculumWrapper(env,
                            intervention_actors=[GoalInterventionActorPolicy()],
                            actives=[(0, 1e9, 1, 0)])
    # Monitor training of agent used in addition with checkpoint callback 
    monitor_file = os.path.join(log_relative_path, file_prefix)
    env = Monitor(env, filename=monitor_file, info_keywords=('fractional_success',))

    # Parameters for RL model
    policy_kwargs = dict(layers=[256, 256])
    sac_config = {
        "gamma": 0.95,
        "tau": 1e-3,
        "ent_coef": 1e-3,
        "target_entropy": 'auto',
        "learning_rate":  1e-4,
        "buffer_size": 1000000,
        "learning_starts": 1000,
        "batch_size": 256
    }

    # Create RL model with parameters and train
    RL_model = SAC(SACMlpPolicy, env, policy_kwargs=policy_kwargs, verbose=1, **sac_config, tensorboard_log=log_relative_path)

    # Init some parameters for training
    total_time_steps=7000000
    validate_every_timesteps=500000
    ckpt_freq = validate_every_timesteps
    checkpoint_callback = CheckpointCallback(save_freq=ckpt_freq, save_path=model_save_path, name_prefix='model_CFRL')
    # list_mse_3d = []

    # Training
    RL_model.learn(int(total_time_steps), callback=checkpoint_callback, reset_num_timesteps=False)
    RL_model.save(save_path=os.path.join(model_save_path, 'CFRL_model_{}_steps'.format(total_time_steps)))

    env.close()

if __name__ == "__main__":
    main()
