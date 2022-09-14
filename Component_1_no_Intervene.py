"""
no_Intervene solution
This tests the baseline performance of the SAC model and ...
uses no interventions and no counterfactuals.
"""
from operator import mod
import tensorflow as tf
import os
import numpy as np
from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from stable_baselines import PPO2, SAC
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SACMlpPolicy
from stable_baselines.bench.monitor import Monitor
from stable_baselines.common.callbacks import CheckpointCallback

def main():
    log_relative_path = "./log_dir"
    # Create new folder you want to save models in.
    model_save_path = "./trained_noIntervene_model"
    file_prefix = "0_noIntervene"

    # init RL environment and model
    task = generate_task(task_generator_id="pushing",
                        variables_space='space_a',
                        dense_reward_weights=np.array([750,250,0]))
    env = CausalWorld(task=task, skip_frame = 3, enable_visualization=False)
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
    checkpoint_callback = CheckpointCallback(save_freq=ckpt_freq, save_path=model_save_path, name_prefix='model_noIntervene')

    # Training
    RL_model.learn(int(total_time_steps), callback=checkpoint_callback, reset_num_timesteps=False)
    RL_model.save(save_path=os.path.join(model_save_path, 'noIntervene_model_{}_steps'.format(total_time_steps)))
    env.close()

if __name__ == "__main__":
    main()
