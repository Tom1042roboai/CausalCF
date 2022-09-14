"""
Evaluates the solutions that did not use a causal representation ...
in the CausalWorld evaluation pipeline.
The solutions tested with this file are: Intervene and no_Intervene.
"""
import os
import numpy as np
from causal_world.task_generators import generate_task
from causal_world.envs.causalworld import CausalWorld
from causal_world.evaluation import EvaluationPipeline
from causal_world.benchmark import PUSHING_BENCHMARK, PICKING_BENCHMARK
import causal_world.evaluation.visualization.visualiser as vis
from stable_baselines import PPO2, SAC

def main():
    # Path for own RL model
    model_path = "./trained_Intervene_models_Picking/model_Intervene_7000000_steps.zip"

    # Evaluation pipeline
    task_params = dict()
    task_params['task_generator_id'] = 'picking'
    world_params = dict()
    world_params['skip_frame'] = 3
    evaluation_protocols = PICKING_BENCHMARK['evaluation_protocols']
    evaluator_1 = EvaluationPipeline(evaluation_protocols=evaluation_protocols,
                                    task_params=task_params,
                                    world_params=world_params,
                                    visualize_evaluation=False)

    # Load model from model_path
    RL_model = SAC.load(model_path)

    def policy_fn(obs):
        return RL_model.predict(obs, deterministic=True)[0]
    
    scores_model = evaluator_1.evaluate_policy(policy_fn, fraction=1.0)
    experiments = dict()
    experiments['Intervene_SAC'] = scores_model
    vis.generate_visual_analysis('./log_dir/', experiments=experiments)

if __name__ == "__main__":
    main()