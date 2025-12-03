"""
MIT License

Copyright (c) 2024 Yorai Shaoul

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Standard imports.
# from torch_robotics.isaac_gym_envs.motion_planning_envs import PandaMotionPlanningIsaacGymEnv, MotionPlanningController

import os
import pickle
from datetime import datetime
import time
from math import ceil
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from typing import Tuple, List
import concurrent.futures

# Project imports.
from experiment_launcher import single_experiment_yaml, run_experiment
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory, CostConstraint
from mmd.models import TemporalUnet, UNET_DIM_MULTS
from mmd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mmd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn
from mmd.trainer import get_dataset, get_model
from mmd.utils.loading import load_params_from_yaml
from torch_robotics.robots import *
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints, \
    compute_average_acceleration, compute_average_acceleration_from_pos_vel, compute_path_length_from_pos
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer
from torch_robotics.robots.robot_planar_disk import RobotPlanarDisk
from mmd.planners.multi_agent import CBS, CBSMasked, PrioritizedPlanning
from mmd.planners.single_agent import MPD, MPDEnsemble
from mmd.common.constraints import MultiPointConstraint, VertexConstraint, EdgeConstraint
from mmd.common.conflicts import VertexConflict, PointConflict, EdgeConflict
from mmd.common.trajectory_utils import smooth_trajs, densify_trajs
from mmd.common import compute_collision_intensity, is_multi_agent_start_goal_states_valid, global_pad_paths, \
    get_start_goal_pos_circle, get_state_pos_column, get_start_goal_pos_boundary, get_start_goal_pos_random_in_env
from mmd.common.pretty_print import *
from mmd.config.mmd_params import MMDParams as params
from mmd.common.experiments import MultiAgentPlanningSingleTrialConfig, MultiAgentPlanningSingleTrialResult, \
    get_result_dir_from_trial_config, TrialSuccessStatus
from torch_robotics.environments import *
from mmd.utils.plot_trajs import create_trajectory_gif, create_masked_trajectory_gif
from typing import Any
import jax.numpy as jnp
import numpy as np

# Import from train_gnn using importlib to avoid modifying sys.path
import importlib.util
import sys as _sys
_train_gnn_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../models/train_gnn.py'))
_spec = importlib.util.spec_from_file_location("train_gnn", _train_gnn_path)
_train_gnn_module = importlib.util.module_from_spec(_spec)
_sys.modules["train_gnn"] = _train_gnn_module
_spec.loader.exec_module(_train_gnn_module)
load_trained_gnn_models = _train_gnn_module.load_trained_gnn_models
GNNSelectionNetwork = _train_gnn_module.GNNSelectionNetwork
del _train_gnn_path, _spec, _train_gnn_module, _sys

allow_ops_in_compiled_graph()

# Convert relative path to absolute path based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODELS_DIR = os.path.join(SCRIPT_DIR, '../../data_trained_models/')
TRAINED_MODELS_DIR = os.path.abspath(TRAINED_MODELS_DIR)
device = 'cuda'
device = get_torch_device(device)
tensor_args = {'device': device, 'dtype': torch.float32}

def get_player_masks(past_trajs: List[torch.Tensor], model, model_state):
    # convert past_trajs to jnp
    past_trajs_jnp = []
    for i in range(len(past_trajs)):
        if hasattr(past_trajs[i], 'detach'):
            arr = past_trajs[i].detach().cpu().numpy()
        else:
            arr = np.array(past_trajs[i])
        past_trajs_jnp.append(jnp.array(arr))

    # transpose to be (batch_size, T_observation, N_agents, state_dim)
    past_x_trajs = jnp.stack(past_trajs_jnp, axis=0)
    batch_past_x_trajs = past_x_trajs[None, ...]
    batch_past_x_trajs = batch_past_x_trajs.transpose(0, 2, 1, 3)

    # call model
    masks = model.apply({'params': model_state['params']}, batch_past_x_trajs, deterministic=True)
    masks = masks.squeeze(0)
    masks_np = np.array(masks)
    masks_torch = torch.from_numpy(masks_np).to(**tensor_args)
    return masks_torch

def create_whole_trajectory_planner(
    test_config: MultiAgentPlanningSingleTrialConfig,
):
    # ============================
    # Start time per agent.
    # ============================
    start_time_l = [i * test_config.stagger_start_time_dt for i in range(test_config.num_agents)]

    # ============================
    # Arguments for the high/low level planner.
    # ============================
    low_level_planner_model_args = {
        'planner_alg': 'mmd',
        'use_guide_on_extra_objects_only': params.use_guide_on_extra_objects_only,
        'n_samples': params.n_samples,
        'n_local_inference_noising_steps': params.n_local_inference_noising_steps,
        'n_local_inference_denoising_steps': params.n_local_inference_denoising_steps,
        'start_guide_steps_fraction': params.start_guide_steps_fraction,
        'n_guide_steps': params.n_guide_steps,
        'n_diffusion_steps_without_noise': params.n_diffusion_steps_without_noise,
        'weight_grad_cost_collision': params.weight_grad_cost_collision,
        'weight_grad_cost_smoothness': params.weight_grad_cost_smoothness,
        'weight_grad_cost_constraints': params.weight_grad_cost_constraints,
        'weight_grad_cost_soft_constraints': params.weight_grad_cost_soft_constraints,
        'factor_num_interpolated_points_for_collision': params.factor_num_interpolated_points_for_collision,
        'trajectory_duration': params.trajectory_duration,
        'device': params.device,
        'debug': params.debug,
        'seed': params.seed,
        'results_dir': params.results_dir,
        'trained_models_dir': TRAINED_MODELS_DIR,
    }
    high_level_planner_model_args = {
        'is_xcbs': True if test_config.multi_agent_planner_class in ["XECBS", "XCBS"] else False,
        'is_ecbs': True if test_config.multi_agent_planner_class in ["ECBS", "XECBS"] else False,
        'start_time_l': start_time_l,
        'runtime_limit': test_config.runtime_limit,
        'conflict_type_to_constraint_types': {PointConflict: {MultiPointConstraint}},
    }

    num_agents = test_config.num_agents

    # ============================
    # Get planning problem.
    # ============================
    # If want to get random starts and goals, then must do that after creating the reference task and robot.
    start_l = test_config.start_state_pos_l
    goal_l = test_config.goal_state_pos_l
    global_model_ids = test_config.global_model_ids
    agent_skeleton_l = test_config.agent_skeleton_l

    # ============================
    # Transforms and model tiles setup.
    # ============================
    # Create a reference planner from which we'll use the task and robot as the reference on in CBS.
    # Those are used for collision checking and visualization. This has a skeleton of all tiles.
    reference_agent_skeleton = [[r, c] for r in range(len(global_model_ids))
                                for c in range(len(global_model_ids[0]))]

    # ============================
    # Transforms from tiles to global frame.
    # ============================
    tile_width = 2.0
    tile_height = 2.0
    global_model_transforms = [[torch.tensor([x * tile_width, -y * tile_height], **tensor_args)
                                for x in range(len(global_model_ids[0]))] for y in range(len(global_model_ids))]

    # ============================
    # Parse the single agent planner class name.
    # ============================
    if test_config.single_agent_planner_class == "MPD":
        low_level_planner_class = MPD
    elif test_config.single_agent_planner_class == "MPDEnsemble":
        low_level_planner_class = MPDEnsemble
    else:
        raise ValueError(f'Unknown single agent planner class: {test_config.single_agent_planner_class}')

    # ============================
    # Create reference agent planner.
    # ============================
    # And for the reference skeleton.
    reference_task = None
    reference_robot = None
    reference_agent_transforms = {}
    reference_agent_model_ids = {}
    for skeleton_step in range(len(reference_agent_skeleton)):
        skeleton_model_coord = reference_agent_skeleton[skeleton_step]
        reference_agent_transforms[skeleton_step] = global_model_transforms[skeleton_model_coord[0]][
            skeleton_model_coord[1]]
        reference_agent_model_ids[skeleton_step] = global_model_ids[skeleton_model_coord[0]][
            skeleton_model_coord[1]]
    reference_agent_model_ids = [reference_agent_model_ids[i] for i in range(len(reference_agent_model_ids))]
    # Create the reference low level planner.
    print("Creating reference agent stuff.")
    low_level_planner_model_args['start_state_pos'] = torch.tensor([0.5, 0.9], **tensor_args)  # This does not matter.
    low_level_planner_model_args['goal_state_pos'] = torch.tensor([-0.5, 0.9], **tensor_args)  # This does not matter.
    low_level_planner_model_args['model_ids'] = reference_agent_model_ids  # This matters.
    low_level_planner_model_args['transforms'] = reference_agent_transforms  # This matters.

    if test_config.single_agent_planner_class == "MPD":
        low_level_planner_model_args['model_id'] = reference_agent_model_ids[0]

    reference_low_level_planner = low_level_planner_class(**low_level_planner_model_args)
    reference_task = reference_low_level_planner.task
    reference_robot = reference_low_level_planner.robot

    # ============================
    # Run trial.
    # ============================
    exp_name = f'mmd_single_trial'

    # Transform starts and goals to the global frame. Right now they are in the local tile frames.
    start_l = [start_l[i] + global_model_transforms[agent_skeleton_l[i][0][0]][agent_skeleton_l[i][0][1]]
               for i in range(num_agents)]
    goal_l = [goal_l[i] + global_model_transforms[agent_skeleton_l[i][-1][0]][agent_skeleton_l[i][-1][1]]
              for i in range(num_agents)]

    # ============================
    # Create global transforms for each agent's skeleton.
    # ============================
    # Each agent has a dict entry. Each entry is a dict with the skeleton steps (0, 1, 2, ...), mapping to the
    # model transform.
    agent_model_transforms_l = []
    agent_model_ids_l = []
    for agent_id in range(num_agents):
        agent_model_transforms = {}
        agent_model_ids = {}
        for skeleton_step in range(len(agent_skeleton_l[agent_id])):
            skeleton_model_coord = agent_skeleton_l[agent_id][skeleton_step]
            agent_model_transforms[skeleton_step] = global_model_transforms[skeleton_model_coord[0]][
                skeleton_model_coord[1]]
            agent_model_ids[skeleton_step] = global_model_ids[skeleton_model_coord[0]][skeleton_model_coord[1]]
        agent_model_transforms_l.append(agent_model_transforms)
        agent_model_ids_l.append(agent_model_ids)
    # Change the dict of the model ids to a list sorted by the skeleton steps.
    agent_model_ids_l = [[agent_model_ids_l[i][j] for j in range(len(agent_model_ids_l[i]))] for i in
                         range(num_agents)]

    # ============================
    # Create the low level planners.
    # ============================
    planners_creation_start_time = time.time()
    low_level_planner_l = []
    for i in range(num_agents):
        low_level_planner_model_args_i = low_level_planner_model_args.copy()
        low_level_planner_model_args_i['start_state_pos'] = start_l[i]
        low_level_planner_model_args_i['goal_state_pos'] = goal_l[i]
        low_level_planner_model_args_i['model_ids'] = agent_model_ids_l[i]
        low_level_planner_model_args_i['transforms'] = agent_model_transforms_l[i]
        if test_config.single_agent_planner_class == "MPD":
            # Set the model_id to the first one.
            low_level_planner_model_args_i['model_id'] = agent_model_ids_l[i][0]
        low_level_planner_l.append(low_level_planner_class(**low_level_planner_model_args_i))
    print('Planners creation time:', time.time() - planners_creation_start_time)
    print("\n\n\n\n")

    # ============================
    # Create the multi agent planner.
    # ============================
    if (test_config.multi_agent_planner_class in ["XECBS", "ECBS", "XCBS", "CBS"]):
        multi_agent_planner_class = CBS
    elif test_config.multi_agent_planner_class == "PP":
        multi_agent_planner_class = PrioritizedPlanning
    else:
        raise ValueError(f'Unknown multi agent planner class: {test_config.multi_agent_planner_class}')
    planner = multi_agent_planner_class(low_level_planner_l,
                                        start_l,
                                        goal_l,
                                        reference_task=reference_task,
                                        reference_robot=reference_robot,
                                        **high_level_planner_model_args)
    return planner

def run_horizon_step(
    test_config,
    horizon_start,
    horizon_goal,
    past_trajs,
    model,
    model_state,
):
    # ============================
    # Start time per agent.
    # ============================
    start_time_l = [i * test_config.stagger_start_time_dt for i in range(test_config.num_agents)]

    # ============================
    # Arguments for the high/low level planner.
    # ============================
    low_level_planner_model_args = {
        'planner_alg': 'mmd',
        'use_guide_on_extra_objects_only': params.use_guide_on_extra_objects_only,
        'n_samples': params.n_samples,
        'n_local_inference_noising_steps': params.n_local_inference_noising_steps,
        'n_local_inference_denoising_steps': params.n_local_inference_denoising_steps,
        'start_guide_steps_fraction': params.start_guide_steps_fraction,
        'n_guide_steps': params.n_guide_steps,
        'n_diffusion_steps_without_noise': params.n_diffusion_steps_without_noise,
        'weight_grad_cost_collision': params.weight_grad_cost_collision,
        'weight_grad_cost_smoothness': params.weight_grad_cost_smoothness,
        'weight_grad_cost_constraints': params.weight_grad_cost_constraints,
        'weight_grad_cost_soft_constraints': params.weight_grad_cost_soft_constraints,
        'factor_num_interpolated_points_for_collision': params.factor_num_interpolated_points_for_collision,
        'trajectory_duration': params.trajectory_duration,
        'device': params.device,
        'debug': params.debug,
        'seed': params.seed,
        'results_dir': params.results_dir,
        'trained_models_dir': TRAINED_MODELS_DIR,
    }

    high_level_planner_model_args = {
        'is_xcbs': True if test_config.multi_agent_planner_class in ["XECBS", "XCBS"] else False,
        'is_ecbs': True if test_config.multi_agent_planner_class in ["ECBS", "XECBS"] else False,
        'start_time_l': start_time_l,
        'runtime_limit': test_config.runtime_limit,
        'conflict_type_to_constraint_types': {PointConflict: {MultiPointConstraint}},
    }

    # ============================
    # Get planning problem.
    # ============================
    # If want to get random starts and goals, then must do that after creating the reference task and robot.
    start_l = horizon_start
    goal_l = horizon_goal
    global_model_ids = test_config.global_model_ids
    agent_skeleton_l = test_config.agent_skeleton_l
    num_agents = test_config.num_agents

    # ============================
    # Transforms and model tiles setup.
    # ============================
    # Create a reference planner from which we'll use the task and robot as the reference on in CBS.
    # Those are used for collision checking and visualization. This has a skeleton of all tiles.
    reference_agent_skeleton = [[r, c] for r in range(len(global_model_ids))
                                for c in range(len(global_model_ids[0]))]

    # ============================
    # Transforms from tiles to global frame.
    # ============================
    tile_width = 2.0
    tile_height = 2.0
    global_model_transforms = [[torch.tensor([x * tile_width, -y * tile_height], **tensor_args)
                                for x in range(len(global_model_ids[0]))] for y in range(len(global_model_ids))]

    # ============================
    # Parse the single agent planner class name.
    # ============================
    if test_config.single_agent_planner_class == "MPD":
        low_level_planner_class = MPD
    elif test_config.single_agent_planner_class == "MPDEnsemble":
        low_level_planner_class = MPDEnsemble
    else:
        raise ValueError(f'Unknown single agent planner class: {test_config.single_agent_planner_class}')

    # ============================
    # Create reference agent planner.
    # ============================
    # And for the reference skeleton.
    reference_task = None
    reference_robot = None
    reference_agent_transforms = {}
    reference_agent_model_ids = {}
    for skeleton_step in range(len(reference_agent_skeleton)):
        skeleton_model_coord = reference_agent_skeleton[skeleton_step]
        reference_agent_transforms[skeleton_step] = global_model_transforms[skeleton_model_coord[0]][
            skeleton_model_coord[1]]
        reference_agent_model_ids[skeleton_step] = global_model_ids[skeleton_model_coord[0]][
            skeleton_model_coord[1]]
    reference_agent_model_ids = [reference_agent_model_ids[i] for i in range(len(reference_agent_model_ids))]
    # Create the reference low level planner.
    print("Creating reference agent stuff.")
    low_level_planner_model_args['start_state_pos'] = torch.tensor([0.5, 0.9], **tensor_args)  # This does not matter.
    low_level_planner_model_args['goal_state_pos'] = torch.tensor([-0.5, 0.9], **tensor_args)  # This does not matter.
    low_level_planner_model_args['model_ids'] = reference_agent_model_ids  # This matters.
    low_level_planner_model_args['transforms'] = reference_agent_transforms  # This matters.

    if test_config.single_agent_planner_class == "MPD":
        low_level_planner_model_args['model_id'] = reference_agent_model_ids[0]

    reference_low_level_planner = low_level_planner_class(**low_level_planner_model_args)
    reference_task = reference_low_level_planner.task
    reference_robot = reference_low_level_planner.robot

    # ============================
    # Run trial.
    # ============================

    # Transform starts and goals to the global frame. Right now they are in the local tile frames.
    start_l = [start_l[i] + global_model_transforms[agent_skeleton_l[i][0][0]][agent_skeleton_l[i][0][1]]
               for i in range(num_agents)]
    goal_l = [goal_l[i] + global_model_transforms[agent_skeleton_l[i][-1][0]][agent_skeleton_l[i][-1][1]]
              for i in range(num_agents)]

    # ============================
    # Create global transforms for each agent's skeleton.
    # ============================
    # Each agent has a dict entry. Each entry is a dict with the skeleton steps (0, 1, 2, ...), mapping to the
    # model transform.
    agent_model_transforms_l = []
    agent_model_ids_l = []
    for agent_id in range(num_agents):
        agent_model_transforms = {}
        agent_model_ids = {}
        for skeleton_step in range(len(agent_skeleton_l[agent_id])):
            skeleton_model_coord = agent_skeleton_l[agent_id][skeleton_step]
            agent_model_transforms[skeleton_step] = global_model_transforms[skeleton_model_coord[0]][
                skeleton_model_coord[1]]
            agent_model_ids[skeleton_step] = global_model_ids[skeleton_model_coord[0]][skeleton_model_coord[1]]
        agent_model_transforms_l.append(agent_model_transforms)
        agent_model_ids_l.append(agent_model_ids)
    # Change the dict of the model ids to a list sorted by the skeleton steps.
    agent_model_ids_l = [[agent_model_ids_l[i][j] for j in range(len(agent_model_ids_l[i]))] for i in
                         range(num_agents)]

    # ============================
    # Create the low level planners.
    # ============================
    planners_creation_start_time = time.time()
    low_level_planner_l = []
    for i in range(num_agents):
        low_level_planner_model_args_i = low_level_planner_model_args.copy()
        low_level_planner_model_args_i['start_state_pos'] = start_l[i]
        low_level_planner_model_args_i['goal_state_pos'] = goal_l[i]
        low_level_planner_model_args_i['model_ids'] = agent_model_ids_l[i]
        low_level_planner_model_args_i['transforms'] = agent_model_transforms_l[i]
        if test_config.single_agent_planner_class == "MPD":
            # Set the model_id to the first one.
            low_level_planner_model_args_i['model_id'] = agent_model_ids_l[i][0]
        low_level_planner_l.append(low_level_planner_class(**low_level_planner_model_args_i))
    print('Planners creation time:', time.time() - planners_creation_start_time)
    print("\n\n\n\n")

    # ============================
    # Create the multi agent planner.
    # ============================
    if (test_config.multi_agent_planner_class in ["XECBS", "ECBS", "XCBS", "CBS"]):
        # use masked version of planner if we are using player selection model
        if model is not None:
            multi_agent_planner_class = CBSMasked
        else:
            multi_agent_planner_class = CBS
    elif test_config.multi_agent_planner_class == "PP":
        multi_agent_planner_class = PrioritizedPlanning
    else:
        raise ValueError(f'Unknown multi agent planner class: {test_config.multi_agent_planner_class}')
    planner = multi_agent_planner_class(low_level_planner_l,
                                        start_l,
                                        goal_l,
                                        reference_task=reference_task,
                                        reference_robot=reference_robot,
                                        **high_level_planner_model_args)
    # ============================
    # Calculate player masks.
    # ============================
    if model is not None:
        player_masks = get_player_masks(past_trajs, model, model_state)
        player_masks = (player_masks >= 0.3).float()
    else:
        player_masks = None

    # ============================
    # Plan.
    # ============================
    if model is not None:
        paths_l, num_ct_expansions, trial_success_status, num_collisions_in_solution = \
            planner.plan(runtime_limit=test_config.runtime_limit, mask_l=player_masks)
    else:
        paths_l, num_ct_expansions, trial_success_status, num_collisions_in_solution = \
            planner.plan(runtime_limit=test_config.runtime_limit)
    return paths_l, player_masks, num_ct_expansions, trial_success_status, num_collisions_in_solution, planner.conflict_search_time 

def generate_intermediate_goals(
    current_start_l,
    sim_goal_l,
    num_agents,
    remaining_timesteps,
    stride,
):
    num_agents = len(current_start_l)
    intermediate_goals = []
    for i in range(num_agents):
        # current_start_l[i] and sim_goal_l[i] are (x, y) coordinates (1D tensors of size 2)
        # We want a linespace of remaining_timesteps points between start and goal (inclusive)
        linear_traj = torch.stack([
            torch.linspace(current_start_l[i][0], sim_goal_l[i][0], steps=remaining_timesteps, device=current_start_l[i].device, dtype=current_start_l[i].dtype),
            torch.linspace(current_start_l[i][1], sim_goal_l[i][1], steps=remaining_timesteps, device=current_start_l[i].device, dtype=current_start_l[i].dtype)
        ], dim=1)  # shape: (remaining_timesteps, 2)
        intermediate_goals.append(linear_traj[min(stride, remaining_timesteps - 1)])
    
    # Convert to tensor for easier manipulation
    intermediate_goals = torch.stack(intermediate_goals)
    
    # adjust intermediate goals so they are always 0.15 apart
    min_distance = 0.15
    max_attempts = 100
    
    for attempt in range(max_attempts):
        # Compute pairwise distances
        pairwise_dists = torch.cdist(intermediate_goals, intermediate_goals)
        # Set diagonal to inf to ignore self-distances
        pairwise_dists = pairwise_dists.fill_diagonal_(float('inf'))
        
        # Find pairs that are too close
        close_mask = pairwise_dists < min_distance
        
        if not torch.any(close_mask):
            break
        
        # For each pair that's too close, push them apart minimally
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if close_mask[i, j]:
                    # Calculate direction vector from j to i
                    direction = intermediate_goals[i] - intermediate_goals[j]
                    distance = torch.norm(direction)
                    
                    if distance > 1e-6:
                        # Normalize direction
                        direction = direction / distance
                    else:
                        # If positions are identical, use random direction
                        direction = torch.randn(
                            2,
                            device=intermediate_goals[i].device,
                            dtype=intermediate_goals[i].dtype
                        )
                        direction = direction / torch.norm(direction)
                    
                    # Calculate minimal push distance needed
                    # Push each position by half the needed distance plus a small buffer
                    push_distance = (min_distance - distance) / 2 + 0.01
                    
                    # Push positions apart
                    intermediate_goals[i] = intermediate_goals[i] + direction * push_distance
                    intermediate_goals[j] = intermediate_goals[j] - direction * push_distance
    
    if attempt >= max_attempts - 1:
        print(f"Warning: Could not fully separate intermediate goals after {max_attempts} attempts")
    
    # Convert back to list of tensors
    return [intermediate_goals[i] for i in range(num_agents)]

def run_multi_agent_trial(
        test_config: MultiAgentPlanningSingleTrialConfig,
        stride: int = 1,
        model_past_horizon: int = 10,
        model_path: str = None,
    ):
    final_planner = create_whole_trajectory_planner(test_config)

    start_l = test_config.start_state_pos_l
    goal_l = test_config.goal_state_pos_l

    # ============================
    # Setup for rendering
    # ============================
    num_agents = test_config.num_agents
    global_model_ids = test_config.global_model_ids
    agent_skeleton_l = test_config.agent_skeleton_l
    
    # Create transforms
    tile_width = 2.0
    tile_height = 2.0
    global_model_transforms = [[torch.tensor([x * tile_width, -y * tile_height], **tensor_args)
                                for x in range(len(global_model_ids[0]))] for y in range(len(global_model_ids))]
    
    # ============================
    # Load model.
    # ============================
    model, model_state = load_trained_gnn_models(model_path, obs_input_type="full")
    
    # Transform starts and goals to global frame
    start_l = [start_l[i] + global_model_transforms[agent_skeleton_l[i][0][0]][agent_skeleton_l[i][0][1]]
               for i in range(num_agents)]
    goal_l = [goal_l[i] + global_model_transforms[agent_skeleton_l[i][-1][0]][agent_skeleton_l[i][-1][1]]
              for i in range(num_agents)]
    
    # Create reference planner for visualization
    reference_agent_skeleton = [[r, c] for r in range(len(global_model_ids))
                                for c in range(len(global_model_ids[0]))]
    reference_agent_transforms = {}
    reference_agent_model_ids = {}
    for skeleton_step in range(len(reference_agent_skeleton)):
        skeleton_model_coord = reference_agent_skeleton[skeleton_step]
        reference_agent_transforms[skeleton_step] = global_model_transforms[skeleton_model_coord[0]][
            skeleton_model_coord[1]]
        reference_agent_model_ids[skeleton_step] = global_model_ids[skeleton_model_coord[0]][
            skeleton_model_coord[1]]
    reference_agent_model_ids = [reference_agent_model_ids[i] for i in range(len(reference_agent_model_ids))]
    
    # ============================
    # Horizon-based planning parameters
    # ============================
    num_timesteps = 64  # Total number of timesteps in final trajectory
    
    # Initialize storage for the full trajectories
    # Start with the initial start positions/velocities for each agent
    # Pad positions with zero velocities if needed to match path state dimensions (pos + vel)
    full_trajectories = []
    for s in start_l:
        if s.shape[0] == 2:
            # Position only, add zero velocities
            initial_state = torch.cat([s, torch.zeros(2, **tensor_args)])
        else:
            # Already has velocities
            initial_state = s.clone()
        full_trajectories.append([initial_state])
    
    # Initialize current starts and goals
    sim_start_l = [s.clone() for s in start_l]
    sim_goal_l = goal_l  # Goals remain the same
    
    # Statistics accumulators
    total_ct_expansions = 0
    total_planning_time = 0.0
    total_collisions = 0
    final_success_status = TrialSuccessStatus.SUCCESS
    
    print(f"{BLUE}Starting horizon-based planning for {num_timesteps} timesteps{RESET}")
    
    # ============================
    # Main horizon-based planning loop
    # ============================
    # Calculate number of planning iterations needed
    # We start with 1 waypoint (initial state) and add stride waypoints each iteration
    num_planning_iterations = (num_timesteps - 1 + stride - 1) // stride  # ceiling division
    print(f"Planning iterations needed: {num_planning_iterations}, stride: {stride}, target timesteps: {num_timesteps}")
    current_start_l = sim_start_l
    remaining_timesteps = num_timesteps
    full_trajectories = [[torch.cat([sim_start_l[i], torch.zeros(2, **tensor_args)])] for i in range(num_agents)]
    sim_masks = []
    
    for iteration in range(num_planning_iterations):
        print(f"\n{CYAN}=== Planning Iteration {iteration + 1}/{num_planning_iterations} ==={RESET}")

        # Generate intermediate goals based on current position and remaining timesteps
        # This ensures goals adapt to the actual path taken
        current_goal_l = generate_intermediate_goals(
            current_start_l, sim_goal_l, num_agents, remaining_timesteps, stride
        )
        
        # Update remaining timesteps for next iteration
        remaining_timesteps = max(stride, remaining_timesteps - stride)

        # calculate past trajectories
        past_trajs = []
        for i in range(num_agents):
            start_ind = max(0, len(full_trajectories[i]) - model_past_horizon)
            agent_past_traj = full_trajectories[i][start_ind:]
            
            # Always stack the trajectory into a tensor
            agent_past_traj = torch.stack(agent_past_traj, dim=0)
            
            # Add padding if needed
            if len(agent_past_traj) < model_past_horizon:
                padding = agent_past_traj[-1:].repeat(model_past_horizon - len(agent_past_traj), 1)
                agent_past_traj = torch.cat([agent_past_traj, padding])
            
            past_trajs.append(agent_past_traj)
        past_trajs = torch.stack(past_trajs)

        # Call run_horizon_step with current start and goal
        paths_l, player_masks, num_ct_expansions, trial_success_status, num_collisions_in_solution, planning_time = \
            run_horizon_step(
                test_config,
                current_start_l,
                current_goal_l,
                past_trajs,
                model,
                model_state,
            )
        
        extend_amount = min(stride, remaining_timesteps)
        if iteration == 0:
            extend_amount += 1 # account for fact full_trajectories is already padded with initial state

        sim_masks.extend([player_masks] * extend_amount)
        total_planning_time += planning_time
        total_ct_expansions += num_ct_expansions
        total_collisions += num_collisions_in_solution
        if trial_success_status != TrialSuccessStatus.SUCCESS:
            trial_success_status = TrialSuccessStatus.FAIL_NO_SOLUTION
        
        # condense trajectories and update current_state_l
        num_steps = num_timesteps // stride
        paths_l = [paths_l[i][::num_steps] for i in range(num_agents)]
        current_start_l = [paths_l[i][-1][:2] for i in range(num_agents)]
        for i in range(num_agents):
            full_trajectories[i].extend(paths_l[i])
    
    # ============================
    # Convert trajectories to tensors
    # ============================
    final_trajectories = []
    for agent_id in range(num_agents):
        if len(full_trajectories[agent_id]) > 0:
            traj = torch.stack(full_trajectories[agent_id])
            final_trajectories.append(traj)
        else:
            final_trajectories.append(None)
    
    # ============================
    # Print summary and save results
    # ============================
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{BLUE}Horizon-based planning completed{RESET}")
    print(f"Total planning time: {total_planning_time:.3f}s")
    print(f"Total CT expansions: {total_ct_expansions}")
    print(f"Total collisions: {total_collisions}")
    print(f"Final status: {final_success_status}")
    print(f"{CYAN}{'='*60}{RESET}\n")
    
    # ============================
    # Render trajectories
    # ============================


    # ============================
    # Create a results directory.
    # ============================
    results_dir = get_result_dir_from_trial_config(test_config, test_config.time_str, test_config.trial_number)
    os.makedirs(results_dir, exist_ok=True)
    exp_name = 'mmd_horizon_based_trial'

    if final_success_status == TrialSuccessStatus.SUCCESS and len(final_trajectories) > 0 and all(t is not None for t in final_trajectories):
        print(f"\n{CYAN}Rendering trajectories...{RESET}")
        plot_starts = torch.stack(start_l)
        plot_goals = torch.stack(goal_l)
        plot_trajs = torch.stack(final_trajectories)
        create_trajectory_gif(plot_trajs, plot_starts, plot_goals, os.path.join(results_dir, f'{exp_name}.gif'), fps=10, figsize=(10, 10), show_velocity_arrows=False, dpi=300)
        if model is not None:
            create_masked_trajectory_gif(plot_trajs, plot_starts, plot_goals, 0, sim_masks, os.path.join(results_dir, f'{exp_name}_masked.gif'), fps=10, figsize=(10, 10), show_velocity_arrows=False, dpi=300)
            final_planner.render_paths_masked(final_trajectories, 0, sim_masks, output_fpath=os.path.join(results_dir, f'{exp_name}_planner_visualization_masked.gif'), plot_trajs=True, animation_duration=10)

    else:
        print(f"{YELLOW}Skipping rendering due to planning failure or empty trajectories{RESET}")

if __name__ == '__main__':
    test_config_single_tile = MultiAgentPlanningSingleTrialConfig()
    test_config_single_tile.num_agents = 5 
    test_config_single_tile.instance_name = "test"
    test_config_single_tile.multi_agent_planner_class = "XECBS"  # Or "ECBS" or "XCBS" or "CBS" or "PP".
    test_config_single_tile.single_agent_planner_class = "MPDEnsemble"  # Or "MPD"
    test_config_single_tile.stagger_start_time_dt = 0
    test_config_single_tile.runtime_limit = 60 * 3  # 3 minutes.
    test_config_single_tile.time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    test_config_single_tile.render_animation = True  # Change the `densify_trajs` call above to create nicer animations.
    
    player_selection_model_path = "/home/alex/gnn_game_planning/log/point_agent_train_runs/gnn_full_MP_2_edge-metric_full_top-k_5/train_n_agents_10_T_50_obs_10_lr_0.0003_bs_32_sigma1_0.11_sigma2_0.11_epochs_50_loss_type_similarity/20251108_103120/psn_best_model.pkl"
    # player_selection_model_path = None
    stride = 4

    example_type = "single_tile"
    # example_type = "multi_tile"
    # ============================
    # Single tile.
    # ============================
    if example_type == "single_tile":
        # Choose the model to use. A model is for a map/robot combination.
        # test_config_single_tile.global_model_ids = [['EnvEmpty2D-RobotPlanarDisk']]
        test_config_single_tile.global_model_ids = [['EnvEmptyNoWait2D-RobotPlanarDisk']]
        # test_config_single_tile.global_model_ids = [['EnvConveyor2D-RobotPlanarDisk']]
        # test_config_single_tile.global_model_ids = [['EnvHighways2D-RobotPlanarDisk']]
        # test_config_single_tile.global_model_ids = [['EnvDropRegion2D-RobotPlanarDisk']]

        # Choose starts and goals.
        test_config_single_tile.agent_skeleton_l = [[[0, 0]]] * test_config_single_tile.num_agents
        torch.random.manual_seed(43)
        test_config_single_tile.start_state_pos_l, test_config_single_tile.goal_state_pos_l = \
        get_start_goal_pos_random_in_env(test_config_single_tile.num_agents,
                                         EnvDropRegion2D,
                                         tensor_args,
                                         margin=0.2,
                                         obstacle_margin=0.11)

        # custom start/end positions
        # test_config_single_tile.start_state_pos_l = [torch.tensor([-0.0796, -0.0326], device='cuda:0'), torch.tensor([0.3682, 0.8917], device='cuda:0'), torch.tensor([-0.3945, -0.0336], device='cuda:0'), torch.tensor([ 0.2185, -0.0062], device='cuda:0'), torch.tensor([-0.0910, -0.8407], device='cuda:0')]
        
        # get_start_goal_pos_circle(test_config_single_tile.num_agents, 0.8)
        # Another option is to get random starts and goals.
        # get_start_goal_pos_random_in_env(test_config_single_tile.num_agents,
        #                                  EnvDropRegion2D,
        #                                  tensor_args,
        #                                  margin=0.2,
        #                                  obstacle_margin=0.11)
        # A third option is to get starts and goals in a "boundary" formation.
        # get_start_goal_pos_boundary(test_config_single_tile.num_agents, 0.85)
        # And a final option is to hard-code starts and goals.
        # (torch.tensor([[-0.8, 0], [0.8, -0]], **tensor_args),
        #  torch.tensor([[0.8, -0], [-0.8, 0]], **tensor_args))
        print("Starts:", test_config_single_tile.start_state_pos_l)
        print("Goals:", test_config_single_tile.goal_state_pos_l)

        run_multi_agent_trial(test_config_single_tile, stride=stride, model_path=player_selection_model_path)
        print(GREEN, 'OK.', RESET)

    # ============================
    # Multiple tiles example.
    # ============================
    if example_type == "multi_tile":
        test_config_multiple_tiles = test_config_single_tile
        test_config_multiple_tiles.num_agents = 4
        test_config_multiple_tiles.stagger_start_time_dt = 5
        test_config_multiple_tiles.global_model_ids = \
            [['EnvEmptyNoWait2D-RobotPlanarDisk', 'EnvEmptyNoWait2D-RobotPlanarDisk']]

        test_config_multiple_tiles.agent_skeleton_l = [[[0, 0], [0, 1]],
                                                       [[0, 1], [0, 0]],
                                                       [[0, 0], [0, 1]],
                                                       [[0, 1], [0, 0]]]
        test_config_multiple_tiles.start_state_pos_l, test_config_multiple_tiles.goal_state_pos_l = \
            (torch.tensor([[0, 0.8], [0, 0.3], [0, -0.3], [0, -0.8]], **tensor_args),
             torch.tensor([[0, -0.8], [0, -0.3], [0, 0.3], [0, 0.8]], **tensor_args))
        print(test_config_multiple_tiles.start_state_pos_l)
        test_config_multiple_tiles.multi_agent_planner_class = "XECBS"
        run_multi_agent_trial(test_config_multiple_tiles, stride=stride)
        print(GREEN, 'OK.', RESET)
