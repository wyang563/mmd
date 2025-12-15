import torch
from mmd.config.mmd_params import MMDParams as params
from mmd.planners.single_agent.mpd_safe import MPDSafe
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../data_trained_models/'))
LOGS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../logs/'))


def centralized_planner(start_l, goal_l, collision_radius, model_id, num_timesteps, t_past=3):
    # go in order to generate trajectories for each agent one by one
    n_agents = len(start_l)
    constraints = [[] for _ in range(num_timesteps)]
    final_trajectories = []
    for i in range(n_agents):
        start_i = start_l[i]
        goal_i = goal_l[i]
        low_level_planner_model_args = {
            "start_state_pos": start_i,
            "goal_state_pos": goal_i,
            "collision_radius": collision_radius,
            "model_id": model_id,
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

        planner = MPDSafe(**low_level_planner_model_args)
        print("MPDSafe planner created successfully!")
        
        print("\n" + "=" * 60)
        print("Running planning with constraints for agent " + str(i) + "...")
        print("=" * 60)
        result = planner(
            start_state_pos=start_i,
            goal_state_pos=goal_i,
            constraints=constraints
        )

        # 1) Extract the best trajectory for this agent and add to final_trajectories
        if result.trajs_final_free is not None and result.trajs_final_free.shape[0] > 0:
            # Use the best collision-free trajectory
            if hasattr(result, 'idx_best_free_traj') and result.idx_best_free_traj is not None:
                best_traj = result.trajs_final_free[result.idx_best_free_traj]  # Shape: [H, D]
            else:
                # Use first collision-free trajectory if no best index
                best_traj = result.trajs_final_free[0]  # Shape: [H, D]
        else:
            # Fall back to best trajectory from all trajectories
            if hasattr(result, 'idx_best_traj') and result.idx_best_traj is not None:
                best_traj = result.trajs_final[result.idx_best_traj]  # Shape: [H, D]
            else:
                best_traj = result.trajs_final[0]  # Shape: [H, D]
        
        final_trajectories.append(best_traj)
        print(f"Agent {i} trajectory shape: {best_traj.shape}")
        
        # 2) For each position at timestep t, add it as a constraint at timesteps t, t-1, t-2, ..., t-t_past
        for t in range(min(num_timesteps, best_traj.shape[0])):
            position = best_traj[t, :2]  # Extract [x, y] position at timestep t
            
            # Convert tensor to Python list for compatibility with constraint processing
            position_list = position.cpu().detach().tolist()
            
            # Add this position as a constraint at timestep t and t_past timesteps before
            for delta in range(t_past + 1):  # delta = 0, 1, 2, ..., t_past
                constraint_timestep = t - delta
                if constraint_timestep >= 0:
                    constraints[constraint_timestep].append(position_list)
        
        print(f"Added constraints for agent {i}")
    
    # 3) Return the final set of trajectories
    return final_trajectories

def detect_conflicts(trajectories, collision_radius=0.1):
    conflicts_by_agent = []
    for agent_i in range(len(trajectories)):
        conflicts_for_agent_i = []
        for agent_j in range(agent_i + 1, len(trajectories)):
            trajectory_i = trajectories[agent_i]
            trajectory_j = trajectories[agent_j]
            for t in range(len(trajectory_i)):
                position_i = trajectory_i[t, :2]
                position_j = trajectory_j[t, :2]
                # Use torch.norm instead of numpy
                distance = torch.norm(position_i - position_j).item()
                if distance < collision_radius:
                    conflicts_for_agent_i.append((position_j.tolist(), t))
        conflicts_by_agent.append(conflicts_for_agent_i)
    return conflicts_by_agent

def decentralized_planner(start_l, goal_l, collision_radius, model_id, num_timesteps, t_past=3):
    """Placeholder for a decentralized planner (not yet implemented)."""
    contraints_by_agent = {}
    n_agents = len(start_l)
    # initialize empty constraint sets
    for agent_i in range(n_agents):
        contraints_by_agent[agent_i] = [[] for _ in range(num_timesteps)]
    
    # generate initial trajectories without constraints for each agent
    first_iter = True
    exist_conflicts = False
    iter_num = 0
    agents_to_generate_trajectories = list(range(n_agents)) # we only want to generate trajectories for agents that have conflicts later on
    trajectories = [None for _ in range(n_agents)]
    while first_iter or exist_conflicts:
        print("Iteration " + str(iter_num) + "...")
        
        for agent_i in agents_to_generate_trajectories:
            start_i = start_l[agent_i]
            goal_i = goal_l[agent_i]
            constraints_i = contraints_by_agent[agent_i]
            low_level_planner_model_args = {
                "start_state_pos": start_i,
                "goal_state_pos": goal_i,
                "collision_radius": collision_radius,
                "model_id": model_id,
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

            planner = MPDSafe(**low_level_planner_model_args)
            result = planner(
                start_state_pos=start_i,
                goal_state_pos=goal_i,
                constraints=constraints_i
            )

            if result.trajs_final_free is not None and result.trajs_final_free.shape[0] > 0:
                if hasattr(result, 'idx_best_free_traj') and result.idx_best_free_traj is not None:
                    best_traj = result.trajs_final_free[result.idx_best_free_traj]  # Shape: [H, D]
                else:
                    # Use first collision-free trajectory if no best index
                    best_traj = result.trajs_final_free[0]  # Shape: [H, D]
            else:
                # Fall back to best trajectory from all trajectories
                if hasattr(result, 'idx_best_traj') and result.idx_best_traj is not None:
                    best_traj = result.trajs_final[result.idx_best_traj]  # Shape: [H, D]
                else:
                    best_traj = result.trajs_final[0]  # Shape: [H, D]
            
            trajectories[agent_i] = best_traj

        # check for conflicts
        detected_conflicts = detect_conflicts(trajectories, collision_radius)
        print("Detected conflicts: " + str(detected_conflicts))
        
        # Count total conflict points for debugging
        total_conflict_points = sum(len(conflicts_for_agent) for conflicts_for_agent in detected_conflicts)
        print(f"Total conflict points detected: {total_conflict_points}")
        
        exist_conflicts = any(len(conflicts_for_agent) > 0 for conflicts_for_agent in detected_conflicts)
        
        agents_to_generate_trajectories = []
        if exist_conflicts:
            for agent_i in range(n_agents):
                if len(detected_conflicts[agent_i]) > 0:
                    agents_to_generate_trajectories.append(agent_i)
                for position, time_conflict in detected_conflicts[agent_i]:
                    start_time = max(0, time_conflict - t_past)
                    for t in range(start_time, time_conflict + 1):
                        contraints_by_agent[agent_i][t].append(position)
        
        first_iter = False
        iter_num += 1
    
    return trajectories

    