"""
Simple test script for MPDSafe class.
Creates a random start/goal and 3 random conflict points as constraints.
"""
import os
import torch
import numpy as np
import random

# NOTES FOR MULTIAGENT TRAJECTORY GENERATION:
# - Make sure conflict zones don't have sharp boundaries ie have a smooth continious area represent conflict zones
# - Use masking along different points of the trajectory to reduce the number of conflict zones needed to solve for at time step t

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../data_trained_models/'))

# Import MPDSafe
from mmd.planners.single_agent.mpd_safe import MPDSafe
from mmd.config.mmd_params import MMDParams as params
from mmd.utils.plot_trajs import create_single_agent_trajectory_gif_with_constraints
from torch_robotics.torch_utils.torch_utils import get_torch_device

# Device setup
device = 'cuda'
device = get_torch_device(device)
tensor_args = {'device': device, 'dtype': torch.float32}


def create_random_start_goal(bounds=(-1.0, 1.0), seed=None):
    """
    Create random start and goal positions within bounds.
    Note: The trained models use a [-1, 1] coordinate system, not [-10, 10].
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create random positions within bounds (scaled from [-10,10] request to [-1,1] actual)
    # The user requested [-10,10] but the models work in [-1,1] range
    scale = (bounds[1] - bounds[0]) / 2
    offset = (bounds[1] + bounds[0]) / 2
    
    start_pos = torch.rand(2, **tensor_args) * 2 - 1  # Random in [-1, 1]
    goal_pos = torch.rand(2, **tensor_args) * 2 - 1   # Random in [-1, 1]
    
    # Scale to the requested bounds
    start_pos = start_pos * scale + offset
    goal_pos = goal_pos * scale + offset
    
    # Ensure start and goal are not too close
    while torch.norm(start_pos - goal_pos) < 0.3:
        goal_pos = torch.rand(2, **tensor_args) * 2 - 1
        goal_pos = goal_pos * scale + offset
    
    return start_pos, goal_pos


def create_random_constraints_3d(num_constraints=3, time_duration=10, num_timesteps=64, bounds=(-1.0, 1.0), seed=None):
    """
    Create random conflict points as a 3D list indexed by [timestep][point_idx][coord].
    
    Returns:
        constraints_3d: 3D list of shape [num_timesteps][num_points_at_timestep][2]
                        - First index: timestep (0 to num_timesteps-1)
                        - Second index: conflict point index at that timestep
                        - Third index: coordinate [x, y]
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    scale = (bounds[1] - bounds[0]) / 2
    offset = (bounds[1] + bounds[0]) / 2
    
    # Initialize 3D list: constraints_3d[timestep][point_idx][coord]
    # Each timestep starts with an empty list of points
    constraints_3d = [[] for _ in range(num_timesteps)]
    
    # Random timesteps for each constraint (avoiding first and last few steps)
    margin = 5
    timesteps = np.random.randint(margin, num_timesteps - margin, size=num_constraints)
    
    # Add each constraint point to its corresponding timestep
    for i in range(num_constraints):
        t0 = int(timesteps[i])
        # Random position in bounds - this is the [x, y] coordinate
        point = ((np.random.rand(2) * 2 - 1) * scale + offset).tolist()
        for t in range(0, num_timesteps):
            constraints_3d[t].append(point)  # point is [x, y]
    
    # Print the 3D structure
    print("\n=== Constraints as 3D list [timestep][point_idx][coord] ===")
    print(f"Total timesteps: {num_timesteps}")
    for t in range(num_timesteps):
        if len(constraints_3d[t]) > 0:
            print(f"  Timestep {t}: {len(constraints_3d[t])} constraint(s)")
            for p_idx, point in enumerate(constraints_3d[t]):
                print(f"    Point {p_idx}: [{point[0]:.4f}, {point[1]:.4f}]")
    
    return constraints_3d


def main():
    print("=" * 60)
    print("MPDSafe Single Run Test")
    print("=" * 60)
    
    # Set random seed for reproducibility
    seed = random.randint(0, 1000000)
    print(f"Seed: {seed}")
    
    # Use bounds within [-1, 1] since that's what the trained models expect
    # The user requested [-10, 10] but the models work in normalized coordinates
    bounds = (-0.95, 0.95)  # Slightly inside [-1, 1] to avoid boundary issues
    constraint_time_duration = 20
    num_constraints = 10
    collision_radius = 0.15
    
    # Create random start and goal
    start_pos, goal_pos = create_random_start_goal(bounds=bounds, seed=seed)
    print(f"\nStart position: {start_pos}")
    print(f"Goal position: {goal_pos}")
    
    # Create 3 random conflict points as constraints
    num_timesteps = 64  # Standard trajectory length
    constraints_3d = create_random_constraints_3d(
        num_constraints=num_constraints, 
        time_duration=constraint_time_duration,
        num_timesteps=num_timesteps,
        bounds=bounds,
        seed=seed + 1
    )
    
    # Model configuration
    model_id = 'EnvEmptyNoWait2D-RobotPlanarDisk'  # Use an empty environment for testing
    
    print(f"\nUsing model: {model_id}")
    print(f"Trained models directory: {TRAINED_MODELS_DIR}")
    
    # MPDSafe planner arguments
    low_level_planner_model_args = {
        "start_state_pos": start_pos,
        "goal_state_pos": goal_pos,
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
    
    print("\n" + "=" * 60)
    print("Creating MPDSafe planner...")
    print("=" * 60)
    
    try:
        # Create the MPDSafe planner
        planner = MPDSafe(**low_level_planner_model_args)
        print("MPDSafe planner created successfully!")
        
        print("\n" + "=" * 60)
        print("Running planning with constraints...")
        print("=" * 60)
        
        # Pass the 3D constraints list directly to the planner
        # Format: constraints_3d[timestep][point_idx][coord] where coord is [x, y]
        result = planner(
            start_state_pos=start_pos,
            goal_state_pos=goal_pos,
            constraints=constraints_3d
        )
        
        print("\n" + "=" * 60)
        print("Planning Results")
        print("=" * 60)
        print(f"Total time: {result.t_total:.3f}s")
        print(f"Number of trajectories generated: {result.trajs_final.shape[0]}")
        print(f"Trajectory shape: {result.trajs_final.shape}")
        
        if result.trajs_final_free is not None:
            print(f"Number of collision-free trajectories: {result.trajs_final_free.shape[0]}")
            print(f"Best trajectory index: {result.idx_best_traj}")
            print(f"Best trajectory cost: {result.cost_best_free_traj:.4f}")
        else:
            print("WARNING: No collision-free trajectories found!")
        
        print("\n" + "=" * 60)
        print("Plotting Trajectories...")
        print("=" * 60)

        # Get the best trajectory
        if result.trajs_final_free is not None and result.trajs_final_free.shape[0] > 0:
            # Use the best collision-free trajectory
            if result.idx_best_free_traj is not None:
                best_traj = result.trajs_final_free[result.idx_best_free_traj]  # Shape: [H, D]
            else:
                # Use first collision-free trajectory if no best index
                best_traj = result.trajs_final_free[0]  # Shape: [H, D]
        else:
            # Fall back to best trajectory from all trajectories
            if result.idx_best_traj is not None:
                best_traj = result.trajs_final[result.idx_best_traj]  # Shape: [H, D]
            else:
                # Use first trajectory if no best index
                best_traj = result.trajs_final[0]  # Shape: [H, D]
        
        # Create output directory if it doesn't exist
        output_dir = params.results_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the gif with constraints
        output_path = os.path.join(output_dir, 'single_agent_trajectory_with_constraints.gif')
        print(f"Creating trajectory GIF with constraints at: {output_path}")
        create_single_agent_trajectory_gif_with_constraints(
            trajectory=best_traj,
            start=start_pos,
            goal=goal_pos,
            constraints=constraints_3d,
            output_path=output_path,
            fps=10,
            figsize=(10, 10),
            agent_radius=params.robot_planar_disk_radius,
            constraint_radius=collision_radius,
            show_velocity_arrows=False,
            title="Single Agent Trajectory with Constraints",
            dpi=300
        )
        print(f"Trajectory GIF with constraints saved successfully!")
        
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: MPDSafe may have missing initialization (self.guide).")
        print("You may need to fix mpd_safe.py to add the guide manager setup.")


if __name__ == '__main__':
    main()

