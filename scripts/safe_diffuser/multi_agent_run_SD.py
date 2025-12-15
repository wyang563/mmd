import os
import torch
import numpy as np
import random
import copy
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# NOTES FOR MULTIAGENT TRAJECTORY GENERATION:
# - Make sure conflict zones don't have sharp boundaries ie have a smooth continious area represent conflict zones
# - Use masking along different points of the trajectory to reduce the number of conflict zones needed to solve for at time step t

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../data_trained_models/'))
LOGS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../logs/'))

# Import MPDSafe
from scripts.safe_diffuser.multi_agent_planners import centralized_planner, decentralized_planner
from torch_robotics.torch_utils.torch_utils import get_torch_device
from mmd.utils.plot_trajs import create_trajectory_gif


# Device setup
device = 'cuda'
device = get_torch_device(device)
tensor_args = {'device': device, 'dtype': torch.float32}

def generate_safe_start_goal_positions(n_agents, bounds=(-1.0, 1.0), seed=None, constraint_radius=0.1):
    """
    Generate start and goal positions for n_agents such that:
    - All positions are within the specified xy coordinate bounds
    - No two start positions are within constraint_radius of each other
    - No two goal positions are within constraint_radius of each other
    - No start position is within constraint_radius of any goal position
    
    Args:
        n_agents: Number of agents
        bounds: Tuple of (min, max) bounds for x and y coordinates
        seed: Random seed for reproducibility
        constraint_radius: Minimum distance between any two positions (start-start, goal-goal, or start-goal)
    
    Returns:
        start_l: List of n_agents start position tensors, each of shape [2]
        goal_l: List of n_agents goal position tensors, each of shape [2]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    # Calculate scale and offset for bounds
    scale = (bounds[1] - bounds[0]) / 2
    offset = (bounds[1] + bounds[0]) / 2
    
    max_tries = 10000  # Maximum attempts to find valid positions
    
    def generate_positions(num_positions, avoid_positions=None):
        """
        Generate num_positions that are all at least constraint_radius apart.
        
        Args:
            num_positions: Number of positions to generate
            avoid_positions: Optional list of positions to avoid (e.g., start positions when generating goals)
        """
        positions = []
        
        for i in range(num_positions):
            attempts = 0
            while attempts < max_tries:
                # Generate random position within bounds
                candidate = torch.rand(2, **tensor_args) * 2 - 1  # Random in [-1, 1]
                candidate = candidate * scale + offset
                
                # Check if candidate is far enough from all existing positions
                is_valid = True
                
                # Check against existing positions in the current list
                if len(positions) > 0:
                    existing_positions = torch.stack(positions)  # Shape: [len(positions), 2]
                    distances = torch.norm(existing_positions - candidate.unsqueeze(0), dim=1)
                    if torch.any(distances < constraint_radius):
                        is_valid = False
                
                # Check against positions to avoid (e.g., start positions when generating goals)
                if is_valid and avoid_positions is not None and len(avoid_positions) > 0:
                    avoid_positions_tensor = torch.stack(avoid_positions)  # Shape: [len(avoid_positions), 2]
                    distances_to_avoid = torch.norm(avoid_positions_tensor - candidate.unsqueeze(0), dim=1)
                    if torch.any(distances_to_avoid < constraint_radius):
                        is_valid = False
                
                if is_valid:
                    positions.append(candidate)
                    break
                
                attempts += 1
            
            if attempts >= max_tries:
                # If we couldn't find a valid position, use the last candidate anyway
                # (this shouldn't happen often if bounds and constraint_radius are reasonable)
                print(f"WARNING: Could not find valid position {i+1}/{num_positions} after {max_tries} attempts, using last candidate")
                positions.append(candidate)
        
        return positions
    
    # Generate start positions
    start_l = generate_positions(n_agents)
    
    # Generate goal positions, ensuring they are also separated from all start positions
    goal_l = generate_positions(n_agents, avoid_positions=start_l)
    
    return start_l, goal_l


# DECENTRALIZED CBS PLANNER

if __name__ == "__main__":
    model_id = 'EnvEmptyNoWait2D-RobotPlanarDisk'  # Use an empty environment for testing
    num_timesteps = 64
    num_agents = 15 
    start_l, goal_l = generate_safe_start_goal_positions(n_agents=num_agents)
    trajectories = decentralized_planner(start_l, goal_l, collision_radius=0.1, model_id=model_id, num_timesteps=num_timesteps)

    # Plot trajectories
    print("\n" + "=" * 60)
    print("Multi-Agent Planning Complete!")
    print("=" * 60)
    print(f"Number of agents: {len(trajectories)}")
    for i, traj in enumerate(trajectories):
        print(f"Agent {i} trajectory shape: {traj.shape}")
    
    # Convert trajectories list to numpy array of shape (num_agents, time_steps, 4)
    trajectories_array = torch.stack(trajectories).cpu().detach().numpy()
    
    # Convert start and goal lists to numpy arrays of shape (num_agents, 2)
    starts_array = torch.stack(start_l).cpu().detach().numpy()
    goals_array = torch.stack(goal_l).cpu().detach().numpy()
    
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Save trajectory gif
    output_path = os.path.join(LOGS_DIR, 'multi_agent_trajectory.gif')
    create_trajectory_gif(
        trajectories=trajectories_array,
        starts=starts_array,
        goals=goals_array,
        output_path=output_path,
        fps=10,
        figsize=(12, 12),
        agent_radius=0.05,
        show_velocity_arrows=False,
        title=f'Multi-Agent Trajectories ({num_agents} agents)',
        dpi=150
    )
    
    print(f"\nGIF saved to: {output_path}")



