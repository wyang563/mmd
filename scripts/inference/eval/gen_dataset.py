"""
Script to generate random start/goal positions for varying numbers of agents.
Generates a fixed number of datasets per agent count.
"""
import os
import json
import torch
from pathlib import Path

# Project imports
from mmd.common import get_start_goal_pos_random_in_env
from torch_robotics.environments import EnvDropRegion2D
from torch_robotics.torch_utils.torch_utils import get_torch_device

# Setup
device = 'cuda'
device = get_torch_device(device)
tensor_args = {'device': device, 'dtype': torch.float32}

def tensor_to_list(tensor):
    """Convert torch tensor to Python list."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().tolist()
    return tensor

def generate_dataset(num_agents_range=(2, 20), datapoints_per_agent=2):
    """
    Generate random start/goal positions for varying numbers of agents.
    Generates a fixed number of datasets for each agent count.
    
    Args:
        num_agents_range: Tuple of (min_agents, max_agents) inclusive
        datapoints_per_agent: Number of datasets to generate for each agent count
    
    Returns:
        Dictionary with structure: {num_agents: [list of {start_positions, goal_positions}]}
    """
    dataset = {}
    min_agents, max_agents = num_agents_range
    
    for num_agents in range(min_agents, max_agents + 1):
        print(f"Generating {datapoints_per_agent} datasets for {num_agents} agents...")
        agent_datasets = []
        
        for dataset_idx in range(datapoints_per_agent):
            # Generate random start/goal positions
            start_state_pos_l, goal_state_pos_l = get_start_goal_pos_random_in_env(
                num_agents,
                EnvDropRegion2D,
                tensor_args,
                margin=0.2,
                obstacle_margin=0.11
            )
            
            # Convert tensors to lists for JSON serialization
            start_positions = [tensor_to_list(pos) for pos in start_state_pos_l]
            goal_positions = [tensor_to_list(pos) for pos in goal_state_pos_l]
            
            agent_datasets.append({
                'start_positions': start_positions,
                'goal_positions': goal_positions
            })
        
        dataset[num_agents] = agent_datasets
        print(f"Completed {num_agents} agents ({datapoints_per_agent} datasets)\n")
    
    return dataset

def save_dataset(dataset, output_path):
    """Save dataset to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {output_path}")

if __name__ == '__main__':
    # Generate dataset
    print("=" * 60)
    print("Generating random start/goal position dataset")
    print(f"Generating {2} datasets per agent count (2-20)")
    print("=" * 60)
    
    dataset = generate_dataset(num_agents_range=(2, 20), datapoints_per_agent=2)
    
    # Save to JSON file in the inference folder
    script_dir = Path(__file__).parent.parent  # Go up from eval/ to inference/
    output_path = script_dir / 'start_goal_positions_dataset.json'
    
    save_dataset(dataset, output_path)
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    total = sum(len(datasets) for datasets in dataset.values())
    print(f"Total datasets: {total}")
    print(f"Number of agents range: {min(dataset.keys())} to {max(dataset.keys())}")
    print(f"Datasets per agent count: {dict((k, len(v)) for k, v in dataset.items())}")
    print("=" * 60)

