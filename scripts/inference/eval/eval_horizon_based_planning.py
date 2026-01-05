"""
Evaluation script for horizon-based multi-agent planning.
Iterates through a dataset of start/goal positions and runs planning trials.
"""
import os
import json
from datetime import datetime
from pathlib import Path

import torch
import sys
import importlib.util

# Import inference_horizon_based_multi_agent using importlib
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
INFERENCE_MODULE_PATH = os.path.join(PARENT_DIR, 'inference_horizon_based_multi_agent.py')

spec = importlib.util.spec_from_file_location("inference_horizon_based_multi_agent", INFERENCE_MODULE_PATH)
inference_module = importlib.util.module_from_spec(spec)
sys.modules["inference_horizon_based_multi_agent"] = inference_module
spec.loader.exec_module(inference_module)

run_multi_agent_trial = inference_module.run_multi_agent_trial
tensor_args = inference_module.tensor_args
from mmd.common.experiments import MultiAgentPlanningSingleTrialConfig
from mmd.common.pretty_print import *

# Configuration
DATASET_PATH = os.path.join(SCRIPT_DIR, 'start_goal_positions_dataset.json')
PLAYER_SELECTION_MODEL_PATH = "/home/alex/gnn_game_planning/best_models/point_agent/gnn_full_MP_2_edge-metric_barrier-function_top-k_5/train_n_agents_20_T_50_obs_10_lr_0.0001_bs_32_sigma1_0.63_sigma2_0.63_sigma3_0.063_noise_std_0.5_epochs_20_loss_type_ego_agent_cost/20251231_015734/psn_best_model.pkl"
# PLAYER_SELECTION_MODEL_PATH = None  # Set to None to disable player selection model

# Planning parameters
STRIDE = 16
MODEL_PAST_HORIZON = 10
RUNTIME_LIMIT = 60 * 3  # 3 minutes
MULTI_AGENT_PLANNER_CLASS = "XECBS"  # Options: "CBS", "ECBS", "XCBS", "XECBS", "PP"
SINGLE_AGENT_PLANNER_CLASS = "MPDEnsemble"  # Options: "MPD", "MPDEnsemble"
STAGGER_START_TIME_DT = 0

# Environment configuration
GLOBAL_MODEL_IDS = [['EnvEmptyNoWait2D-RobotPlanarDisk']]
# GLOBAL_MODEL_IDS = [['EnvEmpty2D-RobotPlanarDisk']]
# GLOBAL_MODEL_IDS = [['EnvConveyor2D-RobotPlanarDisk']]
# GLOBAL_MODEL_IDS = [['EnvHighways2D-RobotPlanarDisk']]
# GLOBAL_MODEL_IDS = [['EnvDropRegion2D-RobotPlanarDisk']]


def load_dataset(dataset_path: str) -> dict:
    """Load the start/goal positions dataset from JSON."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset


def create_test_config(
    num_agents: int,
    start_positions: list,
    goal_positions: list,
    trial_number: int,
    time_str: str,
) -> MultiAgentPlanningSingleTrialConfig:
    """Create a test configuration from start/goal positions."""
    config = MultiAgentPlanningSingleTrialConfig()
    
    # Basic configuration
    config.num_agents = num_agents
    config.instance_name = "eval_horizon_based"
    config.multi_agent_planner_class = MULTI_AGENT_PLANNER_CLASS
    config.single_agent_planner_class = SINGLE_AGENT_PLANNER_CLASS
    config.stagger_start_time_dt = STAGGER_START_TIME_DT
    config.runtime_limit = RUNTIME_LIMIT
    config.time_str = time_str
    config.trial_number = trial_number
    config.render_animation = True
    
    # Environment configuration
    config.global_model_ids = GLOBAL_MODEL_IDS
    config.agent_skeleton_l = [[[0, 0]]] * num_agents
    
    # Convert start/goal positions to tensors
    config.start_state_pos_l = [
        torch.tensor(pos, **tensor_args) for pos in start_positions
    ]
    config.goal_state_pos_l = [
        torch.tensor(pos, **tensor_args) for pos in goal_positions
    ]
    
    return config


def run_evaluation(
    dataset_path: str = DATASET_PATH,
    model_path: str = PLAYER_SELECTION_MODEL_PATH,
    stride: int = STRIDE,
    model_past_horizon: int = MODEL_PAST_HORIZON,
):
    """Run evaluation on all test cases in the dataset."""
    print(f"{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}Starting Horizon-Based Multi-Agent Planning Evaluation{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")
    print(f"Dataset: {dataset_path}")
    print(f"Model path: {model_path if model_path else 'None (no player selection)'}")
    print(f"Stride: {stride}")
    print(f"Model past horizon: {model_past_horizon}")
    print(f"Runtime limit: {RUNTIME_LIMIT}s")
    print(f"Multi-agent planner: {MULTI_AGENT_PLANNER_CLASS}")
    print(f"Single-agent planner: {SINGLE_AGENT_PLANNER_CLASS}")
    print(f"Environment: {GLOBAL_MODEL_IDS[0][0]}")
    print(f"{BLUE}{'='*80}{RESET}\n")
    
    # Load dataset
    dataset = load_dataset(dataset_path)
    
    # Create time string for this evaluation run
    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Statistics
    total_trials = 0
    successful_trials = 0
    failed_trials = 0
    
    # Iterate through all test cases
    for num_agents_str, test_cases in dataset.items():
        num_agents = int(num_agents_str)
        print(f"\n{CYAN}{'='*80}{RESET}")
        print(f"{CYAN}Processing {num_agents} agents: {len(test_cases)} test cases{RESET}")
        print(f"{CYAN}{'='*80}{RESET}\n")
        
        for trial_idx, test_case in enumerate(test_cases):
            trial_number = total_trials
            total_trials += 1
            
            print(f"\n{GREEN}{'='*60}{RESET}")
            print(f"{GREEN}Trial {trial_number + 1}/{sum(len(cases) for cases in dataset.values())}{RESET}")
            print(f"{GREEN}Num agents: {num_agents}, Test case: {trial_idx + 1}/{len(test_cases)}{RESET}")
            print(f"{GREEN}{'='*60}{RESET}")
            
            # Extract start and goal positions
            start_positions = test_case['start_positions']
            goal_positions = test_case['goal_positions']
            
            # Validate positions
            if len(start_positions) != num_agents or len(goal_positions) != num_agents:
                print(f"{YELLOW}Warning: Mismatch in number of positions. Skipping.{RESET}")
                failed_trials += 1
                continue
            
            # Create test configuration
            try:
                test_config = create_test_config(
                    num_agents=num_agents,
                    start_positions=start_positions,
                    goal_positions=goal_positions,
                    trial_number=trial_number,
                    time_str=time_str,
                )
                
                # Run the trial
                print(f"{BLUE}Running planning trial...{RESET}")
                final_trajectories = run_multi_agent_trial(
                    test_config=test_config,
                    stride=stride,
                    model_past_horizon=model_past_horizon,
                    model_path=model_path,
                )

                print(f"Final trajectories: {final_trajectories}")
                
                successful_trials += 1
                print(f"{GREEN}✓ Trial {trial_number + 1} completed successfully{RESET}\n")
                
            except Exception as e:
                failed_trials += 1
                print(f"{RED}✗ Trial {trial_number + 1} failed with error: {e}{RESET}\n")
                import traceback
                traceback.print_exc()
    
    # Print summary
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{CYAN}Evaluation Summary{RESET}")
    print(f"{CYAN}{'='*80}{RESET}")
    print(f"Total trials: {total_trials}")
    print(f"Successful: {successful_trials} ({100 * successful_trials / total_trials if total_trials > 0 else 0:.1f}%)")
    print(f"Failed: {failed_trials} ({100 * failed_trials / total_trials if total_trials > 0 else 0:.1f}%)")
    print(f"{CYAN}{'='*80}{RESET}\n")


if __name__ == '__main__':
    # Run evaluation
    run_evaluation(
        dataset_path=DATASET_PATH,
        model_path=PLAYER_SELECTION_MODEL_PATH,
        stride=STRIDE,
        model_past_horizon=MODEL_PAST_HORIZON,
    )
    print(f"{GREEN}Evaluation complete!{RESET}")

