import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from typing import Optional, Tuple
import os


def create_trajectory_gif(
    trajectories,
    starts,
    goals,
    output_path: str,
    fps: int = 10,
    figsize: Tuple[int, int] = (10, 10),
    agent_radius: float = 0.01,
    show_velocity_arrows: bool = False,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    dpi: int = 100
) -> None:
    """
    Create a gif animation of multi-agent trajectories.
    
    Parameters
    ----------
    trajectories : np.ndarray or torch.Tensor
        Trajectories array of shape (num_agents, time_steps, 4) where the last 
        dimension consists of (px, py, vx, vy):
        - px, py: x, y position
        - vx, vy: x, y velocity
    starts : np.ndarray or torch.Tensor
        Start positions array of shape (num_agents, 2) where each row contains
        (x, y) coordinates of the start position for each agent
    goals : np.ndarray or torch.Tensor
        Goal positions array of shape (num_agents, 2) where each row contains
        (x, y) coordinates of the goal position for each agent
    output_path : str
        Path to save the output gif file
    fps : int, optional
        Frames per second for the gif, by default 10
    figsize : Tuple[int, int], optional
        Figure size in inches, by default (10, 10)
    agent_radius : float, optional
        Radius of agent circles in the plot, by default 0.3
    show_velocity_arrows : bool, optional
        Whether to show velocity arrows, by default False
    title : Optional[str], optional
        Title for the plot, by default None
    xlim : Optional[Tuple[float, float]], optional
        X-axis limits, by default None (auto-computed from data)
    ylim : Optional[Tuple[float, float]], optional
        Y-axis limits, by default None (auto-computed from data)
    dpi : int, optional
        DPI for the output gif, by default 100
    """
    # Convert torch tensors to numpy arrays if needed
    if hasattr(trajectories, 'cpu'):
        trajectories = trajectories.cpu().detach().numpy()
    elif not isinstance(trajectories, np.ndarray):
        trajectories = np.array(trajectories)
    
    if hasattr(starts, 'cpu'):
        starts = starts.cpu().detach().numpy()
    elif not isinstance(starts, np.ndarray):
        starts = np.array(starts)
    
    if hasattr(goals, 'cpu'):
        goals = goals.cpu().detach().numpy()
    elif not isinstance(goals, np.ndarray):
        goals = np.array(goals)
    
    num_agents, time_steps, _ = trajectories.shape
    
    # Generate distinct colors for each agent
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_agents, 10)))
    if num_agents > 10:
        colors = plt.cm.rainbow(np.linspace(0, 1, num_agents))
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute axis limits if not provided
    if xlim is None:
        all_x = trajectories[:, :, 0].flatten()
        x_margin = (all_x.max() - all_x.min()) * 0.1
        xlim = (all_x.min() - x_margin, all_x.max() + x_margin)
    
    if ylim is None:
        all_y = trajectories[:, :, 1].flatten()
        y_margin = (all_y.max() - all_y.min()) * 0.1
        ylim = (all_y.min() - y_margin, all_y.max() + y_margin)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    if title:
        ax.set_title(title)
    
    # Plot start positions and goal positions (static elements)
    for agent_idx in range(num_agents):
        color = colors[agent_idx]
        
        # Start position (circle marker)
        start_x, start_y = starts[agent_idx, 0], starts[agent_idx, 1]
        ax.plot(start_x, start_y, 'o', color=color, markersize=12, 
                label=f'Agent {agent_idx} Start', zorder=5)
        
        # Goal position (star marker)
        goal_x, goal_y = goals[agent_idx, 0], goals[agent_idx, 1]
        ax.plot(goal_x, goal_y, '*', color=color, markersize=20, 
                label=f'Agent {agent_idx} Goal', zorder=5)
    
    # Initialize trajectory lines and agent circles
    trajectory_lines = []
    agent_circles = []
    velocity_arrows = []
    
    for agent_idx in range(num_agents):
        color = colors[agent_idx]
        
        # Trajectory line (will be updated each frame)
        line, = ax.plot([], [], '-', color=color, linewidth=2, alpha=0.6, zorder=3)
        trajectory_lines.append(line)
        
        # Agent circle (current position)
        circle = Circle((0, 0), agent_radius, color=color, alpha=0.8, zorder=10)
        ax.add_patch(circle)
        agent_circles.append(circle)
        
        # Velocity arrow (optional)
        if show_velocity_arrows:
            arrow = ax.arrow(0, 0, 0, 0, head_width=0.3, head_length=0.3, 
                           fc=color, ec=color, alpha=0.7, zorder=8)
            velocity_arrows.append(arrow)
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Animation update function
    def update(frame):
        # Update trajectory lines (show path up to current frame)
        for agent_idx in range(num_agents):
            traj_x = trajectories[agent_idx, :frame+1, 0]
            traj_y = trajectories[agent_idx, :frame+1, 1]
            trajectory_lines[agent_idx].set_data(traj_x, traj_y)
            
            # Update agent position
            current_x = trajectories[agent_idx, frame, 0]
            current_y = trajectories[agent_idx, frame, 1]
            agent_circles[agent_idx].center = (current_x, current_y)
            
            # Update velocity arrows if enabled
            if show_velocity_arrows and frame < time_steps:
                vx = trajectories[agent_idx, frame, 2]
                vy = trajectories[agent_idx, frame, 3]
                # Remove old arrow and create new one
                if velocity_arrows[agent_idx] in ax.patches:
                    velocity_arrows[agent_idx].remove()
                
                arrow = ax.arrow(current_x, current_y, vx, vy, 
                               head_width=0.3, head_length=0.3,
                               fc=colors[agent_idx], ec=colors[agent_idx], 
                               alpha=0.7, zorder=8)
                velocity_arrows[agent_idx] = arrow
        
        # Update time text
        time_text.set_text(f'Time Step: {frame}/{time_steps-1}')
        
        artists = trajectory_lines + agent_circles + [time_text]
        if show_velocity_arrows:
            artists += velocity_arrows
        return artists
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=time_steps, 
        interval=1000/fps, blit=False, repeat=True
    )
    
    # Save as gif
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    
    plt.close(fig)
    print(f"Trajectory gif saved to: {output_path}")
