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
    agent_radius: float = 0.05,
    show_velocity_arrows: bool = False,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    dpi: int = 300
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
    # Use different colormaps for better distinctiveness
    if num_agents <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:num_agents]
    elif num_agents <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:num_agents]
    else:
        # For many agents, use hsv colormap which provides good spread
        colors = plt.cm.hsv(np.linspace(0, 0.95, num_agents))
    
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
        ax.plot(start_x, start_y, 'o', color=color, markersize=12, zorder=5)
        
        # Add agent number text label at start position
        ax.text(start_x, start_y, str(agent_idx), color='white', 
                fontsize=8, fontweight='bold', ha='center', va='center', 
                zorder=6)
        
        # Goal position (star marker)
        goal_x, goal_y = goals[agent_idx, 0], goals[agent_idx, 1]
        ax.plot(goal_x, goal_y, '*', color=color, markersize=20, zorder=5)
        
        # Add agent number text label at goal position
        ax.text(goal_x, goal_y, str(agent_idx), color='white', 
                fontsize=8, fontweight='bold', ha='center', va='center', 
                zorder=6)
    
    # Add legend for start and goal markers (not per-agent)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Start Position',
                   markerfacecolor='gray', markersize=10),
        plt.Line2D([0], [0], marker='*', color='w', label='Goal Position',
                   markerfacecolor='gray', markersize=15)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
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
            # Set empty data to hide trailing lines - only show moving circles
            trajectory_lines[agent_idx].set_data([], [])
            
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


def create_masked_trajectory_gif(
    trajectories,
    starts,
    goals,
    agent_id: int,
    mask,
    output_path: str,
    fps: int = 10,
    figsize: Tuple[int, int] = (10, 10),
    agent_radius: float = 0.05,
    show_velocity_arrows: bool = False,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    dpi: int = 300
) -> None:
    """
    Create a gif animation of multi-agent trajectories with masking visualization.
    
    The ego agent (agent_id) is highlighted in blue, agents in the mask are shown in red,
    and agents not in the mask are shown in gray.
    
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
    agent_id : int
        The ID of the ego agent to highlight in blue
    mask : list of np.ndarray or torch.Tensor
        List of length time_steps, where each element is a binary mask array of shape 
        (num_agents, num_agents). mask[t][i][j] indicates whether agent i can see agent j
        at time step t (1 = visible/in mask, 0 = not visible).
        Can also be a single np.ndarray or torch.Tensor of shape (time_steps, num_agents, num_agents).
    output_path : str
        Path to save the output gif file
    fps : int, optional
        Frames per second for the gif, by default 10
    figsize : Tuple[int, int], optional
        Figure size in inches, by default (10, 10)
    agent_radius : float, optional
        Radius of agent circles in the plot, by default 0.05
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
    
    # Handle mask input - convert list of tensors to numpy array
    if isinstance(mask, list):
        # List of tensors/arrays, each of shape (num_agents, num_agents)
        mask_list = []
        for m in mask:
            if hasattr(m, 'cpu'):
                mask_list.append(m.cpu().detach().numpy())
            elif isinstance(m, np.ndarray):
                mask_list.append(m)
            else:
                mask_list.append(np.array(m))
        # Stack into shape (time_steps, num_agents, num_agents)
        mask_array = np.stack(mask_list, axis=0)
    else:
        # Single array or tensor
        if hasattr(mask, 'cpu'):
            mask_array = mask.cpu().detach().numpy()
        elif isinstance(mask, np.ndarray):
            mask_array = mask
        else:
            mask_array = np.array(mask)
    
    # Extract mask for the specified agent_id
    # mask_array shape: (time_steps, num_agents, num_agents)
    # mask_array[t, agent_id, :] gives which agents are visible to agent_id at time t
    if mask_array.ndim != 3:
        raise ValueError(f"Unexpected mask shape: {mask_array.shape}. Expected (time_steps, num_agents, num_agents)")
    
    if mask_array.shape[0] != time_steps:
        raise ValueError(f"Mask time dimension {mask_array.shape[0]} does not match trajectories time dimension {time_steps}")
    
    if mask_array.shape[1] != num_agents or mask_array.shape[2] != num_agents:
        raise ValueError(f"Mask agent dimensions {mask_array.shape[1:]}, do not match number of agents {num_agents}")
    
    # Extract the mask for the ego agent: shape (time_steps, num_agents)
    agent_mask = mask_array[:, agent_id, :]
    
    # Define colors
    ego_color = 'blue'
    masked_color = 'red'
    unmasked_color = 'gray'
    
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
    else:
        ax.set_title(f'Agent {agent_id} Perspective (Masked View)')
    
    # Plot start positions and goal positions (static elements)
    for agent_idx in range(num_agents):
        if agent_idx == agent_id:
            color = ego_color
        else:
            # Use gray for start/goal (will be colored during animation)
            color = 'lightgray'
        
        # Start position (circle marker)
        start_x, start_y = starts[agent_idx, 0], starts[agent_idx, 1]
        ax.plot(start_x, start_y, 'o', color=color, markersize=12, zorder=5, alpha=0.6)
        
        # Add agent number text label at start position
        ax.text(start_x, start_y, str(agent_idx), color='white', 
                fontsize=8, fontweight='bold', ha='center', va='center', 
                zorder=6)
        
        # Goal position (star marker)
        goal_x, goal_y = goals[agent_idx, 0], goals[agent_idx, 1]
        ax.plot(goal_x, goal_y, '*', color=color, markersize=20, zorder=5, alpha=0.6)
        
        # Add agent number text label at goal position
        ax.text(goal_x, goal_y, str(agent_idx), color='white', 
                fontsize=8, fontweight='bold', ha='center', va='center', 
                zorder=6)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Ego Agent {agent_id}',
                   markerfacecolor=ego_color, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Masked Agents',
                   markerfacecolor=masked_color, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Unmasked Agents',
                   markerfacecolor=unmasked_color, markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Start Position',
                   markerfacecolor='lightgray', markersize=10),
        plt.Line2D([0], [0], marker='*', color='w', label='Goal Position',
                   markerfacecolor='lightgray', markersize=15)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Initialize trajectory lines and agent circles
    trajectory_lines = []
    agent_circles = []
    velocity_arrows = []
    
    for agent_idx in range(num_agents):
        # Initial color (will be updated in animation)
        if agent_idx == agent_id:
            color = ego_color
        else:
            color = unmasked_color
        
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
        # Get mask for current frame
        current_mask = agent_mask[frame]  # Shape: (num_agents,)
        
        # Update trajectory lines (show path up to current frame)
        for agent_idx in range(num_agents):
            # Determine color based on agent role
            if agent_idx == agent_id:
                color = ego_color
            elif current_mask[agent_idx]:
                color = masked_color
            else:
                color = unmasked_color
            
            traj_x = trajectories[agent_idx, :frame+1, 0]
            traj_y = trajectories[agent_idx, :frame+1, 1]
            trajectory_lines[agent_idx].set_data(traj_x, traj_y)
            trajectory_lines[agent_idx].set_color(color)
            
            # Update agent position
            current_x = trajectories[agent_idx, frame, 0]
            current_y = trajectories[agent_idx, frame, 1]
            agent_circles[agent_idx].center = (current_x, current_y)
            agent_circles[agent_idx].set_color(color)
            
            # Update velocity arrows if enabled
            if show_velocity_arrows and frame < time_steps:
                vx = trajectories[agent_idx, frame, 2]
                vy = trajectories[agent_idx, frame, 3]
                # Remove old arrow and create new one
                if velocity_arrows[agent_idx] in ax.patches:
                    velocity_arrows[agent_idx].remove()
                
                arrow = ax.arrow(current_x, current_y, vx, vy, 
                               head_width=0.3, head_length=0.3,
                               fc=color, ec=color, 
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
    print(f"Masked trajectory gif saved to: {output_path}")


def create_single_agent_trajectory_gif_with_constraints(
    trajectory,
    start,
    goal,
    constraints,
    output_path: str,
    fps: int = 10,
    figsize: Tuple[int, int] = (10, 10),
    agent_radius: float = 0.05,
    constraint_radius: float = 0.1,
    show_velocity_arrows: bool = False,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    dpi: int = 300
) -> None:
    """
    Create a gif animation of a single agent trajectory with constraint points.
    
    Constraint points appear at their specified timesteps and remain visible.
    
    Parameters
    ----------
    trajectory : np.ndarray or torch.Tensor
        Trajectory array of shape (time_steps, 4) or (time_steps, 2) where:
        - If shape is (time_steps, 4): (px, py, vx, vy) - position and velocity
        - If shape is (time_steps, 2): (px, py) - position only
    start : np.ndarray or torch.Tensor
        Start position of shape (2,) containing (x, y) coordinates
    goal : np.ndarray or torch.Tensor
        Goal position of shape (2,) containing (x, y) coordinates
    constraints : list of list of list
        Constraints as a 3D list indexed by [timestep][point_idx][coord].
        Format: constraints[timestep][point_idx][coord] where:
        - timestep: time step index (0 to time_steps-1)
        - point_idx: index of constraint point at that timestep
        - coord: [x, y] coordinates of the constraint point
        Each timestep can have zero or more constraint points.
    output_path : str
        Path to save the output gif file
    fps : int, optional
        Frames per second for the gif, by default 10
    figsize : Tuple[int, int], optional
        Figure size in inches, by default (10, 10)
    agent_radius : float, optional
        Radius of agent circle in the plot, by default 0.05
    constraint_radius : float, optional
        Radius of constraint point circles, by default 0.1
    show_velocity_arrows : bool, optional
        Whether to show velocity arrows, by default False
    title : Optional[str], optional
        Title for the plot, by default None
    xlim : Optional[Tuple[float, float]], optional
        X-axis limits, by default None (auto-computed from data)
    ylim : Optional[Tuple[float, float]], optional
        Y-axis limits, by default None (auto-computed from data)
    dpi : int, optional
        DPI for the output gif, by default 300
    """
    # Convert torch tensors to numpy arrays if needed
    if hasattr(trajectory, 'cpu'):
        trajectory = trajectory.cpu().detach().numpy()
    elif not isinstance(trajectory, np.ndarray):
        trajectory = np.array(trajectory)
    
    if hasattr(start, 'cpu'):
        start = start.cpu().detach().numpy()
    elif not isinstance(start, np.ndarray):
        start = np.array(start)
    
    if hasattr(goal, 'cpu'):
        goal = goal.cpu().detach().numpy()
    elif not isinstance(goal, np.ndarray):
        goal = np.array(goal)
    
    # Ensure trajectory is 2D
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(1, -1)
    
    time_steps, traj_dim = trajectory.shape
    
    # Handle trajectory dimensions - add velocity if needed
    if traj_dim == 2:
        # Only position, add zero velocity
        positions = trajectory
        velocities = np.zeros_like(positions)
        if time_steps > 1:
            # Compute velocity from finite differences
            velocities[:-1] = positions[1:] - positions[:-1]
            velocities[-1] = velocities[-2] if time_steps > 1 else velocities[0]
        trajectory_full = np.concatenate([positions, velocities], axis=-1)
    elif traj_dim == 4:
        # Already has position and velocity
        trajectory_full = trajectory
    else:
        raise ValueError(f"Unexpected trajectory dimension: {traj_dim}. Expected 2 (position) or 4 (position+velocity).")
    
    # Ensure start and goal are 1D arrays of length 2
    start = np.array(start).flatten()[:2]
    goal = np.array(goal).flatten()[:2]
    
    # Normalize constraints format - ensure it's a list of lists
    # Convert constraints to a more usable format: list of (timestep, point) tuples
    constraint_points = []  # List of (timestep, x, y) tuples
    for t in range(len(constraints)):
        if t < time_steps:  # Only process constraints within trajectory length
            for point in constraints[t]:
                if len(point) >= 2:
                    constraint_points.append((t, float(point[0]), float(point[1])))
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute axis limits if not provided
    if xlim is None:
        all_x = [trajectory_full[:, 0], start[0], goal[0]]
        for _, x, _ in constraint_points:
            all_x.append(x)
        all_x = np.concatenate([np.array(x).flatten() for x in all_x])
        if len(all_x) > 0:
            x_margin = (all_x.max() - all_x.min()) * 0.1 if all_x.max() != all_x.min() else 0.1
            xlim = (all_x.min() - x_margin, all_x.max() + x_margin)
        else:
            xlim = (-1, 1)
    
    if ylim is None:
        all_y = [trajectory_full[:, 1], start[1], goal[1]]
        for _, _, y in constraint_points:
            all_y.append(y)
        all_y = np.concatenate([np.array(y).flatten() for y in all_y])
        if len(all_y) > 0:
            y_margin = (all_y.max() - all_y.min()) * 0.1 if all_y.max() != all_y.min() else 0.1
            ylim = (all_y.min() - y_margin, all_y.max() + y_margin)
        else:
            ylim = (-1, 1)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Single Agent Trajectory with Constraints')
    
    # Plot start and goal positions (static elements)
    agent_color = 'blue'
    constraint_color = 'red'
    
    # Start position (circle marker)
    ax.plot(start[0], start[1], 'o', color=agent_color, markersize=12, zorder=5, label='Start')
    ax.text(start[0], start[1], 'S', color='white', fontsize=8, fontweight='bold', 
            ha='center', va='center', zorder=6)
    
    # Goal position (star marker)
    ax.plot(goal[0], goal[1], '*', color=agent_color, markersize=20, zorder=5, label='Goal')
    ax.text(goal[0], goal[1], 'G', color='white', fontsize=8, fontweight='bold', 
            ha='center', va='center', zorder=6)
    
    # Initialize trajectory line and agent circle
    trajectory_line, = ax.plot([], [], '-', color=agent_color, linewidth=2, alpha=0.6, zorder=3, label='Trajectory')
    agent_circle = Circle((0, 0), agent_radius, color=agent_color, alpha=0.8, zorder=10)
    ax.add_patch(agent_circle)
    
    # Initialize constraint circles (will be added dynamically)
    constraint_circles = []
    
    # Velocity arrow (optional) - use a list to store it so it can be modified in update
    velocity_arrow_list = [None]
    if show_velocity_arrows:
        velocity_arrow_list[0] = ax.arrow(0, 0, 0, 0, head_width=0.3, head_length=0.3, 
                                          fc=agent_color, ec=agent_color, alpha=0.7, zorder=8)
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Start',
                   markerfacecolor=agent_color, markersize=10),
        plt.Line2D([0], [0], marker='*', color='w', label='Goal',
                   markerfacecolor=agent_color, markersize=15),
        plt.Line2D([0], [0], color=agent_color, label='Trajectory', linewidth=2),
        plt.Line2D([0], [0], marker='o', color='w', label='Constraint',
                   markerfacecolor=constraint_color, markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Animation update function
    def update(frame):
        # Update trajectory line (show path up to current frame)
        # Set to empty arrays to prevent drawing the trajectory trace
        trajectory_line.set_data([], [])
        
        # Update agent position
        current_x = trajectory_full[frame, 0]
        current_y = trajectory_full[frame, 1]
        agent_circle.center = (current_x, current_y)
        
        # Remove old constraint circles from previous frames
        for circle in constraint_circles:
            if circle in ax.patches:
                circle.remove()
        constraint_circles.clear()
        
        # Add constraint points that appear ONLY at this exact timestep
        for t, x, y in constraint_points:
            if t == frame:  # Only show constraint at its exact timestep
                circle = Circle((x, y), constraint_radius, color=constraint_color, 
                               alpha=0.6, zorder=7, edgecolor='darkred', linewidth=1.5)
                ax.add_patch(circle)
                constraint_circles.append(circle)
        
        # Update velocity arrow if enabled
        if show_velocity_arrows and frame < time_steps:
            vx = trajectory_full[frame, 2]
            vy = trajectory_full[frame, 3]
            if velocity_arrow_list[0] is not None and velocity_arrow_list[0] in ax.patches:
                velocity_arrow_list[0].remove()
            
            velocity_arrow_list[0] = ax.arrow(current_x, current_y, vx, vy, 
                                             head_width=0.3, head_length=0.3,
                                             fc=agent_color, ec=agent_color, 
                                             alpha=0.7, zorder=8)
            ax.add_patch(velocity_arrow_list[0])
        
        # Update time text
        time_text.set_text(f'Time Step: {frame}/{time_steps-1}')
        
        artists = [trajectory_line, agent_circle, time_text] + constraint_circles
        if show_velocity_arrows and velocity_arrow_list[0] is not None:
            artists.append(velocity_arrow_list[0])
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
    print(f"Single agent trajectory gif with constraints saved to: {output_path}")
