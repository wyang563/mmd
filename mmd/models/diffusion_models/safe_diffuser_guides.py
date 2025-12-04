import torch
from torch import nn
from torch.autograd import Variable
from qpth.qp import QPFunction, QPSolvers

@torch.no_grad()   # only for sampling
def invariance_time(
    x,
    xp1,
    constraints,
    t,
    constraint_radius: float = 0.15,
    gamma_min: float = -0.5,          # how loose the spec is early in diffusion (negative)
    num_diffusion_steps: int = 1000,  # max diffusion step index (for scheduling)
    alpha_k: float = 1.0,             # CBF class-K gain
    safety_margin: float = 0.0        # small extra margin if desired
):
    """
    Time-Varying-Safe (TVS) diffuser with time-varying constraints.

    This version:
    - Uses b_base(x) = ||x - c||^2 - R^2 as the base barrier.
    - Introduces a time-varying offset gamma(t) with gamma_min < 0 early,
      and gamma(0) = 0 at the end of diffusion.
    - Enforces the CBF:
          ∂b/∂x · u  - γ̇(t) + α (b_base(x) - γ(t)) >= 0
      which matches Eq. (12) in the paper.
    - Solves ONE QP over the whole trajectory (positions at all horizons).

    Args:
        x: Current trajectory state, shape [1, batch, horizon, state_dim]
        xp1: Next trajectory state from diffusion model, same shape as x
        constraints: 3D list [timestep][point_idx][coord] where:
            - timestep: planning timestep k (0 to horizon-1)
            - point_idx: index of constraint point at that timestep
            - coord: [x, y] coordinates in normalized space
        t: Diffusion timestep index (scalar tensor or tensor of shape [batch]).
           Assumed to be in [0, num_diffusion_steps] with large t = early diffusion.
        constraint_radius: Radius of constraint points (in normalized space).
        gamma_min: Negative value for gamma at the *earliest* diffusion step.
        num_diffusion_steps: Total number of diffusion steps for scheduling.
        alpha_k: CBF gain α.
        safety_margin: Optional extra margin added to the obstacle radius.

    Returns:
        rt: Safe trajectory with same shape as xp1.
    """
    # Extract diffusion timestep as scalar float
    if isinstance(t, torch.Tensor):
        t_single = t[0].item() if t.numel() > 0 else 0.0
    else:
        t_single = float(t)

    # Squeeze the leading 1-dim: [1, batch, horizon, state_dim] -> [batch, horizon, state_dim]
    x = x.squeeze(0)
    xp1 = xp1.squeeze(0)

    nBatch, horizon, state_dim = x.shape
    device = x.device

    # --- Extract positions ---
    # Assume first 2 dims are (x, y). If you store only positions, state_dim should be 2.
    pos_dim = 2
    x_pos = x[:, :, :pos_dim]    # [batch, horizon, 2]
    xp1_pos = xp1[:, :, :pos_dim]

    # Nominal change suggested by the diffusion model: u_nominal ≈ xp1_pos - x_pos
    ref = xp1_pos - x_pos        # [batch, horizon, 2]

    # --- Parse constraints by planning timestep k ---
    # constraints format: [timestep][point_idx][coord]
    constraint_points_by_timestep = {}
    for k in range(len(constraints)):
        if k < horizon and len(constraints[k]) > 0:
            points = []
            for point in constraints[k]:
                # point is e.g. [x, y]
                points.append([float(coord) for coord in point])
            if len(points) > 0:
                constraint_points_by_timestep[k] = torch.tensor(
                    points, device=device, dtype=x.dtype
                )  # [n_points, 2]

    # If no constraints, just return the diffuser's suggestion.
    if len(constraint_points_by_timestep) == 0:
        return xp1.unsqueeze(0)

    # --- Time-varying γ(t) schedule ---
    # We assume t_single ∈ [0, num_diffusion_steps], where:
    #   t_single ≈ num_diffusion_steps-1: early diffusion → gamma ≈ gamma_min (negative)
    #   t_single = 0: end of diffusion → gamma = 0.
    denom = max(num_diffusion_steps - 1, 1)
    tau = max(min(t_single, float(denom)), 0.0) / float(denom)  # ∈ [0, 1]
    gamma_t = gamma_min * tau               # negative early, 0 at end
    gamma_dot = gamma_min / float(denom)    # constant derivative wrt t

    # We use:
    #   b_base(x) = ||x - c||^2 - R^2 - safety_margin
    #   b_eff(x,t) = b_base(x) - gamma_t
    #   Constraint: ∂b_base/∂x · u  - γ̇(t) + α b_eff >= 0
    #   ⇒ -∂b_base/∂x · u <= -γ̇(t) + α b_eff

    # We'll solve one big QP over u ∈ R^{horizon * 2} for each batch element.
    n_vars = horizon * pos_dim

    G_rows = []   # to accumulate [batch, 1, n_vars]
    h_rows = []   # to accumulate [batch, 1]

    for k, constraint_points in constraint_points_by_timestep.items():
        # Positions at this planning time k: [batch, 2]
        x_k = x_pos[:, k, :]  # [batch, 2]
        n_points = constraint_points.shape[0]

        # For each constraint point at this timestep
        for point_idx in range(n_points):
            c_center = constraint_points[point_idx]  # [2]

            # diff: [batch, 2]
            diff = x_k - c_center.unsqueeze(0)
            dist_sq = torch.sum(diff ** 2, dim=1, keepdim=True)  # [batch, 1]

            # Base barrier: outside a circle of radius constraint_radius
            b_base = dist_sq - (constraint_radius ** 2 + safety_margin)  # [batch, 1]

            # Effective barrier with time-varying γ(t)
            b_eff = b_base - gamma_t  # [batch, 1], gamma_t is scalar

            # Gradient of b_base wrt x: ∂b_base/∂x = 2(x - c)
            grad_b = 2.0 * diff  # [batch, 2]

            # CBF constraint: ∂b_base/∂x · u - γ̇ + α b_eff >= 0
            # Rearrange: -∂b_base/∂x · u <= -γ̇ + α b_eff
            # so:
            #   G_row = -grad_b (only in block for timestep k)
            #   h_row = -gamma_dot + alpha_k * b_eff

            # Build the full G row over u ∈ R^{horizon*2} by placing -grad_b in the k-th block.
            G_full = torch.zeros(
                nBatch, 1, n_vars, device=device, dtype=x.dtype
            )  # [batch, 1, horizon*2]
            # Block for timestep k: indices [2k, 2k+2)
            start = k * pos_dim
            end = start + pos_dim
            G_full[:, 0, start:end] = -grad_b  # [batch, 2]

            h_full = -gamma_dot + alpha_k * b_eff  # [batch, 1]

            G_rows.append(G_full)
            h_rows.append(h_full)

    # If somehow there are still no constraints after parsing, return xp1
    if len(G_rows) == 0:
        return xp1.unsqueeze(0)

    # Concatenate constraint rows
    # G: [batch, n_constraints, n_vars]
    G = torch.cat(G_rows, dim=1)
    # h: [batch, n_constraints]
    h = torch.cat(h_rows, dim=1)

    # --- Build QP matrices ---
    # Objective: minimize ||u - ref||^2 = (u - ref)^T (u - ref)
    # = u^T u - 2 ref^T u + const
    # ⇒ Q = 2I, q = -2 ref (up to a scaling factor; we can just use Q=I, q=-ref)
    ref_flat = ref.reshape(nBatch, n_vars)  # [batch, horizon*2]
    q = -ref_flat

    Q = torch.eye(n_vars, device=device, dtype=x.dtype)
    Q = Q.unsqueeze(0).expand(nBatch, n_vars, n_vars)  # [batch, n_vars, n_vars]

    # No equality constraints: pass empty tensors
    e = Variable(torch.Tensor())

    # Solve batched QP: u = argmin 0.5 u^T Q u + q^T u  s.t. G u <= h
    u_opt = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(
        Q, q, G, h, e, e
    )  # [batch, n_vars]

    # Reshape and apply correction to positions
    u_opt_pos = u_opt.reshape(nBatch, horizon, pos_dim)  # [batch, horizon, 2]
    rt = xp1.clone()                                     # [batch, horizon, state_dim]
    rt[:, :, :pos_dim] = x_pos + u_opt_pos              # update positions; keep other dims from xp1
    return rt
