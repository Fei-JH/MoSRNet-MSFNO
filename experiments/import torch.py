
import torch


def assemble_condensed_matrices_torch(
    dmgfields: torch.Tensor,
    L: float,
    E: float,
    I: float,
    rho: float,
    A: float,
    n_elements: int,
    mass_dmg_power: float = 2.0,
    device: torch.device | None = None,
):
    """
    Assemble condensed mass and stiffness matrices for a 2-DOF/-node Euler beam.
    Only translational DOFs are retained via static condensation.

    Parameters
    ----------
    dmgfields : Tensor, shape [samples, 1, n_elements] or [samples, n_elements]
        Element-wise damage coefficients (>0). 1 = intact.
    L, E, I, rho, A : float
        Beam geometry & material constants.
    n_elements : int
        Number of finite elements.
    mass_dmg_power : float, optional
        Mass scaling exponent (m_coeff = dmg ** (-mass_dmg_power)).
    device : torch.device, optional
        Target device. Defaults to dmgfields.device.

    Returns
    -------
    M_cond : Tensor, shape [samples, n_nodes, n_nodes]
    K_cond : Tensor, shape [samples, n_nodes, n_nodes]
        Condensed matrices defined only on translational DOFs.
    """
    # ------------------------------------------------------------------ #
    # ---------- basic sanity & reshape -------------------------------- #
    # ------------------------------------------------------------------ #
    if dmgfields.dim() == 3:
        dmgfields = dmgfields.squeeze(1)          # [samples, n_elements]
    if dmgfields.dim() != 2 or dmgfields.size(1) != n_elements:
        raise ValueError("dmgfields must be [samples, n_elements]")

    device = device or dmgfields.device
    dtype = dmgfields.dtype
    samples = dmgfields.size(0)
    n_nodes = n_elements + 1
    dof_total = 2 * n_nodes                     # 2 DOF/node: w, θ

    # ------------------------------------------------------------------ #
    # ---------- element matrices (4×4) -------------------------------- #
    # ------------------------------------------------------------------ #
    Le = L / n_elements
    Ke_base = (E * I / Le**3) * torch.tensor(
        [
            [12, 6 * Le, -12, 6 * Le],
            [6 * Le, 4 * Le**2, -6 * Le, 2 * Le**2],
            [-12, -6 * Le, 12, -6 * Le],
            [6 * Le, 2 * Le**2, -6 * Le, 4 * Le**2],
        ],
        dtype=dtype,
        device=device,
    )
    Me_base = (rho * A * Le / 420) * torch.tensor(
        [
            [156, 22 * Le, 54, -13 * Le],
            [22 * Le, 4 * Le**2, 13 * Le, -3 * Le**2],
            [54, 13 * Le, 156, -22 * Le],
            [-13 * Le, -3 * Le**2, -22 * Le, 4 * Le**2],
        ],
        dtype=dtype,
        device=device,
    )

    # ------------------------------------------------------------------ #
    # ---------- global matrices initialisation ------------------------ #
    # ------------------------------------------------------------------ #
    K_glb = torch.zeros(samples, dof_total, dof_total, dtype=dtype, device=device)
    M_glb = torch.zeros_like(K_glb)

    # pre-compute DOF indices for each element
    elem_dof = torch.tensor(
        [[2 * i, 2 * i + 1, 2 * (i + 1), 2 * (i + 1) + 1] for i in range(n_elements)],
        dtype=torch.long,
        device=device,
    )

    # ------------------------------------------------------------------ #
    # ---------- assembly loop (vectorised over samples) --------------- #
    # ------------------------------------------------------------------ #
    for e in range(n_elements):
        idx = elem_dof[e]                         # (4,)
        dmg = dmgfields[:, e]                    # (samples,)
        K_e = Ke_base * dmg[:, None, None]        # (samples,4,4)
        mass_coeff = dmg.pow(-mass_dmg_power)
        M_e = Me_base * mass_coeff[:, None, None]

        # add element contributions (broadcasted on batch dim)
        K_glb[:, idx[:, None], idx] += K_e
        M_glb[:, idx[:, None], idx] += M_e

    # enforce symmetry (floating error safeguard)
    K_glb = 0.5 * (K_glb + K_glb.transpose(-1, -2))
    M_glb = 0.5 * (M_glb + M_glb.transpose(-1, -2))

    # ------------------------------------------------------------------ #
    # ---------- static condensation ----------------------------------- #
    # ------------------------------------------------------------------ #
    trans = torch.arange(0, dof_total, 2, device=device)   # w DOFs
    rot   = torch.arange(1, dof_total, 2, device=device)   # θ DOFs

    # block extraction
    Kpp, Kpr, Krp, Krr = (
        K_glb[:, trans][:, :, trans],
        K_glb[:, trans][:, :, rot],
        K_glb[:, rot][:, :, trans],
        K_glb[:, rot][:, :, rot],
    )
    Mpp, Mpr, Mrp, Mrr = (
        M_glb[:, trans][:, :, trans],
        M_glb[:, trans][:, :, rot],
        M_glb[:, rot][:, :, trans],
        M_glb[:, rot][:, :, rot],
    )

    # inverse on rotation sub-blocks
    Krr_inv = torch.linalg.inv(Krr)
    Mrr_inv = torch.linalg.inv(Mrr)

    # condensed matrices
    K_cond = Kpp - Kpr @ Krr_inv @ Krp
    M_cond = Mpp - Mpr @ Mrr_inv @ Mrp

    return M_cond, K_cond

if __name__ == "__main__":
    import torch
    import time

    # Beam parameters
    L, E, I = 10.0, 210e9, 8.5e-6
    rho, A = 7850.0, 0.02
    n_elements = 540

    # Device
    device = torch.device("cpu")

    # Toy damage field: shape [batch, 1, n_elements]
    batch_size = 16
    dmg = torch.ones(batch_size, 1, n_elements, dtype=torch.float64, device=device)
    dmg[:, :, 5:10] = 0.8  # weak zone

    # Warm-up
    _ = assemble_condensed_matrices_torch(dmg, L, E, I, rho, A, n_elements, device=device)

    # Timing
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()

    M_cond, K_cond = assemble_condensed_matrices_torch(
        dmg, L, E, I, rho, A, n_elements, device=device
    )

    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.time()

    # Output
    print(f"Shapes: M_cond={M_cond.shape}, K_cond={K_cond.shape}")
    print(f"Elapsed time: {(t1 - t0)*1000:.2f} ms on {device}")

# if __name__ == "__main__":
#     import torch

#     # beam parameters
#     L, E, I = 10.0, 210e9, 8.5e-6
#     rho, A = 7850.0, 0.02
#     n_elements = 540

#     # toy damage field (batch = 4)
#     dmg = torch.ones(4, 1, n_elements)
#     dmg[:, :, 5:10] = 0.8                       # weak zone
#     dmg = dmg.to(torch.float64)                # double precision recommended
#     dmg = dmg.cuda()
#     device=dmg.device
#     M_cond, K_cond = assemble_condensed_matrices_torch(
#         dmg, L, E, I, rho, A, n_elements, mass_dmg_power=2, device=device
#     )
#     print("Shapes:", M_cond.shape, K_cond.shape)  # -> (4, 21, 21)
