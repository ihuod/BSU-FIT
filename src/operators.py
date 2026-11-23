"""
Continuum-consistent adaptive convection-diffusion graph operator (CDQuad)
and discrete GNN baselines (GRAND-nl, CDE-GRAND, and diploma AdaCDE eq. (2.6)).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized, spsolve

from .meshes import MeshData


# 3-point Gauss-Legendre on [0, 1]
_GL3_PTS = 0.5 * (1.0 + np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]))
_GL3_WTS = 0.5 * np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])


@dataclass
class PhysicalLawParams:
    """
    Domain-intrinsic parameters living on Omega = [0, 1]^2 (independent of mesh size N).
    K(x) = R(theta(x))^T diag(kappa1, kappa2) R(theta(x)).
    Velocity v(x): either streamfunction psi(x) -> grad^perp psi, or direct vector field v_fn(x).
    """
    kappa1: float = 0.02
    kappa2: float = 0.02
    theta_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None  # (M,2) -> (M,) angle in rad
    psi_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None    # (M,2) -> (M,) streamfunction (0 on dOmega)
    v_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None      # (M,2) -> (M,2) velocity field
    lambda_r: float = 0.0          # linear reaction damping -lambda_r * u
    allen_cahn_eps: float = 0.0    # optional + allen_cahn_eps * (u - u^3)
    # Level 3 adaptive homogeneity gate parameters (alpha_max = 0 disables adaptation)
    alpha_max: float = 0.0
    r0: float = 0.15               # physical radius of homogeneity kernel K_{r0} on Omega
    gamma_hom: float = 4.0         # contrast sensitivity in exp(-gamma * (u_i - u_j)^2)

    def eval_tensor(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate symmetric positive-definite tensor K(x) -> shape (M, 2, 2)."""
        M = coords.shape[0]
        if self.theta_fn is None or abs(self.kappa1 - self.kappa2) < 1e-15:
            K = np.zeros((M, 2, 2), dtype=np.float64)
            K[:, 0, 0] = self.kappa1
            K[:, 1, 1] = self.kappa2
            if self.theta_fn is not None:
                th = self.theta_fn(coords)
                c, s = np.cos(th), np.sin(th)
                K[:, 0, 0] = self.kappa1 * c * c + self.kappa2 * s * s
                K[:, 1, 1] = self.kappa1 * s * s + self.kappa2 * c * c
                K[:, 0, 1] = (self.kappa1 - self.kappa2) * c * s
                K[:, 1, 0] = K[:, 0, 1]
            return K
        th = self.theta_fn(coords)
        c, s = np.cos(th), np.sin(th)
        K = np.zeros((M, 2, 2), dtype=np.float64)
        K[:, 0, 0] = self.kappa1 * c * c + self.kappa2 * s * s
        K[:, 1, 1] = self.kappa1 * s * s + self.kappa2 * c * c
        K[:, 0, 1] = (self.kappa1 - self.kappa2) * c * s
        K[:, 1, 0] = K[:, 0, 1]
        return K

    def eval_velocity(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate velocity v(x) -> shape (M, 2)."""
        if self.v_fn is not None:
            return self.v_fn(coords)
        if self.psi_fn is not None:
            eps = 1e-6
            dx = np.array([eps, 0.0])
            dy = np.array([0.0, eps])
            dpsi_dx = (self.psi_fn(coords + dx) - self.psi_fn(coords - dx)) / (2.0 * eps)
            dpsi_dy = (self.psi_fn(coords + dy) - self.psi_fn(coords - dy)) / (2.0 * eps)
            return np.column_stack([dpsi_dy, -dpsi_dx])
        return np.zeros_like(coords)

    @property
    def kappa_min(self) -> float:
        return min(self.kappa1, self.kappa2) * (1.0 - self.alpha_max)

    @property
    def kappa_max(self) -> float:
        return max(self.kappa1, self.kappa2)

    def peclet(self, probe_coords: Optional[np.ndarray] = None) -> float:
        if probe_coords is None:
            xs = np.linspace(0.0, 1.0, 41)
            xx, yy = np.meshgrid(xs, xs, indexing="ij")
            probe_coords = np.column_stack([xx.ravel(), yy.ravel()])
        v = self.eval_velocity(probe_coords)
        v_inf = float(np.max(np.linalg.norm(v, axis=1)))
        if self.alpha_max > 0:
            v_inf *= self.alpha_max
        return v_inf * np.sqrt(2.0) / max(self.kappa_min, 1e-15)


def assemble_cdquad_edge_quantities(mesh: MeshData, params: PhysicalLawParams):
    """
    Assemble edge-level symmetric diffusive weights W_e (for K(x)) and antisymmetric
    conservative fluxes F_e (from edge[e,0] -> edge[e,1]) on either:
      - Triangulated mesh (exact P1-FEM stiffness + Donald barycentric dual fluxes)
      - SLIC superpixel RAG (two-point harmonic/centroid approximation + pixel-interface flux)
    Returns:
      edges: (Ne, 2) int array with i = edges[e,0] < j = edges[e,1]
      w_diff: (Ne,) baseline diffusive conductance (>= 0 for acute Delaunay; exact P1 stiffness)
      f_conv: (Ne,) integrated velocity flux int_{Gamma_ij^*} v . nu_{i->j} ds
    """
    n_nodes = mesh.points.shape[0]
    edges = mesh.edges
    n_edges = edges.shape[0]
    edge_keys = edges[:, 0].astype(np.int64) * n_nodes + edges[:, 1].astype(np.int64)

    w_diff = np.zeros(n_edges, dtype=np.float64)
    f_conv = np.zeros(n_edges, dtype=np.float64)

    if mesh.triangles is not None:
        tris = mesh.triangles
        p0 = mesh.points[tris[:, 0]]
        p1 = mesh.points[tris[:, 1]]
        p2 = mesh.points[tris[:, 2]]
        # Twice signed area (guaranteed > 0 from _compute_triangulation_metrics)
        detJ = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
        areas = 0.5 * detJ

        # Gradients of P1 hat basis on each triangle: shape (Nt, 3, 2)
        grads = np.zeros((tris.shape[0], 3, 2), dtype=np.float64)
        grads[:, 0, 0] = (p1[:, 1] - p2[:, 1]) / detJ
        grads[:, 0, 1] = (p2[:, 0] - p1[:, 0]) / detJ
        grads[:, 1, 0] = (p2[:, 1] - p0[:, 1]) / detJ
        grads[:, 1, 1] = (p0[:, 0] - p2[:, 0]) / detJ
        grads[:, 2, 0] = (p0[:, 1] - p1[:, 1]) / detJ
        grads[:, 2, 1] = (p1[:, 0] - p0[:, 0]) / detJ

        # 3-midpoint quadrature on each triangle for K(x) (exact for quadratic K)
        m01 = 0.5 * (p0 + p1)
        m12 = 0.5 * (p1 + p2)
        m20 = 0.5 * (p2 + p0)
        K_avg = (params.eval_tensor(m01) + params.eval_tensor(m12) + params.eval_tensor(m20)) / 3.0

        bary = (p0 + p1 + p2) / 3.0
        mids_local = [m01, m12, m20]
        pairs_local = [(0, 1), (1, 2), (2, 0)]

        # Precompute streamfunction at barycenters & midpoints if available
        use_psi = params.psi_fn is not None
        if use_psi:
            psi_bary = params.psi_fn(bary)
            psi_mids = [params.psi_fn(m) for m in mids_local]

        for idx_pair, (a_loc, b_loc) in enumerate(pairs_local):
            va = tris[:, a_loc]
            vb = tris[:, b_loc]
            ga = grads[:, a_loc, :]  # (Nt, 2)
            gb = grads[:, b_loc, :]  # (Nt, 2)
            # Off-diagonal P1 stiffness s_{ab} = int_T grad(phi_a)^T K grad(phi_b) dx
            # Note: -s_{ab} is the edge conductance W_{ab} in sum_j W_{ij}(u_j - u_i)
            kgb = np.einsum("tij,tj->ti", K_avg, gb)
            s_ab = areas * np.sum(ga * kgb, axis=1)
            w_contrib = -s_ab

            # Donald dual flux from dual cell of va to dual cell of vb along m_{ab} -> bary:
            # In CCW triangle (v0, v1, v2), each cyclic pair (0->1, 1->2, 2->0) has outward normal
            # from va to vb equal to (tau_y, -tau_x) where tau = bary - m_{ab}!
            m_ab = mids_local[idx_pair]
            if use_psi:
                flux_ab = psi_bary - psi_mids[idx_pair]
            else:
                tau = bary - m_ab  # (Nt, 2)
                n_vec = np.column_stack([tau[:, 1], -tau[:, 0]])  # integrated normal (Nt, 2)
                flux_ab = np.zeros(tris.shape[0], dtype=np.float64)
                for gp, gw in zip(_GL3_PTS, _GL3_WTS):
                    q_pts = m_ab + gp * tau
                    v_q = params.eval_velocity(q_pts)
                    flux_ab += gw * np.sum(v_q * n_vec, axis=1)

            i_min = np.minimum(va, vb).astype(np.int64)
            i_max = np.maximum(va, vb).astype(np.int64)
            sgn = np.where(va < vb, 1.0, -1.0)
            e_idx = np.searchsorted(edge_keys, i_min * n_nodes + i_max)
            np.add.at(w_diff, e_idx, w_contrib)
            np.add.at(f_conv, e_idx, sgn * flux_ab)
    else:
        # SLIC Region Adjacency Graph: non-Delaunay irregular partition
        assert mesh.edge_lengths is not None and mesh.edge_flux_normals is not None and mesh.edge_midpoints is not None
        p_i = mesh.points[edges[:, 0]]
        p_j = mesh.points[edges[:, 1]]
        diff = p_j - p_i
        dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
        unit_ij = diff / dist[:, None]
        K_mid = params.eval_tensor(mesh.edge_midpoints)
        # Directional diffusivity along centroid line (demonstrates SLIC RAG geometric mismatch!)
        k_eff = np.einsum("ei,eij,ej->e", unit_ij, K_mid, unit_ij)
        w_diff = k_eff * mesh.edge_lengths / dist

        v_mid = params.eval_velocity(mesh.edge_midpoints)
        f_conv = np.sum(v_mid * mesh.edge_flux_normals, axis=1)

    return edges, w_diff, f_conv


def build_cdquad_matrices(
    mesh: MeshData,
    params: PhysicalLawParams,
    alpha_edge: Optional[np.ndarray] = None,
    scheme: str = "upwind",
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Build sparse mass matrix M_G (diagonal) and spatial generator A_G (such that M_G du/dt = A_G u)
    where A_G incorporates anisotropic diffusion, conservative upwind (or centered) convection,
    and linear damping -lambda_r M_G.
    """
    n = mesh.points.shape[0]
    edges, w_diff, f_conv = assemble_cdquad_edge_quantities(mesh, params)
    if alpha_edge is None:
        if params.alpha_max > 0:
            raise ValueError("For nonlinear alpha_max > 0, pass alpha_edge explicitly per step.")
        mod_diff = np.ones_like(w_diff)
        mod_conv = np.ones_like(f_conv)
    else:
        mod_diff = 1.0 - alpha_edge
        mod_conv = alpha_edge

    wd = mod_diff * w_diff
    fc = mod_conv * f_conv

    ii = edges[:, 0]
    jj = edges[:, 1]

    rows = []
    cols = []
    vals = []

    # Diffusion contribution to M du/dt: sum_{j} wd_ij (u_j - u_i)
    rows.extend([ii, jj, ii, jj])
    cols.extend([jj, ii, ii, jj])
    vals.extend([wd, wd, -wd, -wd])

    # Convection contribution to M du/dt: -sum_{j} Phi_{ij}(u)
    # where Phi_{ij} is flux from i to j (so cell i loses Phi_{ij}, cell j gains Phi_{ij})
    if scheme == "upwind":
        f_pos = np.maximum(fc, 0.0)   # flow i -> j uses u_i
        f_neg = np.maximum(-fc, 0.0)  # flow j -> i uses u_j
        # Cell i: -(f_pos * u_i - f_neg * u_j) = -f_pos * u_i + f_neg * u_j
        # Cell j: +(f_pos * u_i - f_neg * u_j) = +f_pos * u_i - f_neg * u_j
        rows.extend([ii, ii, jj, jj])
        cols.extend([ii, jj, ii, jj])
        vals.extend([-f_pos, f_neg, f_pos, -f_neg])
    elif scheme == "centered":
        half = 0.5 * fc
        # Cell i: -0.5 * fc * (u_i + u_j); Cell j: +0.5 * fc * (u_i + u_j)
        rows.extend([ii, ii, jj, jj])
        cols.extend([ii, jj, ii, jj])
        vals.extend([-half, -half, half, half])
    else:
        raise ValueError(f"Unknown convection scheme: {scheme}")

    if params.lambda_r != 0.0:
        idx_all = np.arange(n)
        rows.append(idx_all)
        cols.append(idx_all)
        vals.append(-params.lambda_r * mesh.mass)

    r_cat = np.concatenate(rows)
    c_cat = np.concatenate(cols)
    v_cat = np.concatenate(vals)
    A_mat = sparse.csr_matrix((v_cat, (r_cat, c_cat)), shape=(n, n))
    M_mat = sparse.diags(mesh.mass, format="csr")
    return M_mat, A_mat


_GLOBAL_SPATIAL_KERNEL_CACHE: dict[tuple[str, int, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def compute_continuum_homogeneity_alpha(
    mesh: MeshData,
    u: np.ndarray,
    alpha_max: float,
    r0: float,
    gamma_hom: float,
    spatial_kernel_cache: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute quadrature-consistent nonlocal homogeneity gate:
      h_i(u) = (sum_j m_j K_{r0}(x_i, x_j) exp(-gamma * (u_i - u_j)^2)) / (sum_j m_j K_{r0}(x_i, x_j))
      alpha_i(u) = alpha_max * (1 - h_i(u)) in [0, alpha_max].
    Returns (alpha_node, alpha_edge).
    """
    n_pts = mesh.points.shape[0]
    cache_key = (mesh.name, n_pts, float(r0))
    if cache_key in _GLOBAL_SPATIAL_KERNEL_CACHE:
        r_idx, c_idx, w_norm = _GLOBAL_SPATIAL_KERNEL_CACHE[cache_key]
    else:
        pts = mesh.points
        cutoff2 = (2.2 * r0) ** 2
        inv_r02 = 1.0 / (r0 * r0)
        r_list = []
        c_list = []
        w_list = []
        chunk = 1024
        for start in range(0, n_pts, chunk):
            end = min(n_pts, start + chunk)
            dx = pts[start:end, None, 0] - pts[None, :, 0]
            dy = pts[start:end, None, 1] - pts[None, :, 1]
            d2 = dx * dx + dy * dy
            rr, cc = np.where(d2 <= cutoff2)
            vals = np.exp(-d2[rr, cc] * inv_r02) * mesh.mass[cc]
            denom_chunk = np.zeros(end - start, dtype=np.float64)
            np.add.at(denom_chunk, rr, vals)
            wn = vals / np.maximum(denom_chunk[rr], 1e-15)
            r_list.append(rr + start)
            c_list.append(cc)
            w_list.append(wn)
        r_idx = np.concatenate(r_list)
        c_idx = np.concatenate(c_list)
        w_norm = np.concatenate(w_list)
        _GLOBAL_SPATIAL_KERNEL_CACHE[cache_key] = (r_idx, c_idx, w_norm)

    du2 = (u[r_idx] - u[c_idx]) ** 2
    sim = w_norm * np.exp(-gamma_hom * du2)
    h_node = np.zeros(mesh.points.shape[0], dtype=np.float64)
    np.add.at(h_node, r_idx, sim)
    alpha_node = np.clip(alpha_max * (1.0 - h_node), 0.0, alpha_max)
    alpha_edge = 0.5 * (alpha_node[mesh.edges[:, 0]] + alpha_node[mesh.edges[:, 1]])
    return alpha_node, alpha_edge


def solve_cdquad_trajectory(
    mesh: MeshData,
    u0: np.ndarray,
    params: PhysicalLawParams,
    T: float,
    n_steps: int,
    scheme: str = "upwind",
    save_every: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate CDQuad from t=0 to t=T:
      - Linear Level 1 & Level 2 (alpha_max == 0, allen_cahn_eps == 0):
        2nd-order Crank-Nicolson with pre-factored sparse LU (unconditionally L^2-stable).
      - Nonlinear Level 3 (alpha_max > 0 or allen_cahn_eps != 0):
        Semi-implicit IMEX-BDF1 (implicit diffusion + implicit upwind transport at frozen gate alpha(u^n)).
    Returns (times, traj) where traj has shape (n_saved, N).
    """
    dt = T / n_steps
    u = u0.astype(np.float64).copy()
    times = [0.0]
    snaps = [u.copy()]

    if params.alpha_max == 0.0 and params.allen_cahn_eps == 0.0:
        M_mat, A_mat = build_cdquad_matrices(mesh, params, alpha_edge=None, scheme=scheme)
        lhs = (M_mat - 0.5 * dt * A_mat).tocsc()
        rhs_op = (M_mat + 0.5 * dt * A_mat).tocsr()
        solve_lu = factorized(lhs)
        for step in range(1, n_steps + 1):
            u = solve_lu(rhs_op @ u)
            if step % save_every == 0 or step == n_steps:
                times.append(step * dt)
                snaps.append(u.copy())
    else:
        edges, w_diff, f_conv = assemble_cdquad_edge_quantities(mesh, params)
        ii = edges[:, 0]
        jj = edges[:, 1]
        n = mesh.points.shape[0]
        M_diag = mesh.mass
        M_mat = sparse.diags(M_diag, format="csr")
        k_cache: dict = {}

        for step in range(1, n_steps + 1):
            if params.alpha_max > 0.0:
                _, alpha_edge = compute_continuum_homogeneity_alpha(
                    mesh, u, params.alpha_max, params.r0, params.gamma_hom, k_cache
                )
            else:
                alpha_edge = np.zeros(edges.shape[0], dtype=np.float64)

            wd = (1.0 - alpha_edge) * w_diff
            fc = alpha_edge * f_conv
            f_pos = np.maximum(fc, 0.0)
            f_neg = np.maximum(-fc, 0.0)

            rows = [ii, jj, ii, jj, ii, ii, jj, jj]
            cols = [jj, ii, ii, jj, ii, jj, ii, jj]
            vals = [wd, wd, -wd, -wd, -f_pos, f_neg, f_pos, -f_neg]
            if params.lambda_r != 0.0:
                idx_all = np.arange(n)
                rows.append(idx_all)
                cols.append(idx_all)
                vals.append(-params.lambda_r * M_diag)

            A_step = sparse.csr_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                shape=(n, n),
            )
            rhs = M_diag * u
            if params.allen_cahn_eps != 0.0:
                rhs = rhs + dt * params.allen_cahn_eps * M_diag * (u - u ** 3)

            u = spsolve((M_mat - dt * A_step).tocsc(), rhs)
            if step % save_every == 0 or step == n_steps:
                times.append(step * dt)
                snaps.append(u.copy())

    return np.asarray(times), np.vstack(snaps)


# ==============================================================================
# Negative-Class Discrete Graph Models (GRAND-nl, CDE-GRAND, Diploma AdaCDE (2.6))
# ==============================================================================

@dataclass
class DiscreteGNNParams:
    w_q: float = 1.2
    w_k: float = 1.2
    w_v: float = 1.8
    w_rbf: float = 2.0
    beta: float = 0.8
    rate_scale: float = 1.0  # overall time scale


def _build_directed_neighbor_lists(mesh: MeshData):
    n = mesh.points.shape[0]
    src = np.concatenate([mesh.edges[:, 0], mesh.edges[:, 1]])
    dst = np.concatenate([mesh.edges[:, 1], mesh.edges[:, 0]])
    deg = np.bincount(src, minlength=n).astype(np.float64)
    return n, src, dst, deg


def rhs_grand_nl(mesh: MeshData, u: np.ndarray, p: DiscreteGNNParams, nav=None) -> np.ndarray:
    """GRAND-nl: edge-softmax attention diffusion sum_{j in N_i} a_ij(u) (u_j - u_i)."""
    if nav is None:
        nav = _build_directed_neighbor_lists(mesh)
    n, src, dst, _ = nav
    # Include self-loop in softmax as in standard GAT/GRAND
    src_all = np.concatenate([src, np.arange(n)])
    dst_all = np.concatenate([dst, np.arange(n)])
    scores = (p.w_q * u[src_all]) * (p.w_k * u[dst_all])
    max_s = np.full(n, -np.inf)
    np.maximum.at(max_s, src_all, scores)
    exp_s = np.exp(scores - max_s[src_all])
    sum_s = np.zeros(n)
    np.add.at(sum_s, src_all, exp_s)
    attn = exp_s / np.maximum(sum_s[src_all], 1e-15)

    out = np.zeros(n, dtype=np.float64)
    np.add.at(out, src_all, attn * (u[dst_all] - u[src_all]))
    return p.rate_scale * out


def rhs_cde_grand(mesh: MeshData, u: np.ndarray, p: DiscreteGNNParams, nav=None) -> np.ndarray:
    """CDE-GRAND (Zhao et al., IJCAI 2023; bachelors.pdf eq. (2.1))."""
    if nav is None:
        nav = _build_directed_neighbor_lists(mesh)
    n, src, dst, deg = nav
    deg_tilde = deg + 1.0
    norm_ij = 1.0 / np.sqrt(deg_tilde[src] * deg_tilde[dst])
    diff_term = np.zeros(n, dtype=np.float64)
    np.add.at(diff_term, src, norm_ij * (u[dst] - u[src]))

    v_ij = np.tanh(p.w_v * (u[dst] - u[src]))
    conv_term = np.zeros(n, dtype=np.float64)
    np.add.at(conv_term, src, v_ij * u[dst])
    return p.rate_scale * (diff_term + p.beta * conv_term)


def rhs_adacde_diploma(mesh: MeshData, u: np.ndarray, p: DiscreteGNNParams, nav=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact Diploma AdaCDE equation (2.6) (bachelors.pdf pp. 31-35, scalar state without LayerNorm):
      h_i^local = (1 / |N_i|) sum_{j in N_i} exp(-(w_rbf * (u_i - u_j))^2)
      alpha_i = 1 - h_i^local
      du_i/dt = (1 - alpha_i) sum_j a_ij(u)(u_j - u_i) + alpha_i * beta * sum_{j in N_i} tanh(w_v(u_j - u_i)) * u_j.
    Returns (du_dt, alpha_i).
    """
    if nav is None:
        nav = _build_directed_neighbor_lists(mesh)
    n, src, dst, deg = nav

    rbf = np.exp(-((p.w_rbf * (u[src] - u[dst])) ** 2))
    h_loc = np.zeros(n, dtype=np.float64)
    np.add.at(h_loc, src, rbf)
    h_loc /= np.maximum(deg, 1.0)
    alpha = 1.0 - h_loc

    # Attention diffusion
    src_all = np.concatenate([src, np.arange(n)])
    dst_all = np.concatenate([dst, np.arange(n)])
    scores = (p.w_q * u[src_all]) * (p.w_k * u[dst_all])
    max_s = np.full(n, -np.inf)
    np.maximum.at(max_s, src_all, scores)
    exp_s = np.exp(scores - max_s[src_all])
    sum_s = np.zeros(n)
    np.add.at(sum_s, src_all, exp_s)
    attn = exp_s / np.maximum(sum_s[src_all], 1e-15)
    diff_term = np.zeros(n, dtype=np.float64)
    np.add.at(diff_term, src_all, attn * (u[dst_all] - u[src_all]))

    # Unnormalized neighbor-sum tanh convection
    v_ij = np.tanh(p.w_v * (u[dst] - u[src]))
    conv_term = np.zeros(n, dtype=np.float64)
    np.add.at(conv_term, src, v_ij * u[dst])

    du = p.rate_scale * ((1.0 - alpha) * diff_term + alpha * p.beta * conv_term)
    return du, alpha


def solve_discrete_gnn_rk4(
    model_name: str,
    mesh: MeshData,
    u0: np.ndarray,
    p: DiscreteGNNParams,
    T: float,
    n_steps: int,
) -> np.ndarray:
    """Explicit RK4 integration (as in bachelors.pdf eq. (2.9)) returning final state u(T)."""
    nav = _build_directed_neighbor_lists(mesh)
    dt = T / n_steps
    u = u0.astype(np.float64).copy()

    def f_eval(state: np.ndarray) -> np.ndarray:
        if model_name == "GRAND-nl":
            return rhs_grand_nl(mesh, state, p, nav)
        elif model_name == "CDE-GRAND":
            return rhs_cde_grand(mesh, state, p, nav)
        elif model_name == "AdaCDE":
            du, _ = rhs_adacde_diploma(mesh, state, p, nav)
            return du
        else:
            raise ValueError(model_name)

    for _ in range(n_steps):
        k1 = f_eval(u)
        k2 = f_eval(u + 0.5 * dt * k1)
        k3 = f_eval(u + 0.5 * dt * k2)
        k4 = f_eval(u + dt * k3)
        u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return u
