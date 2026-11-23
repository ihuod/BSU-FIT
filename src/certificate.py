"""
Computable Elliptic Dual Quadrature Defect delta_{-1}(G), naive L^2 defect delta_{L2}(G),
and Split-Conformal Error Certificate without running the reference solver on test meshes.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

from .meshes import MeshData
from .operators import PhysicalLawParams, build_cdquad_matrices, assemble_cdquad_edge_quantities


def neumann_cosine_mode(coords: np.ndarray, p: int, q: int):
    """
    Exact eigenfunction psi_{p,q}(x, y) = cos(pi p x) cos(pi q y) satisfying d_nu psi = 0 on d[0,1]^2,
    plus exact gradient and Hessian.
    """
    x = coords[:, 0]
    y = coords[:, 1]
    px = np.pi * p
    qy = np.pi * q
    cx, sx = np.cos(px * x), np.sin(px * x)
    cy, sy = np.cos(qy * y), np.sin(qy * y)

    val = cx * cy
    gx = -px * sx * cy
    gy = -qy * cx * sy
    gxx = -(px ** 2) * cx * cy
    gyy = -(qy ** 2) * cx * cy
    gxy = px * qy * sx * sy
    grad = np.column_stack([gx, gy])
    # H^2([0,1]^2) norm analytic
    norm_l2_sq = (1.0 if p == 0 else 0.5) * (1.0 if q == 0 else 0.5)
    lam = px ** 2 + qy ** 2
    h2_norm = float(np.sqrt(norm_l2_sq * (1.0 + lam + lam ** 2)))
    return val, grad, (gxx, gxy, gyy), h2_norm


def eval_continuum_operator_analytic(coords: np.ndarray, p: int, q: int, params: PhysicalLawParams) -> tuple[np.ndarray, float]:
    """
    Evaluate exact continuum operator (L psi_{p,q})(x) = div(K(x) grad psi) - v(x) . grad psi - lambda_r psi
    analytically / via centered 1e-6 derivative of K(x), with ZERO reference mesh!
    """
    val, grad, (gxx, gxy, gyy), h2_norm = neumann_cosine_mode(coords, p, q)
    K = params.eval_tensor(coords)
    # K : D^2 psi = K_00 gxx + 2 K_01 gxy + K_11 gyy
    diff_hess = K[:, 0, 0] * gxx + 2.0 * K[:, 0, 1] * gxy + K[:, 1, 1] * gyy

    # Add (div K) . grad psi for spatially varying K(x)
    eps = 1e-6
    Kp_x = params.eval_tensor(np.clip(coords + np.array([eps, 0.0]), 0.0, 1.0))
    Km_x = params.eval_tensor(np.clip(coords - np.array([eps, 0.0]), 0.0, 1.0))
    step_x = np.maximum(
        np.clip(coords[:, 0] + eps, 0.0, 1.0) - np.clip(coords[:, 0] - eps, 0.0, 1.0),
        1e-12,
    )
    dK00_dx = (Kp_x[:, 0, 0] - Km_x[:, 0, 0]) / step_x
    dK10_dx = (Kp_x[:, 1, 0] - Km_x[:, 1, 0]) / step_x

    Kp_y = params.eval_tensor(np.clip(coords + np.array([0.0, eps]), 0.0, 1.0))
    Km_y = params.eval_tensor(np.clip(coords - np.array([0.0, eps]), 0.0, 1.0))
    step_y = np.maximum(
        np.clip(coords[:, 1] + eps, 0.0, 1.0) - np.clip(coords[:, 1] - eps, 0.0, 1.0),
        1e-12,
    )
    dK01_dy = (Kp_y[:, 0, 1] - Km_y[:, 0, 1]) / step_y
    dK11_dy = (Kp_y[:, 1, 1] - Km_y[:, 1, 1]) / step_y

    divK_x = dK00_dx + dK01_dy
    divK_y = dK10_dx + dK11_dy
    diff_part = diff_hess + divK_x * grad[:, 0] + divK_y * grad[:, 1]

    v = params.eval_velocity(coords)
    conv_part = -(v[:, 0] * grad[:, 0] + v[:, 1] * grad[:, 1])
    reac_part = -params.lambda_r * val
    return diff_part + conv_part + reac_part, h2_norm


def compute_mesh_defects(
    mesh: MeshData,
    params: PhysicalLawParams,
    max_freq: int = 3,
) -> dict[str, float]:
    """
    Compute both:
      1) Elliptic Dual Defect delta_{-1}(G) in H_G^{-1} (consistent with Theorem 1!)
      2) Naive L^2 Operator Defect delta_{L2}(G) (which stagnates at O(1) on unstructured upwind meshes!)
    Uses Linearized / Effective Law (if alpha_max > 0, evaluates at representative gate alpha_eff = 0.5 * alpha_max).
    """
    eff_params = PhysicalLawParams(
        kappa1=params.kappa1 * (1.0 - 0.5 * params.alpha_max),
        kappa2=params.kappa2 * (1.0 - 0.5 * params.alpha_max),
        theta_fn=params.theta_fn,
        psi_fn=(lambda c: (0.5 * params.alpha_max) * params.psi_fn(c)) if (params.alpha_max > 0 and params.psi_fn is not None) else params.psi_fn,
        v_fn=(lambda c: (0.5 * params.alpha_max) * params.v_fn(c)) if (params.alpha_max > 0 and params.v_fn is not None) else params.v_fn,
        lambda_r=params.lambda_r,
        alpha_max=0.0,
    )

    M_mat, A_mat = build_cdquad_matrices(mesh, eff_params, alpha_edge=None, scheme="upwind")
    # Build symmetric elliptic Riesz map E_G = M_G + (1 / kappa_min) S_G(K)
    edges, w_diff, _ = assemble_cdquad_edge_quantities(mesh, eff_params)
    ii, jj = edges[:, 0], edges[:, 1]
    n = mesh.points.shape[0]
    S_mat = sparse.csr_matrix(
        (
            np.concatenate([-w_diff, -w_diff, w_diff, w_diff]),
            (np.concatenate([ii, jj, ii, jj]), np.concatenate([jj, ii, ii, jj])),
        ),
        shape=(n, n),
    )
    k0 = max(eff_params.kappa_min, 1e-8)
    riesz_op = (M_mat + (1.0 / k0) * S_mat).tocsc()
    solve_riesz = factorized(riesz_op)

    # Cell-averaged continuum action R_G(L psi): for triangles use 3-midpoint average per triangle, else nodal
    dual_vals = []
    l2_vals = []
    for p in range(max_freq + 1):
        for q in range(max_freq + 1):
            if p == 0 and q == 0:
                continue
            psi_nodal, _, _, h2_norm = neumann_cosine_mode(mesh.points, p, q)

            if mesh.triangles is not None:
                tris = mesh.triangles
                p0, p1, p2 = mesh.points[tris[:, 0]], mesh.points[tris[:, 1]], mesh.points[tris[:, 2]]
                areas = 0.5 * np.abs(
                    (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
                )
                m01, m12, m20 = 0.5 * (p0 + p1), 0.5 * (p1 + p2), 0.5 * (p2 + p0)
                l_01, _ = eval_continuum_operator_analytic(m01, p, q, eff_params)
                l_12, _ = eval_continuum_operator_analytic(m12, p, q, eff_params)
                l_20, _ = eval_continuum_operator_analytic(m20, p, q, eff_params)
                # Exact P1 hat quadrature on each triangle using edge midpoints:
                # On a triangle, phi_0 at (m01, m12, m20) is (1/2, 0, 1/2), so int_T f phi_0 dx = |T|/3 * (f(m01)+f(m20))/2 * ...
                # Wait: midpoint quadrature weight on triangle is |T|/3 at each edge midpoint!
                # At m01: (phi_0, phi_1, phi_2) = (1/2, 1/2, 0)!
                rhs_exact = np.zeros(n, dtype=np.float64)
                np.add.at(rhs_exact, tris[:, 0], (areas / 3.0) * 0.5 * (l_01 + l_20))
                np.add.at(rhs_exact, tris[:, 1], (areas / 3.0) * 0.5 * (l_01 + l_12))
                np.add.at(rhs_exact, tris[:, 2], (areas / 3.0) * 0.5 * (l_12 + l_20))

                # Subtract boundary co-normal flux int_{dOmega} (K grad psi . nu) phi_i ds
                # so <L_weak psi, phi_i> = -int (grad psi)^T K grad phi_i - int (v.grad psi + lambda_r psi) phi_i
                tol_b = 1e-12
                pts = mesh.points
                for e_idx in range(edges.shape[0]):
                    ia, ib = int(edges[e_idx, 0]), int(edges[e_idx, 1])
                    pa, pb = pts[ia], pts[ib]
                    nu = None
                    if abs(pa[0]) < tol_b and abs(pb[0]) < tol_b:
                        nu = np.array([-1.0, 0.0])
                    elif abs(pa[0] - 1.0) < tol_b and abs(pb[0] - 1.0) < tol_b:
                        nu = np.array([1.0, 0.0])
                    elif abs(pa[1]) < tol_b and abs(pb[1]) < tol_b:
                        nu = np.array([0.0, -1.0])
                    elif abs(pa[1] - 1.0) < tol_b and abs(pb[1] - 1.0) < tol_b:
                        nu = np.array([0.0, 1.0])
                    if nu is not None:
                        elen = float(np.linalg.norm(pb - pa))
                        # 2-point Gauss-Legendre on boundary edge [pa, pb]
                        for gp, gw in ((0.5 - 0.5 / np.sqrt(3.0), 0.5), (0.5 + 0.5 / np.sqrt(3.0), 0.5)):
                            xq = (1.0 - gp) * pa + gp * pb
                            _, gq, _, _ = neumann_cosine_mode(xq[None, :], p, q)
                            Kq = eff_params.eval_tensor(xq[None, :])[0]
                            conorm = float(np.dot(Kq @ gq[0], nu))
                            rhs_exact[ia] -= gw * elen * (1.0 - gp) * conorm
                            rhs_exact[ib] -= gw * elen * gp * conorm
            else:
                l_nodal, _ = eval_continuum_operator_analytic(mesh.points, p, q, eff_params)
                rhs_exact = mesh.mass * l_nodal

            rhs_disc = A_mat @ psi_nodal
            res_load = rhs_disc - rhs_exact  # integrated residual <L_G psi - L psi, phi_i>

            # Dual H_G^{-1} norm: sqrt(res_load^T E_G^{-1} res_load)
            z = solve_riesz(res_load)
            dual_sq = float(np.maximum(np.dot(res_load, z), 0.0))
            dual_vals.append(np.sqrt(dual_sq) / h2_norm)

            # Pointwise discrete L^2 norm: sqrt(sum_i (res_load_i / m_i)^2 * m_i)
            l2_sq = float(np.sum((res_load ** 2) / np.maximum(mesh.mass, 1e-15)))
            l2_vals.append(np.sqrt(l2_sq) / h2_norm)

    return {
        "delta_dual_max": float(np.max(dual_vals)),
        "delta_dual_rms": float(np.sqrt(np.mean(np.square(dual_vals)))),
        "delta_l2_max": float(np.max(l2_vals)),
        "delta_l2_rms": float(np.sqrt(np.mean(np.square(l2_vals)))),
    }


def analytic_proxy_bound(
    mesh: MeshData,
    params: PhysicalLawParams,
    T: float,
    delta_dual: float,
    e0_est: float = 0.0,
) -> float:
    """
    Un-calibrated analytical structure B_hat(G; T) from Theorem 1:
      B_hat(G; T) = e0_est + delta_{-1}(G) * sqrt((1 - exp(-mu * T)) / mu)
    where mu = 0.25 * pi^2 * kappa_min + 2 * lambda_r.
    """
    mu = 0.25 * (np.pi ** 2) * params.kappa_min + 2.0 * params.lambda_r
    if mu > 1e-10:
        time_factor = float(np.sqrt((1.0 - np.exp(-mu * T)) / mu))
    else:
        time_factor = float(np.sqrt(T))
    return float(e0_est + delta_dual * time_factor)


@dataclass
class ConformalCertificate:
    alpha_miscoverage: float
    q_quantile: float
    n_calib: int
    scores_calib: np.ndarray

    @classmethod
    def fit(cls, e_true_calib: np.ndarray, b_hat_calib: np.ndarray, alpha_miscoverage: float = 0.10) -> "ConformalCertificate":
        B = len(e_true_calib)
        scores = np.asarray(e_true_calib, dtype=np.float64) / np.maximum(b_hat_calib, 1e-15)
        scores_sorted = np.sort(scores)
        k = int(np.ceil((B + 1) * (1.0 - alpha_miscoverage)))
        if k > B:
            q = float("inf")
        else:
            q = float(scores_sorted[k - 1])
        return cls(alpha_miscoverage=alpha_miscoverage, q_quantile=q, n_calib=B, scores_calib=scores)

    def predict_bound(self, b_hat: np.ndarray | float) -> np.ndarray | float:
        return self.q_quantile * b_hat
