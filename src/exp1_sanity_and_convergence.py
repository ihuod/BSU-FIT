"""
Experiment Suite 1 (Steps A & C):
  1A. Heat equation sanity check against exact analytical solution + Donald-dual div=0 check.
  1B. Level 2 Anisotropic Convection-Diffusion convergence across random Delaunay meshes,
      Peclet sweep Pe in {0, 5, 15, 40, 100}, and long-time horizon T in [0.1, 8.0].
  1C. Level 3 Nonlinear Adaptive CDQuad cross-resolution convergence.
  1D. Class boundary (Theorem 3): CDQuad vs Diploma AdaCDE (eq. 2.6), CDE-GRAND, GRAND-nl,
      including 1-hop homogeneity gate collapse alpha_G -> 0 and graded-mesh drift.
"""
from __future__ import annotations

import json
import os
import time
import numpy as np

from src.meshes import (
    build_structured_mesh,
    build_jittered_delaunay_mesh,
    build_random_delaunay_mesh,
    build_two_to_one_graded_mesh,
    ProlongationGrid,
)
from src.operators import (
    PhysicalLawParams,
    DiscreteGNNParams,
    assemble_cdquad_edge_quantities,
    compute_continuum_homogeneity_alpha,
    solve_cdquad_trajectory,
    rhs_adacde_diploma,
    solve_discrete_gnn_rk4,
)


def exact_heat_solution(coords: np.ndarray, t: float, kappa: float = 0.02) -> np.ndarray:
    x = coords[:, 0]
    y = coords[:, 1]
    lam = (1.0 + 4.0) * (np.pi ** 2)
    decay = np.exp(-kappa * lam * t)
    return 0.5 + decay * (0.35 * np.cos(np.pi * x) * np.cos(2.0 * np.pi * y) - 0.25 * np.cos(2.0 * np.pi * x) * np.cos(np.pi * y))


def initial_field_smooth(coords: np.ndarray) -> np.ndarray:
    x = coords[:, 0]
    y = coords[:, 1]
    b1 = np.exp(-((x - 0.32) ** 2 + (y - 0.36) ** 2) / (2.0 * 0.11 ** 2))
    b2 = 0.65 * np.exp(-((x - 0.68) ** 2 + (y - 0.64) ** 2) / (2.0 * 0.13 ** 2))
    return b1 + b2


def make_aniso_cd_params(u_vel: float, alpha_max: float = 0.0) -> PhysicalLawParams:
    kappa1 = 0.025
    kappa2 = 0.005  # anisotropy ratio chi(K) = 5.0
    theta_fn = lambda c: (np.pi / 4.0) * np.sin(np.pi * c[:, 0]) * np.cos(np.pi * c[:, 1])
    # Streamfunction psi(x, y) = (u_vel / pi) * sin(pi x) * sin(pi y) -> ||v||_inf = u_vel
    psi_fn = lambda c: (u_vel / np.pi) * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1])
    return PhysicalLawParams(
        kappa1=kappa1,
        kappa2=kappa2,
        theta_fn=theta_fn,
        psi_fn=psi_fn,
        lambda_r=0.0,
        alpha_max=alpha_max,
        r0=0.16,
        gamma_hom=6.0,
    )


def run_all():
    t0_wall = time.time()
    os.makedirs("results", exist_ok=True)
    pgrid = ProlongationGrid(m_grid=160)
    results: dict = {"seed_master": 20261122, "eval_grid": 160}

    # ==========================================================================
    # 1A. Step A: Heat equation sanity check + Donald dual zero-divergence check
    # ==========================================================================
    print("--- Running 1A: Heat equation sanity check & Donald dual div=0 check ---")
    kappa_heat = 0.02
    T_heat = 0.25
    heat_params = PhysicalLawParams(kappa1=kappa_heat, kappa2=kappa_heat)
    u_exact_eval = exact_heat_solution(pgrid.coords, T_heat, kappa=kappa_heat)

    heat_struct = []
    heat_jitter = []
    for nx in [8, 16, 32, 64]:
        n_steps = max(40, (nx * nx) // 4)
        m_s = build_structured_mesh(nx, nx, diagonal="right")
        _, tr_s = solve_cdquad_trajectory(m_s, exact_heat_solution(m_s.points, 0.0, kappa_heat), heat_params, T_heat, n_steps)
        err_s = pgrid.l2_norm(pgrid.prolong(m_s, tr_s[-1]) - u_exact_eval)
        heat_struct.append({"nx": nx, "n_nodes": int(m_s.points.shape[0]), "h_max": m_s.h_max, "l2_err": err_s})

        errs_j = []
        hs_j = []
        for s in range(3):
            m_j = build_jittered_delaunay_mesh(nx, nx, jitter=0.28, seed=100 + 10 * nx + s)
            _, tr_j = solve_cdquad_trajectory(m_j, exact_heat_solution(m_j.points, 0.0, kappa_heat), heat_params, T_heat, n_steps)
            errs_j.append(pgrid.l2_norm(pgrid.prolong(m_j, tr_j[-1]) - u_exact_eval))
            hs_j.append(m_j.h_max)
        heat_jitter.append({
            "nx": nx,
            "n_nodes": int(m_j.points.shape[0]),
            "h_max_mean": float(np.mean(hs_j)),
            "l2_err_mean": float(np.mean(errs_j)),
            "l2_err_std": float(np.std(errs_j)),
        })

    # Empirical log-log convergence slopes
    slope_struct = float(np.polyfit(
        np.log([r["h_max"] for r in heat_struct]),
        np.log([r["l2_err"] for r in heat_struct]),
        1,
    )[0])
    slope_jitter = float(np.polyfit(
        np.log([r["h_max_mean"] for r in heat_jitter]),
        np.log([r["l2_err_mean"] for r in heat_jitter]),
        1,
    )[0])

    # Check discrete divergence sum_{j in N_i} F_ij = 0 on irregular Delaunay mesh
    m_div_test = build_random_delaunay_mesh(450, seed=2026, lloyd_iters=1)
    p_div_test = make_aniso_cd_params(u_vel=0.5, alpha_max=0.0)
    edges_d, _, f_conv_d = assemble_cdquad_edge_quantities(m_div_test, p_div_test)
    cell_div = np.zeros(m_div_test.points.shape[0])
    np.add.at(cell_div, edges_d[:, 0], f_conv_d)
    np.add.at(cell_div, edges_d[:, 1], -f_conv_d)
    max_abs_div = float(np.max(np.abs(cell_div)))
    print(f"  Heat slope (structured): {slope_struct:.3f}, (jittered Delaunay): {slope_jitter:.3f}")
    print(f"  Max discrete Donald-dual flux divergence on random Delaunay (N={m_div_test.points.shape[0]}): {max_abs_div:.3e}")

    results["step_1A_heat"] = {
        "structured": heat_struct,
        "jittered_delaunay": heat_jitter,
        "slope_structured": slope_struct,
        "slope_jittered": slope_jitter,
        "donald_dual_max_abs_div": max_abs_div,
    }

    # ==========================================================================
    # 1B. Level 2 Anisotropic CD: mesh refinement, Peclet sweep, and long-time T
    # ==========================================================================
    print("--- Running 1B: Level 2 Anisotropic CD convergence & Peclet sweep ---")
    mesh_ref = build_structured_mesh(96, 96, diagonal="alternating")
    u0_ref = initial_field_smooth(mesh_ref.points)
    T_main = 0.40

    # Mesh convergence at moderate-high Peclet (u_vel = 0.106 -> Pe = 0.106 * sqrt(2) / 0.005 = 30.0)
    u_vel_conv = 0.10606601717798213  # exact Pe = 30.0
    params_pe30 = make_aniso_cd_params(u_vel=u_vel_conv, alpha_max=0.0)
    _, tr_ref_pe30 = solve_cdquad_trajectory(mesh_ref, u0_ref, params_pe30, T_main, n_steps=320)
    u_ref_eval_pe30 = pgrid.prolong(mesh_ref, tr_ref_pe30[-1])
    mass_ref_init = float(np.sum(mesh_ref.mass * u0_ref))
    mass_ref_final = float(np.sum(mesh_ref.mass * tr_ref_pe30[-1]))

    conv_pe30 = []
    for nx in [10, 14, 20, 28, 40, 56]:
        errs_seed = []
        hs_seed = []
        for s in range(5):
            m = build_jittered_delaunay_mesh(nx, nx, jitter=0.28, seed=500 + 20 * nx + s)
            _, tr = solve_cdquad_trajectory(m, initial_field_smooth(m.points), params_pe30, T_main, n_steps=160)
            errs_seed.append(pgrid.l2_norm(pgrid.prolong(m, tr[-1]) - u_ref_eval_pe30))
            hs_seed.append(m.h_max)
        conv_pe30.append({
            "nx": nx,
            "n_nodes": int((nx + 1) ** 2),
            "h_max_mean": float(np.mean(hs_seed)),
            "l2_err_mean": float(np.mean(errs_seed)),
            "l2_err_std": float(np.std(errs_seed)),
        })

    slope_pe30 = float(np.polyfit(
        np.log([r["h_max_mean"] for r in conv_pe30]),
        np.log([r["l2_err_mean"] for r in conv_pe30]),
        1,
    )[0])
    print(f"  Anisotropic CD (Pe=30, chi=5) slope on jittered Delaunay: {slope_pe30:.3f}, mass drift: {abs(mass_ref_final - mass_ref_init):.2e}")

    # Peclet sweep: Pe in {0, 5, 15, 30, 60, 100} comparing coarse G1 (jitter 16x16) vs fine G2 (jitter 48x48) & ref
    peclet_targets = [0.0, 5.0, 15.0, 30.0, 60.0, 100.0]
    peclet_sweep = []
    for pe_val in peclet_targets:
        u_v = pe_val * 0.005 / np.sqrt(2.0)
        p_pe = make_aniso_cd_params(u_vel=u_v, alpha_max=0.0)
        _, tr_r = solve_cdquad_trajectory(mesh_ref, u0_ref, p_pe, T_main, n_steps=400)
        ur_eval = pgrid.prolong(mesh_ref, tr_r[-1])

        pair_diffs_up = []
        ref_errs_up = []
        ref_errs_cent = []
        overshoots_cent = []
        overshoots_up = []
        u0_min, u0_max = float(np.min(u0_ref)), float(np.max(u0_ref))

        for s in range(4):
            g1 = build_jittered_delaunay_mesh(16, 16, jitter=0.26, seed=900 + s)
            g2 = build_jittered_delaunay_mesh(48, 48, jitter=0.26, seed=950 + s)
            _, tr1_up = solve_cdquad_trajectory(g1, initial_field_smooth(g1.points), p_pe, T_main, n_steps=200, scheme="upwind")
            _, tr2_up = solve_cdquad_trajectory(g2, initial_field_smooth(g2.points), p_pe, T_main, n_steps=200, scheme="upwind")
            _, tr1_ce = solve_cdquad_trajectory(g1, initial_field_smooth(g1.points), p_pe, T_main, n_steps=200, scheme="centered")

            u1_eval = pgrid.prolong(g1, tr1_up[-1])
            u2_eval = pgrid.prolong(g2, tr2_up[-1])
            u1_ce_eval = pgrid.prolong(g1, tr1_ce[-1])

            pair_diffs_up.append(pgrid.l2_norm(u1_eval - u2_eval))
            ref_errs_up.append(pgrid.l2_norm(u1_eval - ur_eval))
            ref_errs_cent.append(pgrid.l2_norm(u1_ce_eval - ur_eval))
            overshoots_up.append(max(0.0, float(np.max(tr1_up[-1]) - u0_max), float(u0_min - np.min(tr1_up[-1]))))
            overshoots_cent.append(max(0.0, float(np.max(tr1_ce[-1]) - u0_max), float(u0_min - np.min(tr1_ce[-1]))))

        peclet_sweep.append({
            "Pe": pe_val,
            "u_vel": u_v,
            "pair_diff_16_vs_48_mean": float(np.mean(pair_diffs_up)),
            "pair_diff_16_vs_48_std": float(np.std(pair_diffs_up)),
            "coarse_ref_err_upwind_mean": float(np.mean(ref_errs_up)),
            "coarse_ref_err_centered_mean": float(np.mean(ref_errs_cent)),
            "max_overshoot_upwind": float(np.max(overshoots_up)),
            "max_overshoot_centered": float(np.max(overshoots_cent)),
        })
        print(f"  Pe={pe_val:5.1f}: ||P_G1 u1 - P_G2 u2||={np.mean(pair_diffs_up):.4f}, ref_up={np.mean(ref_errs_up):.4f}, ref_cent={np.mean(ref_errs_cent):.4f}, overshoot_cent={np.max(overshoots_cent):.4f}")

    # Long-time horizon sweep t in [0, 8.0] at Pe = 30 showing NO exp(Pe * t) blowup!
    T_long = 8.0
    n_steps_long = 1600
    g_coarse_lt = build_jittered_delaunay_mesh(18, 18, jitter=0.26, seed=1201)
    g_fine_lt = build_jittered_delaunay_mesh(54, 54, jitter=0.26, seed=1202)
    times_lt, tr_c_lt = solve_cdquad_trajectory(
        g_coarse_lt, initial_field_smooth(g_coarse_lt.points), params_pe30, T_long, n_steps_long, save_every=20
    )
    _, tr_f_lt = solve_cdquad_trajectory(
        g_fine_lt, initial_field_smooth(g_fine_lt.points), params_pe30, T_long, n_steps_long, save_every=20
    )
    probe_times = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    long_time_curve = []
    for tp in probe_times:
        idx = int(np.argmin(np.abs(times_lt - tp)))
        err_t = pgrid.l2_diff(g_coarse_lt, tr_c_lt[idx], g_fine_lt, tr_f_lt[idx])
        std_field = float(np.sqrt(pgrid.weight * np.sum((pgrid.prolong(g_fine_lt, tr_f_lt[idx]) - np.mean(tr_f_lt[idx])) ** 2)))
        long_time_curve.append({"t": float(times_lt[idx]), "l2_transfer_diff": err_t, "signal_std": std_field})
    print("  Long-time curve:", [(r["t"], round(r["l2_transfer_diff"], 5)) for r in long_time_curve])

    results["step_1B_aniso_cd"] = {
        "convergence_pe30": conv_pe30,
        "slope_pe30": slope_pe30,
        "mass_drift_ref": float(abs(mass_ref_final - mass_ref_init)),
        "peclet_sweep": peclet_sweep,
        "long_time_curve": long_time_curve,
    }

    # ==========================================================================
    # 1C. Level 3 Nonlinear Adaptive CDQuad convergence across resolutions
    # ==========================================================================
    print("--- Running 1C: Level 3 Nonlinear Adaptive CDQuad convergence ---")
    params_adapt = make_aniso_cd_params(u_vel=0.18, alpha_max=0.65)
    mesh_ref_ad = build_structured_mesh(72, 72, diagonal="alternating")
    _, tr_ref_ad = solve_cdquad_trajectory(
        mesh_ref_ad, initial_field_smooth(mesh_ref_ad.points), params_adapt, T=0.30, n_steps=180
    )
    u_ref_ad_eval = pgrid.prolong(mesh_ref_ad, tr_ref_ad[-1])

    adapt_conv = []
    for nx in [10, 14, 20, 28, 40]:
        errs_ad = []
        hs_ad = []
        alphas_mean = []
        for s in range(3):
            m = build_jittered_delaunay_mesh(nx, nx, jitter=0.25, seed=1500 + 10 * nx + s)
            u0_m = initial_field_smooth(m.points)
            a_node, _ = compute_continuum_homogeneity_alpha(m, u0_m, params_adapt.alpha_max, params_adapt.r0, params_adapt.gamma_hom)
            alphas_mean.append(float(np.sum(m.mass * a_node)))
            _, tr_m = solve_cdquad_trajectory(m, u0_m, params_adapt, T=0.30, n_steps=120)
            errs_ad.append(pgrid.l2_norm(pgrid.prolong(m, tr_m[-1]) - u_ref_ad_eval))
            hs_ad.append(m.h_max)
        adapt_conv.append({
            "nx": nx,
            "n_nodes": int((nx + 1) ** 2),
            "h_max_mean": float(np.mean(hs_ad)),
            "l2_err_mean": float(np.mean(errs_ad)),
            "l2_err_std": float(np.std(errs_ad)),
            "continuum_gate_mean_u0": float(np.mean(alphas_mean)),
        })
    slope_adapt = float(np.polyfit(
        np.log([r["h_max_mean"] for r in adapt_conv]),
        np.log([r["l2_err_mean"] for r in adapt_conv]),
        1,
    )[0])
    print(f"  Level 3 Adaptive CDQuad slope: {slope_adapt:.3f}; gate mean across meshes: {[round(r['continuum_gate_mean_u0'], 4) for r in adapt_conv]}")
    results["step_1C_adaptive_cdquad"] = {
        "convergence": adapt_conv,
        "slope_adaptive": slope_adapt,
    }

    # ==========================================================================
    # 1D. Class Boundary (Theorem 3): CDQuad vs Diploma AdaCDE, CDE-GRAND, GRAND-nl
    # ==========================================================================
    print("--- Running 1D: Class boundary comparison (CDQuad vs AdaCDE, CDE-GRAND, GRAND-nl) ---")
    gnn_p = DiscreteGNNParams(w_q=1.4, w_k=1.4, w_v=2.0, w_rbf=2.5, beta=1.2, rate_scale=2.5)
    gnn_p_h2 = DiscreteGNNParams(w_q=1.4, w_k=1.4, w_v=2.0, w_rbf=2.5, beta=1.2, rate_scale=2.5)  # we will scale rate by n^2 below too!
    T_cls = 0.30

    mesh_base = build_structured_mesh(12, 12, diagonal="right")
    u0_base = initial_field_smooth(mesh_base.points)
    _, tr_cdq_base = solve_cdquad_trajectory(mesh_base, u0_base, params_adapt, T=T_cls, n_steps=120)
    u_cdq_base_eval = pgrid.prolong(mesh_base, tr_cdq_base[-1])

    u_ada_base_eval = pgrid.prolong(mesh_base, solve_discrete_gnn_rk4("AdaCDE", mesh_base, u0_base, gnn_p, T_cls, 120))
    u_cde_base_eval = pgrid.prolong(mesh_base, solve_discrete_gnn_rk4("CDE-GRAND", mesh_base, u0_base, gnn_p, T_cls, 120))
    u_grn_base_eval = pgrid.prolong(mesh_base, solve_discrete_gnn_rk4("GRAND-nl", mesh_base, u0_base, gnn_p, T_cls, 120))
    u0_eval = initial_field_smooth(pgrid.coords)
    base_evolution_mag = pgrid.l2_norm(u_ada_base_eval - u0_eval)

    class_refinement = []
    for nx in [12, 24, 36, 48, 64]:
        m_u = build_structured_mesh(nx, nx, diagonal="right")
        u0_u = initial_field_smooth(m_u.points)
        _, alpha_diploma = rhs_adacde_diploma(m_u, u0_u, gnn_p)
        alpha_cont, _ = compute_continuum_homogeneity_alpha(m_u, u0_u, params_adapt.alpha_max, params_adapt.r0, params_adapt.gamma_hom)

        _, tr_cdq_u = solve_cdquad_trajectory(m_u, u0_u, params_adapt, T=T_cls, n_steps=120)
        u_ada_u = solve_discrete_gnn_rk4("AdaCDE", m_u, u0_u, gnn_p, T_cls, 120)
        u_cde_u = solve_discrete_gnn_rk4("CDE-GRAND", m_u, u0_u, gnn_p, T_cls, 120)
        u_grn_u = solve_discrete_gnn_rk4("GRAND-nl", m_u, u0_u, gnn_p, T_cls, 120)

        # Also check graded 2:1 mesh vs uniform nx mesh at identical parameter set!
        m_g = build_two_to_one_graded_mesh(nx)
        u0_g = initial_field_smooth(m_g.points)
        _, tr_cdq_g = solve_cdquad_trajectory(m_g, u0_g, params_adapt, T=T_cls, n_steps=120)

        # If someone tries to "rescue" AdaCDE/GRAND-nl by scaling rate_scale ~ (nx / 12)^2:
        gnn_scaled = DiscreteGNNParams(
            w_q=gnn_p.w_q, w_k=gnn_p.w_k, w_v=gnn_p.w_v, w_rbf=gnn_p.w_rbf, beta=gnn_p.beta,
            rate_scale=gnn_p.rate_scale * ((nx / 12.0) ** 2) * 0.08,
        )
        # Operator discrepancy at t=0 between uniform nx and graded 2:1 nx under 1/h^2 rescaling
        du_ada_u, _ = rhs_adacde_diploma(m_u, u0_u, gnn_scaled)
        du_ada_g, _ = rhs_adacde_diploma(m_g, u0_g, gnn_scaled)
        rhs_interface_jump_adacde = pgrid.l2_diff(m_u, du_ada_u, m_g, du_ada_g)

        class_refinement.append({
            "nx": nx,
            "h": 1.0 / nx,
            "diploma_alpha_mean": float(np.sum(m_u.mass * alpha_diploma)),
            "cdquad_alpha_mean": float(np.sum(m_u.mass * alpha_cont)),
            # Discrepancy between uniform mesh nx and graded 2:1 mesh nx at t=T:
            "uniform_vs_graded_cdquad": pgrid.l2_diff(m_u, tr_cdq_u[-1], m_g, tr_cdq_g[-1]),
            # Drift from coarse 12x12 model prediction as nx grows:
            "drift_from_12_cdquad": pgrid.l2_norm(pgrid.prolong(m_u, tr_cdq_u[-1]) - u_cdq_base_eval),
            "drift_from_12_adacde": pgrid.l2_norm(pgrid.prolong(m_u, u_ada_u) - u_ada_base_eval),
            "drift_from_12_cde_grand": pgrid.l2_norm(pgrid.prolong(m_u, u_cde_u) - u_cde_base_eval),
            "drift_from_12_grand_nl": pgrid.l2_norm(pgrid.prolong(m_u, u_grn_u) - u_grn_base_eval),
            # Freezing check ||u_n(T) - u_0||_L2:
            "evolution_norm_cdquad": pgrid.l2_norm(pgrid.prolong(m_u, tr_cdq_u[-1]) - u0_eval),
            "evolution_norm_adacde": pgrid.l2_norm(pgrid.prolong(m_u, u_ada_u) - u0_eval),
            "evolution_norm_cde_grand": pgrid.l2_norm(pgrid.prolong(m_u, u_cde_u) - u0_eval),
            "evolution_norm_grand_nl": pgrid.l2_norm(pgrid.prolong(m_u, u_grn_u) - u0_eval),
            "rescaled_adacde_graded_rhs_jump": rhs_interface_jump_adacde,
        })

    for r in class_refinement:
        print(
            f"  nx={r['nx']:2d}: alpha_AdaCDE={r['diploma_alpha_mean']:.5f} (vs CDQuad {r['cdquad_alpha_mean']:.4f}), "
            f"||u(T)-u0||: CDQuad={r['evolution_norm_cdquad']:.4f}, AdaCDE={r['evolution_norm_adacde']:.5f}, "
            f"unif-vs-graded CDQuad={r['uniform_vs_graded_cdquad']:.5f}, rescaled-AdaCDE-jump={r['rescaled_adacde_graded_rhs_jump']:.4f}"
        )

    results["step_1D_class_boundary"] = {
        "base_adacde_evolution_norm": base_evolution_mag,
        "refinement_table": class_refinement,
    }
    results["elapsed_sec"] = round(time.time() - t0_wall, 2)

    with open("results/exp1_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results/exp1_results.json in {results['elapsed_sec']}s")


if __name__ == "__main__":
    run_all()
