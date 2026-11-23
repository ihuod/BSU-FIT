"""
Experiment Suite 2 (Step D):
  2A. Elliptic Dual Defect delta_{-1}(G) vs Naive L^2 Defect delta_{L2}(G) on irregular meshes.
  2B. Split-Conformal Calibration on held-out random Delaunay meshes (B=80 cal, M=80 test,
      plus 200 random splits) at alpha in {0.20, 0.10, 0.05}, and comparison vs unnormalized CP.
  2C. Transfer of error certificate to Lipschitz morphometric descriptors (soft area, centroid)
      and demonstration of non-Lipschitz jump on hard-threshold indicator area.
  2D. Out-of-distribution (OOD) stress tests breaking exchangeability:
      (i) anisotropically stretched meshes (power_x = 3.0),
      (ii) SLIC superpixel Region Adjacency Graphs (non-Delaunay / non-Donald).
  2E. High-Peclet sharp-front test (Pe = 400): centered dispersion overshoot vs upwind monotonicity.
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
    build_stretched_ood_mesh,
    build_slic_graph,
    ProlongationGrid,
)
from src.operators import PhysicalLawParams, solve_cdquad_trajectory
from src.certificate import compute_mesh_defects, analytic_proxy_bound, ConformalCertificate
from src.descriptors import (
    soft_area_grid,
    hard_area_grid,
    soft_centroid_grid,
    lipschitz_const_area,
    lipschitz_const_centroid,
)
from src.exp1_sanity_and_convergence import initial_field_smooth, make_aniso_cd_params


def sample_in_distribution_mesh(idx: int, seed_base: int = 40000):
    rng = np.random.default_rng(seed_base + idx)
    # Mix random Lloyd-Delaunay and jittered Delaunay across N in [144, 1024]
    if idx % 2 == 0:
        nx = int(rng.integers(11, 31))
        jit = float(rng.uniform(0.16, 0.34))
        m = build_jittered_delaunay_mesh(nx, nx, jitter=jit, seed=seed_base + 1000 + idx)
        m.name = f"id_jit_{idx}_nx{nx}"
        return m
    else:
        n_approx = int(rng.integers(150, 950))
        lloyd = int(rng.integers(1, 3))
        m = build_random_delaunay_mesh(n_approx, seed=seed_base + 2000 + idx, lloyd_iters=lloyd)
        m.name = f"id_rnd_{idx}_N{m.points.shape[0]}"
        return m


def evaluate_mesh_record(mesh, params, T, n_steps, pgrid, u_ref_eval, theta_desc=0.35, eta_desc=0.06):
    u0 = initial_field_smooth(mesh.points)
    u0_eval = pgrid.prolong(mesh, u0)
    u0_exact_eval = initial_field_smooth(pgrid.coords)
    e0_est = pgrid.l2_norm(u0_eval - u0_exact_eval)

    defects = compute_mesh_defects(mesh, params, max_freq=3)
    # Pure operator-defect proxy (zero peeking at u0_exact or reference!)
    b_hat = analytic_proxy_bound(mesh, params, T, defects["delta_dual_rms"], e0_est=0.0)

    _, tr = solve_cdquad_trajectory(mesh, u0, params, T=T, n_steps=n_steps)
    uT_eval = pgrid.prolong(mesh, tr[-1])
    e_true = pgrid.l2_norm(uT_eval - u_ref_eval)

    area_g = soft_area_grid(pgrid, uT_eval, theta=theta_desc, eta=eta_desc)
    hard_g = hard_area_grid(pgrid, uT_eval, theta=theta_desc)
    cent_g = soft_centroid_grid(pgrid, uT_eval, theta=theta_desc, eta=eta_desc)

    return {
        "name": mesh.name,
        "n_nodes": int(mesh.points.shape[0]),
        "h_max": mesh.h_max,
        "h_mean": mesh.h_mean,
        "shape_reg": None if np.isnan(mesh.shape_reg) else mesh.shape_reg,
        "e0": e0_est,
        "delta_dual_rms": defects["delta_dual_rms"],
        "delta_dual_max": defects["delta_dual_max"],
        "delta_l2_rms": defects["delta_l2_rms"],
        "b_hat": b_hat,
        "e_true": e_true,
        "soft_area": area_g,
        "hard_area": hard_g,
        "centroid_x": float(cent_g[0]),
        "centroid_y": float(cent_g[1]),
    }


def run_all():
    t0 = time.time()
    os.makedirs("results", exist_ok=True)
    pgrid = ProlongationGrid(m_grid=160)

    # Anisotropic convection-diffusion with reaction damping lambda_r = 0.15, Pe = 35, chi = 5
    u_vel = 35.0 * 0.005 / np.sqrt(2.0)
    params = make_aniso_cd_params(u_vel=u_vel, alpha_max=0.0)
    params.lambda_r = 0.15
    T = 0.40

    print("--- Computing fine 96x96 reference trajectory ---")
    mesh_ref = build_structured_mesh(96, 96, diagonal="alternating")
    _, tr_ref = solve_cdquad_trajectory(mesh_ref, initial_field_smooth(mesh_ref.points), params, T=T, n_steps=360)
    u_ref_eval = pgrid.prolong(mesh_ref, tr_ref[-1])

    theta_desc = 0.35
    eta_desc = 0.06
    area_ref = soft_area_grid(pgrid, u_ref_eval, theta=theta_desc, eta=eta_desc)
    hard_ref = hard_area_grid(pgrid, u_ref_eval, theta=theta_desc)
    cent_ref = soft_centroid_grid(pgrid, u_ref_eval, theta=theta_desc, eta=eta_desc)
    L_area = lipschitz_const_area(eta_desc)
    # Conservative m0 lower bound computed purely from coarse mass conservation (>= 0.8 * area_0)
    m0_lower = 0.12
    L_cent = lipschitz_const_centroid(eta_desc, m0_lower=m0_lower)

    # ==========================================================================
    # 2A & 2B: Generate 160 exchangeable ID meshes (80 calibration + 80 test)
    # ==========================================================================
    print("--- Evaluating 160 in-distribution random unstructured meshes ---")
    id_records = []
    for idx in range(160):
        m = sample_in_distribution_mesh(idx, seed_base=50000)
        rec = evaluate_mesh_record(m, params, T, n_steps=160, pgrid=pgrid, u_ref_eval=u_ref_eval, theta_desc=theta_desc, eta_desc=eta_desc)
        rec["area_err"] = abs(rec["soft_area"] - area_ref)
        rec["hard_area_err"] = abs(rec["hard_area"] - hard_ref)
        rec["cent_err"] = float(np.hypot(rec["centroid_x"] - cent_ref[0], rec["centroid_y"] - cent_ref[1]))
        id_records.append(rec)
        if (idx + 1) % 40 == 0:
            print(f"  Completed {idx + 1}/160 ID meshes...")

    e_all = np.array([r["e_true"] for r in id_records])
    d_dual_all = np.array([r["delta_dual_rms"] for r in id_records])
    d_l2_all = np.array([r["delta_l2_rms"] for r in id_records])
    b_all = np.array([r["b_hat"] for r in id_records])
    h_all = np.array([r["h_mean"] for r in id_records])

    corr_dual = float(np.corrcoef(d_dual_all, e_all)[0, 1])
    corr_bhat = float(np.corrcoef(b_all, e_all)[0, 1])
    corr_l2 = float(np.corrcoef(d_l2_all, e_all)[0, 1])
    slope_dual_vs_h = float(np.polyfit(np.log(h_all), np.log(d_dual_all), 1)[0])
    slope_l2_vs_h = float(np.polyfit(np.log(h_all), np.log(d_l2_all), 1)[0])
    print(f"  Correlation with e_true: B_hat={corr_bhat:.4f}, delta_dual={corr_dual:.4f}, naive delta_L2={corr_l2:.4f}")
    print(f"  Scaling vs h_mean: delta_dual slope={slope_dual_vs_h:.3f}, naive delta_L2 slope={slope_l2_vs_h:.3f}")

    # Primary split: seeded exchangeable 80/80 random split (seed=20261122)
    perm_primary = np.random.default_rng(20261122).permutation(160)
    cal_recs = [id_records[i] for i in perm_primary[:80]]
    test_recs = [id_records[i] for i in perm_primary[80:]]
    e_cal = np.array([r["e_true"] for r in cal_recs])
    b_cal = np.array([r["b_hat"] for r in cal_recs])
    e_test = np.array([r["e_true"] for r in test_recs])
    b_test = np.array([r["b_hat"] for r in test_recs])
    area_err_test = np.array([r["area_err"] for r in test_recs])
    cent_err_test = np.array([r["cent_err"] for r in test_recs])

    alphas = [0.20, 0.10, 0.05]
    conformal_id_table = []
    rng_mc = np.random.default_rng(202611)
    for alpha in alphas:
        cert = ConformalCertificate.fit(e_cal, b_cal, alpha_miscoverage=alpha)
        bounds_test = cert.predict_bound(b_test)
        cov_primary = float(np.mean(e_test <= bounds_test))

        # Compare against naive constant conformal bound (b_hat = 1)
        cert_const = ConformalCertificate.fit(e_cal, np.ones_like(b_cal), alpha_miscoverage=alpha)
        bounds_const = cert_const.predict_bound(np.ones_like(b_test))
        cov_const = float(np.mean(e_test <= bounds_const))

        # 200 random 80/80 splits to measure mean & std of coverage
        mc_covs = []
        for _ in range(200):
            perm = rng_mc.permutation(160)
            c_idx, t_idx = perm[:80], perm[80:]
            c_fit = ConformalCertificate.fit(e_all[c_idx], b_all[c_idx], alpha_miscoverage=alpha)
            mc_covs.append(float(np.mean(e_all[t_idx] <= c_fit.predict_bound(b_all[t_idx]))))

        # Theorem 2 induced descriptor intervals: L_Phi * q * B_hat
        area_bounds = L_area * bounds_test
        cent_bounds = L_cent * bounds_test
        cov_area_lip = float(np.mean(area_err_test <= area_bounds))
        cov_cent_lip = float(np.mean(cent_err_test <= cent_bounds))

        conformal_id_table.append({
            "alpha": alpha,
            "target_coverage": 1.0 - alpha,
            "q_quantile": cert.q_quantile,
            "empirical_coverage_primary": cov_primary,
            "mc_200_coverage_mean": float(np.mean(mc_covs)),
            "mc_200_coverage_std": float(np.std(mc_covs)),
            "mean_width_dual_norm": float(np.mean(bounds_test)),
            "mean_width_const_cp": float(np.mean(bounds_const)),
            "const_cp_coverage_id": cov_const,
            "mean_tightness_ratio_bound_over_err": float(np.mean(bounds_test / np.maximum(e_test, 1e-12))),
            "descriptor_area_lipschitz_coverage": cov_area_lip,
            "descriptor_area_mean_bound": float(np.mean(area_bounds)),
            "descriptor_area_mean_true_err": float(np.mean(area_err_test)),
            "descriptor_centroid_lipschitz_coverage": cov_cent_lip,
            "descriptor_centroid_mean_bound": float(np.mean(cent_bounds)),
            "descriptor_centroid_mean_true_err": float(np.mean(cent_err_test)),
        })
        print(
            f"  alpha={alpha:.2f} (target {1-alpha:.2f}): q={cert.q_quantile:.3f}, "
            f"cov={cov_primary:.3f} (MC {np.mean(mc_covs):.3f}±{np.std(mc_covs):.3f}), "
            f"mean_bound={np.mean(bounds_test):.5f} (vs const {np.mean(bounds_const):.5f}), "
            f"area_cov={cov_area_lip:.3f}, cent_cov={cov_cent_lip:.3f}"
        )

    # Non-Lipschitz counterexample check on hard threshold area:
    # Perturb reference field near level set theta by tiny delta in L^2 on flat plateau
    u_plateau = np.full(pgrid.coords.shape[0], theta_desc - 0.002)
    u_plateau_pert = np.full(pgrid.coords.shape[0], theta_desc + 0.002)
    l2_pert = pgrid.l2_norm(u_plateau_pert - u_plateau)
    hard_jump = abs(hard_area_grid(pgrid, u_plateau_pert, theta_desc) - hard_area_grid(pgrid, u_plateau, theta_desc))
    soft_jump = abs(soft_area_grid(pgrid, u_plateau_pert, theta_desc, eta_desc) - soft_area_grid(pgrid, u_plateau, theta_desc, eta_desc))

    # ==========================================================================
    # 2D. OOD Stress Tests: (i) Stretched graded meshes, (ii) SLIC Superpixels
    # ==========================================================================
    print("--- Evaluating OOD Stress Tests (stretched meshes & SLIC RAGs) ---")
    cert_90 = ConformalCertificate.fit(e_cal, b_cal, alpha_miscoverage=0.10)
    cert_const_90 = ConformalCertificate.fit(e_cal, np.ones_like(b_cal), alpha_miscoverage=0.10)

    ood_stretched = []
    for s in range(30):
        nx_s = int(np.random.default_rng(7000 + s).integers(10, 18))
        ny_s = int(np.random.default_rng(7100 + s).integers(12, 22))
        m_st = build_stretched_ood_mesh(nx_s, ny_s, power_x=3.0, seed=7200 + s)
        rec = evaluate_mesh_record(m_st, params, T, n_steps=160, pgrid=pgrid, u_ref_eval=u_ref_eval)
        ood_stretched.append(rec)

    e_st = np.array([r["e_true"] for r in ood_stretched])
    b_st = np.array([r["b_hat"] for r in ood_stretched])
    cov_st_dual = float(np.mean(e_st <= cert_90.predict_bound(b_st)))
    cov_st_const = float(np.mean(e_st <= cert_const_90.predict_bound(np.ones_like(b_st))))

    # SLIC Superpixel RAGs built on 128x128 synthetic textured/noisy initial image
    rng_slic = np.random.default_rng(8800)
    H_img, W_img = 128, 128
    xs_i = (np.arange(W_img) + 0.5) / W_img
    ys_i = (np.arange(H_img) + 0.5) / H_img
    xx_i, yy_i = np.meshgrid(xs_i, ys_i, indexing="xy")
    base_img = initial_field_smooth(np.column_stack([xx_i.ravel(), yy_i.ravel()])).reshape(H_img, W_img)

    ood_slic = []
    for s in range(30):
        n_seg = int(rng_slic.integers(180, 480))
        tex = 0.08 * np.sin(2 * np.pi * 9 * xx_i + rng_slic.uniform(0, 6)) * np.cos(2 * np.pi * 9 * yy_i)
        noisy_guide = np.clip(base_img + tex + rng_slic.normal(0, 0.03, size=base_img.shape), 0.0, 1.5)
        m_sl = build_slic_graph(noisy_guide, n_segments=n_seg, compactness=0.08)
        m_sl.name = f"ood_slic_{s}_K{m_sl.points.shape[0]}"
        rec = evaluate_mesh_record(m_sl, params, T, n_steps=160, pgrid=pgrid, u_ref_eval=u_ref_eval)
        ood_slic.append(rec)

    e_sl = np.array([r["e_true"] for r in ood_slic])
    b_sl = np.array([r["b_hat"] for r in ood_slic])
    cov_sl_dual = float(np.mean(e_sl <= cert_90.predict_bound(b_sl)))
    cov_sl_const = float(np.mean(e_sl <= cert_const_90.predict_bound(np.ones_like(b_sl))))

    # OOD 3: Localized sensor blind-spot mesh (fine 28x28 globally, but nodes inside
    # the active front zone [0.16, 0.54] x [0.18, 0.56] are 92% depleted!)
    from scipy.spatial import Delaunay
    from src.meshes import MeshData, _compute_triangulation_metrics
    ood_hole = []
    for s in range(30):
        rng_h = np.random.default_rng(9300 + s)
        base_m = build_jittered_delaunay_mesh(28, 28, jitter=0.22, seed=9400 + s)
        pts = base_m.points
        in_box = (pts[:, 0] > 0.16) & (pts[:, 0] < 0.54) & (pts[:, 1] > 0.18) & (pts[:, 1] < 0.56)
        keep = (~in_box) | (rng_h.uniform(0.0, 1.0, size=pts.shape[0]) < 0.06)
        pts_h = pts[keep]
        tri_h = Delaunay(pts_h).simplices
        tris_h, mass_h, edges_h, hmax_h, hmean_h, sreg_h = _compute_triangulation_metrics(pts_h, tri_h)
        m_hole = MeshData(
            name=f"ood_blindspot_{s}",
            points=pts_h,
            triangles=tris_h,
            mass=mass_h,
            edges=edges_h,
            h_max=hmax_h,
            h_mean=hmean_h,
            shape_reg=sreg_h,
        )
        # Note: certificate only sees u0 on the coarse blind-spot mesh (so use 0 initial proxy or coarse interpolation proxy)
        defects_h = compute_mesh_defects(m_hole, params, max_freq=3)
        b_hat_h = analytic_proxy_bound(m_hole, params, T, defects_h["delta_dual_rms"], e0_est=0.0)
        _, tr_h = solve_cdquad_trajectory(m_hole, initial_field_smooth(m_hole.points), params, T=T, n_steps=160)
        e_true_h = pgrid.l2_norm(pgrid.prolong(m_hole, tr_h[-1]) - u_ref_eval)
        ood_hole.append({"e_true": e_true_h, "b_hat": b_hat_h, "h_max": hmax_h, "h_mean": hmean_h})

    e_ho = np.array([r["e_true"] for r in ood_hole])
    b_ho = np.array([r["b_hat"] for r in ood_hole])
    cov_ho_dual = float(np.mean(e_ho <= cert_90.predict_bound(b_ho)))
    cov_ho_const = float(np.mean(e_ho <= cert_const_90.predict_bound(np.ones_like(b_ho))))

    print(f"  OOD Stretched (p=3.0, target 0.90): dual-cert cov={cov_st_dual:.3f}, const-CP cov={cov_st_const:.3f}, mean e_true={np.mean(e_st):.5f}")
    print(f"  OOD SLIC RAG  (target 0.90):        dual-cert cov={cov_sl_dual:.3f}, const-CP cov={cov_sl_const:.3f}, mean e_true={np.mean(e_sl):.5f}, mean bound={np.mean(cert_90.predict_bound(b_sl)):.5f}")
    print(f"  OOD BlindSpot (target 0.90):        dual-cert cov={cov_ho_dual:.3f}, const-CP cov={cov_ho_const:.3f}, mean e_true={np.mean(e_ho):.5f}, mean bound={np.mean(cert_90.predict_bound(b_ho)):.5f}")

    # ==========================================================================
    # 2E. High-Peclet Sharp-Front Stress Test (Pe = 400): Centered vs Upwind
    # ==========================================================================
    print("--- Running 2E: High-Peclet (Pe=400) sharp front test ---")
    p_hi = PhysicalLawParams(
        kappa1=0.0015,
        kappa2=0.0005,  # kappa_min = 0.0005
        theta_fn=lambda c: np.full(c.shape[0], np.pi / 6.0),
        psi_fn=lambda c: (0.141421356 / np.pi) * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1]),  # Pe = 400
    )
    sharp_u0_fn = lambda c: 1.0 / (1.0 + np.exp((np.hypot(c[:, 0] - 0.35, c[:, 1] - 0.35) - 0.14) / 0.012))
    m_hi_coarse = build_jittered_delaunay_mesh(18, 18, jitter=0.25, seed=9901)
    u0_hi = sharp_u0_fn(m_hi_coarse.points)
    _, tr_hi_up = solve_cdquad_trajectory(m_hi_coarse, u0_hi, p_hi, T=0.50, n_steps=250, scheme="upwind")
    _, tr_hi_ce = solve_cdquad_trajectory(m_hi_coarse, u0_hi, p_hi, T=0.50, n_steps=250, scheme="centered")
    min_up, max_up = float(np.min(tr_hi_up[-1])), float(np.max(tr_hi_up[-1]))
    min_ce, max_ce = float(np.min(tr_hi_ce[-1])), float(np.max(tr_hi_ce[-1]))
    print(f"  Pe=400 sharp front range (initial [0,1]): Upwind=[{min_up:.4f}, {max_up:.4f}], Centered=[{min_ce:.4f}, {max_ce:.4f}]")

    out = {
        "defect_comparison_160_meshes": {
            "corr_bhat_vs_etrue": corr_bhat,
            "corr_delta_dual_vs_etrue": corr_dual,
            "corr_delta_l2_vs_etrue": corr_l2,
            "log_slope_delta_dual_vs_h": slope_dual_vs_h,
            "log_slope_delta_l2_vs_h": slope_l2_vs_h,
            "sample_coarse_vs_fine": [
                id_records[0],
                id_records[1],
                id_records[10],
                id_records[11],
            ],
        },
        "conformal_id_results": conformal_id_table,
        "non_lipschitz_hard_area_demo": {
            "l2_perturbation": l2_pert,
            "hard_area_jump": hard_jump,
            "hard_ratio": hard_jump / l2_pert,
            "soft_area_jump": soft_jump,
            "soft_ratio": soft_jump / l2_pert,
            "theoretical_L_area": L_area,
        },
        "ood_stress_tests_alpha_010": {
            "stretched_power3": {
                "n_meshes": 30,
                "dual_cert_coverage": cov_st_dual,
                "const_cp_coverage": cov_st_const,
                "mean_e_true": float(np.mean(e_st)),
                "mean_dual_bound": float(np.mean(cert_90.predict_bound(b_st))),
                "const_bound": float(cert_const_90.q_quantile),
            },
            "slic_superpixel_rag": {
                "n_meshes": 30,
                "dual_cert_coverage": cov_sl_dual,
                "const_cp_coverage": cov_sl_const,
                "mean_e_true": float(np.mean(e_sl)),
                "mean_dual_bound": float(np.mean(cert_90.predict_bound(b_sl))),
                "const_bound": float(cert_const_90.q_quantile),
            },
            "localized_sensor_blindspot": {
                "n_meshes": 30,
                "dual_cert_coverage": cov_ho_dual,
                "const_cp_coverage": cov_ho_const,
                "mean_e_true": float(np.mean(e_ho)),
                "mean_dual_bound": float(np.mean(cert_90.predict_bound(b_ho))),
                "const_bound": float(cert_const_90.q_quantile),
            },
        },
        "high_peclet_pe400_sharp_front": {
            "upwind_min": min_up,
            "upwind_max": max_up,
            "centered_min": min_ce,
            "centered_max": max_ce,
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }

    with open("results/exp2_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results/exp2_results.json in {out['elapsed_sec']}s")


if __name__ == "__main__":
    run_all()
