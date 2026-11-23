"""
Experiment Suite 3 (Step E — Applied Morphometry of Dynamic Structures on Images):
  3A. Synthetic Dynamic Microscopy Benchmark (108x108 ground truth):
      Train CDQuad, Diploma AdaCDE (eq. 2.6), CDE-GRAND, and GRAND-nl EXCLUSIVELY on
      coarse 18x18 downsampled image sequences; evaluate zero-shot transfer on:
        (i)   coarse 18x18 grid (in-resolution),
        (ii)  fine 54x54 grid (9x node density increase),
        (iii) irregular Delaunay sensor mesh (N ~ 1225),
        (iv)  SLIC superpixel RAG (K ~ 300),
      measuring L^2 trajectory error, soft-threshold area M_0, centroid M_1,
      front migration speed V, and split-conformal error certificate coverage.
  3B. Real Live-Cell Fluorescence Time-Lapse Microscopy (NPCsingleNucleus.tif,
      15 frames, 180x180 nuclear protein transport):
      Fit models on coarse 30x30 early frame transitions (k=0..4), calibrate on k=5..7,
      and evaluate zero-shot morphometric stability & cross-resolution discrepancy
      between 30x30 and 90x90 (9x resolution increase) + SLIC RAG on held-out frames k=8..11.
"""
from __future__ import annotations

import json
import os
import time
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
import tifffile

from src.meshes import (
    build_structured_mesh,
    build_jittered_delaunay_mesh,
    build_slic_graph,
    ProlongationGrid,
)
from src.operators import (
    PhysicalLawParams,
    DiscreteGNNParams,
    solve_cdquad_trajectory,
    solve_discrete_gnn_rk4,
)
from src.descriptors import compute_all_descriptors, lipschitz_const_area, lipschitz_const_centroid
from src.certificate import compute_mesh_defects, analytic_proxy_bound, ConformalCertificate


def make_synthetic_initial_state(coords: np.ndarray, seq_id: int) -> np.ndarray:
    rng = np.random.default_rng(30000 + seq_id)
    cx1 = rng.uniform(0.26, 0.42)
    cy1 = rng.uniform(0.28, 0.46)
    r1 = rng.uniform(0.09, 0.125)
    amp1 = rng.uniform(0.85, 1.0)

    cx2 = cx1 + rng.uniform(0.14, 0.24)
    cy2 = cy1 + rng.uniform(0.08, 0.20)
    r2 = rng.uniform(0.07, 0.10)
    amp2 = rng.uniform(0.45, 0.70)

    x, y = coords[:, 0], coords[:, 1]
    u = amp1 * np.exp(-((x - cx1) ** 2 + (y - cy1) ** 2) / (2.0 * r1 ** 2))
    u += amp2 * np.exp(-((x - cx2) ** 2 + (y - cy2) ** 2) / (2.0 * r2 ** 2))
    return np.clip(u, 0.0, 1.2)


def build_parametric_law(vec: np.ndarray) -> PhysicalLawParams:
    """vec = [log_k1, log_k2, theta, u_vel, logit_alpha]"""
    k1 = float(np.exp(np.clip(vec[0], -8.5, -2.0)))
    k2 = float(np.exp(np.clip(vec[1], -8.5, -2.0)))
    theta0 = float(vec[2])
    u_vel = float(vec[3])
    alpha_max = float(0.85 / (1.0 + np.exp(-np.clip(vec[4], -6.0, 6.0))))
    return PhysicalLawParams(
        kappa1=k1,
        kappa2=k2,
        theta_fn=lambda c, th=theta0: np.full(c.shape[0], th),
        psi_fn=lambda c, uv=u_vel: (uv / np.pi) * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1]),
        lambda_r=0.0,
        alpha_max=alpha_max,
        r0=0.16,
        gamma_hom=6.0,
    )


def build_gnn_params(vec: np.ndarray) -> DiscreteGNNParams:
    return DiscreteGNNParams(
        w_q=float(vec[0]),
        w_k=float(vec[1]),
        w_v=float(vec[2]),
        w_rbf=float(np.exp(np.clip(vec[3], -3.0, 3.0))),
        beta=float(vec[4]),
        rate_scale=float(np.exp(np.clip(vec[5], -4.0, 4.0))),
    )


def sample_image_at_coords(img_2d: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Bilinear / cell sampling of 2D image (H, W) at coords in [0, 1]^2 (x=col/W, y=row/H)."""
    from scipy.ndimage import map_coordinates
    H, W = img_2d.shape
    cols = coords[:, 0] * W - 0.5
    rows = coords[:, 1] * H - 0.5
    return map_coordinates(img_2d, [rows, cols], order=1, mode="nearest")


def run_synthetic_benchmark(pgrid: ProlongationGrid):
    print("=== Running 3A: Synthetic Dynamic Microscopy Cross-Resolution Benchmark ===")
    true_params = PhysicalLawParams(
        kappa1=0.022,
        kappa2=0.0055,
        theta_fn=lambda c: np.full(c.shape[0], 0.55),
        psi_fn=lambda c: (0.20 / np.pi) * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1]),
        lambda_r=0.0,
        alpha_max=0.55,
        r0=0.16,
        gamma_hom=6.0,
    )
    T = 0.35
    theta_d = 0.32
    eta_d = 0.06

    mesh_gt = build_structured_mesh(90, 90, diagonal="alternating")
    mesh_coarse = build_structured_mesh(18, 18, diagonal="right")
    mesh_fine = build_structured_mesh(54, 54, diagonal="right")

    # Generate 22 independent dynamic image sequences: 6 train, 8 calib, 8 test
    print("  Generating 22 ground-truth 90x90 trajectories...", flush=True)
    n_total = 22
    gt_u0_eval = []
    gt_uT_eval = []
    coarse_u0 = []
    coarse_uT_obs = []

    rng_noise = np.random.default_rng(777)
    for sid in range(n_total):
        u0_gt = make_synthetic_initial_state(mesh_gt.points, sid)
        _, tr_gt = solve_cdquad_trajectory(mesh_gt, u0_gt, true_params, T=T, n_steps=80)
        gt_u0_eval.append(pgrid.prolong(mesh_gt, u0_gt))
        gt_uT_eval.append(pgrid.prolong(mesh_gt, tr_gt[-1]))

        # Coarse observations (with realistic low sensor noise sigma=0.004 on training frames)
        u0_c = make_synthetic_initial_state(mesh_coarse.points, sid)
        from scipy.interpolate import LinearNDInterpolator
        interp_gt = LinearNDInterpolator(mesh_gt.points, tr_gt[-1])
        uT_c_clean = interp_gt(mesh_coarse.points)
        if sid < 6:
            uT_c = uT_c_clean + rng_noise.normal(0.0, 0.004, size=uT_c_clean.shape)
        else:
            uT_c = uT_c_clean
        coarse_u0.append(u0_c)
        coarse_uT_obs.append(uT_c)

    # 1. Fit CDQuad ONLY on the 6 coarse 18x18 training sequences!
    print("  Fitting CDQuad on 6 coarse 18x18 sequences...", flush=True)
    def loss_cdquad(vec: np.ndarray) -> float:
        p = build_parametric_law(vec)
        tot = 0.0
        for sid in range(6):
            _, tr = solve_cdquad_trajectory(mesh_coarse, coarse_u0[sid], p, T=T, n_steps=25)
            diff = tr[-1] - coarse_uT_obs[sid]
            tot += float(np.sum(mesh_coarse.mass * (diff ** 2)))
        return tot / 6.0

    x0_cdq = np.array([np.log(0.015), np.log(0.008), 0.35, 0.14, 0.0])
    res_cdq = minimize(loss_cdquad, x0_cdq, method="Powell", options={"maxfev": 160, "ftol": 1e-4})
    learned_cdq = build_parametric_law(res_cdq.x)
    print(
        f"  Learned CDQuad on 18x18: k1={learned_cdq.kappa1:.5f} (true 0.022), "
        f"k2={learned_cdq.kappa2:.5f} (true 0.0055), theta={(res_cdq.x[2] % np.pi):.3f} (true 0.55), "
        f"U={res_cdq.x[3]:.3f} (true 0.20), alpha_max={learned_cdq.alpha_max:.3f} (true 0.55)",
        flush=True,
    )

    # 2. Fit Diploma AdaCDE, CDE-GRAND, and GRAND-nl on the EXACT SAME 6 coarse 18x18 sequences!
    learned_gnns: dict[str, DiscreteGNNParams] = {}
    for mname in ["AdaCDE", "CDE-GRAND", "GRAND-nl"]:
        def loss_gnn(vec: np.ndarray, model=mname) -> float:
            gp = build_gnn_params(vec)
            tot = 0.0
            for sid in range(6):
                pred = solve_discrete_gnn_rk4(model, mesh_coarse, coarse_u0[sid], gp, T=T, n_steps=25)
                if not np.all(np.isfinite(pred)):
                    return 1e6
                diff = pred - coarse_uT_obs[sid]
                tot += float(np.sum(mesh_coarse.mass * (diff ** 2)))
            return tot / 6.0

        x0_g = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        res_g = minimize(loss_gnn, x0_g, method="Powell", options={"maxfev": 160, "ftol": 1e-4})
        learned_gnns[mname] = build_gnn_params(res_g.x)
        print(f"  Trained {mname:9s} on 18x18: train_mse={res_g.fun:.6f}, rate={learned_gnns[mname].rate_scale:.3f}", flush=True)

    # 3. Calibrate Split-Conformal Certificate on sequences 6..13 (8 sequences x 2 irregular meshes = 16 pairs)
    cal_e = []
    cal_b = []
    for idx_c, sid in enumerate(range(6, 14)):
        for m_type in range(2):
            nx_m = 18 if m_type == 0 else 36
            m_cal = build_jittered_delaunay_mesh(nx_m, nx_m, jitter=0.25, seed=61000 + 10 * idx_c + m_type)
            defects = compute_mesh_defects(m_cal, learned_cdq, max_freq=3)
            b_val = analytic_proxy_bound(m_cal, learned_cdq, T, defects["delta_dual_rms"], e0_est=0.0)
            _, tr_c = solve_cdquad_trajectory(m_cal, make_synthetic_initial_state(m_cal.points, sid), learned_cdq, T=T, n_steps=80)
            e_val = pgrid.l2_norm(pgrid.prolong(m_cal, tr_c[-1]) - gt_uT_eval[sid])
            cal_e.append(e_val)
            cal_b.append(b_val)

    cert_90 = ConformalCertificate.fit(np.array(cal_e), np.array(cal_b), alpha_miscoverage=0.10)
    L_area = lipschitz_const_area(eta_d)
    L_cent = lipschitz_const_centroid(eta_d, m0_lower=0.08)

    # 4. Zero-Shot Evaluation on Held-Out Test Sequences 14..21 across 4 graph types!
    test_sids = list(range(14, 22))
    summary_by_regime: dict[str, dict] = {}

    H_slic, W_slic = 108, 108
    xs_s = (np.arange(W_slic) + 0.5) / W_slic
    ys_s = (np.arange(H_slic) + 0.5) / H_slic
    xx_s, yy_s = np.meshgrid(xs_s, ys_s, indexing="xy")
    coords_slic_px = np.column_stack([xx_s.ravel(), yy_s.ravel()])

    for regime in ["coarse_18x18", "fine_54x54", "delaunay_N1225", "slic_K300"]:
        metrics = {m: {"l2": [], "area": [], "cent": [], "speed": []} for m in ["CDQuad", "AdaCDE", "CDE-GRAND", "GRAND-nl"]}
        cert_cov_l2 = []
        cert_cov_area = []
        cert_cov_cent = []
        cert_widths_l2 = []
        cert_widths_area = []

        for sid in test_sids:
            if regime == "coarse_18x18":
                m_eval = mesh_coarse
            elif regime == "fine_54x54":
                m_eval = mesh_fine
            elif regime == "delaunay_N1225":
                m_eval = build_jittered_delaunay_mesh(34, 34, jitter=0.27, seed=65000 + sid)
            elif regime == "slic_K300":
                img0 = make_synthetic_initial_state(coords_slic_px, sid).reshape(H_slic, W_slic)
                tex = 0.05 * np.sin(2 * np.pi * 8 * xx_s) * np.cos(2 * np.pi * 8 * yy_s)
                m_eval = build_slic_graph(np.clip(img0 + tex, 0.0, 1.2), n_segments=300, compactness=0.10)
                m_eval.name = f"test_slic_{sid}"
            else:
                raise ValueError(regime)

            u0_m = make_synthetic_initial_state(m_eval.points, sid)

            # Ground truth descriptors
            from src.descriptors import soft_area_grid, soft_centroid_grid
            gt_area = soft_area_grid(pgrid, gt_uT_eval[sid], theta=theta_d, eta=eta_d)
            gt_c0 = soft_centroid_grid(pgrid, gt_u0_eval[sid], theta=theta_d, eta=eta_d)
            gt_cT = soft_centroid_grid(pgrid, gt_uT_eval[sid], theta=theta_d, eta=eta_d)
            gt_speed = float(np.linalg.norm((gt_cT - gt_c0) / T))

            # CDQuad prediction + certificate
            _, tr_cdq = solve_cdquad_trajectory(m_eval, u0_m, learned_cdq, T=T, n_steps=90)
            uT_cdq_grid = pgrid.prolong(m_eval, tr_cdq[-1])
            err_l2_cdq = pgrid.l2_norm(uT_cdq_grid - gt_uT_eval[sid])
            desc_cdq = compute_all_descriptors(pgrid, m_eval, u0_m, tr_cdq[-1], T=T, theta=theta_d, eta=eta_d)

            metrics["CDQuad"]["l2"].append(err_l2_cdq)
            metrics["CDQuad"]["area"].append(abs(float(desc_cdq["soft_area"]) - gt_area))
            metrics["CDQuad"]["cent"].append(float(np.linalg.norm(desc_cdq["centroid_T"] - gt_cT)))
            metrics["CDQuad"]["speed"].append(abs(float(desc_cdq["front_speed"]) - gt_speed))

            defects_m = compute_mesh_defects(m_eval, learned_cdq, max_freq=3)
            b_m = analytic_proxy_bound(m_eval, learned_cdq, T, defects_m["delta_dual_rms"], e0_est=0.0)
            bound_l2 = float(cert_90.predict_bound(b_m))
            cert_widths_l2.append(bound_l2)
            cert_widths_area.append(L_area * bound_l2)
            cert_cov_l2.append(float(err_l2_cdq <= bound_l2))
            cert_cov_area.append(float(abs(float(desc_cdq["soft_area"]) - gt_area) <= L_area * bound_l2))
            cert_cov_cent.append(float(np.linalg.norm(desc_cdq["centroid_T"] - gt_cT) <= L_cent * bound_l2))

            # Negative class GNN baselines
            for mname in ["AdaCDE", "CDE-GRAND", "GRAND-nl"]:
                uT_g = solve_discrete_gnn_rk4(mname, m_eval, u0_m, learned_gnns[mname], T=T, n_steps=90)
                uT_g_grid = pgrid.prolong(m_eval, uT_g)
                desc_g = compute_all_descriptors(pgrid, m_eval, u0_m, uT_g, T=T, theta=theta_d, eta=eta_d)
                metrics[mname]["l2"].append(pgrid.l2_norm(uT_g_grid - gt_uT_eval[sid]))
                metrics[mname]["area"].append(abs(float(desc_g["soft_area"]) - gt_area))
                metrics[mname]["cent"].append(float(np.linalg.norm(desc_g["centroid_T"] - gt_cT)))
                metrics[mname]["speed"].append(abs(float(desc_g["front_speed"]) - gt_speed))

        summary_by_regime[regime] = {
            "models": {
                m: {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in d.items()}
                for m, d in metrics.items()
            },
            "certificate_90": {
                "l2_coverage": float(np.mean(cert_cov_l2)),
                "area_coverage": float(np.mean(cert_cov_area)),
                "centroid_coverage": float(np.mean(cert_cov_cent)),
                "mean_l2_bound": float(np.mean(cert_widths_l2)),
                "mean_area_bound": float(np.mean(cert_widths_area)),
            },
        }
        print(
            f"  Regime {regime:15s} | L2 err: CDQuad={np.mean(metrics['CDQuad']['l2']):.4f}, "
            f"AdaCDE={np.mean(metrics['AdaCDE']['l2']):.4f}, CDE-GRAND={np.mean(metrics['CDE-GRAND']['l2']):.4f} | "
            f"Area err: CDQuad={np.mean(metrics['CDQuad']['area']):.4f}, AdaCDE={np.mean(metrics['AdaCDE']['area']):.4f} | "
            f"Cert90 cov={np.mean(cert_cov_l2):.2f} (bound={np.mean(cert_widths_l2):.4f})"
        )

    return {
        "learned_cdquad_params": {
            "kappa1": learned_cdq.kappa1,
            "kappa2": learned_cdq.kappa2,
            "theta": float(res_cdq.x[2] % np.pi),
            "u_vel": float(res_cdq.x[3]),
            "alpha_max": learned_cdq.alpha_max,
            "conformal_q90": cert_90.q_quantile,
        },
        "regimes": summary_by_regime,
    }


def run_real_microscopy_benchmark(pgrid: ProlongationGrid):
    print("=== Running 3B: Real Live-Cell Fluorescence Time-Lapse (NPCsingleNucleus.tif) ===")
    raw = tifffile.imread("data/NPCsingleNucleus.tif")  # (15, 2, 180, 183)
    # Channel 1: cargo protein fluorescence across 15 frames; crop square 180x180
    ch1 = raw[:, 1, :180, :180].astype(np.float64)
    # Mild sigma=1.2 px denoising of photon shot noise and normalize globally to [0, 1]
    frames = np.array([gaussian_filter(f, sigma=1.2) for f in ch1])
    f_min, f_max = float(np.percentile(frames, 1.0)), float(np.percentile(frames, 99.5))
    frames = np.clip((frames - f_min) / max(f_max - f_min, 1e-8), 0.0, 1.0)

    # Build 12 multi-frame transitions: frame k -> frame k+3 (T_phys = 0.30)
    # k = 0..4 (5 transitions): Train on coarse 30x30 (6x downsampled per axis -> 36x fewer pixels)
    # k = 5..7 (3 transitions): Calibration
    # k = 8..11 (4 transitions): Held-out Test on coarse 30x30, fine 90x90 (9x more nodes), and SLIC K~320!
    T_real = 0.30
    theta_r = 0.42
    eta_r = 0.07

    mesh_c30 = build_structured_mesh(29, 29, diagonal="right")  # 30x30 = 900 nodes
    mesh_f90 = build_structured_mesh(89, 89, diagonal="right")  # 90x90 = 8100 nodes (9x more nodes!)

    # Downsample frames by 6x6 box-averaging for 30x30, and 2x2 box-averaging for 90x90
    frames_30 = frames.reshape(15, 30, 6, 30, 6).mean(axis=(2, 4))
    frames_90 = frames.reshape(15, 90, 2, 90, 2).mean(axis=(2, 4))

    train_u0 = [sample_image_at_coords(frames_30[k], mesh_c30.points) for k in range(5)]
    train_uT = [sample_image_at_coords(frames_30[k + 3], mesh_c30.points) for k in range(5)]

    # Fit CDQuad with linear reaction term lambda_r (accounts for photobleaching / out-of-plane flux!)
    def unpack_real_cdq(vec: np.ndarray) -> PhysicalLawParams:
        k1 = float(np.exp(np.clip(vec[0], -9.0, -2.5)))
        k2 = float(np.exp(np.clip(vec[1], -9.0, -2.5)))
        th = float(vec[2])
        vx = float(vec[3])
        vy = float(vec[4])
        lam_r = float(vec[5])  # may be positive (bleaching) or slightly negative (influx)
        return PhysicalLawParams(
            kappa1=k1,
            kappa2=k2,
            theta_fn=lambda c, t=th: np.full(c.shape[0], t),
            v_fn=lambda c, ux=vx, uy=vy: np.column_stack([
                ux * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1]),
                uy * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1]),
            ]),
            lambda_r=max(lam_r, -0.5),
            alpha_max=0.35,
            r0=0.15,
            gamma_hom=5.0,
        )

    def loss_real_cdq(vec: np.ndarray) -> float:
        p = unpack_real_cdq(vec)
        tot = 0.0
        for k in range(5):
            _, tr = solve_cdquad_trajectory(mesh_c30, train_u0[k], p, T=T_real, n_steps=25)
            tot += float(np.sum(mesh_c30.mass * ((tr[-1] - train_uT[k]) ** 2)))
        return tot / 5.0

    res_rcdq = minimize(
        loss_real_cdq,
        np.array([np.log(0.004), np.log(0.002), 0.3, 0.02, -0.02, 0.05]),
        method="Powell",
        options={"maxfev": 160, "ftol": 1e-4},
    )
    real_cdq = unpack_real_cdq(res_rcdq.x)
    print(f"  Real microscopy CDQuad fit on 30x30: train_mse={res_rcdq.fun:.6f}, k1={real_cdq.kappa1:.5f}, k2={real_cdq.kappa2:.5f}, lambda_r={real_cdq.lambda_r:.4f}", flush=True)

    real_gnns: dict[str, DiscreteGNNParams] = {}
    for mname in ["AdaCDE", "CDE-GRAND", "GRAND-nl"]:
        def loss_rgnn(vec: np.ndarray, model=mname) -> float:
            gp = build_gnn_params(vec)
            tot = 0.0
            for k in range(5):
                pred = solve_discrete_gnn_rk4(model, mesh_c30, train_u0[k], gp, T=T_real, n_steps=25)
                if not np.all(np.isfinite(pred)):
                    return 1e6
                tot += float(np.sum(mesh_c30.mass * ((pred - train_uT[k]) ** 2)))
            return tot / 5.0

        res_rg = minimize(loss_rgnn, np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.2]), method="Powell", options={"maxfev": 160, "ftol": 1e-4})
        real_gnns[mname] = build_gnn_params(res_rg.x)
        print(f"  Real microscopy {mname:9s} fit on 30x30: train_mse={res_rg.fun:.6f}", flush=True)

    # Calibrate cross-resolution transfer discrepancy ||P_30 u_30 - P_90 u_90||_L2 & full-frame error on k=5..7
    cal_errs_r = []
    cal_b_r = []
    def_c30 = compute_mesh_defects(mesh_c30, real_cdq, max_freq=3)
    b_c30 = analytic_proxy_bound(mesh_c30, real_cdq, T_real, def_c30["delta_dual_rms"], e0_est=0.0)
    def_f90 = compute_mesh_defects(mesh_f90, real_cdq, max_freq=3)
    b_f90 = analytic_proxy_bound(mesh_f90, real_cdq, T_real, def_f90["delta_dual_rms"], e0_est=0.0)

    for k in [5, 6, 7]:
        u0_30 = sample_image_at_coords(frames_30[k], mesh_c30.points)
        u0_90 = sample_image_at_coords(frames_90[k], mesh_f90.points)
        _, tr30 = solve_cdquad_trajectory(mesh_c30, u0_30, real_cdq, T=T_real, n_steps=50)
        _, tr90 = solve_cdquad_trajectory(mesh_f90, u0_90, real_cdq, T=T_real, n_steps=50)
        diff_30_90 = pgrid.l2_diff(mesh_c30, tr30[-1], mesh_f90, tr90[-1])
        cal_errs_r.append(diff_30_90)
        cal_b_r.append(b_c30 + b_f90)

    cert_real_pair = ConformalCertificate.fit(np.array(cal_errs_r), np.array(cal_b_r), alpha_miscoverage=0.25)  # B=3 -> (3+1)*0.75=3rd order stat (max of 3!)

    # Evaluate on held-out windows k = 8, 9, 10, 11
    from src.descriptors import soft_area_grid, soft_centroid_grid
    heldout_models = {
        m: {"l2_30": [], "l2_90": [], "area_90": [], "cent_90": [], "cross_res_drift_30_vs_90": [], "slic_drift_vs_90": []}
        for m in ["CDQuad", "AdaCDE", "CDE-GRAND", "GRAND-nl"]
    }
    pair_cert_covered = []

    for k in [8, 9, 10, 11]:
        gt_grid = sample_image_at_coords(frames[k + 3], pgrid.coords)
        gt_area = soft_area_grid(pgrid, gt_grid, theta=theta_r, eta=eta_r)
        gt_cent = soft_centroid_grid(pgrid, gt_grid, theta=theta_r, eta=eta_r)

        u0_30 = sample_image_at_coords(frames_30[k], mesh_c30.points)
        u0_90 = sample_image_at_coords(frames_90[k], mesh_f90.points)
        m_slic_r = build_slic_graph(frames[k], n_segments=320, compactness=0.10)
        m_slic_r.name = f"real_slic_k{k}"
        u0_slic = sample_image_at_coords(frames[k], m_slic_r.points)

        # CDQuad
        _, tr30 = solve_cdquad_trajectory(mesh_c30, u0_30, real_cdq, T=T_real, n_steps=50)
        _, tr90 = solve_cdquad_trajectory(mesh_f90, u0_90, real_cdq, T=T_real, n_steps=50)
        _, tr_sl = solve_cdquad_trajectory(m_slic_r, u0_slic, real_cdq, T=T_real, n_steps=50)
        p30 = pgrid.prolong(mesh_c30, tr30[-1])
        p90 = pgrid.prolong(mesh_f90, tr90[-1])
        psl = pgrid.prolong(m_slic_r, tr_sl[-1])

        drift_cdq = pgrid.l2_norm(p30 - p90)
        pair_cert_covered.append(float(drift_cdq <= cert_real_pair.predict_bound(b_c30 + b_f90)))

        heldout_models["CDQuad"]["l2_30"].append(pgrid.l2_norm(p30 - gt_grid))
        heldout_models["CDQuad"]["l2_90"].append(pgrid.l2_norm(p90 - gt_grid))
        heldout_models["CDQuad"]["area_90"].append(abs(soft_area_grid(pgrid, p90, theta_r, eta_r) - gt_area))
        heldout_models["CDQuad"]["cent_90"].append(float(np.linalg.norm(soft_centroid_grid(pgrid, p90, theta_r, eta_r) - gt_cent)))
        heldout_models["CDQuad"]["cross_res_drift_30_vs_90"].append(drift_cdq)
        heldout_models["CDQuad"]["slic_drift_vs_90"].append(pgrid.l2_norm(psl - p90))

        for mname in ["AdaCDE", "CDE-GRAND", "GRAND-nl"]:
            g30 = solve_discrete_gnn_rk4(mname, mesh_c30, u0_30, real_gnns[mname], T=T_real, n_steps=50)
            g90 = solve_discrete_gnn_rk4(mname, mesh_f90, u0_90, real_gnns[mname], T=T_real, n_steps=50)
            gsl = solve_discrete_gnn_rk4(mname, m_slic_r, u0_slic, real_gnns[mname], T=T_real, n_steps=50)
            pg30 = pgrid.prolong(mesh_c30, g30)
            pg90 = pgrid.prolong(mesh_f90, g90)
            pgsl = pgrid.prolong(m_slic_r, gsl)
            heldout_models[mname]["l2_30"].append(pgrid.l2_norm(pg30 - gt_grid))
            heldout_models[mname]["l2_90"].append(pgrid.l2_norm(pg90 - gt_grid))
            heldout_models[mname]["area_90"].append(abs(soft_area_grid(pgrid, pg90, theta_r, eta_r) - gt_area))
            heldout_models[mname]["cent_90"].append(float(np.linalg.norm(soft_centroid_grid(pgrid, pg90, theta_r, eta_r) - gt_cent)))
            heldout_models[mname]["cross_res_drift_30_vs_90"].append(pgrid.l2_norm(pg30 - pg90))
            heldout_models[mname]["slic_drift_vs_90"].append(pgrid.l2_norm(pgsl - pg90))

    real_summary = {
        m: {k: float(np.mean(v)) for k, v in d.items()}
        for m, d in heldout_models.items()
    }
    for m, d in real_summary.items():
        print(
            f"  Real HeldOut {m:9s} | L2(30x30)={d['l2_30']:.4f} -> L2(90x90)={d['l2_90']:.4f} | "
            f"CrossResDrift(30v90)={d['cross_res_drift_30_vs_90']:.4f} | AreaErr90={d['area_90']:.4f}, CentErr90={d['cent_90']:.4f}"
        )

    return {
        "dataset": "NPCsingleNucleus.tif (skimage protein_transport, 15 frames, 180x180, channel 1)",
        "models_heldout_mean": real_summary,
        "pair_transfer_cert_bound": float(cert_real_pair.predict_bound(b_c30 + b_f90)),
        "pair_transfer_cert_coverage": float(np.mean(pair_cert_covered)),
    }


def run_all():
    t0 = time.time()
    os.makedirs("results", exist_ok=True)
    pgrid = ProlongationGrid(m_grid=160)
    res_3a = run_synthetic_benchmark(pgrid)
    res_3b = run_real_microscopy_benchmark(pgrid)
    out = {
        "benchmark_3A_synthetic_microscopy": res_3a,
        "benchmark_3B_real_fluorescence_timelapse": res_3b,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    with open("results/exp3_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results/exp3_results.json in {out['elapsed_sec']}s")


if __name__ == "__main__":
    run_all()
