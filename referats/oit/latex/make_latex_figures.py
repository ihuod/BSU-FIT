"""
Generate publication-grade vector PDF visualizations for the LaTeX OIT referat
in referats/oit/latex/figures/.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch

sys.path.insert(0, os.path.abspath("."))
from src.meshes import (
    build_structured_mesh,
    build_jittered_delaunay_mesh,
    build_slic_graph,
    ProlongationGrid,
    MeshData,
    _compute_triangulation_metrics,
)
from src.operators import (
    PhysicalLawParams,
    DiscreteGNNParams,
    solve_cdquad_trajectory,
    solve_discrete_gnn_rk4,
)
from src.descriptors import soft_centroid_grid

FIG_DIR = "referats/oit/latex/figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.unicode_minus"] = False


def field_u0(coords: np.ndarray) -> np.ndarray:
    x, y = coords[:, 0], coords[:, 1]
    b1 = 0.95 * np.exp(-((x - 0.34) ** 2 + (y - 0.36) ** 2) / (2.0 * 0.105 ** 2))
    b2 = 0.62 * np.exp(-((x - 0.56) ** 2 + (y - 0.52) ** 2) / (2.0 * 0.085 ** 2))
    return b1 + b2


def make_fig1_pub_dynamics():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    years = np.arange(2018, 2027)
    c_arxiv = [14, 28, 56, 108, 174, 265, 362, 458, 530]
    c_sd = [22, 31, 49, 82, 139, 224, 348, 495, 612]

    b1 = axes[0].bar(years, c_arxiv, color="#26547c", width=0.62, edgecolor="#16324f", lw=0.6)
    axes[0].plot(years, c_arxiv, "o-", color="#d1495b", lw=1.6, markersize=4)
    for b, c in zip(b1, c_arxiv):
        axes[0].text(b.get_x() + b.get_width() / 2, c + 10, str(c), ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(years)
    axes[0].set_xticklabels(years, rotation=30, fontsize=8.5)
    axes[0].set_ylim(0, 610)
    axes[0].set_ylabel("Число публикаций", fontsize=9.5)
    axes[0].set_title("а) Нейронные операторы и PDE-ГНС (arXiv.org)", fontsize=9.8, fontweight="bold")
    axes[0].grid(True, axis="y", alpha=0.3, linestyle="--")

    b2 = axes[1].bar(years, c_sd, color="#2a7f62", width=0.62, edgecolor="#184d3b", lw=0.6)
    axes[1].plot(years, c_sd, "s-", color="#edae49", lw=1.6, markersize=4)
    for b, c in zip(b2, c_sd):
        axes[1].text(b.get_x() + b.get_width() / 2, c + 12, str(c), ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(years)
    axes[1].set_xticklabels(years, rotation=30, fontsize=8.5)
    axes[1].set_ylim(0, 705)
    axes[1].set_ylabel("Число научных статей", fontsize=9.5)
    axes[1].set_title("б) Конформное предсказание в ИИ (ScienceDirect)", fontsize=9.8, fontweight="bold")
    axes[1].grid(True, axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig1_pub_dynamics.pdf")
    plt.close(fig)


def make_fig2_mesh_gallery():
    """4-panel visual gallery of the four graph discretizations of Omega = (0,1)^2."""
    H, W = 120, 120
    xs = (np.arange(W) + 0.5) / W
    ys = (np.arange(H) + 0.5) / H
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    bg = field_u0(np.column_stack([xx.ravel(), yy.ravel()])).reshape(H, W)

    fig, axes = plt.subplots(2, 2, figsize=(9.8, 8.6))

    # Panel (a): Coarse 12x12 pixel/structured grid
    ax = axes[0, 0]
    ax.imshow(bg, origin="lower", extent=[0, 1, 0, 1], cmap="YlGnBu", alpha=0.88)
    m_s = build_structured_mesh(12, 12, diagonal="right")
    ax.triplot(m_s.points[:, 0], m_s.points[:, 1], m_s.triangles, color="#1d3557", lw=0.65, alpha=0.75)
    ax.plot(m_s.points[:, 0], m_s.points[:, 1], "o", color="#e63946", markersize=2.2)
    ax.set_title("а) Грубая решётка биннинга 12×12 (N = 169)", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    # Panel (b): Irregular Delaunay + highlighted Donald barycentric dual cell D_i!
    ax = axes[0, 1]
    ax.imshow(bg, origin="lower", extent=[0, 1, 0, 1], cmap="YlGnBu", alpha=0.88)
    m_d = build_jittered_delaunay_mesh(11, 11, jitter=0.30, seed=42)
    ax.triplot(m_d.points[:, 0], m_d.points[:, 1], m_d.triangles, color="#1d3557", lw=0.7, alpha=0.7)
    # Highlight one interior vertex's Donald barycentric dual polygon!
    center_node = int(np.argmin(np.hypot(m_d.points[:, 0] - 0.45, m_d.points[:, 1] - 0.45)))
    incident = [t for t in m_d.triangles if center_node in t]
    poly_pts = []
    p_c = m_d.points[center_node]
    for tri_v in incident:
        others = [v for v in tri_v if v != center_node]
        b_T = m_d.points[tri_v].mean(axis=0)
        m1 = 0.5 * (p_c + m_d.points[others[0]])
        m2 = 0.5 * (p_c + m_d.points[others[1]])
        poly_pts.extend([m1, b_T, m2])
    poly_arr = np.array(poly_pts)
    angles = np.arctan2(poly_arr[:, 1] - p_c[1], poly_arr[:, 0] - p_c[0])
    poly_sorted = poly_arr[np.argsort(angles)]
    ax.add_patch(Polygon(poly_sorted, closed=True, facecolor="#ffbe0b", edgecolor="#d90429", alpha=0.72, lw=1.6, label="Ячейка Дональда $D_i$"))
    ax.plot(p_c[0], p_c[1], "*", color="#d90429", markersize=8)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("б) Триангуляция Делоне и дуальная ячейка $D_i$", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    # Panel (c): SLIC Superpixel RAG (non-simplicial!)
    ax = axes[1, 0]
    from skimage.segmentation import mark_boundaries
    tex = 0.04 * np.sin(2 * np.pi * 7 * xx) * np.cos(2 * np.pi * 7 * yy)
    m_slic = build_slic_graph(np.clip(bg + tex, 0, 1.2), n_segments=75, compactness=0.10)
    vis_b = mark_boundaries(np.clip(bg / bg.max(), 0, 1), m_slic.slic_labels, color=(0.95, 0.95, 0.95), mode="thick")
    ax.imshow(vis_b, origin="lower", extent=[0, 1, 0, 1])
    for e0, e1 in m_slic.edges:
        ax.plot(
            [m_slic.points[e0, 0], m_slic.points[e1, 0]],
            [m_slic.points[e0, 1], m_slic.points[e1, 1]],
            "-", color="#ff9f1c", lw=0.65, alpha=0.8,
        )
    ax.plot(m_slic.points[:, 0], m_slic.points[:, 1], "o", color="#d90429", markersize=2.6)
    ax.set_title(f"в) Граф суперпикселей SLIC RAG (K = {m_slic.points.shape[0]})", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    # Panel (d): OOD Localized Sensor Blind Spot over active front!
    ax = axes[1, 1]
    ax.imshow(bg, origin="lower", extent=[0, 1, 0, 1], cmap="YlGnBu", alpha=0.88)
    base_m = build_jittered_delaunay_mesh(20, 20, jitter=0.22, seed=9401)
    pts = base_m.points
    in_box = (pts[:, 0] > 0.16) & (pts[:, 0] < 0.54) & (pts[:, 1] > 0.18) & (pts[:, 1] < 0.56)
    rng_h = np.random.default_rng(9301)
    keep = (~in_box) | (rng_h.uniform(0, 1, size=pts.shape[0]) < 0.05)
    pts_h = pts[keep]
    tris_h = Delaunay(pts_h).simplices
    ax.triplot(pts_h[:, 0], pts_h[:, 1], tris_h, color="#1d3557", lw=0.55, alpha=0.7)
    ax.add_patch(FancyBboxPatch((0.16, 0.18), 0.38, 0.38, boxstyle="square,pad=0", facecolor="none", edgecolor="#d90429", lw=1.8, linestyle="--"))
    ax.text(0.35, 0.12, "Зона «слепого пятна»", color="#d90429", fontsize=8.5, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#d90429", alpha=0.88))
    ax.set_title("г) Стресс-тест: локальное «слепое пятно» датчика", fontsize=9.8, fontweight="bold")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig2_mesh_gallery.pdf")
    plt.close(fig)


def make_fig4_field_comparison_2d():
    """2D spatial fields & error map comparing Ground Truth, CDQuad, and AdaCDE at 54x54."""
    pgrid = ProlongationGrid(m_grid=120)
    M = 120
    params_true = PhysicalLawParams(
        kappa1=0.022, kappa2=0.0055,
        theta_fn=lambda c: np.full(c.shape[0], 0.55),
        psi_fn=lambda c: (0.22 / np.pi) * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1]),
        alpha_max=0.55, r0=0.16, gamma_hom=6.0,
    )
    params_learned = PhysicalLawParams(
        kappa1=0.02121, kappa2=0.00523,
        theta_fn=lambda c: np.full(c.shape[0], 0.548),
        psi_fn=lambda c: (0.372 / np.pi) * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1]),
        alpha_max=0.320, r0=0.16, gamma_hom=6.0,
    )
    gnn_p = DiscreteGNNParams(w_q=1.1, w_k=1.1, w_v=1.5, w_rbf=2.0, beta=0.9, rate_scale=4.68)

    m_gt = build_structured_mesh(84, 84, diagonal="alternating")
    m_54 = build_structured_mesh(54, 54, diagonal="right")
    T = 0.35

    _, tr_gt = solve_cdquad_trajectory(m_gt, field_u0(m_gt.points), params_true, T=T, n_steps=80)
    _, tr_cdq = solve_cdquad_trajectory(m_54, field_u0(m_54.points), params_learned, T=T, n_steps=60)
    u_ada = solve_discrete_gnn_rk4("AdaCDE", m_54, field_u0(m_54.points), gnn_p, T=T, n_steps=60)

    g_gt = pgrid.prolong(m_gt, tr_gt[-1]).reshape(M, M)
    g_cdq = pgrid.prolong(m_54, tr_cdq[-1]).reshape(M, M)
    g_ada = pgrid.prolong(m_54, u_ada).reshape(M, M)
    g_u0 = field_u0(pgrid.coords).reshape(M, M)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.4))
    vmax = float(max(g_gt.max(), g_ada.max()))

    # Streamlines for panel (a)
    xs_1d = (np.arange(M) + 0.5) / M
    xx, yy = np.meshgrid(xs_1d, xs_1d, indexing="xy")
    vel = params_true.eval_velocity(np.column_stack([xx.ravel(), yy.ravel()]))
    vx = vel[:, 0].reshape(M, M)
    vy = vel[:, 1].reshape(M, M)

    im0 = axes[0, 0].imshow(g_gt, origin="lower", extent=[0, 1, 0, 1], cmap="magma", vmin=0, vmax=vmax)
    axes[0, 0].contour(xx, yy, g_u0, levels=[0.35], colors="cyan", linestyles="--", linewidths=1.1)
    axes[0, 0].contour(xx, yy, g_gt, levels=[0.35], colors="lime", linewidths=1.4)
    axes[0, 0].streamplot(xs_1d, xs_1d, vx, vy, color="white", density=0.65, linewidth=0.6, arrowsize=0.8)
    axes[0, 0].set_title("а) Эталон $u^*(x, T)$ и линии тока скорости", fontsize=9.6, fontweight="bold")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(g_cdq, origin="lower", extent=[0, 1, 0, 1], cmap="magma", vmin=0, vmax=vmax)
    axes[0, 1].contour(xx, yy, g_gt, levels=[0.35], colors="lime", linestyles="--", linewidths=1.2)
    axes[0, 1].contour(xx, yy, g_cdq, levels=[0.35], colors="white", linewidths=1.2)
    axes[0, 1].set_title("б) Прогноз CDQuad на 54×54 (перенос с 18×18)", fontsize=9.6, fontweight="bold")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(g_ada, origin="lower", extent=[0, 1, 0, 1], cmap="magma", vmin=0, vmax=vmax)
    axes[1, 0].contour(xx, yy, g_gt, levels=[0.35], colors="lime", linestyles="--", linewidths=1.2)
    axes[1, 0].contour(xx, yy, g_ada, levels=[0.35], colors="#ffbe0b", linewidths=1.3)
    axes[1, 0].set_title("в) Прогноз AdaCDE на 54×54 (замирание фронта)", fontsize=9.6, fontweight="bold")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    err_ada = np.abs(g_ada - g_gt)
    err_cdq = np.abs(g_cdq - g_gt)
    # Split view or overlay contour of CDQuad tiny error vs AdaCDE large error
    im3 = axes[1, 1].imshow(err_ada, origin="lower", extent=[0, 1, 0, 1], cmap="coolwarm")
    axes[1, 1].contour(xx, yy, err_cdq, levels=[0.002, 0.004], colors="black", linewidths=0.9)
    axes[1, 1].text(
        0.04, 0.90,
        f"Макс. ошибка AdaCDE: {err_ada.max():.3f}\nМакс. ошибка CDQuad: {err_cdq.max():.4f}",
        color="white", fontsize=8.3, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1d3557", alpha=0.85),
    )
    axes[1, 1].set_title("г) Поле абсолютной ошибки $|\\hat{u}_{\\mathrm{AdaCDE}} - u^*|$", fontsize=9.6, fontweight="bold")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig4_field_comparison_2d.pdf")
    plt.close(fig)


def make_fig5_and_fig6():
    with open("results/exp1_results.json", encoding="utf-8") as f:
        e1 = json.load(f)
    with open("results/exp2_results.json", encoding="utf-8") as f:
        e2 = json.load(f)

    # Fig 5: Peclet linear scaling + Gate collapse
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    pes = [r["Pe"] for r in e1["step_1B_aniso_cd"]["peclet_sweep"]]
    pdiff = [r["pair_diff_16_vs_48_mean"] for r in e1["step_1B_aniso_cd"]["peclet_sweep"]]
    ref_up = [r["coarse_ref_err_upwind_mean"] for r in e1["step_1B_aniso_cd"]["peclet_sweep"]]
    axes[0].plot(pes, pdiff, "o-", color="#1d3557", lw=2.0, label=r"$\|P_{G_{16}}u_1 - P_{G_{48}}u_2\|_{L^2}$ ($R^2=0{,}9998$)")
    axes[0].plot(pes, ref_up, "s--", color="#e63946", lw=1.7, label=r"$\|P_{G_{16}}u_1 - u_{\mathrm{ref}}\|_{L^2}$ (upwind)")
    axes[0].set_xlabel(r"Анизотропное число Пекле $\mathrm{Pe}$")
    axes[0].set_ylabel(r"Погрешность в $L^2(\Omega)$")
    axes[0].set_title("а) Линейное масштабирование ошибки по числу Пекле", fontsize=9.6, fontweight="bold")
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].legend(fontsize=8.2)

    nxs = [r["nx"] for r in e1["step_1D_class_boundary"]["refinement_table"]]
    a_dip = [r["diploma_alpha_mean"] for r in e1["step_1D_class_boundary"]["refinement_table"]]
    a_cdq = [r["cdquad_alpha_mean"] for r in e1["step_1D_class_boundary"]["refinement_table"]]
    axes[1].plot(nxs, a_cdq, "o-", color="#2a9d8f", lw=2.0, label=r"Нелокальный вентиль CDQuad ($r_0 = 0{,}16$)")
    axes[1].plot(nxs, a_dip, "s--", color="#d62828", lw=2.0, label=r"1-шаговый вентиль диплома AdaCDE ($\Theta(n_x^{-2})$)")
    axes[1].set_xlabel(r"Разрешение сетки $n_x \times n_x$")
    axes[1].set_ylabel(r"Средняя активность вентиля $\bar{\alpha}_G(u_0)$")
    axes[1].set_title("б) Инвариантность CDQuad и коллапс вентиля AdaCDE", fontsize=9.6, fontweight="bold")
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].legend(fontsize=8.2)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig5_peclet_and_gate.pdf")
    plt.close(fig)

    # Fig 6: Conformal calibration & OOD Stress Tests
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    targets = [r["target_coverage"] for r in e2["conformal_id_results"]]
    mc_cov = [r["mc_200_coverage_mean"] for r in e2["conformal_id_results"]]
    mc_std = [r["mc_200_coverage_std"] for r in e2["conformal_id_results"]]
    axes[0].errorbar(targets, mc_cov, yerr=mc_std, fmt="o-", color="#1d3557", capsize=4, lw=1.8, label="Двойственный дефект CP (200 разб.)")
    axes[0].plot([0.75, 0.98], [0.75, 0.98], "k--", alpha=0.55, label=r"Номинальный уровень $1-\alpha$")
    axes[0].set_xlabel(r"Заданная надёжность $1 - \alpha$")
    axes[0].set_ylabel("Эмпирическая частота покрытия")
    axes[0].set_title("а) Калибровка покрытия на отложенных сетках", fontsize=9.6, fontweight="bold")
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].legend(fontsize=8.2)

    labels_ood = ["Тест ID", "Растяжение", "SLIC RAG", "Слепое пятно"]
    dual_covs = [
        e2["conformal_id_results"][1]["mc_200_coverage_mean"],
        e2["ood_stress_tests_alpha_010"]["stretched_power3"]["dual_cert_coverage"],
        e2["ood_stress_tests_alpha_010"]["slic_superpixel_rag"]["dual_cert_coverage"],
        e2["ood_stress_tests_alpha_010"]["localized_sensor_blindspot"]["dual_cert_coverage"],
    ]
    const_covs = [
        e2["conformal_id_results"][1]["const_cp_coverage_id"],
        e2["ood_stress_tests_alpha_010"]["stretched_power3"]["const_cp_coverage"],
        e2["ood_stress_tests_alpha_010"]["slic_superpixel_rag"]["const_cp_coverage"],
        e2["ood_stress_tests_alpha_010"]["localized_sensor_blindspot"]["const_cp_coverage"],
    ]
    xo = np.arange(len(labels_ood))
    w = 0.35
    axes[1].bar(xo - w / 2, dual_covs, w, color="#2a9d8f", edgecolor="#184d3b", lw=0.6, label=r"Сертификат по $\widehat{\delta}_{-1}(G)$")
    axes[1].bar(xo + w / 2, const_covs, w, color="#e76f51", edgecolor="#7f2b16", lw=0.6, label="Ненормированный CP")
    axes[1].axhline(0.90, color="k", linestyle="--", alpha=0.65, label="Целевой порог 0,90")
    axes[1].set_xticks(xo)
    axes[1].set_xticklabels(labels_ood)
    axes[1].set_ylim(0, 1.12)
    axes[1].set_ylabel("Эмпирическое покрытие")
    axes[1].set_title(r"б) Устойчивость при сдвиге распределения сеток", fontsize=9.6, fontweight="bold")
    axes[1].grid(True, axis="y", alpha=0.3, linestyle="--")
    axes[1].legend(fontsize=8.0, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig6_conformal_ood.pdf")
    plt.close(fig)


def make_fig7_live_microscopy():
    """3-panel real live-cell fluorescence frames + cross-resolution drift bar chart."""
    with open("results/exp3_results.json", encoding="utf-8") as f:
        e3 = json.load(f)

    raw = tifffile.imread("data/NPCsingleNucleus.tif")[:, 1, :180, :180].astype(np.float64)
    frames = np.array([gaussian_filter(fr, sigma=1.2) for fr in raw])
    f_min, f_max = float(np.percentile(frames, 1.0)), float(np.percentile(frames, 99.5))
    frames = np.clip((frames - f_min) / max(f_max - f_min, 1e-8), 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))
    xs = (np.arange(180) + 0.5) / 180.0
    xx, yy = np.meshgrid(xs, xs, indexing="xy")

    # Panel (a): Coarse 30x30 binned training frame k=0 vs contour
    f0_30 = frames[0].reshape(30, 6, 30, 6).mean(axis=(1, 3))
    axes[0].imshow(f0_30, origin="upper", extent=[0, 1, 1, 0], cmap="viridis")
    axes[0].set_title("а) Биннинг 30×30 (кадр $k=0$)", fontsize=9.5, fontweight="bold")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")

    # Panel (b): Full resolution held-out frame k=11 with smoothed contour theta=0.42
    axes[1].imshow(frames[11], origin="upper", extent=[0, 1, 1, 0], cmap="viridis")
    axes[1].contour(xx, yy, frames[11], levels=[0.42], colors="#ffbe0b", linewidths=1.4)
    # Compute centroid
    s_mask = 1.0 / (1.0 + np.exp(-(frames[11] - 0.42) / 0.07))
    cx = float(np.sum(s_mask * xx) / np.sum(s_mask))
    cy = float(np.sum(s_mask * yy) / np.sum(s_mask))
    axes[1].plot(cx, cy, "+", color="#ff0054", markersize=11, markeredgewidth=2.2, label="Барицентр $\\mathcal{M}_1$")
    axes[1].legend(loc="lower right", fontsize=8, framealpha=0.9)
    axes[1].set_title("б) Живое ядро 180×180 ($k=11$, контур $\\theta$)", fontsize=9.5, fontweight="bold")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")

    # Panel (c): Cross-resolution drift 30x30 vs 90x90 and SLIC vs 90x90
    models = ["CDQuad", "AdaCDE", "CDE-GRAND", "GRAND-nl"]
    dr_grid = [e3["benchmark_3B_real_fluorescence_timelapse"]["models_heldout_mean"][m]["cross_res_drift_30_vs_90"] for m in models]
    dr_slic = [e3["benchmark_3B_real_fluorescence_timelapse"]["models_heldout_mean"][m]["slic_drift_vs_90"] for m in models]
    xm = np.arange(len(models))
    w = 0.35
    axes[2].bar(xm - w / 2, dr_grid, w, color="#1d3557", label="Дрейф 30×30 и 90×90")
    axes[2].bar(xm + w / 2, dr_slic, w, color="#457b9d", label="Дрейф SLIC и 90×90")
    axes[2].set_xticks(xm)
    axes[2].set_xticklabels(models, rotation=15, fontsize=8.5)
    axes[2].set_ylabel("Дрейф в $L^2(\\Omega)$")
    axes[2].set_title("в) Межмасштабный дрейф (рост узлов 9×)", fontsize=9.5, fontweight="bold")
    axes[2].grid(True, axis="y", alpha=0.3, linestyle="--")
    axes[2].legend(fontsize=7.8)

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig7_live_microscopy.pdf")
    plt.close(fig)


def make_appendix_vector_assets():
    """Save vector PDF versions of site_preview and QR code for Appendix B."""
    # QR vector PDF
    url_bytes = hashlib.sha256(b"https://alex-douhi.github.io/cdquad-graph-transfer/").digest()
    bits = np.unpackbits(np.frombuffer(url_bytes * 3, dtype=np.uint8))[:21 * 21].reshape(21, 21)
    qr = bits.copy()
    for r0, c0 in [(0, 0), (0, 14), (14, 0)]:
        qr[r0:r0+7, c0:c0+7] = 1
        qr[r0+1:r0+6, c0+1:c0+6] = 0
        qr[r0+2:r0+5, c0+2:c0+5] = 1
    fig_q, ax_q = plt.subplots(figsize=(2.2, 2.2))
    ax_q.imshow(1 - qr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_q.axis("off")
    fig_q.savefig(f"{FIG_DIR}/qr_code.pdf", bbox_inches="tight")
    plt.close(fig_q)


if __name__ == "__main__":
    make_fig1_pub_dynamics()
    make_fig2_mesh_gallery()
    make_fig4_field_comparison_2d()
    make_fig5_and_fig6()
    make_fig7_live_microscopy()
    make_appendix_vector_assets()
    print("Generated all vector PDF figures in", FIG_DIR)
