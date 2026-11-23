"""Generate summary figures in results/figures/ from results/exp{1,2,3}_results.json."""
from __future__ import annotations

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    os.makedirs("results/figures", exist_ok=True)
    with open("results/exp1_results.json", encoding="utf-8") as f:
        e1 = json.load(f)
    with open("results/exp2_results.json", encoding="utf-8") as f:
        e2 = json.load(f)
    with open("results/exp3_results.json", encoding="utf-8") as f:
        e3 = json.load(f)

    # Figure 1: Convergence rates & Peclet linear scaling
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    hs_h = [r["h_max_mean"] for r in e1["step_1A_heat"]["jittered_delaunay"]]
    er_h = [r["l2_err_mean"] for r in e1["step_1A_heat"]["jittered_delaunay"]]
    hs_cd = [r["h_max_mean"] for r in e1["step_1B_aniso_cd"]["convergence_pe30"]]
    er_cd = [r["l2_err_mean"] for r in e1["step_1B_aniso_cd"]["convergence_pe30"]]
    hs_ad = [r["h_max_mean"] for r in e1["step_1C_adaptive_cdquad"]["convergence"]]
    er_ad = [r["l2_err_mean"] for r in e1["step_1C_adaptive_cdquad"]["convergence"]]
    ax.loglog(hs_h, er_h, "o-", label=f"Heat (slope={e1['step_1A_heat']['slope_jittered']:.2f})")
    ax.loglog(hs_cd, er_cd, "s-", label=f"Aniso CD Pe=30 (slope={e1['step_1B_aniso_cd']['slope_pe30']:.2f})")
    ax.loglog(hs_ad, er_ad, "^-", label=f"Adaptive CDQuad (slope={e1['step_1C_adaptive_cdquad']['slope_adaptive']:.2f})")
    ax.set_xlabel(r"Max mesh diameter $h_G$")
    ax.set_ylabel(r"$L^2(\Omega)$ trajectory error")
    ax.set_title("Mesh Convergence on Jittered Delaunay Graphs")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.5)

    ax2 = axes[1]
    pes = [r["Pe"] for r in e1["step_1B_aniso_cd"]["peclet_sweep"]]
    pdiff = [r["pair_diff_16_vs_48_mean"] for r in e1["step_1B_aniso_cd"]["peclet_sweep"]]
    ref_up = [r["coarse_ref_err_upwind_mean"] for r in e1["step_1B_aniso_cd"]["peclet_sweep"]]
    ax2.plot(pes, pdiff, "o-", label=r"$\|P_{G_{16}}u_1 - P_{G_{48}}u_2\|_{L^2}$")
    ax2.plot(pes, ref_up, "s--", label=r"$\|P_{G_{16}}u_1 - u_{\mathrm{ref}}\|_{L^2}$")
    ax2.set_xlabel(r"Anisotropic Péclet number $\mathrm{Pe}$")
    ax2.set_ylabel(r"$L^2(\Omega)$ discrepancy at $T=0.40$")
    ax2.set_title("Linear Péclet Scaling of Transfer Discrepancy")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig("results/figures/fig1_convergence_and_peclet.pdf")
    fig.savefig("results/figures/fig1_convergence_and_peclet.png", dpi=180)
    plt.close(fig)

    # Figure 2: Class boundary (homogeneity gate collapse) & long-time decay
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    nxs = [r["nx"] for r in e1["step_1D_class_boundary"]["refinement_table"]]
    a_dip = [r["diploma_alpha_mean"] for r in e1["step_1D_class_boundary"]["refinement_table"]]
    a_cdq = [r["cdquad_alpha_mean"] for r in e1["step_1D_class_boundary"]["refinement_table"]]
    ax.plot(nxs, a_cdq, "o-", label=r"CDQuad continuum gate $\bar{\alpha}_G(u_0)$")
    ax.plot(nxs, a_dip, "s--", label=r"Diploma AdaCDE 1-hop gate $\bar{\alpha}_G(u_0)$")
    ax.set_xlabel(r"Grid resolution $n_x$")
    ax.set_ylabel(r"Domain-integrated gate $\bar{\alpha}_G(u_0)$")
    ax.set_title("1-Hop Gate Collapse in Raw AdaCDE vs CDQuad")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)

    ax2 = axes[1]
    ts = [r["t"] for r in e1["step_1B_aniso_cd"]["long_time_curve"]]
    errs_lt = [r["l2_transfer_diff"] for r in e1["step_1B_aniso_cd"]["long_time_curve"]]
    ax2.plot(ts, errs_lt, "o-", color="tab:green", label=r"$\|P_{G_{18}}u_1(t) - P_{G_{54}}u_2(t)\|_{L^2}$")
    ax2.set_xlabel(r"Time horizon $t$")
    ax2.set_ylabel(r"$L^2(\Omega)$ cross-mesh error")
    ax2.set_title(r"Long-Time Dissipation ($\mathrm{Pe}=30$, No $\exp(\mathrm{Pe}\,t)$ Growth)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig("results/figures/fig2_class_boundary_and_longtime.pdf")
    fig.savefig("results/figures/fig2_class_boundary_and_longtime.png", dpi=180)
    plt.close(fig)

    # Figure 3: Conformal coverage & OOD stress tests
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    targets = [r["target_coverage"] for r in e2["conformal_id_results"]]
    mc_cov = [r["mc_200_coverage_mean"] for r in e2["conformal_id_results"]]
    mc_std = [r["mc_200_coverage_std"] for r in e2["conformal_id_results"]]
    ax.errorbar(targets, mc_cov, yerr=mc_std, fmt="o-", capsize=4, label="Dual-defect CP (200 splits)")
    ax.plot([0.75, 0.98], [0.75, 0.98], "k--", alpha=0.5, label=r"Ideal $1-\alpha$")
    ax.set_xlabel(r"Nominal coverage $1 - \alpha$")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("In-Distribution Marginal Coverage")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)

    ax2 = axes[1]
    labels_ood = ["ID Test", "OOD Stretch", "OOD SLIC", "OOD BlindSpot"]
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
    x = np.arange(len(labels_ood))
    w = 0.35
    ax2.bar(x - w / 2, dual_covs, w, label=r"Dual-defect CP ($\widehat{\delta}_{-1}$)")
    ax2.bar(x + w / 2, const_covs, w, label="Unnormalized Constant CP")
    ax2.axhline(0.90, color="k", linestyle="--", alpha=0.6, label="Target 0.90")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_ood, rotation=12)
    ax2.set_ylabel("Empirical coverage")
    ax2.set_title(r"Coverage under Mesh Distribution Shifts ($\alpha=0.10$)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/fig3_conformal_and_ood.pdf")
    fig.savefig("results/figures/fig3_conformal_and_ood.png", dpi=180)
    plt.close(fig)

    # Figure 4: Applied Cross-Resolution Morphometry (Synthetic & Real Microscopy)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ax = axes[0]
    regimes = ["coarse_18x18", "fine_54x54", "delaunay_N1225", "slic_K300"]
    nl = chr(10)
    reg_labels = [f"18x18{nl}(train)", f"54x54{nl}(9x zero-shot)", f"Delaunay{nl}(N=1225)", f"SLIC RAG{nl}(K=300)"]
    models = ["CDQuad", "AdaCDE", "CDE-GRAND", "GRAND-nl"]
    x = np.arange(len(regimes))
    bw = 0.19
    for idx_m, m in enumerate(models):
        vals = [e3["benchmark_3A_synthetic_microscopy"]["regimes"][rg]["models"][m]["l2"]["mean"] for rg in regimes]
        ax.bar(x + (idx_m - 1.5) * bw, vals, bw, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(reg_labels)
    ax.set_ylabel(r"$L^2(\Omega)$ prediction error")
    ax.set_title("Synthetic Benchmark: Coarse-Trained Zero-Shot Transfer")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax2 = axes[1]
    drifts_30_90 = [e3["benchmark_3B_real_fluorescence_timelapse"]["models_heldout_mean"][m]["cross_res_drift_30_vs_90"] for m in models]
    drifts_slic = [e3["benchmark_3B_real_fluorescence_timelapse"]["models_heldout_mean"][m]["slic_drift_vs_90"] for m in models]
    x2 = np.arange(len(models))
    ax2.bar(x2 - 0.18, drifts_30_90, 0.35, label=r"$\|P_{30}u_{30} - P_{90}u_{90}\|_{L^2}$")
    ax2.bar(x2 + 0.18, drifts_slic, 0.35, label=r"$\|P_{\mathrm{SLIC}}u_{\mathrm{SLIC}} - P_{90}u_{90}\|_{L^2}$")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(models)
    ax2.set_ylabel(r"Cross-resolution drift in $L^2(\Omega)$")
    ax2.set_title("Real Microscopy (NPCsingleNucleus): Resolution Stability")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/fig4_applied_morphometry.pdf")
    fig.savefig("results/figures/fig4_applied_morphometry.png", dpi=180)
    plt.close(fig)
    print("Generated 4 PDF/PNG figures in results/figures/")


if __name__ == "__main__":
    main()
