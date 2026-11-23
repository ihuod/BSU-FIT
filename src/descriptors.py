"""
Continuum-regularized morphometric descriptors on L^2(Omega) with explicit,
mesh-independent Lipschitz constants (and non-Lipschitz hard-threshold control).
"""
from __future__ import annotations

import numpy as np
from .meshes import ProlongationGrid, MeshData


def sigmoid_eta(u: np.ndarray, theta: float, eta: float) -> np.ndarray:
    z = np.clip((u - theta) / eta, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def soft_area_grid(pgrid: ProlongationGrid, u_grid: np.ndarray, theta: float = 0.35, eta: float = 0.06) -> float:
    """M_0(u) = int_Omega sigma_eta(u(x)) dx; Lipschitz constant L = 1 / (4 * eta)."""
    return float(pgrid.weight * np.sum(sigmoid_eta(u_grid, theta, eta)))


def hard_area_grid(pgrid: ProlongationGrid, u_grid: np.ndarray, theta: float = 0.35) -> float:
    """Non-Lipschitz indicator area int_Omega 1_{u > theta} dx."""
    return float(pgrid.weight * np.sum(u_grid > theta))


def soft_centroid_grid(
    pgrid: ProlongationGrid,
    u_grid: np.ndarray,
    theta: float = 0.35,
    eta: float = 0.06,
) -> np.ndarray:
    """
    M_1(u) = (int_Omega x sigma_eta(u(x)) dx) / M_0(u) in R^2;
    Sharp L^2-Lipschitz constant on {M_0 >= m0}: sqrt(2) / (4 * eta * m0).
    """
    s = sigmoid_eta(u_grid, theta, eta)
    m0 = float(pgrid.weight * np.sum(s))
    if m0 < 1e-12:
        return np.array([0.5, 0.5], dtype=np.float64)
    num = pgrid.weight * (s @ pgrid.coords)
    return num / m0


def soft_gyration_radius_grid(
    pgrid: ProlongationGrid,
    u_grid: np.ndarray,
    theta: float = 0.35,
    eta: float = 0.06,
) -> float:
    """Second central moment spread R_g^2(u) = int |x - M_1(u)|^2 sigma_eta(u) dx / M_0(u)."""
    s = sigmoid_eta(u_grid, theta, eta)
    m0 = float(pgrid.weight * np.sum(s))
    if m0 < 1e-12:
        return 0.0
    c = (pgrid.weight * (s @ pgrid.coords)) / m0
    r2 = np.sum((pgrid.coords - c[None, :]) ** 2, axis=1)
    return float(pgrid.weight * np.sum(s * r2) / m0)


def lipschitz_const_area(eta: float) -> float:
    return 1.0 / (4.0 * eta)


def lipschitz_const_centroid(eta: float, m0_lower: float) -> float:
    return float(np.sqrt(2.0) / (4.0 * eta * max(m0_lower, 1e-12)))


def compute_all_descriptors(
    pgrid: ProlongationGrid,
    mesh: MeshData,
    u_0: np.ndarray,
    u_T: np.ndarray,
    T: float,
    theta: float = 0.35,
    eta: float = 0.06,
) -> dict[str, float | np.ndarray]:
    u0_grid = pgrid.prolong(mesh, u_0)
    uT_grid = pgrid.prolong(mesh, u_T)
    area_T = soft_area_grid(pgrid, uT_grid, theta=theta, eta=eta)
    hard_T = hard_area_grid(pgrid, uT_grid, theta=theta)
    c0 = soft_centroid_grid(pgrid, u0_grid, theta=theta, eta=eta)
    cT = soft_centroid_grid(pgrid, uT_grid, theta=theta, eta=eta)
    vel = (cT - c0) / max(T, 1e-12)
    speed = float(np.linalg.norm(vel))
    rg2 = soft_gyration_radius_grid(pgrid, uT_grid, theta=theta, eta=eta)
    return {
        "soft_area": area_T,
        "hard_area": hard_T,
        "centroid_0": c0,
        "centroid_T": cT,
        "front_velocity_vec": vel,
        "front_speed": speed,
        "gyration_sq": rg2,
    }
