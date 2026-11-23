"""
Mesh generation, Donald barycentric dual volumes, P1 prolongation/restriction,
and SLIC superpixel Region Adjacency Graphs on Omega = [0, 1]^2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.spatial import Delaunay
from scipy import sparse


@dataclass
class MeshData:
    name: str
    points: np.ndarray          # (N, 2) coordinates in [0, 1]^2
    triangles: Optional[np.ndarray]  # (Nt, 3) int indices, or None for SLIC RAG
    mass: np.ndarray            # (N,) lumped cell/dual areas, sum(mass) = 1.0
    edges: np.ndarray           # (Ne, 2) undirected edges (i < j)
    h_max: float                # max element/cell diameter
    h_mean: float               # mean element/cell diameter
    shape_reg: float            # max(h_T / r_in_T) for triangulations, else nan
    slic_labels: Optional[np.ndarray] = None  # (H, W) int labels if superpixel RAG
    edge_lengths: Optional[np.ndarray] = None # (Ne,) interface lengths |Gamma_ij|
    edge_flux_normals: Optional[np.ndarray] = None  # (Ne, 2) unit normal i -> j
    edge_midpoints: Optional[np.ndarray] = None     # (Ne, 2) interface midpoints


def _compute_triangulation_metrics(points: np.ndarray, triangles: np.ndarray):
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]

    # Ensure counter-clockwise orientation
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    flip = cross < 0
    if np.any(flip):
        triangles = triangles.copy()
        triangles[flip, 1], triangles[flip, 2] = triangles[flip, 2].copy(), triangles[flip, 1].copy()
        cross = -cross * flip + cross * (~flip)

    areas = 0.5 * np.abs(cross)
    l01 = np.linalg.norm(p1 - p0, axis=1)
    l12 = np.linalg.norm(p2 - p1, axis=1)
    l20 = np.linalg.norm(p0 - p2, axis=1)
    h_T = np.maximum(np.maximum(l01, l12), l20)
    perim = l01 + l12 + l20
    r_in = 2.0 * areas / np.maximum(perim, 1e-15)

    n_nodes = points.shape[0]
    mass = np.zeros(n_nodes, dtype=np.float64)
    for a in range(3):
        np.add.at(mass, triangles[:, a], areas / 3.0)

    # Extract unique undirected edges (i < j)
    raw_edges = np.vstack([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]],
    ])
    raw_edges = np.sort(raw_edges, axis=1)
    edges = np.unique(raw_edges, axis=0)

    return triangles, mass, edges, float(np.max(h_T)), float(np.mean(h_T)), float(np.max(h_T / np.maximum(r_in, 1e-15)))


def build_structured_mesh(nx: int, ny: Optional[int] = None, diagonal: str = "right") -> MeshData:
    """Uniform structured triangulation on [0, 1]^2 with (nx+1)*(ny+1) nodes."""
    if ny is None:
        ny = nx
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel()])

    def idx(i: int, j: int) -> int:
        return i * (ny + 1) + j

    tris = []
    for i in range(nx):
        for j in range(ny):
            v00 = idx(i, j)
            v10 = idx(i + 1, j)
            v01 = idx(i, j + 1)
            v11 = idx(i + 1, j + 1)
            use_right = (diagonal == "right") or (diagonal == "alternating" and (i + j) % 2 == 0)
            if use_right:
                tris.append([v00, v10, v11])
                tris.append([v00, v11, v01])
            else:
                tris.append([v00, v10, v01])
                tris.append([v10, v11, v01])
    triangles = np.asarray(tris, dtype=np.int64)
    triangles, mass, edges, h_max, h_mean, shape_reg = _compute_triangulation_metrics(points, triangles)
    return MeshData(
        name=f"struct_{nx}x{ny}_{diagonal}",
        points=points,
        triangles=triangles,
        mass=mass,
        edges=edges,
        h_max=h_max,
        h_mean=h_mean,
        shape_reg=shape_reg,
    )


def build_jittered_delaunay_mesh(nx: int, ny: Optional[int] = None, jitter: float = 0.28, seed: int = 0) -> MeshData:
    """Jittered interior + boundary-conforming Delaunay mesh on [0, 1]^2."""
    if ny is None:
        ny = nx
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    hx = 1.0 / nx
    hy = 1.0 / ny

    dx = rng.uniform(-jitter * hx, jitter * hx, size=xx.shape)
    dy = rng.uniform(-jitter * hy, jitter * hy, size=yy.shape)
    # Keep boundary nodes on boundary; fix 4 corners
    dx[0, :] = 0.0
    dx[-1, :] = 0.0
    dy[:, 0] = 0.0
    dy[:, -1] = 0.0
    dx[:, 0] *= 0.5
    dx[:, -1] *= 0.5
    dy[0, :] *= 0.5
    dy[-1, :] *= 0.5

    pts = np.column_stack([(xx + dx).ravel(), (yy + dy).ravel()])
    pts = np.clip(pts, 0.0, 1.0)
    tri = Delaunay(pts)
    triangles = tri.simplices.copy()
    # Filter degenerate near-collinear boundary slivers if any
    p0, p1, p2 = pts[triangles[:, 0]], pts[triangles[:, 1]], pts[triangles[:, 2]]
    areas = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
    triangles = triangles[areas > 1e-13]
    triangles, mass, edges, h_max, h_mean, shape_reg = _compute_triangulation_metrics(pts, triangles)
    return MeshData(
        name=f"jitter_{nx}x{ny}_s{seed}",
        points=pts,
        triangles=triangles,
        mass=mass,
        edges=edges,
        h_max=h_max,
        h_mean=h_mean,
        shape_reg=shape_reg,
    )


def build_random_delaunay_mesh(n_approx: int, seed: int = 0, lloyd_iters: int = 2) -> MeshData:
    """Random point cloud + boundary nodes + Lloyd centroidal smoothing + Delaunay."""
    rng = np.random.default_rng(seed)
    n_side = max(4, int(round(np.sqrt(n_approx))))
    t_line = np.linspace(0.0, 1.0, n_side + 1)
    b0 = np.column_stack([t_line, np.zeros_like(t_line)])
    b1 = np.column_stack([t_line, np.ones_like(t_line)])
    b2 = np.column_stack([np.zeros_like(t_line[1:-1]), t_line[1:-1]])
    b3 = np.column_stack([np.ones_like(t_line[1:-1]), t_line[1:-1]])
    boundary = np.vstack([b0, b1, b2, b3])

    n_int = max(4, n_approx - boundary.shape[0])
    margin = 0.5 / n_side
    interior = rng.uniform(margin, 1.0 - margin, size=(n_int, 2))
    pts = np.vstack([boundary, interior])
    n_bnd = boundary.shape[0]

    for _ in range(lloyd_iters):
        tri = Delaunay(pts)
        simplices = tri.simplices
        p0, p1, p2 = pts[simplices[:, 0]], pts[simplices[:, 1]], pts[simplices[:, 2]]
        areas = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
        valid = areas > 1e-13
        simplices = simplices[valid]
        areas = areas[valid]
        bary = (pts[simplices[:, 0]] + pts[simplices[:, 1]] + pts[simplices[:, 2]]) / 3.0
        w_sum = np.zeros(pts.shape[0])
        c_sum = np.zeros_like(pts)
        for a in range(3):
            np.add.at(w_sum, simplices[:, a], areas)
            np.add.at(c_sum, simplices[:, a], areas[:, None] * bary)
        mask = np.arange(pts.shape[0]) >= n_bnd
        mask &= w_sum > 1e-14
        pts[mask] = np.clip(c_sum[mask] / w_sum[mask, None], margin * 0.5, 1.0 - margin * 0.5)

    tri = Delaunay(pts)
    triangles = tri.simplices
    p0, p1, p2 = pts[triangles[:, 0]], pts[triangles[:, 1]], pts[triangles[:, 2]]
    areas = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0]))
    triangles = triangles[areas > 1e-13]
    triangles, mass, edges, h_max, h_mean, shape_reg = _compute_triangulation_metrics(pts, triangles)
    return MeshData(
        name=f"rand_N{pts.shape[0]}_s{seed}",
        points=pts,
        triangles=triangles,
        mass=mass,
        edges=edges,
        h_max=h_max,
        h_mean=h_mean,
        shape_reg=shape_reg,
    )


def build_stretched_ood_mesh(nx: int, ny: int, power_x: float = 2.6, seed: int = 0) -> MeshData:
    """
    Out-of-distribution (OOD) graded/stretched anisotropic mesh:
    coarse & high-aspect-ratio elements in part of [0,1]^2.
    """
    rng = np.random.default_rng(seed)
    sx = np.linspace(0.0, 1.0, nx + 1) ** power_x
    sy = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(sx, sy, indexing="ij")
    if seed >= 0:
        # slight interior perturbation
        dx = np.diff(sx)
        dy = np.diff(sy)
        for i in range(1, nx):
            for j in range(1, ny):
                xx[i, j] += rng.uniform(-0.15, 0.15) * min(dx[i - 1], dx[i])
                yy[i, j] += rng.uniform(-0.15, 0.15) * min(dy[j - 1], dy[j])
    points = np.column_stack([xx.ravel(), yy.ravel()])
    tris = []
    for i in range(nx):
        for j in range(ny):
            v00 = i * (ny + 1) + j
            v10 = (i + 1) * (ny + 1) + j
            v01 = i * (ny + 1) + (j + 1)
            v11 = (i + 1) * (ny + 1) + (j + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])
    triangles = np.asarray(tris, dtype=np.int64)
    triangles, mass, edges, h_max, h_mean, shape_reg = _compute_triangulation_metrics(points, triangles)
    return MeshData(
        name=f"ood_stretch_{nx}x{ny}_p{power_x:.1f}_s{seed}",
        points=points,
        triangles=triangles,
        mass=mass,
        edges=edges,
        h_max=h_max,
        h_mean=h_mean,
        shape_reg=shape_reg,
    )


def build_two_to_one_graded_mesh(n_base: int) -> MeshData:
    """
    Mesh with 2*n_base steps on [0, 0.5] and n_base/2 steps on [0.5, 1.0],
    creating a sharp step-ratio interface at x = 0.5 while h -> 0 as n_base -> inf.
    """
    n_left = 2 * n_base
    n_right = max(2, n_base // 2)
    xs = np.concatenate([
        np.linspace(0.0, 0.5, n_left + 1)[:-1],
        np.linspace(0.5, 1.0, n_right + 1),
    ])
    ys = np.linspace(0.0, 1.0, n_base + 1)
    nx = len(xs) - 1
    ny = len(ys) - 1
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    tris = []
    for i in range(nx):
        for j in range(ny):
            v00 = i * (ny + 1) + j
            v10 = (i + 1) * (ny + 1) + j
            v01 = i * (ny + 1) + (j + 1)
            v11 = (i + 1) * (ny + 1) + (j + 1)
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])
    triangles = np.asarray(tris, dtype=np.int64)
    triangles, mass, edges, h_max, h_mean, shape_reg = _compute_triangulation_metrics(points, triangles)
    return MeshData(
        name=f"graded2to1_{n_base}",
        points=points,
        triangles=triangles,
        mass=mass,
        edges=edges,
        h_max=h_max,
        h_mean=h_mean,
        shape_reg=shape_reg,
    )


def build_slic_graph(image: np.ndarray, n_segments: int, compactness: float = 0.12) -> MeshData:
    """
    Build a Region Adjacency Graph (RAG) from SLIC superpixels on a 2D grayscale image in [0,1].
    Coordinates: x1 = col / W in (0,1), x2 = row / H in (0,1) (cell-centered).
    """
    from skimage.segmentation import slic

    H, W = image.shape
    labels = slic(
        image.astype(np.float64),
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=None,
    )
    # Relabel contiguously 0..K-1
    uniq, inv = np.unique(labels, return_inverse=True)
    labels = inv.reshape(H, W)
    K = len(uniq)

    hx = 1.0 / W
    hy = 1.0 / H
    xs = (np.arange(W) + 0.5) * hx
    ys = (np.arange(H) + 0.5) * hy
    xx, yy = np.meshgrid(xs, ys, indexing="xy")  # shape (H, W): xx is x1, yy is x2

    counts = np.bincount(labels.ravel(), minlength=K).astype(np.float64)
    mass = counts * (hx * hy)
    cx = np.bincount(labels.ravel(), weights=xx.ravel(), minlength=K) / np.maximum(counts, 1.0)
    cy = np.bincount(labels.ravel(), weights=yy.ravel(), minlength=K) / np.maximum(counts, 1.0)
    points = np.column_stack([cx, cy])

    # Compute max diameter proxy per superpixel from bounding box
    min_x = np.full(K, np.inf)
    max_x = np.full(K, -np.inf)
    min_y = np.full(K, np.inf)
    max_y = np.full(K, -np.inf)
    np.minimum.at(min_x, labels.ravel(), xx.ravel() - 0.5 * hx)
    np.maximum.at(max_x, labels.ravel(), xx.ravel() + 0.5 * hx)
    np.minimum.at(min_y, labels.ravel(), yy.ravel() - 0.5 * hy)
    np.maximum.at(max_y, labels.ravel(), yy.ravel() + 0.5 * hy)
    diams = np.hypot(max_x - min_x, max_y - min_y)

    # Pixel-grid shared boundary interfaces (horizontal & vertical pixel faces)
    edge_len_dict = {}
    edge_flux_vec = {}
    edge_mid_sum = {}
    edge_cnt = {}

    # Vertical faces between (r, c) and (r, c+1): length hy, normal (+1, 0) from left to right
    l_left = labels[:, :-1]
    l_right = labels[:, 1:]
    diff_v = l_left != l_right
    rs, cs = np.where(diff_v)
    for r, c in zip(rs, cs):
        a, b = int(l_left[r, c]), int(l_right[r, c])
        sign = 1.0
        if a > b:
            a, b = b, a
            sign = -1.0
        key = (a, b)
        edge_len_dict[key] = edge_len_dict.get(key, 0.0) + hy
        edge_flux_vec[key] = edge_flux_vec.get(key, np.zeros(2)) + np.array([sign * hy, 0.0])
        edge_mid_sum[key] = edge_mid_sum.get(key, np.zeros(2)) + np.array([(c + 1) * hx, (r + 0.5) * hy])
        edge_cnt[key] = edge_cnt.get(key, 0) + 1

    # Horizontal faces between (r, c) and (r+1, c): length hx, normal (0, +1) from row r to r+1
    l_top = labels[:-1, :]
    l_bot = labels[1:, :]
    diff_h = l_top != l_bot
    rs, cs = np.where(diff_h)
    for r, c in zip(rs, cs):
        a, b = int(l_top[r, c]), int(l_bot[r, c])
        sign = 1.0
        if a > b:
            a, b = b, a
            sign = -1.0
        key = (a, b)
        edge_len_dict[key] = edge_len_dict.get(key, 0.0) + hx
        edge_flux_vec[key] = edge_flux_vec.get(key, np.zeros(2)) + np.array([0.0, sign * hx])
        edge_mid_sum[key] = edge_mid_sum.get(key, np.zeros(2)) + np.array([(c + 0.5) * hx, (r + 1) * hy])
        edge_cnt[key] = edge_cnt.get(key, 0) + 1

    keys = sorted(edge_len_dict.keys())
    edges = np.array(keys, dtype=np.int64)
    edge_lengths = np.array([edge_len_dict[k] for k in keys], dtype=np.float64)
    # Net oriented normal vector integrated over shared interface (exact Green's theorem vector!)
    net_vecs = np.array([edge_flux_vec[k] for k in keys], dtype=np.float64)
    mids = np.array([edge_mid_sum[k] / edge_cnt[k] for k in keys], dtype=np.float64)

    return MeshData(
        name=f"slic_K{K}",
        points=points,
        triangles=None,
        mass=mass,
        edges=edges,
        h_max=float(np.max(diams)),
        h_mean=float(np.mean(diams)),
        shape_reg=float("nan"),
        slic_labels=labels,
        edge_lengths=edge_lengths,
        edge_flux_normals=net_vecs,  # note: stores integral_{\Gamma_ij} nu_ij ds!
        edge_midpoints=mids,
    )


class ProlongationGrid:
    """
    Common high-resolution quadrature grid on [0, 1]^2 to compute exact L^2(Omega)
    distances ||P_{G_1} u_1 - P_{G_2} u_2||_{L^2(Omega)} between arbitrary meshes.
    """
    def __init__(self, m_grid: int = 160):
        self.m_grid = m_grid
        h = 1.0 / m_grid
        xs = (np.arange(m_grid) + 0.5) * h
        ys = (np.arange(m_grid) + 0.5) * h
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        self.coords = np.column_stack([xx.ravel(), yy.ravel()])  # (M^2, 2)
        self.weight = h * h
        self._cache: dict[str, sparse.csr_matrix] = {}

    def get_matrix(self, mesh: MeshData) -> sparse.csr_matrix:
        if mesh.name in self._cache:
            return self._cache[mesh.name]

        n_eval = self.coords.shape[0]
        n_nodes = mesh.points.shape[0]

        if mesh.triangles is not None:
            # Build sparse P1 barycentric interpolation matrix (n_eval, n_nodes)
            from matplotlib.tri import Triangulation
            tri_obj = Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.triangles)
            finder = tri_obj.get_trifinder()
            t_idx = finder(self.coords[:, 0], self.coords[:, 1])
            # Fallback to nearest triangle for any boundary roundoff (-1)
            missing = t_idx < 0
            if np.any(missing):
                bary_tris = mesh.points[mesh.triangles].mean(axis=1)
                dists = np.linalg.norm(self.coords[missing, None, :] - bary_tris[None, :, :], axis=2)
                t_idx[missing] = np.argmin(dists, axis=1)

            tris = mesh.triangles[t_idx]
            p0 = mesh.points[tris[:, 0]]
            p1 = mesh.points[tris[:, 1]]
            p2 = mesh.points[tris[:, 2]]
            v0 = p1 - p0
            v1 = p2 - p0
            v2 = self.coords - p0
            denom = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
            l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / denom
            l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / denom
            l0 = 1.0 - l1 - l2

            rows = np.repeat(np.arange(n_eval), 3)
            cols = tris.ravel()
            vals = np.column_stack([l0, l1, l2]).ravel()
            mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n_eval, n_nodes))
        else:
            # SLIC piecewise-constant prolongation on [0, 1]^2
            assert mesh.slic_labels is not None
            H, W = mesh.slic_labels.shape
            cols_px = np.clip((self.coords[:, 0] * W).astype(int), 0, W - 1)
            rows_px = np.clip((self.coords[:, 1] * H).astype(int), 0, H - 1)
            lbl = mesh.slic_labels[rows_px, cols_px]
            mat = sparse.csr_matrix(
                (np.ones(n_eval, dtype=np.float64), (np.arange(n_eval), lbl)),
                shape=(n_eval, n_nodes),
            )

        self._cache[mesh.name] = mat
        return mat

    def prolong(self, mesh: MeshData, u_g: np.ndarray) -> np.ndarray:
        return self.get_matrix(mesh) @ u_g

    def l2_norm(self, f_vals: np.ndarray) -> float:
        return float(np.sqrt(self.weight * np.sum(f_vals ** 2)))

    def l2_diff(self, mesh1: MeshData, u1: np.ndarray, mesh2: MeshData, u2: np.ndarray) -> float:
        return self.l2_norm(self.prolong(mesh1, u1) - self.prolong(mesh2, u2))
