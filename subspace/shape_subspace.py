"""
Shape Subspace Module for Dent Detection
=========================================
Implementation based on Fukui et al.'s Shape Subspace method.

Reference:
- Fukui, K., et al. "Second-order difference subspace" (2024)
- Fukui, K., & Maki, A. "Difference subspace and its generalization for subspace-based methods"
  IEEE TPAMI, 37(11):2164-2177, 2015.

This module implements:
1. Shape Subspace computation from 3D point clouds
2. Canonical angles between subspaces (Mag(D) on Vt)
3. Magnitude of Difference Subspace
4. Difference Subspace projection (DS method on U)
5. Dent detection using subspace comparison

Three detection methods:
- S[2]: Third singular value (original method)
- Mag(D): Magnitude of difference subspace using canonical angles
- DS: Difference subspace projection magnitude
"""

import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import numpy as np
from scipy.linalg import eig
from scipy.interpolate import griddata
from typing import Tuple, List, Optional


def compute_shape_subspace(points: np.ndarray, center: bool = True, dim: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute shape subspace from 3D point cloud.

    Args:
        points: (N, 3) array of 3D points
        center: Whether to center the points
        dim: Dimension of subspace to return

    Returns:
        U: (N, dim) Left singular vectors (shape subspace basis)
        S: (dim,) Singular values
        Vt: (dim, 3) Right singular vectors (principal directions)
    """
    if center:
        centered_points = points - np.mean(points, axis=0)
    else:
        centered_points = points

    U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)

    return U[:, :dim], S[:dim], Vt[:dim, :]


def compute_canonical_angles(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """
    Compute canonical angles between two subspaces.

    Args:
        U1: (N, d1) Orthonormal basis of first subspace
        U2: (N, d2) Orthonormal basis of second subspace

    Returns:
        canonical_angles: Array of canonical angles in radians
    """
    correlation = U1.T @ U2
    _, sigma, _ = np.linalg.svd(correlation, full_matrices=False)
    sigma = np.clip(sigma, -1.0, 1.0)
    canonical_angles = np.arccos(sigma)
    return canonical_angles


def magnitude_of_difference_subspace(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Compute the magnitude of difference subspace between two subspaces.
    Mag(D(S1, S2)) = Trace(2(I - Σ)) = Σ ||d̄_i||²

    Args:
        U1: (N, d) Orthonormal basis of first subspace
        U2: (N, d) Orthonormal basis of second subspace

    Returns:
        magnitude: Scalar value representing the difference magnitude
    """
    correlation = U1.T @ U2
    _, sigma, _ = np.linalg.svd(correlation, full_matrices=False)
    sigma = np.clip(sigma, 0, 1)
    magnitude = np.sum(2 * (1 - sigma))
    return magnitude


def grassmann_distance(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Compute geodesic distance on Grassmann manifold.
    d(S1, S2) = ||θ||_F = sqrt(sum(θ_i²))
    """
    canonical_angles = compute_canonical_angles(U1, U2)
    return np.sqrt(np.sum(canonical_angles ** 2))


def max_canonical_angle(U1: np.ndarray, U2: np.ndarray) -> float:
    """Compute maximum canonical angle between two subspaces."""
    canonical_angles = compute_canonical_angles(U1, U2)
    return np.max(canonical_angles)


def subspace_similarity(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Compute similarity between two subspaces.
    Similarity = cos(max_canonical_angle)
    """
    max_angle = max_canonical_angle(U1, U2)
    return np.cos(max_angle)


def compute_difference_subspace(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """
    Compute the Difference Subspace (DS) between two subspaces.

    The difference subspace contains vectors that are partially in one
    subspace but not fully in both. Found by eigendecomposition of
    G = U1 @ U1.T + U2 @ U2.T (sum of projection matrices).

    Eigenvalue interpretation:
    - λ = 2: vector is in both subspaces (intersection)
    - λ = 1: vector is in exactly one subspace
    - 0 < λ < 1: vector is partially in one or both
    - λ = 0: vector is in neither

    Args:
        U1: (N, d) Orthonormal basis of first subspace
        U2: (N, d) Orthonormal basis of second subspace

    Returns:
        DS: (N, k) Difference subspace basis (eigenvectors with 0 < λ < 1)
             Returns empty array (N, 0) if no difference subspace found
    """
    # Sum of projection matrices
    G = U1 @ U1.T + U2 @ U2.T

    # Eigendecomposition
    eigen_val, eigen_vec = eig(G)
    eigen_val = np.real(eigen_val)
    eigen_vec = np.real(eigen_vec)

    # Find eigenvectors with eigenvalues strictly between 0 and 1
    # These represent the "difference" between subspaces
    # NOTE: Use < 1 (not < 1-eps) to match main6.py behavior
    idx = np.where((1e-6 < eigen_val) & (eigen_val < 1))[0]

    if len(idx) == 0:
        # Return empty array with 0 columns (not zeros!)
        return np.zeros((U1.shape[0], 0))

    return eigen_vec[:, idx]


def difference_subspace_projection_magnitude(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Compute the magnitude of U1 projected onto the difference subspace between U1 and U2.

    This measures how much of U1 lies in the "difference" between the two subspaces.
    Higher values indicate more difference between the subspaces.

    Matches main6.py implementation:
        DS = gen_shape_difference_subspace(S1, S2, cfg)
        if DS.shape[1] > 0:
            P = DS @ DS.T
            V = P @ S1
            magnitude = np.linalg.norm(V)

    Args:
        U1: (N, d) Orthonormal basis of first subspace (the one being projected)
        U2: (N, d) Orthonormal basis of second subspace

    Returns:
        magnitude: Frobenius norm of projection (0 if no difference subspace)
    """
    DS = compute_difference_subspace(U1, U2)

    # Check if DS has any columns (like main6.py: if DS.shape[1] > 0)
    if DS.shape[1] == 0:
        return 0.0

    # Projection matrix onto difference subspace
    P = DS @ DS.T

    # Project U1 onto difference subspace
    V = P @ U1

    # Return Frobenius norm
    return np.linalg.norm(V)


def compute_neighbor_average_ds(patches_U: List[np.ndarray],
                                 n_patches_x: int,
                                 n_patches_y: int,
                                 use_8_neighbors: bool = True) -> np.ndarray:
    """
    Compute DS magnitude for each patch as average of comparisons with neighbors.

    Args:
        patches_U: List of U matrices for each patch
        n_patches_x: Number of patches in x direction
        n_patches_y: Number of patches in y direction
        use_8_neighbors: If True, use 8 neighbors; if False, use 4 neighbors

    Returns:
        magnitudes: Array of average DS magnitudes for each patch
    """
    n_patches = len(patches_U)
    magnitudes = np.zeros(n_patches)

    def get_neighbors(idx):
        row = idx // n_patches_x
        col = idx % n_patches_x
        neighbors = []

        # 4-connectivity
        if row > 0:
            neighbors.append(idx - n_patches_x)  # Up
        if row < n_patches_y - 1:
            neighbors.append(idx + n_patches_x)  # Down
        if col > 0:
            neighbors.append(idx - 1)  # Left
        if col < n_patches_x - 1:
            neighbors.append(idx + 1)  # Right

        # 8-connectivity (add diagonals)
        if use_8_neighbors:
            if row > 0 and col > 0:
                neighbors.append(idx - n_patches_x - 1)  # Up-Left
            if row > 0 and col < n_patches_x - 1:
                neighbors.append(idx - n_patches_x + 1)  # Up-Right
            if row < n_patches_y - 1 and col > 0:
                neighbors.append(idx + n_patches_x - 1)  # Down-Left
            if row < n_patches_y - 1 and col < n_patches_x - 1:
                neighbors.append(idx + n_patches_x + 1)  # Down-Right

        return neighbors

    for i in range(n_patches):
        neighbors = get_neighbors(i)
        if len(neighbors) > 0:
            neighbor_mags = []
            for j in neighbors:
                mag = difference_subspace_projection_magnitude(patches_U[i], patches_U[j])
                neighbor_mags.append(mag)
            magnitudes[i] = np.mean(neighbor_mags)

    return magnitudes


def compute_neighbor_average_magd(patches_U: List[np.ndarray],
                                   n_patches_x: int,
                                   n_patches_y: int,
                                   use_8_neighbors: bool = True) -> np.ndarray:
    """
    Compute Mag(D) for each patch as average of comparisons with neighbors.

    Args:
        patches_U: List of U matrices for each patch
        n_patches_x: Number of patches in x direction
        n_patches_y: Number of patches in y direction
        use_8_neighbors: If True, use 8 neighbors; if False, use 4 neighbors

    Returns:
        magnitudes: Array of average Mag(D) for each patch
    """
    n_patches = len(patches_U)
    magnitudes = np.zeros(n_patches)

    def get_neighbors(idx):
        row = idx // n_patches_x
        col = idx % n_patches_x
        neighbors = []

        # 4-connectivity
        if row > 0:
            neighbors.append(idx - n_patches_x)
        if row < n_patches_y - 1:
            neighbors.append(idx + n_patches_x)
        if col > 0:
            neighbors.append(idx - 1)
        if col < n_patches_x - 1:
            neighbors.append(idx + 1)

        # 8-connectivity
        if use_8_neighbors:
            if row > 0 and col > 0:
                neighbors.append(idx - n_patches_x - 1)
            if row > 0 and col < n_patches_x - 1:
                neighbors.append(idx - n_patches_x + 1)
            if row < n_patches_y - 1 and col > 0:
                neighbors.append(idx + n_patches_x - 1)
            if row < n_patches_y - 1 and col < n_patches_x - 1:
                neighbors.append(idx + n_patches_x + 1)

        return neighbors

    for i in range(n_patches):
        neighbors = get_neighbors(i)
        if len(neighbors) > 0:
            neighbor_mags = []
            for j in neighbors:
                mag = magnitude_of_difference_subspace(patches_U[i], patches_U[j])
                neighbor_mags.append(mag)
            magnitudes[i] = np.mean(neighbor_mags)

    return magnitudes


def compute_reference_flat_subspace(patches_U: List[np.ndarray],
                                     n_patches_x: int,
                                     n_patches_y: int,
                                     method: str = 'edges') -> np.ndarray:
    """
    Compute a reference "flat" subspace from edge/corner patches.

    This reference represents the expected flat surface subspace.
    Patches are compared against this reference to detect deviations.

    Args:
        patches_U: List of U matrices for each patch
        n_patches_x: Number of patches in x direction
        n_patches_y: Number of patches in y direction
        method: 'edges' (use edge patches), 'corners' (use corner patches only),
                'karcher' (use Karcher mean of all patches)

    Returns:
        reference_U: Reference subspace basis matrix
    """
    n_patches = len(patches_U)

    if method == 'corners':
        # Use only corner patches (4 corners)
        corner_indices = [
            0,  # top-left
            n_patches_x - 1,  # top-right
            (n_patches_y - 1) * n_patches_x,  # bottom-left
            n_patches_y * n_patches_x - 1  # bottom-right
        ]
        reference_patches = [patches_U[i] for i in corner_indices if i < n_patches]

    elif method == 'edges':
        # Use all edge patches (top, bottom, left, right rows/columns)
        edge_indices = set()
        for j in range(n_patches_x):
            edge_indices.add(j)  # top row
            edge_indices.add((n_patches_y - 1) * n_patches_x + j)  # bottom row
        for i in range(n_patches_y):
            edge_indices.add(i * n_patches_x)  # left column
            edge_indices.add(i * n_patches_x + n_patches_x - 1)  # right column
        reference_patches = [patches_U[i] for i in edge_indices if i < n_patches]

    else:  # karcher - use all patches
        reference_patches = patches_U

    if len(reference_patches) == 0:
        return patches_U[0]

    # Compute Karcher mean of reference patches
    # Start with first patch as initial estimate
    mean = reference_patches[0].copy()

    for iteration in range(50):
        tangent_sum = np.zeros_like(mean)
        for U in reference_patches:
            # Project U onto tangent space at mean
            correlation = mean.T @ U
            diff = U - mean @ correlation
            tangent_sum += diff

        tangent_avg = tangent_sum / len(reference_patches)

        if np.linalg.norm(tangent_avg) < 1e-6:
            break

        # Update mean by moving in tangent direction
        mean_new = mean + 0.5 * tangent_avg
        mean, _ = np.linalg.qr(mean_new)

    return mean


def compute_reference_based_magd(patches_U: List[np.ndarray],
                                  reference_U: np.ndarray) -> np.ndarray:
    """
    Compute Mag(D) for each patch compared to a reference subspace.

    This method detects ANY deviation from the reference (flat) surface,
    including gradual bumps that neighbor-based methods miss.

    Args:
        patches_U: List of U matrices for each patch
        reference_U: Reference subspace basis matrix

    Returns:
        magnitudes: Array of Mag(D) values for each patch vs reference
    """
    n_patches = len(patches_U)
    magnitudes = np.zeros(n_patches)

    for i in range(n_patches):
        magnitudes[i] = magnitude_of_difference_subspace(patches_U[i], reference_U)

    return magnitudes


def compute_reference_based_ds(patches_U: List[np.ndarray],
                                reference_U: np.ndarray) -> np.ndarray:
    """
    Compute DS projection magnitude for each patch compared to a reference subspace.

    This method detects ANY deviation from the reference (flat) surface,
    including gradual bumps that neighbor-based methods miss.

    Args:
        patches_U: List of U matrices for each patch
        reference_U: Reference subspace basis matrix

    Returns:
        magnitudes: Array of DS projection magnitudes for each patch vs reference
    """
    n_patches = len(patches_U)
    magnitudes = np.zeros(n_patches)

    for i in range(n_patches):
        magnitudes[i] = difference_subspace_projection_magnitude(patches_U[i], reference_U)

    return magnitudes


def compute_all_detection_methods(patches: List[np.ndarray],
                                   n_patches_x: int,
                                   n_patches_y: int,
                                   reference_method: str = 'edges') -> dict:
    """
    Compute all detection methods: S[2], Mag(D), DS with both neighbor and reference approaches.

    Args:
        patches: List of point cloud patches
        n_patches_x: Number of patches in x direction
        n_patches_y: Number of patches in y direction
        reference_method: Method to compute reference ('edges', 'corners', 'karcher')

    Returns:
        Dictionary with all detection scores:
        - sv2: S[2] scores (absolute Z variance)
        - magd_neighbor: Mag(D) neighbor-based scores
        - ds_neighbor: DS neighbor-based scores
        - magd_reference: Mag(D) reference-based scores
        - ds_reference: DS reference-based scores
    """
    # Compute subspaces for all patches
    patch_data = [compute_shape_subspace(patch) for patch in patches]
    patches_U = [d[0] for d in patch_data]
    patches_S = [d[1] for d in patch_data]

    # Method 1: S[2] scores (absolute)
    sv2_scores = np.array([S[2] for S in patches_S])

    # Method 2 & 3: Neighbor-based Mag(D) and DS
    magd_neighbor = compute_neighbor_average_magd(patches_U, n_patches_x, n_patches_y)
    ds_neighbor = compute_neighbor_average_ds(patches_U, n_patches_x, n_patches_y)

    # Method 4 & 5: Reference-based Mag(D) and DS
    reference_U = compute_reference_flat_subspace(patches_U, n_patches_x, n_patches_y, reference_method)
    magd_reference = compute_reference_based_magd(patches_U, reference_U)
    ds_reference = compute_reference_based_ds(patches_U, reference_U)

    return {
        'sv2': sv2_scores,
        'magd_neighbor': magd_neighbor,
        'ds_neighbor': ds_neighbor,
        'magd_reference': magd_reference,
        'ds_reference': ds_reference,
        'patches_U': patches_U,
        'patches_S': patches_S,
        'reference_U': reference_U
    }


class ShapeSubspaceDetector:
    """
    Dent detection using Shape Subspace method with canonical angles.
    """

    def __init__(self, subspace_dim: int = 3, detection_metric: str = 'magnitude'):
        self.subspace_dim = subspace_dim
        self.detection_metric = detection_metric
        self.reference_subspace = None
        self.patch_subspaces = []

    def compute_patch_subspaces(self, patches: List[np.ndarray]) -> List[np.ndarray]:
        """Compute shape subspace for each patch."""
        self.patch_subspaces = []
        for patch in patches:
            U, S, Vt = compute_shape_subspace(patch, center=True, dim=self.subspace_dim)
            self.patch_subspaces.append(U)
        return self.patch_subspaces

    def compute_reference_subspace(self, method: str = 'mean') -> np.ndarray:
        """Compute reference subspace (Karcher mean of all patches)."""
        if len(self.patch_subspaces) == 0:
            raise ValueError("No patch subspaces computed.")

        # Simple mean approximation for Karcher mean
        mean = self.patch_subspaces[0].copy()
        for _ in range(50):  # Iterative refinement
            tangent_sum = np.zeros_like(mean)
            for U in self.patch_subspaces:
                correlation = mean.T @ U
                diff = U - mean @ correlation
                tangent_sum += diff
            tangent_avg = tangent_sum / len(self.patch_subspaces)
            if np.linalg.norm(tangent_avg) < 1e-6:
                break
            mean_new = mean + 0.5 * tangent_avg
            mean, _ = np.linalg.qr(mean_new)

        self.reference_subspace = mean
        return self.reference_subspace

    def detect(self, patches: List[np.ndarray], threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Detect dents using Shape Subspace method.

        Returns:
            scores: Raw detection scores for each patch
            z_scores: Z-scores for each patch
            detected_indices: Indices of patches detected as dents
        """
        self.compute_patch_subspaces(patches)
        self.compute_reference_subspace()

        scores = np.zeros(len(patches))
        for i, U in enumerate(self.patch_subspaces):
            if self.detection_metric == 'magnitude':
                scores[i] = magnitude_of_difference_subspace(U, self.reference_subspace)
            elif self.detection_metric == 'max_angle':
                scores[i] = max_canonical_angle(U, self.reference_subspace)
            elif self.detection_metric == 'grassmann':
                scores[i] = grassmann_distance(U, self.reference_subspace)

        mean_score = np.mean(scores)
        std_score = np.std(scores)
        z_scores = (scores - mean_score) / std_score if std_score > 0 else np.zeros_like(scores)
        detected_indices = list(np.where(z_scores > threshold)[0])

        return scores, z_scores, detected_indices


def resample_patch_to_grid(patch: np.ndarray, patch_bounds: Tuple[float, float, float, float],
                           grid_size: int = 10) -> np.ndarray:
    """
    Resample a patch to a standardized grid using interpolation.

    This ensures all patches have points at the same relative positions,
    making U matrices directly comparable (fixes the point ordering problem).

    Args:
        patch: (N, 3) array of XYZ points in the patch
        patch_bounds: (x_min, x_max, y_min, y_max) bounds of the patch
        grid_size: Number of grid points in each direction (grid_size x grid_size total)

    Returns:
        grid_points: (grid_size*grid_size, 3) array of resampled points
    """
    x_min, x_max, y_min, y_max = patch_bounds

    # Handle empty or nearly empty patches
    if len(patch) < 4:
        grid_x = np.linspace(x_min, x_max, grid_size)
        grid_y = np.linspace(y_min, y_max, grid_size)
        gx, gy = np.meshgrid(grid_x, grid_y)
        gz = np.zeros_like(gx)
        return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    # Create grid with small margin to avoid edge interpolation issues
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    grid_x = np.linspace(x_min + margin_x, x_max - margin_x, grid_size)
    grid_y = np.linspace(y_min + margin_y, y_max - margin_y, grid_size)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_xy = np.column_stack([gx.ravel(), gy.ravel()])

    # Interpolate Z values from original points
    gz = griddata(patch[:, :2], patch[:, 2], grid_xy, method='linear', fill_value=0.0)
    gz = np.nan_to_num(gz, nan=0.0)

    return np.column_stack([grid_xy, gz])


def divide_surface_to_patches(points: np.ndarray, n_patches_x: int, n_patches_y: int,
                              use_grid_resampling: bool = False,
                              grid_size: int = 10) -> Tuple[List[np.ndarray], List[Tuple]]:
    """
    Divide a surface point cloud into patches.

    Args:
        points: (N, 3) array of XYZ points
        n_patches_x: Number of patches in X direction
        n_patches_y: Number of patches in Y direction
        use_grid_resampling: If True, resample each patch to a standardized grid
        grid_size: Grid size for resampling (only used if use_grid_resampling=True)

    Returns:
        patches: List of point arrays for each patch
        bounds: List of (x_min, x_max, y_min, y_max) bounds for each patch
    """
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()

    patch_width = (x_max - x_min) / n_patches_x
    patch_height = (y_max - y_min) / n_patches_y

    patches = []
    bounds = []

    for j in range(n_patches_y):
        for i in range(n_patches_x):
            px_min = x_min + i * patch_width
            px_max = x_min + (i + 1) * patch_width
            py_min = y_min + j * patch_height
            py_max = y_min + (j + 1) * patch_height

            patch_bounds = (px_min, px_max, py_min, py_max)
            bounds.append(patch_bounds)

            # Get points in this patch
            mask = ((points[:, 0] >= px_min) & (points[:, 0] < px_max) &
                    (points[:, 1] >= py_min) & (points[:, 1] < py_max))

            # Handle right and top edges
            if i == n_patches_x - 1:
                mask = mask | ((points[:, 0] == px_max) &
                              (points[:, 1] >= py_min) & (points[:, 1] < py_max))
            if j == n_patches_y - 1:
                mask = mask | ((points[:, 1] == py_max) &
                              (points[:, 0] >= px_min) & (points[:, 0] < px_max))

            patch = points[mask]

            if use_grid_resampling and len(patch) >= 4:
                patch = resample_patch_to_grid(patch, patch_bounds, grid_size)

            patches.append(patch)

    return patches, bounds


def normalize_patch_sizes(patches: List[np.ndarray], target_size: int = 50) -> List[np.ndarray]:
    """
    Normalize all patches to have the same number of points.

    This is needed when NOT using grid resampling, to ensure U matrices
    have the same dimensions for comparison.

    Args:
        patches: List of point arrays (variable sizes)
        target_size: Target number of points per patch

    Returns:
        normalized: List of point arrays (all same size)
    """
    normalized = []
    for patch in patches:
        if len(patch) == 0:
            normalized.append(np.zeros((target_size, 3)))
        elif len(patch) >= target_size:
            indices = np.random.choice(len(patch), target_size, replace=False)
            normalized.append(patch[indices])
        else:
            indices = np.random.choice(len(patch), target_size, replace=True)
            normalized.append(patch[indices])
    return normalized


def compute_all_detection_methods_with_grid(points: np.ndarray,
                                            n_patches_x: int,
                                            n_patches_y: int,
                                            use_grid_resampling: bool = True,
                                            grid_size: int = 10,
                                            reference_method: str = 'edges') -> dict:
    """
    Complete pipeline: divide surface, optionally resample to grid, compute all detection methods.

    Args:
        points: (N, 3) array of XYZ surface points
        n_patches_x: Number of patches in X direction
        n_patches_y: Number of patches in Y direction
        use_grid_resampling: If True, use standardized grid resampling (recommended)
        grid_size: Grid size for resampling
        reference_method: Method for reference subspace ('edges', 'corners', 'karcher')

    Returns:
        Dictionary with detection scores and metadata:
        - sv2: S[2] scores
        - magd_neighbor: Mag(D) neighbor-based
        - ds_neighbor: DS neighbor-based
        - magd_reference: Mag(D) reference-based
        - ds_reference: DS reference-based
        - patches: List of patches used
        - bounds: Patch bounds
    """
    # Divide surface into patches
    patches, bounds = divide_surface_to_patches(
        points, n_patches_x, n_patches_y,
        use_grid_resampling=use_grid_resampling,
        grid_size=grid_size
    )

    # If not using grid resampling, normalize patch sizes
    if not use_grid_resampling:
        patches = normalize_patch_sizes(patches)

    # Compute detection methods
    results = compute_all_detection_methods(
        patches, n_patches_x, n_patches_y, reference_method
    )

    results['patches'] = patches
    results['bounds'] = bounds
    results['use_grid_resampling'] = use_grid_resampling
    results['grid_size'] = grid_size

    return results


if __name__ == "__main__":
    print("Shape Subspace Module Test")
    print("=" * 50)

    # Create test point clouds
    np.random.seed(42)
    n_points = 100
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    z = np.random.normal(0, 0.01, n_points)
    flat_points = np.column_stack([x, y, z])

    dent_points = flat_points.copy()
    center_mask = (x**2 + y**2) < 0.3**2
    dent_points[center_mask, 2] -= 0.5

    U_flat, S_flat, _ = compute_shape_subspace(flat_points)
    U_dent, S_dent, _ = compute_shape_subspace(dent_points)

    print(f"\nFlat surface singular values: {S_flat}")
    print(f"Dented surface singular values: {S_dent}")

    mag_diff = magnitude_of_difference_subspace(U_flat, U_dent)
    max_angle = max_canonical_angle(U_flat, U_dent)
    similarity = subspace_similarity(U_flat, U_dent)

    print(f"\nComparison (flat vs dented):")
    print(f"  Magnitude of Difference Subspace: {mag_diff:.6f}")
    print(f"  Max Canonical Angle: {np.degrees(max_angle):.2f} degrees")
    print(f"  Subspace Similarity: {similarity:.6f}")
    print("\n" + "=" * 50)
