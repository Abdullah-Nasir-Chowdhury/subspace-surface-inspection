"""
DentNet Inference Script with RANSAC Preprocessing
===================================================
Uses RANSAC to fit the quadric surface, making it robust to dent outliers.
This ensures dents appear as negative residuals (like in training data).
"""

import torch
import numpy as np
try:
    import open3d as o3d
except ImportError:  # PLY-only stand-in in common/
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "common"))
    import open3d as o3d
from scipy.optimize import least_squares
from scipy.ndimage import binary_closing
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin

from model import DentNet
import config


class QuadricRegressor(BaseEstimator, RegressorMixin):
    """Custom regressor for quadric surface fitting"""
    def __init__(self):
        self.coef_ = None

    def fit(self, XY, Z):
        """Fit quadric surface z = a + bx + cy + dx^2 + exy + fy^2"""
        X = XY[:, 0]
        Y = XY[:, 1]

        # Design matrix for quadric
        A = np.column_stack([
            np.ones_like(X),  # a
            X,                 # b
            Y,                 # c
            X**2,              # d
            X*Y,               # e
            Y**2               # f
        ])

        # Solve least squares
        self.coef_, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        return self

    def predict(self, XY):
        """Predict Z values"""
        X = XY[:, 0]
        Y = XY[:, 1]
        a, b, c, d, e, f = self.coef_
        return a + b*X + c*Y + d*X**2 + e*X*Y + f*Y**2


def fit_quadric_ransac(X, Y, Z, residual_threshold=0.3, min_samples=0.5):
    """
    Fit quadric surface using RANSAC to be robust to dent outliers.

    Args:
        X, Y, Z: Coordinate arrays
        residual_threshold: Maximum residual for inliers (mm)
        min_samples: Fraction of points to use for each RANSAC iteration

    Returns:
        residuals: Z - Z_fitted (dents will be negative)
        inlier_mask: Boolean mask of inliers
    """
    # Prepare data
    XY = np.column_stack([X.flatten(), Y.flatten()])
    Z_flat = Z.flatten()

    # RANSAC with custom quadric regressor
    ransac = RANSACRegressor(
        estimator=QuadricRegressor(),
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=100,
        random_state=42
    )

    ransac.fit(XY, Z_flat)

    # Get fitted surface
    Z_fitted = ransac.predict(XY)

    # Compute residuals
    residuals = Z_flat - Z_fitted

    # Reshape to original grid shape
    residuals = residuals.reshape(X.shape)
    inlier_mask = ransac.inlier_mask_.reshape(X.shape)

    return residuals, inlier_mask


def preprocess_point_cloud_ransac(points, use_ransac=True, ransac_threshold=0.3, negate_residuals=False):
    """
    Preprocess point cloud data using RANSAC for robust quadric fitting.

    Args:
        points: Nx3 numpy array (X, Y, Z coordinates)
        use_ransac: Whether to use RANSAC (True) or standard least squares (False)
        ransac_threshold: Residual threshold for RANSAC inliers
        negate_residuals: If True, negate residuals (for data where dents are convex/positive Z)

    Returns:
        residuals: 2D array ready for model input
        X, Y, Z: Original coordinates (for visualization)
        inlier_mask: Which points were used for fitting (if RANSAC)
    """
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    # Reshape to grid
    num_points = len(X)
    h = config.H_PIXELS
    w = config.W_PIXELS

    print(f"Reshaping to grid: {h}x{w}")

    X_grid = X[:h*w].reshape(h, w)
    Y_grid = Y[:h*w].reshape(h, w)
    Z_grid = Z[:h*w].reshape(h, w)

    if use_ransac:
        print("Using RANSAC for robust quadric fitting...")
        residuals, inlier_mask = fit_quadric_ransac(
            X_grid, Y_grid, Z_grid,
            residual_threshold=ransac_threshold
        )
        inlier_pct = 100 * inlier_mask.mean()
        print(f"RANSAC inliers: {inlier_pct:.1f}%")
    else:
        # Standard least squares (for comparison)
        def quadric_surface(params, X, Y):
            a, b, c, d, e, f = params
            return a + b*X + c*Y + d*X**2 + e*X*Y + f*Y**2

        def residuals_func(params, Z, X, Y):
            return Z - quadric_surface(params, X, Y)

        initial_params = np.zeros(6)
        result = least_squares(residuals_func, initial_params,
                              args=(Z_grid.flatten(), X_grid.flatten(), Y_grid.flatten()))

        Z_fitted = quadric_surface(result.x, X_grid, Y_grid)
        residuals = Z_grid - Z_fitted
        inlier_mask = np.ones_like(residuals, dtype=bool)

    # Negate residuals if needed (for data where dents appear as positive Z)
    if negate_residuals:
        residuals = -residuals
        print("Residuals NEGATED (dents now appear as negative)")

    return residuals, X_grid, Y_grid, Z_grid, inlier_mask


def load_model(model_path='best_dentnet_real_scale.pth', device='cuda'):
    """Load trained DentNet model"""
    model = DentNet().to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle DataParallel models
    if isinstance(checkpoint, dict) and any(k.startswith('module.') for k in checkpoint.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        checkpoint = new_state_dict

    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def predict_dents(model, residuals, device='cuda', threshold=0.5, fill_holes=True, closing_kernel=7):
    """
    Run inference on preprocessed residuals.

    Args:
        model: Trained DentNet model
        residuals: 2D numpy array of residuals
        device: 'cuda' or 'cpu'
        threshold: Probability threshold for dent classification
        fill_holes: If True, apply morphological closing to fill center holes
        closing_kernel: Kernel size for morphological closing (0 to disable)

    Returns:
        prediction_mask: Binary mask (1=dent, 0=normal)
        probabilities: Raw probability map
    """
    original_h, original_w = residuals.shape

    # Pad to multiple of 16
    pad_h = (16 - original_h % 16) % 16
    pad_w = (16 - original_w % 16) % 16

    residuals_padded = np.pad(residuals, ((0, pad_h), (0, pad_w)), mode='edge')

    input_tensor = torch.from_numpy(residuals_padded).float().unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        dent_prob = probs[0, 1, :, :].cpu().numpy()
        dent_prob = dent_prob[:original_h, :original_w]
        prediction_mask = (dent_prob > threshold).astype(np.uint8)

    # Post-processing: fill holes in detected dents
    if fill_holes and closing_kernel > 0:
        struct = np.ones((closing_kernel, closing_kernel))
        prediction_mask = binary_closing(prediction_mask, structure=struct).astype(np.uint8)

    return prediction_mask, dent_prob


def visualize_results(residuals, prediction_mask, probabilities, X_grid, Y_grid,
                     inlier_mask=None, save_path=None):
    """Visualize input residuals and prediction results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Residuals
    im1 = axes[0,0].imshow(residuals, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto',
                           extent=[X_grid.min(), X_grid.max(), Y_grid.max(), Y_grid.min()])
    axes[0,0].set_title(f'Residuals (RANSAC)\nMin={residuals.min():.2f}, Max={residuals.max():.2f} mm')
    axes[0,0].set_xlabel('X (mm)')
    axes[0,0].set_ylabel('Y (mm)')
    plt.colorbar(im1, ax=axes[0,0], label='Residual (mm)')

    # Plot 2: Probability map
    im2 = axes[0,1].imshow(probabilities, cmap='hot', vmin=0, vmax=1, aspect='auto',
                           extent=[X_grid.min(), X_grid.max(), Y_grid.max(), Y_grid.min()])
    axes[0,1].set_title('Dent Probability')
    axes[0,1].set_xlabel('X (mm)')
    axes[0,1].set_ylabel('Y (mm)')
    plt.colorbar(im2, ax=axes[0,1], label='Probability')

    # Plot 3: Binary prediction
    im3 = axes[1,0].imshow(prediction_mask, cmap='Greens', vmin=0, vmax=1, aspect='auto',
                           extent=[X_grid.min(), X_grid.max(), Y_grid.max(), Y_grid.min()])
    dent_pct = 100 * prediction_mask.mean()
    axes[1,0].set_title(f'Predicted Dent Mask\n{prediction_mask.sum()} pixels ({dent_pct:.1f}%)')
    axes[1,0].set_xlabel('X (mm)')
    axes[1,0].set_ylabel('Y (mm)')
    plt.colorbar(im3, ax=axes[1,0], label='Dent')

    # Plot 4: RANSAC inliers (if available)
    if inlier_mask is not None:
        im4 = axes[1,1].imshow(inlier_mask.astype(int), cmap='RdYlGn', vmin=0, vmax=1, aspect='auto',
                               extent=[X_grid.min(), X_grid.max(), Y_grid.max(), Y_grid.min()])
        inlier_pct = 100 * inlier_mask.mean()
        axes[1,1].set_title(f'RANSAC Inliers\n{inlier_pct:.1f}% used for surface fitting')
        axes[1,1].set_xlabel('X (mm)')
        axes[1,1].set_ylabel('Y (mm)')
        plt.colorbar(im4, ax=axes[1,1], label='Inlier')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def save_colored_point_cloud(points, prediction_mask, output_path):
    """Save point cloud with dents colored in green"""
    mask_flat = prediction_mask.flatten()
    num_points = len(points)

    if len(mask_flat) < num_points:
        mask_flat = np.pad(mask_flat, (0, num_points - len(mask_flat)), constant_values=0)
    elif len(mask_flat) > num_points:
        mask_flat = mask_flat[:num_points]

    colors = np.zeros((num_points, 3))
    colors[mask_flat == 0] = [0.5, 0.5, 0.5]  # Gray for normal
    colors[mask_flat == 1] = [0.0, 1.0, 0.0]  # Green for dents

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Colored point cloud saved to {output_path}")


def run_inference_ransac(input_file, model_path='best_dentnet_real_scale.pth',
                         output_dir='output', threshold=0.5,
                         ransac_threshold=0.3, negate_residuals=False,
                         fill_holes=True, closing_kernel=7, device=None):
    """
    Complete inference pipeline with RANSAC preprocessing.

    Args:
        input_file: Path to input PLY file
        model_path: Path to trained model
        output_dir: Directory to save results
        threshold: Probability threshold for dent detection
        ransac_threshold: Residual threshold for RANSAC inliers
        negate_residuals: If True, negate residuals (for real data with +Z dents)
        fill_holes: If True, apply morphological closing to fill center holes
        closing_kernel: Kernel size for morphological closing
        device: 'cuda' or 'cpu' (auto-detected if None)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    print(f"\n{'='*60}")
    print(f"Running inference (RANSAC) on: {input_file}")
    print(f"{'='*60}\n")

    # 1. Load point cloud
    print("Step 1: Loading point cloud...")
    pcd = o3d.io.read_point_cloud(input_file)
    points = np.asarray(pcd.points)
    print(f"Loaded {len(points)} points")

    # 2. Preprocess with RANSAC
    print("\nStep 2: Preprocessing (RANSAC quadric fitting)...")
    residuals, X, Y, Z, inlier_mask = preprocess_point_cloud_ransac(
        points, use_ransac=True, ransac_threshold=ransac_threshold,
        negate_residuals=negate_residuals
    )
    print(f"Residuals computed: shape {residuals.shape}")
    print(f"Residual range: [{residuals.min():.4f}, {residuals.max():.4f}] mm")

    # 3. Load model
    print("\nStep 3: Loading model...")
    model = load_model(model_path, device)

    # 4. Predict
    print("\nStep 4: Running inference...")
    prediction_mask, probabilities = predict_dents(
        model, residuals, device, threshold,
        fill_holes=fill_holes, closing_kernel=closing_kernel
    )
    if fill_holes:
        print(f"Post-processing: morphological closing with {closing_kernel}x{closing_kernel} kernel")

    total_pixels = prediction_mask.size
    dent_pixels = np.sum(prediction_mask)
    dent_percentage = (dent_pixels / total_pixels) * 100

    print(f"Prediction complete!")
    print(f"  - Total pixels: {total_pixels}")
    print(f"  - Dent pixels: {dent_pixels} ({dent_percentage:.2f}%)")
    print(f"  - Max dent probability: {probabilities.max():.4f}")

    # 5. Save results
    print("\nStep 5: Saving results...")

    viz_path = os.path.join(output_dir, f"{base_name}_ransac_prediction.png")
    visualize_results(residuals, prediction_mask, probabilities, X, Y,
                     inlier_mask=inlier_mask, save_path=viz_path)

    ply_path = os.path.join(output_dir, f"{base_name}_ransac_predicted.ply")
    save_colored_point_cloud(points, prediction_mask, ply_path)

    mask_path = os.path.join(output_dir, f"{base_name}_ransac_mask.npy")
    np.save(mask_path, prediction_mask)
    print(f"Prediction mask saved to {mask_path}")

    # Also save residuals for comparison
    residuals_path = os.path.join(output_dir, f"{base_name}_ransac_residuals.npy")
    np.save(residuals_path, residuals)
    print(f"Residuals saved to {residuals_path}")

    print(f"\n{'='*60}")
    print("Inference complete!")
    print(f"{'='*60}\n")

    return prediction_mask, probabilities, residuals


def compare_methods(input_file):
    """Compare standard vs RANSAC preprocessing"""
    print("="*70)
    print("COMPARING STANDARD VS RANSAC PREPROCESSING")
    print("="*70)

    # Load point cloud
    pcd = o3d.io.read_point_cloud(input_file)
    points = np.asarray(pcd.points)

    # Standard preprocessing
    print("\n1. Standard least squares fitting:")
    residuals_std, X, Y, Z, _ = preprocess_point_cloud_ransac(points, use_ransac=False)
    print(f"   Residual range: [{residuals_std.min():.4f}, {residuals_std.max():.4f}] mm")

    # RANSAC preprocessing
    print("\n2. RANSAC fitting:")
    residuals_ransac, X, Y, Z, inlier_mask = preprocess_point_cloud_ransac(points, use_ransac=True)
    print(f"   Residual range: [{residuals_ransac.min():.4f}, {residuals_ransac.max():.4f}] mm")

    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im1 = axes[0].imshow(residuals_std, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    axes[0].set_title(f'Standard LS Residuals\n[{residuals_std.min():.2f}, {residuals_std.max():.2f}] mm')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(residuals_ransac, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    axes[1].set_title(f'RANSAC Residuals\n[{residuals_ransac.min():.2f}, {residuals_ransac.max():.2f}] mm')
    plt.colorbar(im2, ax=axes[1])

    diff = residuals_ransac - residuals_std
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[2].set_title(f'Difference (RANSAC - Standard)\n[{diff.min():.2f}, {diff.max():.2f}] mm')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig('output/ransac_comparison.png', dpi=150)
    print("\nSaved comparison to output/ransac_comparison.png")
    plt.show()


if __name__ == "__main__":
    input_file = "realSamples/A.ply"

    if os.path.exists(input_file):
        # Run inference with RANSAC and NEGATED residuals
        # (A.ply has inverted convention - dents appear as positive Z)
        print("="*70)
        print("INFERENCE WITH NEGATED RESIDUALS")
        print("(Real data has dents as positive Z, training data has negative Z)")
        print("="*70)

        run_inference_ransac(
            input_file=input_file,
            model_path='best_dentnet_real_scale.pth',
            output_dir='output',
            threshold=0.5,
            ransac_threshold=0.3,
            negate_residuals=True  # KEY: Negate residuals for real data
        )
    else:
        print(f"Error: {input_file} not found!")
