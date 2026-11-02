"""
Data Generation Script - REAL SCALE (A.ply characteristics)
===========================================================
This version generates training data matching A.ply's spatial characteristics:
- Grid: 640 x 760 pixels (H x W)
- X range: -131.70 to 10.76 mm (142.46 mm total)
- Y range: -80.66 to 36.84 mm (117.50 mm total)
- Pixel density: ~0.187 mm/pixel

This matches real aircraft scan data, unlike the original main.py which used:
- Grid: 640 x 960 pixels
- Metric: 500 x 330 mm
- Pixel density: ~0.52 mm/pixel

Key features:
- NO gt_mask in quadric fitting (consistent with inference)
- Dent sizes scaled for the smaller metric range
- Same noise and dent depth characteristics

Author: Generated for thesis research
Date: December 2024
"""
import numpy as np
import math
import os
from scipy.optimize import least_squares
from tqdm import tqdm

# --- Configuration matching A.ply ---
W_PIXELS = 760  # A.ply width
H_PIXELS = 640  # A.ply height

# Coordinate ranges from A.ply analysis
X_MIN = -131.70
X_MAX = 10.76
Y_MIN = -80.66
Y_MAX = 36.84

# Derived metric ranges
W_METRIC = X_MAX - X_MIN  # 142.46 mm
H_METRIC = Y_MAX - Y_MIN  # 117.50 mm

# Noise and dent parameters
GAUSSIAN_NOISE_STD_DEV = 0.15  # mm
NUM_DENTS_MAX = 3
DENT_PROBABILITY = 0.8
TOTAL_SAMPLES = 30000

# Output directory
OUTPUT_DIR = "D:/pointcloud_db_real_scale"


# --- Helper Functions ---

def generate_grid_coords():
    """Generates coordinate grid matching A.ply exactly."""
    x_coords = np.linspace(X_MIN, X_MAX, W_PIXELS)
    y_coords = np.linspace(Y_MIN, Y_MAX, H_PIXELS)
    X, Y = np.meshgrid(x_coords, y_coords)
    return X, Y


def generate_base_surface(X, Y):
    """Models the base surface z_s = alpha*x^2 + beta*y^2."""
    alpha = np.random.uniform(-0.005, 0.005)
    beta = np.random.uniform(-0.005, 0.005)
    Z_s = alpha * X**2 + beta * Y**2
    return Z_s


def generate_dent(X, Y):
    """
    Models a single dent with sizes appropriate for the real scale.
    Dent sizes are scaled down to match the smaller metric range.
    """
    H, W = X.shape

    # Scaling - adjusted for smaller metric range
    # Original: 20-100mm for 500mm width
    # Scaled: ~6-30mm for 142mm width (factor of ~3.5x smaller)
    scale_x = np.random.uniform(10, 45)
    scale_y = np.random.uniform(10, 45)

    # Shifting - within the actual coordinate range
    # Keep dents away from edges
    margin_x = W_METRIC * 0.2
    margin_y = H_METRIC * 0.2
    shift_x = np.random.uniform(X_MIN + margin_x, X_MAX - margin_x)
    shift_y = np.random.uniform(Y_MIN + margin_y, Y_MAX - margin_y)

    # Rotation
    angle = np.random.uniform(0, 2 * np.pi)

    # Apply transformation
    X_shifted = X - shift_x
    Y_shifted = Y - shift_y

    X_rotated = X_shifted * np.cos(angle) - Y_shifted * np.sin(angle)
    Y_rotated = X_shifted * np.sin(angle) + Y_shifted * np.cos(angle)

    # Calculate elliptical radius
    R = np.sqrt((X_rotated / scale_x)**2 + (Y_rotated / scale_y)**2)

    Z_d = np.zeros((H, W))
    mask = R < 1

    # Depth parameters - same as before
    REQUIRED_SCALING = 6.91
    target_depth = np.random.uniform(0.2, 1.8)
    depth_factor = target_depth * REQUIRED_SCALING

    # Dent depth formula
    Z_d[mask] = depth_factor * (1/10) * (1 - np.exp(1 - R[mask]**2))

    return Z_d, mask


def dent_count_probability(current_count):
    """Exponentially decreasing probability for further dents."""
    return DENT_PROBABILITY * math.exp(-0.5 * current_count)


def quadric_surface(params, X, Y):
    """z(x,y) = a + bx + cy + dx^2 + exy + fy^2"""
    a, b, c, d, e, f = params
    return a + b*X + c*Y + d*X**2 + e*X*Y + f*Y**2


def residuals_func(params, Z, X, Y):
    """Function to minimize: (Z - Z_fitted)"""
    return Z - quadric_surface(params, X, Y)


def fit_quadric_no_gtmask(X, Y, Z_final):
    """
    Fits quadric surface to ALL points (no gt_mask exclusion).
    This matches inference preprocessing where gt_mask is not available.
    """
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z_final.flatten()

    initial_params = np.zeros(6)
    result = least_squares(residuals_func, initial_params, args=(Z_flat, X_flat, Y_flat))
    fitted_params = result.x

    Z_fitted = quadric_surface(fitted_params, X, Y)
    residuals = Z_final - Z_fitted

    return residuals


# --- Main Generation Function ---

def generate_single_sample_data(sample_id):
    """Generates FCN Input (Residuals) and Target (GT Mask) for one sample."""

    # 1. Generate Coordinates
    X, Y = generate_grid_coords()

    # Add jitter
    XY_JITTER_STD = 0.01
    X_noisy = X + np.random.normal(0, XY_JITTER_STD, X.shape)
    Y_noisy = Y + np.random.normal(0, XY_JITTER_STD, Y.shape)

    # 2. Generate Base Surface
    Z_s = generate_base_surface(X_noisy, Y_noisy)

    # 3. Generate Dents and Ground Truth
    Z_d_max = np.zeros(X.shape)
    GT_mask = np.zeros(X.shape, dtype=np.uint8)
    num_dents = 0

    while num_dents < NUM_DENTS_MAX and np.random.rand() < dent_count_probability(num_dents):
        Z_d_new, mask_new = generate_dent(X_noisy, Y_noisy)

        Z_d_max = np.minimum(Z_d_max, Z_d_new)
        GT_mask = np.logical_or(GT_mask, mask_new).astype(np.uint8)

        num_dents += 1

    # 4. Add Gaussian Noise
    Z_w = np.random.normal(0, GAUSSIAN_NOISE_STD_DEV, X.shape)

    # 5. Final Surface
    Z_final = Z_s + Z_d_max + Z_w

    # 6. Compute residuals WITHOUT gt_mask (consistent with inference)
    residuals = fit_quadric_no_gtmask(X_noisy, Y_noisy, Z_final)

    return X_noisy, Y_noisy, residuals, GT_mask


def run_generation_loop():
    """Generates and saves all samples."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*70)
    print("DATA GENERATION - REAL SCALE (A.ply characteristics)")
    print("="*70)
    print(f"\nGrid: {H_PIXELS} x {W_PIXELS} (matching A.ply)")
    print(f"X range: {X_MIN} to {X_MAX} mm ({W_METRIC:.2f} mm)")
    print(f"Y range: {Y_MIN} to {Y_MAX} mm ({H_METRIC:.2f} mm)")
    print(f"Pixel density: {W_METRIC/W_PIXELS:.3f} mm/pixel")
    print(f"Noise std: {GAUSSIAN_NOISE_STD_DEV} mm")
    print(f"Dent scale: 10-45 mm (scaled for real data)")
    print(f"\nTotal samples: {TOTAL_SAMPLES}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70 + "\n")

    for i in tqdm(range(TOTAL_SAMPLES)):
        X_noisy, Y_noisy, residuals, gt_mask = generate_single_sample_data(i)

        # Save files
        base_name = os.path.join(OUTPUT_DIR, f"{i:05d}")
        np.save(f"{base_name}_input.npy", residuals.astype(np.float32))
        np.save(f"{base_name}_gt.npy", gt_mask)
        np.save(f"{base_name}_X.npy", X_noisy.astype(np.float32))
        np.save(f"{base_name}_Y.npy", Y_noisy.astype(np.float32))

    print("\n" + "="*70)
    print("Generation complete!")
    print(f"Total samples: {TOTAL_SAMPLES}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*70)
    print("\nNext steps:")
    print("1. Update config.py:")
    print(f'   DATASET_PATH = "{OUTPUT_DIR}"')
    print("   W_PIXELS = 760")
    print("   H_PIXELS = 640")
    print("2. Run training: python train_hybrid.py")
    print("3. Test on realSamples/A.ply")
    print("="*70)


if __name__ == "__main__":
    run_generation_loop()
