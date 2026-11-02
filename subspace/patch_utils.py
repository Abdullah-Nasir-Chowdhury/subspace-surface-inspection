"""
Patch-Based Processing Utilities
=================================
Functions for extracting, processing, and reconstructing patches
"""

import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import numpy as np
import torch
from typing import List, Tuple, Dict


class PatchExtractor:
    """Extract patches from 2D residual maps"""

    def __init__(self, patch_size: int = 64, stride: int = 32, ignore_edge_patches: bool = True):
        """
        Args:
            patch_size: Size of square patches (e.g., 64 = 64×64)
            stride: Step size for sliding window (smaller = more overlap)
            ignore_edge_patches: If True, ignore patches at image boundaries
        """
        self.patch_size = patch_size
        self.stride = stride
        self.ignore_edge_patches = ignore_edge_patches

    def extract_patches(self, residuals: np.ndarray, X: np.ndarray = None, Y: np.ndarray = None) -> Dict:
        """
        Extract patches from residual map

        Args:
            residuals: 2D array (H, W) of residuals
            X, Y: Optional coordinate grids (for spatial info)

        Returns:
            Dictionary containing:
                - patches: List of (patch_size, patch_size) arrays
                - positions: List of (y, x) top-left positions
                - valid_mask: Boolean array indicating which patches to use
        """
        H, W = residuals.shape
        patches = []
        positions = []
        valid_mask = []

        # Calculate edge margin to ignore
        edge_margin = self.patch_size // 2 if self.ignore_edge_patches else 0

        # Slide window across image
        for y in range(0, H - self.patch_size + 1, self.stride):
            for x in range(0, W - self.patch_size + 1, self.stride):

                # Extract patch
                patch = residuals[y:y+self.patch_size, x:x+self.patch_size]

                # Check if patch is complete (not truncated)
                if patch.shape != (self.patch_size, self.patch_size):
                    continue

                # Check if patch is too close to edge
                is_edge = (y < edge_margin or
                          x < edge_margin or
                          y + self.patch_size > H - edge_margin or
                          x + self.patch_size > W - edge_margin)

                patches.append(patch)
                positions.append((y, x))
                valid_mask.append(not is_edge)

        print(f"Extracted {len(patches)} patches ({sum(valid_mask)} valid, {len(patches)-sum(valid_mask)} edge)")

        return {
            'patches': patches,
            'positions': positions,
            'valid_mask': valid_mask,
            'shape': (H, W)
        }

    def extract_patch_labels(self, ground_truth: np.ndarray, positions: List[Tuple[int, int]],
                            threshold: float = 0.1) -> np.ndarray:
        """
        Extract ground truth labels for patches

        Args:
            ground_truth: 2D array (H, W) with 0=normal, 1=dent
            positions: List of (y, x) positions from extract_patches
            threshold: Minimum fraction of dent pixels to label patch as "dent"

        Returns:
            labels: Array of 0/1 labels for each patch
        """
        labels = []

        for (y, x) in positions:
            patch_gt = ground_truth[y:y+self.patch_size, x:x+self.patch_size]

            # Calculate fraction of dent pixels
            dent_fraction = patch_gt.sum() / (self.patch_size * self.patch_size)

            # Label as dent if fraction exceeds threshold
            label = 1 if dent_fraction > threshold else 0
            labels.append(label)

        return np.array(labels)


class PatchReconstructor:
    """Reconstruct full segmentation mask from patch predictions"""

    def __init__(self, patch_size: int = 64, stride: int = 32):
        self.patch_size = patch_size
        self.stride = stride

    def reconstruct_from_patches(self, patch_labels: np.ndarray,
                                 positions: List[Tuple[int, int]],
                                 original_shape: Tuple[int, int],
                                 valid_mask: np.ndarray = None) -> np.ndarray:
        """
        Reconstruct full segmentation mask from patch predictions

        Args:
            patch_labels: Array of 0/1 labels for each patch
            positions: List of (y, x) top-left positions
            original_shape: (H, W) of original image
            valid_mask: If provided, only use valid patches

        Returns:
            segmentation_mask: (H, W) array with 0=normal, 1=dent
        """
        H, W = original_shape

        # Use voting: accumulate predictions and counts
        vote_sum = np.zeros((H, W), dtype=np.float32)
        vote_count = np.zeros((H, W), dtype=np.int32)

        for idx, (y, x) in enumerate(positions):
            # Skip invalid patches
            if valid_mask is not None and not valid_mask[idx]:
                continue

            # Add this patch's vote
            vote_sum[y:y+self.patch_size, x:x+self.patch_size] += patch_labels[idx]
            vote_count[y:y+self.patch_size, x:x+self.patch_size] += 1

        # Calculate average vote (handles overlaps)
        segmentation_mask = np.zeros((H, W), dtype=np.uint8)
        valid_pixels = vote_count > 0
        segmentation_mask[valid_pixels] = (vote_sum[valid_pixels] / vote_count[valid_pixels] > 0.5).astype(np.uint8)

        return segmentation_mask


def visualize_patches(residuals: np.ndarray, patch_info: Dict, num_samples: int = 9):
    """
    Visualize a grid of extracted patches

    Args:
        residuals: Original residual map
        patch_info: Output from PatchExtractor.extract_patches()
        num_samples: Number of patches to display
    """
    import matplotlib.pyplot as plt

    patches = patch_info['patches']
    positions = patch_info['positions']
    valid_mask = patch_info['valid_mask']

    # Sample patches
    indices = np.random.choice(len(patches), min(num_samples, len(patches)), replace=False)

    # Create subplot grid
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        patch = patches[idx]
        pos = positions[idx]
        is_valid = valid_mask[idx]

        axes[i].imshow(patch, cmap='RdBu_r', vmin=-3, vmax=3)
        axes[i].set_title(f"Pos: {pos}\n{'Valid' if is_valid else 'Edge'}", fontsize=8)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.suptitle(f'Sample Patches (patch_size={patch_info["patches"][0].shape[0]})',
                 fontsize=14, y=1.0)
    plt.show()


if __name__ == "__main__":
    # Test patch extraction
    print("Testing patch extraction...")

    # Create dummy data
    H, W = 640, 960
    dummy_residuals = np.random.randn(H, W)

    # Extract patches
    extractor = PatchExtractor(patch_size=64, stride=32, ignore_edge_patches=True)
    patch_info = extractor.extract_patches(dummy_residuals)

    print(f"Shape: {patch_info['shape']}")
    print(f"Total patches: {len(patch_info['patches'])}")
    print(f"Valid patches: {sum(patch_info['valid_mask'])}")

    # Test reconstruction
    reconstructor = PatchReconstructor(patch_size=64, stride=32)
    dummy_labels = np.random.randint(0, 2, len(patch_info['patches']))
    reconstructed = reconstructor.reconstruct_from_patches(
        dummy_labels,
        patch_info['positions'],
        patch_info['shape'],
        patch_info['valid_mask']
    )

    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Dent pixels: {reconstructed.sum()} ({reconstructed.sum()/reconstructed.size*100:.2f}%)")
    print("\n✓ Patch utilities working correctly!")
