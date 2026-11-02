"""
Subspace Builder
================
Build dent and normal subspaces from training data using PCA
"""

import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import numpy as np
import pickle
import os
from sklearn.decomposition import PCA
from typing import Dict, Tuple
from tqdm import tqdm


class SubspaceBuilder:
    """Build and save subspaces for dent and normal patches"""

    def __init__(self, n_components: int = 50):
        """
        Args:
            n_components: Number of principal components for each subspace
        """
        self.n_components = n_components
        self.dent_subspace = None
        self.normal_subspace = None
        self.dent_mean = None
        self.normal_mean = None

    def build_subspaces(self, dent_features: np.ndarray, normal_features: np.ndarray):
        """
        Build subspaces from feature vectors

        Args:
            dent_features: (num_dent_patches, feature_dim) array
            normal_features: (num_normal_patches, feature_dim) array
        """
        print(f"\nBuilding subspaces...")
        print(f"  Dent patches: {len(dent_features)}")
        print(f"  Normal patches: {len(normal_features)}")

        # Build dent subspace
        print(f"  Building dent subspace (n_components={self.n_components})...")
        self.dent_mean = np.mean(dent_features, axis=0)
        dent_centered = dent_features - self.dent_mean

        dent_pca = PCA(n_components=self.n_components)
        dent_pca.fit(dent_centered)

        self.dent_subspace = {
            'components': dent_pca.components_,  # (n_components, feature_dim)
            'explained_variance': dent_pca.explained_variance_ratio_.sum(),
            'mean': self.dent_mean
        }

        # Build normal subspace
        print(f"  Building normal subspace (n_components={self.n_components})...")
        self.normal_mean = np.mean(normal_features, axis=0)
        normal_centered = normal_features - self.normal_mean

        normal_pca = PCA(n_components=self.n_components)
        normal_pca.fit(normal_centered)

        self.normal_subspace = {
            'components': normal_pca.components_,  # (n_components, feature_dim)
            'explained_variance': normal_pca.explained_variance_ratio_.sum(),
            'mean': self.normal_mean
        }

        print(f"\n✓ Subspaces built:")
        print(f"  Dent subspace: {self.n_components} components, "
              f"explained variance: {self.dent_subspace['explained_variance']:.3f}")
        print(f"  Normal subspace: {self.n_components} components, "
              f"explained variance: {self.normal_subspace['explained_variance']:.3f}")

    def save(self, filepath: str):
        """Save subspaces to file"""
        data = {
            'dent_subspace': self.dent_subspace,
            'normal_subspace': self.normal_subspace,
            'n_components': self.n_components
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"✓ Subspaces saved to: {filepath}")

    def load(self, filepath: str):
        """Load subspaces from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.dent_subspace = data['dent_subspace']
        self.normal_subspace = data['normal_subspace']
        self.n_components = data['n_components']
        self.dent_mean = self.dent_subspace['mean']
        self.normal_mean = self.normal_subspace['mean']

        print(f"✓ Subspaces loaded from: {filepath}")

    def project_to_subspace(self, features: np.ndarray, subspace_dict: Dict) -> np.ndarray:
        """
        Project features onto a subspace

        Args:
            features: (feature_dim,) or (num_samples, feature_dim)
            subspace_dict: Dictionary with 'components' and 'mean'

        Returns:
            coefficients: Projection coefficients
        """
        # Handle single vector or batch
        if features.ndim == 1:
            features = features.reshape(1, -1)
            single = True
        else:
            single = False

        # Center features
        centered = features - subspace_dict['mean']

        # Project onto subspace basis
        components = subspace_dict['components']  # (n_components, feature_dim)
        coefficients = np.dot(centered, components.T)  # (num_samples, n_components)

        if single:
            coefficients = coefficients.flatten()

        return coefficients

    def reconstruction_error(self, features: np.ndarray, subspace_dict: Dict) -> float:
        """
        Calculate reconstruction error (distance from subspace)

        Args:
            features: (feature_dim,) or (num_samples, feature_dim)
            subspace_dict: Dictionary with 'components' and 'mean'

        Returns:
            error: Reconstruction error (distance from subspace)
        """
        # Handle single vector or batch
        if features.ndim == 1:
            features = features.reshape(1, -1)
            single = True
        else:
            single = False

        # Project to subspace
        coefficients = self.project_to_subspace(features, subspace_dict)

        # Reconstruct from subspace
        components = subspace_dict['components']
        mean = subspace_dict['mean']

        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)

        reconstructed = np.dot(coefficients, components) + mean

        # Calculate error
        error = np.linalg.norm(features - reconstructed, axis=1)

        if single:
            error = error[0]

        return error

    def classify_features(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify features as dent or normal based on subspace distance

        Args:
            features: (num_samples, feature_dim) or (feature_dim,)

        Returns:
            predictions: Array of 0 (normal) or 1 (dent)
            confidence: Distance difference (positive = confident dent, negative = confident normal)
        """
        # Calculate distances to both subspaces
        dist_to_dent = self.reconstruction_error(features, self.dent_subspace)
        dist_to_normal = self.reconstruction_error(features, self.normal_subspace)

        # Classify: closer to dent subspace → dent (1), closer to normal → normal (0)
        predictions = (dist_to_dent < dist_to_normal).astype(np.int32)

        # Confidence: difference in distances
        confidence = dist_to_normal - dist_to_dent

        return predictions, confidence


def build_subspaces_from_dataset(dataset_path: str,
                                 feature_extractor,
                                 patch_extractor,
                                 num_samples: int = 1000,
                                 n_components: int = 50,
                                 save_path: str = 'subspaces.pkl') -> SubspaceBuilder:
    """
    Build subspaces from synthetic training dataset

    Args:
        dataset_path: Path to "D:/pointcloud db"
        feature_extractor: DentNetFeatureExtractor instance
        patch_extractor: PatchExtractor instance
        num_samples: Number of training samples to use
        n_components: Number of PCA components
        save_path: Where to save subspaces

    Returns:
        SubspaceBuilder instance with built subspaces
    """
    print(f"\n{'='*70}")
    print(f"Building Subspaces from Training Data")
    print(f"{'='*70}\n")

    dent_features_list = []
    normal_features_list = []

    # Use validation set (last 20%)
    val_start = int(30000 * 0.8)
    sample_ids = range(val_start, val_start + num_samples)

    print(f"Processing {num_samples} samples (IDs: {val_start} to {val_start + num_samples - 1})...")

    for sample_id in tqdm(sample_ids, desc="Extracting features"):
        try:
            # Load sample
            base_path = os.path.join(dataset_path, f"{sample_id:05d}")
            residuals = np.load(f"{base_path}_input.npy")
            ground_truth = np.load(f"{base_path}_gt.npy")

            # Extract patches
            patch_info = patch_extractor.extract_patches(residuals)
            patches = patch_info['patches']
            positions = patch_info['positions']
            valid_mask = patch_info['valid_mask']

            # Get patch labels
            patch_labels = patch_extractor.extract_patch_labels(ground_truth, positions)

            # Filter to valid patches only
            valid_patches = [p for i, p in enumerate(patches) if valid_mask[i]]
            valid_labels = patch_labels[valid_mask]

            if len(valid_patches) == 0:
                continue

            # Extract features
            features = feature_extractor.extract_features_batch(valid_patches)

            # Separate by label
            dent_mask = valid_labels == 1
            normal_mask = valid_labels == 0

            if dent_mask.sum() > 0:
                dent_features_list.append(features[dent_mask])
            if normal_mask.sum() > 0:
                normal_features_list.append(features[normal_mask])

        except Exception as e:
            print(f"  Warning: Failed to process sample {sample_id}: {e}")
            continue

    # Concatenate all features
    dent_features = np.vstack(dent_features_list)
    normal_features = np.vstack(normal_features_list)

    print(f"\nCollected features:")
    print(f"  Dent patches: {len(dent_features)}")
    print(f"  Normal patches: {len(normal_features)}")

    # Build subspaces
    builder = SubspaceBuilder(n_components=n_components)
    builder.build_subspaces(dent_features, normal_features)

    # Save
    builder.save(save_path)

    print(f"\n{'='*70}")
    print(f"Subspace building complete!")
    print(f"{'='*70}\n")

    return builder


if __name__ == "__main__":
    print("Subspace builder module loaded.")
    print("Use build_subspaces_from_dataset() to build subspaces from training data.")
