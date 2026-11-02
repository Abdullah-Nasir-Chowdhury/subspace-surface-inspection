"""
Feature Extraction Using DentNet Encoder
=========================================
Extract deep features from patches using trained DentNet encoder
"""

import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import List
import sys
import os
from model import DentNet


class DentNetFeatureExtractor:
    """Extract features using DentNet encoder (without decoder)"""

    def __init__(self, model_path: str = 'best_dentnet.pth', device: str = None):
        """
        Args:
            model_path: Path to trained DentNet model
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading DentNet encoder on device: {self.device}")

        # Load full model
        model = DentNet().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle DataParallel (remove "module." prefix)
        if isinstance(checkpoint, dict) and any(k.startswith('module.') for k in checkpoint.keys()):
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            checkpoint = new_state_dict

        model.load_state_dict(checkpoint)
        model.eval()

        # Extract encoder only (we don't need decoder for feature extraction)
        self.encoder = self._build_encoder_only(model)
        self.encoder.eval()

        print(f"✓ Feature extractor ready")

    def _build_encoder_only(self, full_model: DentNet) -> nn.Module:
        """
        Extract just the encoder layers from DentNet
        We'll use features from the bottleneck (deepest layer)
        """
        class EncoderOnly(nn.Module):
            def __init__(self, full_model):
                super().__init__()
                # Copy encoder layers
                self.encoder_convIn0 = full_model.encoder_convIn0
                self.encoder_downConv1A = full_model.encoder_downConv1A
                self.encoder_conv1B = full_model.encoder_conv1B
                self.encoder_downConv2A = full_model.encoder_downConv2A
                self.encoder_conv2B = full_model.encoder_conv2B
                self.encoder_downConv3A = full_model.encoder_downConv3A
                self.encoder_conv3B = full_model.encoder_conv3B
                self.encoder_downConv4A = full_model.encoder_downConv4A
                self.encoder_conv4B = full_model.encoder_conv4B

            def forward(self, x):
                """
                Forward pass through encoder
                Returns: Bottleneck features (1024 channels at H/16 × W/16)
                """
                x = self.encoder_convIn0(x)
                x = self.encoder_downConv1A(x)
                x = self.encoder_conv1B(x)
                x = self.encoder_downConv2A(x)
                x = self.encoder_conv2B(x)
                x = self.encoder_downConv3A(x)
                x = self.encoder_conv3B(x)
                x = self.encoder_downConv4A(x)
                x = self.encoder_conv4B(x)
                return x  # Bottleneck features

        return EncoderOnly(full_model)

    def extract_features_from_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Extract features from a single patch

        Args:
            patch: (patch_size, patch_size) numpy array

        Returns:
            features: 1D feature vector
        """
        # Convert to tensor
        patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
        patch_tensor = patch_tensor.to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.encoder(patch_tensor)  # (1, 1024, H/16, W/16)

            # Global average pooling to get fixed-size vector
            features = torch.mean(features, dim=[2, 3])  # (1, 1024)

            features = features.cpu().numpy().flatten()

        return features

    def extract_features_batch(self, patches: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """
        Extract features from multiple patches (batched for efficiency)

        Args:
            patches: List of (patch_size, patch_size) numpy arrays
            batch_size: Number of patches to process at once

        Returns:
            features: (num_patches, feature_dim) array
        """
        all_features = []

        # Process in batches
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size]

            # Stack into batch tensor
            batch_tensor = torch.stack([
                torch.from_numpy(p).float() for p in batch_patches
            ]).unsqueeze(1)  # (batch, 1, H, W)

            batch_tensor = batch_tensor.to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.encoder(batch_tensor)  # (batch, 1024, H/16, W/16)

                # Global average pooling
                features = torch.mean(features, dim=[2, 3])  # (batch, 1024)

                features = features.cpu().numpy()

            all_features.append(features)

        # Concatenate all batches
        all_features = np.vstack(all_features)

        return all_features


if __name__ == "__main__":
    # Test feature extraction
    print("Testing feature extraction...")

    # Create dummy patch
    dummy_patch = np.random.randn(64, 64).astype(np.float32)

    # Initialize extractor
    extractor = DentNetFeatureExtractor(model_path='best_dentnet.pth')

    # Extract features from single patch
    features = extractor.extract_features_from_patch(dummy_patch)
    print(f"Single patch features shape: {features.shape}")

    # Extract features from batch
    dummy_patches = [np.random.randn(64, 64).astype(np.float32) for _ in range(10)]
    features_batch = extractor.extract_features_batch(dummy_patches)
    print(f"Batch features shape: {features_batch.shape}")

    print("\n✓ Feature extraction working correctly!")
