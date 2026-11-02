"""PointNet++ patch classifier on residual-height windows (set abstraction 32-32-64, then
a global stage 64-128-128, two-layer head). The grouping radius is a constructor argument
and is selected on held-out training samples (radius_sweep.py)."""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnpp_layers import PointNetSetAbstraction  # noqa: E402

NORM_RADIUS = 2.0


class PatchClassifier(nn.Module):
    def __init__(self, norm_radius=NORM_RADIUS):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(
            npoint=32, radius=norm_radius, nsample=16, in_channel=3,
            mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=64 + 3,
            mlp=[64, 128, 128], group_all=True)
        self.fc1 = nn.Linear(128, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, xyz):
        l1_xyz, l1_pts = self.sa1(xyz, None)
        _, l2_pts = self.sa2(l1_xyz, l1_pts)
        x = l2_pts.view(l2_pts.shape[0], -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)
