"""Minimal open3d shim backed by plyfile.

open3d publishes no aarch64/py3.12 wheel, so this file sits in the repo root
and satisfies `import open3d as o3d` for the API surface this repo
actually uses: PLY point-cloud read/write with optional colors.
"""
import os, sys  # noqa: E401
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("dentnet", "subspace", "common", "pointnetpp", "synthetic", "mvtec", "real3d", "figures"):
    if os.path.join(_ROOT, _d) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _d))
import numpy as np
from plyfile import PlyData, PlyElement


def Vector3dVector(arr):
    return np.asarray(arr, dtype=np.float64)


class PointCloud:
    def __init__(self):
        self.points = np.zeros((0, 3))
        self.colors = np.zeros((0, 3))

    def has_colors(self):
        return len(self.colors) > 0

    def has_points(self):
        return len(self.points) > 0


class _IO:
    @staticmethod
    def read_point_cloud(path):
        ply = PlyData.read(str(path))
        v = ply["vertex"]
        pcd = PointCloud()
        pcd.points = np.column_stack(
            [v["x"], v["y"], v["z"]]).astype(np.float64)
        names = v.data.dtype.names or ()
        if {"red", "green", "blue"} <= set(names):
            pcd.colors = np.column_stack(
                [v["red"], v["green"], v["blue"]]).astype(np.float64) / 255.0
        return pcd

    @staticmethod
    def write_point_cloud(path, pcd):
        pts = np.asarray(pcd.points, dtype=np.float32)
        cols = np.asarray(pcd.colors)
        if len(cols) == len(pts) and len(pts) > 0:
            rgb = np.clip(cols * 255.0, 0, 255).astype(np.uint8)
            data = np.empty(len(pts), dtype=[
                ("x", "f4"), ("y", "f4"), ("z", "f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1")])
            data["red"] = rgb[:, 0]
            data["green"] = rgb[:, 1]
            data["blue"] = rgb[:, 2]
        else:
            data = np.empty(
                len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        data["x"] = pts[:, 0]
        data["y"] = pts[:, 1]
        data["z"] = pts[:, 2]
        PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))
        return True


class _Geometry:
    PointCloud = PointCloud


class _Utility:
    Vector3dVector = staticmethod(Vector3dVector)


io = _IO()
geometry = _Geometry()
utility = _Utility()
