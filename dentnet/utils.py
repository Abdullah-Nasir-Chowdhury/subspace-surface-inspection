"""
Utility functions for DentNet
"""
import torch
import torch.nn as nn
import numpy as np

class PatchConv2d(nn.Module):
    """
    Patch-based convolution (same as regular Conv2d for this implementation)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        return self.conv(x)

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_iou(pred, target, threshold=0.5):
    """Compute Intersection over Union (IoU) for binary segmentation"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()