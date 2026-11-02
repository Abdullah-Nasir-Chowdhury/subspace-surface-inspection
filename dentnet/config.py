"""
Configuration file for DentNet
Based on the paper specifications
"""

# Input/Output Configuration
NUM_INPUT_FEATURES = 1   # Single channel: residual map
NUM_OUTPUT_FEATURES = 2  # Binary classification: dent/non-dent

# Dataset Configuration - REAL SCALE (matching A.ply)
W_PIXELS = 760   # Match A.ply
H_PIXELS = 640   # Match A.ply
W_METRIC = 142.46  # mm (A.ply X range)
H_METRIC = 117.50  # mm (A.ply Y range)

# Training Configuration
BATCH_SIZE = 4  # Reduced due to OOM with larger images (760x640 vs 960x640)
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

# Data paths
# DATASET_PATH = "D:/pointcloud db"  # Original (with gt_mask preprocessing)
# DATASET_PATH = "D:/pointcloud_db_no_gtmask"  # No gt_mask (960x640 scale)
DATASET_PATH = "D:/pointcloud_db_real_scale"  # Real scale (760x640, matching A.ply)