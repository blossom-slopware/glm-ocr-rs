#!/usr/bin/env python3
"""End-to-end OCR test with a real image."""
import sys
sys.path.insert(0, "/Users/daisy/develop/GLM-OCR")

import numpy as np
import mlx.core as mx
from mlx_vlm import load as mlx_vlm_load
from PIL import Image
import mlx_ocr_2

MODEL_DIR = "/Users/daisy/develop/GLM-OCR/models/GLM-OCR-bf16"

print("Loading models...")
py_model, processor = mlx_vlm_load(MODEL_DIR, lazy=False)
rust_vision = mlx_ocr_2.PyVisionModel.load(MODEL_DIR)

# Create a simple test image with text
img = Image.new('RGB', (224, 56), color=(255, 255, 255))
from PIL import ImageDraw
draw = ImageDraw.Draw(img)
draw.text((10, 10), "Hello World 123", fill=(0, 0, 0))

# Process through Python
print("\nProcessing image through Python pipeline...")
# Use the processor to get pixel_values and grid_thw
processed = processor.image_processor([img], return_tensors="np")
pixel_values_np = processed['pixel_values']
grid_thw_np = processed['image_grid_thw']
print(f"pixel_values shape: {pixel_values_np.shape}")
print(f"grid_thw: {grid_thw_np}")

# Python vision forward
mx_pv = mx.array(pixel_values_np)
mx_grid = mx.array(grid_thw_np)
py_features = py_model.vision_tower(mx_pv, grid_thw=mx_grid)
mx.eval(py_features)
py_f_np = np.array(py_features.astype(mx.float32))
print(f"Python features: shape={py_f_np.shape}, range=[{py_f_np.min():.4f}, {py_f_np.max():.4f}]")

# Rust vision forward
pv_flat = pixel_values_np.astype(np.float32).flatten()
gt_flat = grid_thw_np.astype(np.int32).flatten()
rust_out, rust_shape = rust_vision.forward(
    pv_flat, list(pixel_values_np.shape),
    gt_flat, list(grid_thw_np.shape)
)
rust_f_np = np.array(rust_out).reshape(rust_shape)
print(f"Rust features:   shape={rust_f_np.shape}, range=[{rust_f_np.min():.4f}, {rust_f_np.max():.4f}]")

diff = np.abs(py_f_np - rust_f_np)
cos_sim = np.dot(py_f_np.flatten(), rust_f_np.flatten()) / (
    np.linalg.norm(py_f_np.flatten()) * np.linalg.norm(rust_f_np.flatten()))
print(f"\nMax diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")
print(f"Cosine similarity: {cos_sim:.6f}")
