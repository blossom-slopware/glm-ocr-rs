#!/usr/bin/env python3
"""Test Rust vision encoder against Python reference implementation."""

import sys
sys.path.insert(0, "/Users/daisy/develop/GLM-OCR")

import numpy as np
import mlx.core as mx
import mlx_ocr_2

MODEL_DIR = "/Users/daisy/develop/GLM-OCR/models/GLM-OCR-bf16"

def load_python_vision():
    """Load the Python vision model."""
    from transformers import AutoProcessor
    import mlx_vlm

    model, processor = mlx_vlm.load(MODEL_DIR, lazy=False)
    return model, processor

def test_vision_forward():
    print("Loading Python model...")
    py_model, processor = load_python_vision()

    print("Loading Rust vision model...")
    rust_vision = mlx_ocr_2.PyVisionModel.load(MODEL_DIR)

    # Create a small test image (synthetic)
    # Use grid_thw that's compatible: h and w must be divisible by spatial_merge_size=2
    # Let's use a small grid: t=1, h=4, w=4 -> 16 patches -> 4 merged tokens
    t, h, w = 1, 4, 4
    n_patches = t * h * w  # 16
    patch_dim = 3 * 2 * 14 * 14  # 1176

    # Random pixel values
    np.random.seed(42)
    pixel_values = np.random.randn(n_patches, patch_dim).astype(np.float32)
    grid_thw = np.array([[t, h, w]], dtype=np.int32)

    # Python forward
    print("Running Python vision forward...")
    mx_pixel_values = mx.array(pixel_values)
    mx_grid_thw = mx.array(grid_thw)
    py_output = py_model.vision_tower(mx_pixel_values, grid_thw=mx_grid_thw)
    mx.eval(py_output)
    py_out_np = np.array(py_output)
    print(f"  Python output shape: {py_out_np.shape}, dtype: {py_out_np.dtype}")
    print(f"  Python output range: [{py_out_np.min():.4f}, {py_out_np.max():.4f}]")

    # Rust forward
    print("Running Rust vision forward...")
    pv_flat = pixel_values.flatten()
    pv_shape = list(pixel_values.shape)
    gt_flat = grid_thw.flatten()
    gt_shape = list(grid_thw.shape)

    rust_out_flat, rust_out_shape = rust_vision.forward(pv_flat, pv_shape, gt_flat, gt_shape)
    rust_out_np = np.array(rust_out_flat).reshape(rust_out_shape)
    print(f"  Rust output shape: {rust_out_np.shape}")
    print(f"  Rust output range: [{rust_out_np.min():.4f}, {rust_out_np.max():.4f}]")

    # Compare
    diff = np.abs(py_out_np.astype(np.float32) - rust_out_np)
    print(f"\n  Max absolute diff: {diff.max():.6f}")
    print(f"  Mean absolute diff: {diff.mean():.6f}")

    if diff.max() < 0.2:
        print("\n✅ PASS: Rust vision output matches Python within tolerance")
    else:
        print("\n❌ FAIL: Outputs differ too much")
        return False

    return True

if __name__ == "__main__":
    success = test_vision_forward()
    sys.exit(0 if success else 1)
