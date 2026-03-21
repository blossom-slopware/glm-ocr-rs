"""Test logit equivalence between Python MLX and Rust mlx-rs implementations."""
import sys
import os
import json
import numpy as np

# Add mlx-vlm to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mlx-vlm"))

import mlx.core as mx
import mlx.nn as nn

MODEL_DIR = "/Users/daisy/develop/GLM-OCR/models/GLM-OCR-bf16"


def load_python_model():
    """Load the Python MLX language model."""
    from mlx_vlm.models.glm_ocr.language import LanguageModel
    from mlx_vlm.models.glm_ocr.config import TextConfig

    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)

    text_cfg = TextConfig(**config["text_config"])
    model = LanguageModel(text_cfg)

    # Load weights - only language_model weights
    weight_files = []
    index_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        files = set(index["weight_map"].values())
        weight_files = [os.path.join(MODEL_DIR, f) for f in files]
    else:
        weight_files = [os.path.join(MODEL_DIR, "model.safetensors")]

    weights = {}
    for wf in weight_files:
        w = mx.load(wf)
        for k, v in w.items():
            if k.startswith("language_model."):
                # Strip "language_model." prefix for the LanguageModel
                new_key = k[len("language_model."):]
                weights[new_key] = v

    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model


def test_logit_equivalence():
    print("Loading Python model...")
    py_model = load_python_model()

    print("Loading Rust model...")
    import mlx_ocr_2
    rs_model = mlx_ocr_2.PyGlmOcrModel.load(MODEL_DIR)

    # Create test input
    token_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)  # [1, 5]
    position_ids = np.zeros((3, 1, 5), dtype=np.int32)
    for i in range(3):
        position_ids[i, 0, :] = np.arange(5)

    # Python forward
    print("Running Python forward...")
    tok_mx = mx.array(token_ids)
    pos_mx = mx.array(position_ids)
    py_out = py_model(tok_mx, position_ids=pos_mx)
    py_logits = np.array(py_out.logits.astype(mx.float32))  # [1, 5, vocab_size]
    print(f"  Python logits shape: {py_logits.shape}")
    print(f"  Python logits[:5]: {py_logits[0, -1, :5]}")

    # Rust forward
    print("Running Rust forward...")
    flat_tokens = token_ids.flatten()
    flat_pos = position_ids.flatten()
    rs_logits_flat, rs_shape = rs_model.forward_with_shape(
        flat_tokens, list(token_ids.shape),
        flat_pos, list(position_ids.shape),
    )
    rs_logits = np.array(rs_logits_flat).reshape(rs_shape)
    print(f"  Rust logits shape: {rs_logits.shape}")
    print(f"  Rust logits[:5]: {rs_logits[0, -1, :5]}")

    # Compare
    max_diff = np.abs(rs_logits - py_logits).max()
    mean_diff = np.abs(rs_logits - py_logits).mean()
    print(f"\n  Max abs diff: {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")

    np.testing.assert_allclose(rs_logits, py_logits, rtol=1e-2, atol=1e-2)
    print("\n✓ Logits match within tolerance!")


if __name__ == "__main__":
    test_logit_equivalence()
