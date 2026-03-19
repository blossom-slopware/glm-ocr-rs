"""Test prefill/decode with KV cache: Rust vs Python."""

import sys

sys.path.insert(0, "/Users/daisy/develop/GLM-OCR/rust/mlx-vlm")

import numpy as np
import mlx.core as mx
from mlx_vlm.models.glm_ocr.language import LanguageModel, TextConfig
import json
import mlx_ocr_2

MODEL_DIR = "/Users/daisy/develop/GLM-OCR/models/GLM-OCR-bf16"


def load_python_model():
    with open(f"{MODEL_DIR}/config.json") as f:
        raw = json.load(f)
    text_cfg = TextConfig.from_dict(raw.get("text_config", raw))
    model = LanguageModel(text_cfg)
    from mlx.utils import tree_unflatten
    from mlx_vlm.utils import load_model as _lm

    # Load weights
    import glob

    weight_files = sorted(glob.glob(f"{MODEL_DIR}/model*.safetensors"))
    weights = {}
    for p in weight_files:
        w = mx.load(p)
        for k, v in w.items():
            if k.startswith("language_model."):
                k = k[len("language_model.") :]
            weights[k] = v
    # Filter to only language model weights
    lang_weights = {k: v for k, v in weights.items() if not k.startswith("vision_tower.")}
    model.load_weights(list(lang_weights.items()))
    mx.eval(model.parameters())
    return model, text_cfg


def test_prefill_decode():
    py_model, text_cfg = load_python_model()
    rust_model = mlx_ocr_2.PyGlmOcrModel.load(MODEL_DIR)

    # Test tokens
    token_ids = [1, 100, 200, 300, 400, 500]
    seq_len = len(token_ids)
    batch = 1

    # Position ids: [3, batch, seq_len] — simple sequential
    pos = np.arange(seq_len, dtype=np.int32)
    position_ids_np = np.stack([pos, pos, pos]).reshape(3, 1, seq_len)
    position_ids_mx = mx.array(position_ids_np)

    # === PREFILL (full sequence) ===
    tokens_mx = mx.array(np.array(token_ids, dtype=np.int32).reshape(1, seq_len))

    # Python prefill with cache
    from mlx_lm.models.cache import KVCache

    py_cache = [KVCache() for _ in range(text_cfg.num_hidden_layers)]
    py_out = py_model(tokens_mx, position_ids=position_ids_mx, cache=py_cache)
    py_logits = py_out.logits
    mx.eval(py_logits)
    # py_cache is now populated

    # Rust prefill
    tokens_flat = np.array(token_ids, dtype=np.int32)
    pos_flat = position_ids_np.flatten().astype(np.int32)
    rust_logits_flat, rust_shape = rust_model.prefill(
        tokens_flat,
        [1, seq_len],
        pos_flat,
        [3, 1, seq_len],
    )
    rust_logits = np.array(rust_logits_flat).reshape(rust_shape)
    py_logits_np = np.array(py_logits.astype(mx.float32))

    prefill_diff = np.max(np.abs(rust_logits - py_logits_np))
    print(f"Prefill max diff: {prefill_diff}")
    assert prefill_diff < 1e-2, f"Prefill diff too large: {prefill_diff}"

    # === DECODE (token by token) ===
    # Use the last token's logits to pick next token (or just use a fixed one)
    next_tokens = [600, 700, 800]

    for step, next_tok in enumerate(next_tokens):
        step_pos = seq_len + step
        # Position ids for single token: [3, 1, 1]
        pos_single = np.array([step_pos, step_pos, step_pos], dtype=np.int32).reshape(3, 1, 1)
        pos_single_mx = mx.array(pos_single)

        tok_mx = mx.array(np.array([[next_tok]], dtype=np.int32))

        py_out = py_model(tok_mx, position_ids=pos_single_mx, cache=py_cache)
        py_logits = py_out.logits
        mx.eval(py_logits)

        # Rust decode
        tok_flat = np.array([next_tok], dtype=np.int32)
        pos_flat = pos_single.flatten().astype(np.int32)
        rust_logits_flat, rust_shape = rust_model.decode(
            tok_flat,
            [1, 1],
            pos_flat,
            [3, 1, 1],
        )
        rust_logits = np.array(rust_logits_flat).reshape(rust_shape)
        py_logits_np = np.array(py_logits.astype(mx.float32))

        decode_diff = np.max(np.abs(rust_logits - py_logits_np))
        print(f"Decode step {step} (token={next_tok}, pos={step_pos}) max diff: {decode_diff}")
        assert decode_diff < 1e-2, f"Decode step {step} diff too large: {decode_diff}"

    # === CONSISTENCY CHECK ===
    # Full forward on entire sequence (Python, no cache) should match
    # full forward (Rust, no cache)
    all_tokens = token_ids + next_tokens
    all_len = len(all_tokens)
    all_pos = np.arange(all_len, dtype=np.int32)
    all_pos_ids = np.stack([all_pos, all_pos, all_pos]).reshape(3, 1, all_len)

    all_tokens_flat = np.array(all_tokens, dtype=np.int32)
    all_pos_flat = all_pos_ids.flatten().astype(np.int32)
    rust_full_flat, rust_full_shape = rust_model.forward_with_shape(
        all_tokens_flat,
        [1, all_len],
        all_pos_flat,
        [3, 1, all_len],
    )
    rust_full = np.array(rust_full_flat).reshape(rust_full_shape)

    all_tokens_mx = mx.array(all_tokens_flat.reshape(1, all_len))
    all_pos_mx = mx.array(all_pos_ids)
    py_full_out = py_model(all_tokens_mx, position_ids=all_pos_mx, cache=[None] * text_cfg.num_hidden_layers)
    py_full = np.array(py_full_out.logits.astype(mx.float32))

    full_diff = np.max(np.abs(rust_full - py_full))
    print(f"Full forward consistency max diff: {full_diff}")
    # Note: Rust wraps MLX C++ v0.30.6 while Python uses MLX v0.31.1.
    # Different MLX versions have different Metal sdpa kernel implementations,
    # which causes small numerical differences (< 1.0) for longer sequences.
    assert full_diff < 1e-2, f"Full forward diff too large: {full_diff}"

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_prefill_decode()
