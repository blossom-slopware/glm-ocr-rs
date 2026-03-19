# GLM-OCR Rust Port: Status & Lessons Learned

## What Has Been Done

### Language Model (Complete)

The full GLM-OCR language model is ported to Rust via `mlx-rs` and verified numerically against the Python reference implementation.

**Ported components:**
- `config.rs` — `TextConfig` deserialized from `config.json` (rope parameters, mrope sections, etc.)
- `model/mrope.rs` — Multimodal 3D Rotary Position Embedding (M-RoPE) with `mrope_section=[16,24,24]`, interleaved rotation style
- `model/mlp.rs` — Fused SwiGLU MLP (`gate_up_proj` → split → silu(gate)*up → `down_proj`)
- `model/attention.rs` — Grouped Query Attention (16 query heads, 8 KV heads, head_dim=128) with KV cache support
- `model/decoder.rs` — 4-RMSNorm sandwich decoder layer (unique to this model: input_layernorm → attn → post_self_attn_layernorm → residual → post_attention_layernorm → mlp → post_mlp_layernorm → residual)
- `model/text_model.rs` — Transformer stack: embed_tokens → rotary_emb → 16 decoder layers → final norm. Supports both `forward_with_positions` (token ids) and `forward_with_embeds` (pre-computed embeddings from vision encoder)
- `model/language_model.rs` — `LanguageModel` (text_model + lm_head) and `GlmOcrModel` (outer wrapper matching safetensors key prefix `language_model.`)
- `loader.rs` — Safetensors weight loading (handles both single-file and sharded via `model.safetensors.index.json`)
- `py_bindings.rs` — PyO3 interface: `load`, `prefill`, `decode`, `prefill_embeds`, `continue_prefill_embeds`, `cache_offset`, `num_layers`

**Python integration layer:**
- `rust_lm_wrapper.py` — `RustLanguageModel` drop-in replacement for Python's `LanguageModel`, handles position_ids computation and prefill/decode routing to Rust via numpy bridge
- `run_server.py` — Server launcher that monkey-patches `mlx_vlm.server.load_model_resources` to swap in the Rust language model while keeping the Python vision encoder

### Vision Encoder (Complete)

The full vision pipeline is ported to Rust via `mlx-rs` and verified numerically against the Python reference (cosine similarity ~0.999).

**Ported components:**
- `vision/config.rs` — `VisionConfig` deserialized from `config.json` with serde defaults for missing fields
- `vision/patch_embed.rs` — Conv3d patch embedding (temporal_patch_size=2, patch_size=14, 3→1024)
- `vision/rotary.rs` — 1D RoPE for vision (dim=head_dim//2=32, theta=10000), `rotate_half`, `apply_rotary_pos_emb_vision`
- `vision/attention.rs` — Fused QKV + per-head RMSNorm + RoPE + per-image SDPA via cu_seqlens split
- `vision/mlp.rs` — SwiGLU with bias (gate_proj, up_proj, down_proj)
- `vision/block.rs` — Pre-norm residual (norm1→attn, norm2→mlp)
- `vision/merger.rs` — Linear→LayerNorm→GELU→SwiGLU (LayerNorm, not RmsNorm)
- `vision/vision_model.rs` — Full pipeline: patch_embed → rotary → 24 blocks → post_layernorm → Conv2d downsample → merger, plus `VisionTower` wrapper for key prefix matching
- `vision/mod.rs` — Module declarations

### Config (Complete)

`FullConfig` has both `text_config` and `vision_config`, plus `image_token_id`.

### Weight Sanitization (Complete)

`VisionModel::sanitize_weights()` conditionally transposes Conv3d/Conv2d weights from PyTorch layout to MLX channels-last layout by checking shape dimensions. Called after `load_safetensors` in `load_vision_model`.

### Verification

If numerical difference in a reasonable range, run end2end tests, and compare the quality of output.

---

## What Remains To Do

Nothing — the full model is implemented in Rust.

### Image-Text Embedding Merge (Complete)

`full_model.rs` — `Model::merge_input_ids_with_image_features` replaces `<image>` token positions in text embeddings with vision features. Uses `ops::which` for conditional selection and `ops::cumsum` for index mapping.

### Full Model Wrapper (Complete)

`full_model.rs` — `Model` struct combining `VisionTower` + `GlmOcrModel` + embedding merge + `get_rope_index` for M-RoPE position_ids computation. Methods: `get_input_embeddings`, `compute_position_ids`, `forward`.

### Sampling (Complete)

`sampler.rs` — Temperature, top-p (nucleus), top-k, min-p, repetition penalty. Uses `ops::argsort_axis`, `ops::cumsum`, `ops::argpartition_axis`, `take_along_axis`/`put_along_axis` for filtering.

### Generate Loop (Complete)

`generate.rs` — Chunked prefill (configurable `prefill_step_size`) + autoregressive decode loop with EOS/stop token checking. Returns generated token IDs.

### PyO3 Interface (Complete)

`py_bindings.rs` — Single `PyFullModel` class exposing: `load`, `forward`, `generate`, `prefill`, `decode`, `prefill_embeds`, `continue_prefill_embeds`, `reset_cache`, `cache_offset`, `num_layers`, `eos_token_ids`.

---

## Lessons Learned

### 1. mlx-rs API Discovery

**Always read mlx-rs source code for APIs.** The Rust bindings don't always mirror the Python MLX API 1:1.

- `StrideBy::new()` doesn't exist. Use `(..).stride_by(2)` via the `IntoStrideBy` trait from `mlx_rs::ops::indexing`.
- `eval(&[&arr])` has type issues. Use `arr.eval()?` method instead.
- `Array` contains `*mut c_void` (not Send/Sync). Any PyO3 class holding MLX arrays must use `#[pyclass(unsendable)]`.
- `nn::Rope` is not suitable for M-RoPE. The 3D multimodal position embedding with section selection must be implemented from scratch.
- `scaled_dot_product_attention` mask can be `ScaledDotProductAttentionMask::Causal` (built-in) or `ScaledDotProductAttentionMask::Array` (explicit). Using `Causal` matches Python's `mask="causal"` behavior.

**Do NOT coin nonexistent APIs**.

### 2. Weight Key Matching

Struct hierarchy must mirror the safetensors key prefixes exactly. Wrapping the model as `GlmOcrModel { language_model: LanguageModel { model: GlmOcrTextModel { layers: [...] } } }` automatically matches keys like `language_model.model.layers.0.self_attn.q_proj.weight`. No manual remapping needed.

### 3. Metal Compatibility

macOS 26 (Tahoe) reports Metal Language Version 4.0 support, but the runtime metal compiler in MLX v0.30.6 only supports up to 3.2. The fix is a `PATCH_COMMAND` in `mlx-c/CMakeLists.txt` to sed-replace `LanguageVersion4_0` → `LanguageVersion3_2` in the fetched mlx C++ source.

### 4. mlx-c / mlx C++ Version Compatibility

The `mlx-c` wrapper and `mlx` C++ library must be version-matched. The `mlx-c` submodule at HEAD targets mlx v0.31.1 but `CMakeLists.txt` fetches v0.30.6. The v0.31.1 `mlx-c` calls `quantize`/`dequantize` with a `global_scale` parameter that doesn't exist in v0.30.6. **Solution: checkout `mlx-c` to commit `786d1a2` (the v0.30.6-matching version), then re-apply the Metal 3.2 patch.**

After cleaning `target/` the build will need a full from-scratch compilation of mlx C++ (~5 minutes). Do not casually `rm -rf target/`.

### 6. KV Cache Initialization

`ConcatKeyValueCache::new()` creates an empty but initialized cache. Cache entries must be `Some(ConcatKeyValueCache::new())` for prefill/decode to work. Using `None` entries means `update_and_fetch` is never called and KV pairs are never stored.

### 7. Build Environment

- `maturin develop --release` is required (not `cargo build`) because PyO3 cdylib needs Python symbols.
- If both `CONDA_PREFIX` and `VIRTUAL_ENV` are set, maturin refuses to build. `unset CONDA_PREFIX` first.
- The virtual env is at `/Users/daisy/develop/GLM-OCR/.venv-mlx`.

### 8. Causal Mask: String vs Array

Python MLX's `scaled_dot_product_attention` accepts `mask="causal"` (string hint for built-in Metal causal masking). Passing an explicit additive causal mask array produces the same result for small sequences but can diverge for larger ones due to different Metal kernel code paths. In Rust, use `ScaledDotProductAttentionMask::Causal` when offset=0 and seq_len>1 (prefill without prior cache). Use explicit `Array` mask only when there's a cache offset (chunked prefill or continuation).

### 9. Python Integration Architecture

For the server integration, the cleanest approach is:
- Keep the Python `Model` class (vision tower + `get_input_embeddings`) intact
- Replace only `model.language_model` with a `RustLanguageModel` wrapper
- The wrapper replicates position_ids/rope_delta computation in Python, then calls Rust for the actual transformer forward
- Dummy `_DummyCache` objects track offset for position_ids computation; actual KV storage is in Rust's `ConcatKeyValueCache`
- The wrapper must implement `make_cache()` to return dummy caches (checked by `make_prompt_cache`)

### 10. Vision PatchEmbed Reshape Order

The pixel_values input `[N, 1176]` has data packed as `C*T*H*W = 3*2*14*14`. You must reshape to `[N, C, T, H, W]` first (PyTorch layout), then `transpose_axes([0, 2, 3, 4, 1])` to get channels-last `[N, T, H, W, C]` for MLX Conv3d. Directly reshaping to `[N, T, H, W, C]` reinterprets the data incorrectly because the channel dimension is packed contiguously in the original data.

### 11. mlx-rs Builder APIs Take Positional Args

`Conv3dBuilder::new(in_channels, out_channels, kernel_size)` and `Conv2dBuilder::new(in_channels, out_channels, kernel_size)` take required arguments positionally in the constructor. Optional settings like `.stride()`, `.bias()` are chained. Do NOT use `Conv3dBuilder::new().input_channels(...)`.

### 12. mlx-rs Type Patterns

- `expand_dims(0)` takes a single `i32`, not a slice. NOT `expand_dims(&[0])`.
- `as_dtype(Dtype::Float32)` uses the `Dtype` enum. NOT `as_dtype::<f32>()`.
- `Array::from_f32(val)` for scalar creation. NOT `Array::from_float(val)`.
- `nn::gelu(&x)` is the public path (re-exported from private `activation` module). NOT `nn::activation::gelu(&x)`.
- `LayerNorm::new(dim)` is the constructor. NOT `LayerNormBuilder::new().dimensions(d)`.

### 13. VisionTower Wrapper for Key Prefix Matching

Wrap `VisionModel` in a `VisionTower` struct with a field named `vision_tower` to automatically match safetensors key prefix `vision_tower.*`. Same pattern as `GlmOcrModel { language_model: ... }`.

### 14. Variable-Length Attention via cu_seqlens Split

mlx-rs has no native cu_seqlens support in SDPA. Implement by computing segment lengths from cu_seqlens, using `ops::split_sections` to split q/k/v along the sequence axis, running `scaled_dot_product_attention` per segment (no mask needed — each segment gets full attention), then `ops::concatenate_axis` to reassemble.

### 15. Conv Weight Sanitization is Conditional

The bf16 safetensors may already have Conv weights in MLX channels-last layout. Check shape to determine if transpose is needed: for Conv3d, if `shape[1] == in_channels` it's PyTorch layout; for Conv2d, if `shape[1] > shape[2]` it's PyTorch layout.

