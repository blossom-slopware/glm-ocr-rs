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

`FullConfig` has `text_config`, `vision_config`, `image_token_id`, `video_token_id`, `image_start_token_id`, `eos_token_id`.

### Weight Sanitization (Complete)

`VisionModel::sanitize_weights()` conditionally transposes Conv3d/Conv2d weights from PyTorch layout to MLX channels-last layout by checking shape dimensions. Called after `load_safetensors` in `load_vision_model`.

### Image-Text Embedding Merge (Complete)

`full_model.rs` — `Model::merge_input_ids_with_image_features` replaces `<image>` token positions in text embeddings with vision features. Uses `ops::which` for conditional selection and `ops::cumsum` for index mapping.

### Full Model Wrapper (Complete)

`full_model.rs` — `Model` struct combining `VisionTower` + `GlmOcrModel` + embedding merge + `get_rope_index` for M-RoPE position_ids computation. Methods: `get_input_embeddings`, `compute_position_ids`, `forward`.

### Sampling (Complete)

`sampler.rs` — Temperature, top-p (nucleus), top-k, min-p, repetition penalty. Uses `ops::argsort_axis`, `ops::cumsum`, `ops::argpartition_axis`, `take_along_axis`/`put_along_axis` for filtering.

### Generate Loop (Complete)

`generate.rs` — Chunked prefill (configurable `prefill_step_size`) + autoregressive decode loop with EOS/stop token checking. Returns generated token IDs.

### PyO3 Interface (Complete)

`py_bindings.rs` — Single `PyFullModel` class exposing: `load`, `forward`, `generate`, `get_text_embeddings`, `get_input_embeddings`, `prefill`, `decode`, `prefill_embeds`, `continue_prefill_embeds`, `reset_cache`, `cache_offset`, `num_layers`, `eos_token_ids`.

### Python Integration (Complete — Rust powers entire model)

`run_server.py` replaces the **entire** Python `Model` object with `RustFullModel`, which routes:
- `model.get_input_embeddings()` → Rust vision encoder + embedding merge via `PyFullModel.get_input_embeddings()`
- `model.language_model(...)` → Rust LM forward via `PyFullModel.prefill/decode/prefill_embeds`

Verified via `RUST_LOG=info`: all vision blocks (24), patch_embed, downsample, merger, and LM run in Rust. Zero Python-side model computation.

## What Remains To Do

### Eliminate Python Entirely

The final goal is a standalone Rust binary with no Python dependency. Remaining Python-side work:

1. **Image preprocessing** — Python's `processor` (from `transformers`) handles image loading, resizing, padding, pixel normalization, and `grid_thw` computation. Port to Rust (use `image` crate for loading, reimplement the Qwen2VL image processor logic).

2. **Tokenization** — Python's `processor.tokenizer` handles chat template formatting, BPE tokenization, and special token insertion (`<image_start>`, `<image>`, etc.). Port to Rust (use `tokenizers` crate with the model's `tokenizer.json`).

3. **Detokenization + streaming** — Token IDs → text decoding, incremental detokenization for streaming output. The `tokenizers` crate handles this.

4. **HTTP server** — Replace `mlx_vlm.server` (FastAPI/uvicorn) with a Rust HTTP server (e.g., `axum`) serving the OpenAI-compatible `/chat/completions` endpoint.

5. **Chat template** — The Jinja2 chat template in `tokenizer_config.json` needs to be replicated in Rust for formatting multi-turn conversations with image placeholders.

Once these are done, the entire pipeline runs as a single Rust binary: HTTP request → image preprocess → tokenize → vision encoder → merge → LM → sample → detokenize → HTTP response.

## Verification

Run end2end tests with `test_ocr.py` and compare the quality of output.

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

**THE SERVER (`mlx_vlm`) CALLS `model.get_input_embeddings()` THEN `model.language_model(...)`. TO REPLACE THE FULL MODEL WITH RUST, YOU MUST REPLACE THE ENTIRE `model` OBJECT, NOT JUST `model.language_model`. IF YOU ONLY SWAP `model.language_model`, THE PYTHON VISION ENCODER AND EMBEDDING MERGE STILL RUN IN PYTHON AND YOUR RUST VISION/MERGE CODE IS DEAD CODE.**

The correct approach:
- Replace the entire `model` with `RustFullModel` which implements both `get_input_embeddings()` (routing to Rust vision + merge) and `.language_model(...)` (routing to Rust LM)
- `RustFullModel.get_input_embeddings()` calls `PyFullModel.get_input_embeddings()` for vision+merge, or `PyFullModel.get_text_embeddings()` for text-only
- `RustFullModel.language_model` is a `RustLanguageModel` that calls `PyFullModel.prefill/decode/prefill_embeds`
- Dummy `_DummyCache` objects track offset for position_ids computation; actual KV storage is in Rust's `ConcatKeyValueCache`
- The wrapper must satisfy `mlx_vlm` server contract: `model.config`, `model.get_input_embeddings(input_ids, pixel_values, **kwargs) -> InputEmbeddingsFeatures`, `model.language_model(inputs, inputs_embeds=, cache=, **kwargs) -> LanguageModelOutput`

### 10. Vision PatchEmbed Reshape Order

The pixel_values input `[N, 1176]` has data packed as `C*T*H*W = 3*2*14*14`. You must reshape to `[N, C, T, H, W]` first (PyTorch layout), then `transpose_axes([0, 2, 3, 4, 1])` to get channels-last `[N, T, H, W, C]` for MLX Conv3d. Directly reshaping to `[N, T, H, W, C]` reinterprets the data incorrectly because the channel dimension is packed contiguously in the original data.

### 11. mlx-rs Builder APIs Take Positional Args

`Conv3dBuilder::new(in_channels, out_channels, kernel_size)` and `Conv2dBuilder::new(in_channels, out_channels, kernel_size)` take required arguments positionally in the constructor. Optional settings like `.stride()`, `.bias()` are chained. Do NOT use `Conv3dBuilder::new().input_channels(...)`.

### 12. mlx-rs Type Patterns

- `expand_dims(0)` takes a single `i32`, not a slice. NOT `expand_dims(&[0])`.
- `as_dtype(Dtype::Float32)` uses the `Dtype` enum. NOT `as_dtype::<f32>()`.
- `Array::from_f32(val)` for scalar creation. NOT `Array::from_float(val)`.
- `Array::from_int(val)` for i32 scalar. NOT `Array::from_i32(val)`.
- `nn::gelu(&x)` is the public path (re-exported from private `activation` module). NOT `nn::activation::gelu(&x)`.
- `LayerNorm::new(dim)` is the constructor. NOT `LayerNormBuilder::new().dimensions(d)`.

### 13. VisionTower Wrapper for Key Prefix Matching

Wrap `VisionModel` in a `VisionTower` struct with a field named `vision_tower` to automatically match safetensors key prefix `vision_tower.*`. Same pattern as `GlmOcrModel { language_model: ... }`.

### 14. Variable-Length Attention via cu_seqlens Split

mlx-rs has no native cu_seqlens support in SDPA. Implement by computing segment lengths from cu_seqlens, using `ops::split_sections` to split q/k/v along the sequence axis, running `scaled_dot_product_attention` per segment (no mask needed — each segment gets full attention), then `ops::concatenate_axis` to reassemble.

### 15. Conv Weight Sanitization is Conditional

The bf16 safetensors may already have Conv weights in MLX channels-last layout. Check shape to determine if transpose is needed: for Conv3d, if `shape[1] == in_channels` it's PyTorch layout; for Conv2d, if `shape[1] > shape[2]` it's PyTorch layout.

### 16. mlx-rs Generated Macro Signatures

The `#[generate_macro]` + `#[default_device]` pattern strips the `stream` parameter. So free functions like `ops::sum(array, keep_dims)`, `ops::max(array, keep_dims)`, `ops::cumsum(a, axis, reverse, inclusive)` — NOT the `_device` variants. `ops::which(cond, a, b)` is the correct conditional select (NOT `ops::r#where` which requires stream). `argmax_axis` lives in `ops::indexing`, not `ops`.
