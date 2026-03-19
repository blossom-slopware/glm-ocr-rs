"""Full Rust model wrapper — replaces the ENTIRE Python Model, not just language_model.

The mlx_vlm server calls:
  1. model.get_input_embeddings(input_ids, pixel_values, **kwargs) → InputEmbeddingsFeatures
     This must run vision encoder + embedding merge IN RUST.
  2. model.language_model(inputs, inputs_embeds=, cache=, **kwargs) → LanguageModelOutput
     This must run the LM forward IN RUST.

Both are routed through PyFullModel which owns vision + LM + merge in Rust.
"""

import numpy as np
import mlx.core as mx
import mlx_ocr_2


class _DummyCache:
    """Tracks offset for the server's cache management. Actual KV lives in Rust."""
    def __init__(self):
        self.offset = 0
        self.keys = None
        self.values = None

    def update_and_fetch(self, keys, values):
        self.offset += keys.shape[2]
        return keys, values

    @property
    def state(self):
        return mx.zeros((1,)), mx.zeros((1,))

    @state.setter
    def state(self, v):
        pass

    def is_trimmable(self):
        return False

    def trim(self, n):
        return 0


class _FakeTextModel:
    """Provides .layers for make_prompt_cache."""
    def __init__(self, num_layers):
        self.layers = [None] * num_layers


class RustLanguageModel:
    """Drop-in for model.language_model — routes LM calls to Rust PyFullModel."""

    def __init__(self, rust_full_model, original_language_model):
        self.rust_model = rust_full_model
        self.args = original_language_model.args
        self.config = original_language_model.config
        self.model_type = original_language_model.model_type
        self.model = _FakeTextModel(self.args.num_hidden_layers)
        self._rope_deltas = None
        self._position_ids = None
        self._original_lm = original_language_model

    def make_cache(self):
        return [_DummyCache() for _ in range(self.args.num_hidden_layers)]

    @property
    def layers(self):
        return self.model.layers

    def get_rope_index(self, *args, **kwargs):
        return self._original_lm.get_rope_index(*args, **kwargs)

    def __call__(
        self,
        inputs,
        inputs_embeds=None,
        mask=None,
        cache=None,
        **kwargs,
    ):
        from mlx_vlm.models.base import LanguageModelOutput

        position_ids = kwargs.pop("position_ids", None)
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        if pixel_values is not None:
            self._rope_deltas = None

        cache_offset = 0
        if cache and cache[0] is not None:
            offset = cache[0].offset
            if isinstance(offset, int):
                cache_offset = offset
            elif isinstance(offset, mx.array):
                cache_offset = (offset if offset.ndim == 0 else offset[0]).item()
            else:
                raise ValueError(f"Unexpected cache offset type: {type(offset)}")

        rope_mask = mask
        if mask is not None and mask.shape[-1] != inputs.shape[-1]:
            rope_mask = None

        if position_ids is None and (rope_mask is None or rope_mask.ndim == 2):
            if (
                (cache is not None and cache[0] is not None and (cache_offset == 0))
                or self._rope_deltas is None
                or cache is None
            ):
                if self._position_ids is not None:
                    seq_length = inputs.shape[1]
                    position_ids = self._position_ids[
                        :, :, cache_offset : cache_offset + seq_length
                    ]
                else:
                    position_ids, rope_deltas = self.get_rope_index(
                        inputs, image_grid_thw, video_grid_thw, rope_mask
                    )
                    self._rope_deltas = rope_deltas
                    self._position_ids = position_ids
            else:
                batch_size, seq_length = inputs.shape
                delta = mx.array(
                    cache_offset + self._rope_deltas if cache is not None else 0
                )
                position_ids = mx.arange(seq_length).reshape(1, -1)
                position_ids = mx.broadcast_to(position_ids, (batch_size, seq_length))

                if cache_offset is not None:
                    if delta.ndim == 0:
                        delta = mx.expand_dims(delta, axis=0)
                    if delta.shape[0] < batch_size:
                        delta = mx.tile(delta, (batch_size, 1))
                    else:
                        delta = delta[:batch_size]

                position_ids = mx.add(position_ids, delta)[None, ...]
                position_ids = mx.broadcast_to(
                    position_ids, (3, batch_size, seq_length)
                )

        mx.eval(position_ids)
        pos_np = np.array(position_ids.astype(mx.int32))
        pos_flat = pos_np.flatten().astype(np.int32)
        pos_shape = list(pos_np.shape)

        is_prefill = (inputs_embeds is not None) or (inputs.shape[1] > 1 and cache_offset == 0)

        if inputs_embeds is not None:
            mx.eval(inputs_embeds)
            embeds_f32 = np.array(inputs_embeds.astype(mx.float32))
            embeds_flat = embeds_f32.flatten().astype(np.float32)
            embeds_shape = list(embeds_f32.shape)

            if is_prefill and cache_offset == 0:
                logits_flat, logits_shape = self.rust_model.prefill_embeds(
                    embeds_flat, embeds_shape, pos_flat, pos_shape
                )
            else:
                logits_flat, logits_shape = self.rust_model.continue_prefill_embeds(
                    embeds_flat, embeds_shape, pos_flat, pos_shape
                )
        else:
            mx.eval(inputs)
            tok_np = np.array(inputs.astype(mx.int32))
            tok_flat = tok_np.flatten().astype(np.int32)
            tok_shape = list(tok_np.shape)

            if is_prefill and cache_offset == 0:
                logits_flat, logits_shape = self.rust_model.prefill(
                    tok_flat, tok_shape, pos_flat, pos_shape
                )
            else:
                logits_flat, logits_shape = self.rust_model.decode(
                    tok_flat, tok_shape, pos_flat, pos_shape
                )

        logits_np = np.array(logits_flat).reshape(logits_shape)
        logits_mx = mx.array(logits_np)

        seq_len = inputs_embeds.shape[1] if inputs_embeds is not None else inputs.shape[1]
        if cache:
            for c in cache:
                if c is not None:
                    c.offset += seq_len

        return LanguageModelOutput(logits=logits_mx)

    def parameters(self):
        return {}

    def sanitize(self, weights):
        return weights

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads


class RustFullModel:
    """Drop-in replacement for the ENTIRE Python Model class.

    Routes get_input_embeddings() through Rust vision+merge,
    and language_model() through Rust LM.
    """

    def __init__(self, rust_full_model, original_model):
        self.rust_model = rust_full_model
        self.config = original_model.config
        # Replace language_model with Rust-backed wrapper
        self.language_model = RustLanguageModel(rust_full_model, original_model.language_model)

    @property
    def layers(self):
        return self.language_model.layers

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        """Run vision encoder + merge IN RUST, return InputEmbeddingsFeatures."""
        from mlx_vlm.models.base import InputEmbeddingsFeatures

        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            # Text-only: reset position state, return text embeddings
            self.language_model._position_ids = None
            self.language_model._rope_deltas = None
            # We need to get embeddings from Rust — call forward with just input_ids
            mx.eval(input_ids)
            tok_np = np.array(input_ids.astype(mx.int32))
            tok_flat = tok_np.flatten().astype(np.int32)
            tok_shape = list(tok_np.shape)
            embeds_flat, embeds_shape = self.rust_model.get_text_embeddings(
                tok_flat, tok_shape
            )
            embeds_np = np.array(embeds_flat).reshape(embeds_shape)
            inputs_embeds = mx.array(embeds_np)
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        # Vision path: run full vision+merge pipeline in Rust
        mx.eval(input_ids)
        tok_np = np.array(input_ids.astype(mx.int32))
        tok_flat = tok_np.flatten().astype(np.int32)
        tok_shape = list(tok_np.shape)

        mx.eval(pixel_values)
        pv_np = np.array(pixel_values.astype(mx.float32))
        pv_flat = pv_np.flatten().astype(np.float32)
        pv_shape = list(pv_np.shape)

        mx.eval(grid_thw)
        grid_np = np.array(grid_thw.astype(mx.int32))
        grid_flat = grid_np.flatten().astype(np.int32)
        grid_shape = list(grid_np.shape)

        embeds_flat, embeds_shape = self.rust_model.get_input_embeddings(
            tok_flat, tok_shape, pv_flat, pv_shape, grid_flat, grid_shape
        )
        embeds_np = np.array(embeds_flat).reshape(embeds_shape)
        inputs_embeds = mx.array(embeds_np)

        # Pre-calculate position_ids (like Python Model.get_input_embeddings)
        if image_grid_thw is not None or video_grid_thw is not None:
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, mask
            )
            self.language_model._position_ids = position_ids
            self.language_model._rope_deltas = rope_deltas

        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def sanitize(self, weights):
        return weights

    def parameters(self):
        return {}
