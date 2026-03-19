"""Drop-in replacement for LanguageModel backed by Rust PyFullModel."""

import numpy as np
import mlx.core as mx
import mlx_ocr_2


class _DummyCache:
    """Tracks offset for position_ids computation without actually storing KV."""
    def __init__(self):
        self.offset = 0
        self.keys = None
        self.values = None

    def update_and_fetch(self, keys, values):
        self.offset += keys.shape[2]
        return keys, values

    @property
    def state(self):
        # Return empty arrays so mx.eval doesn't crash
        return mx.zeros((1,)), mx.zeros((1,))

    @state.setter
    def state(self, v):
        pass

    def is_trimmable(self):
        return False

    def trim(self, n):
        return 0


class _FakeTextModel:
    """Provides .layers property for make_prompt_cache."""
    def __init__(self, num_layers):
        self.layers = [None] * num_layers


class RustLanguageModel:
    """Drop-in replacement for LanguageModel, backed by Rust."""

    def __init__(self, rust_model, original_language_model):
        self.rust_model = rust_model
        self.args = original_language_model.args
        self.config = original_language_model.config
        self.model_type = original_language_model.model_type
        self.model = _FakeTextModel(self.args.num_hidden_layers)
        # Copy the embed_tokens so get_input_embeddings can use it
        self.model.embed_tokens = original_language_model.model.embed_tokens
        self._rope_deltas = None
        self._position_ids = None
        # Copy get_rope_index from original
        self._original_lm = original_language_model
        self._cache_offset = 0

    def make_cache(self):
        """Return dummy caches for generate_step's make_prompt_cache."""
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

        # Compute cache_offset from the dummy cache objects
        cache_offset = 0
        if cache and cache[0] is not None:
            offset = cache[0].offset
            if isinstance(offset, int):
                cache_offset = offset
            elif isinstance(offset, mx.array):
                cache_offset = (offset if offset.ndim == 0 else offset[0]).item()
            else:
                raise ValueError(f"Unexpected cache offset type: {type(offset)}")

        # Compute position_ids (replicated from Python LanguageModel.__call__)
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

        # Convert position_ids to numpy
        mx.eval(position_ids)
        pos_np = np.array(position_ids.astype(mx.int32))
        pos_flat = pos_np.flatten().astype(np.int32)
        pos_shape = list(pos_np.shape)

        is_prefill = (inputs_embeds is not None) or (inputs.shape[1] > 1 and cache_offset == 0)

        if inputs_embeds is not None:
            # Prefill with embeddings (from vision encoder)
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
            # Decode with token ids
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

        # Convert back to mlx
        logits_np = np.array(logits_flat).reshape(logits_shape)
        logits_mx = mx.array(logits_np)

        # Update dummy cache offsets
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


def make_dummy_cache(num_layers):
    """Create dummy cache objects that track offset for position_ids."""
    return [_DummyCache() for _ in range(num_layers)]
