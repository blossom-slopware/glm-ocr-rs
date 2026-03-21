use mlx_rs::error::Exception;
use mlx_rs::ops::{self, concatenate_axis, indexing::{Ellipsis, IndexMutOp, IndexOp}};
use mlx_rs::Array;
use mlx_lm::cache::KeyValueCache;

/// Pre-allocated KV Cache that grows in fixed-size steps.
///
/// Mirrors Python mlx-lm's `KVCache`: pre-allocates buffers in chunks of `STEP`
/// (256) slots along the sequence axis, then writes new keys/values via slice
/// updates. This avoids the O(n²) copying of `ConcatKeyValueCache`.
const STEP: i32 = 256;

#[derive(Debug, Clone)]
pub struct KVCache {
    keys: Array,
    values: Array,
    offset: i32,
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            keys: Array::from_f32(0.0),
            values: Array::from_f32(0.0),
            offset: 0,
        }
    }

    pub fn keys(&self) -> &Array {
        &self.keys
    }

    pub fn values(&self) -> &Array {
        &self.values
    }
}

impl KeyValueCache for KVCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let prev = self.offset;
        let new_seq_len = keys.shape()[keys.ndim() - 2];

        // Check if we need to grow the buffer
        let need_grow = if self.keys.ndim() == 0 {
            true
        } else {
            (prev + new_seq_len) > self.keys.shape()[self.keys.ndim() - 2]
        };

        if need_grow {
            let k_shape = keys.shape();
            let v_shape = values.shape();
            let k_ndim = k_shape.len();
            let v_ndim = v_shape.len();

            // Allocate n_steps * STEP new slots (ceil division)
            let n_steps = (STEP + new_seq_len - 1) / STEP;
            let new_slots = n_steps * STEP;

            // Build shape for new zero buffer: same as keys but with new_slots in seq dim
            let mut new_k_shape: Vec<i32> = k_shape.to_vec();
            new_k_shape[k_ndim - 2] = new_slots;
            let mut new_v_shape: Vec<i32> = v_shape.to_vec();
            new_v_shape[v_ndim - 2] = new_slots;

            let new_k = ops::zeros_dtype(&new_k_shape, keys.dtype())?;
            let new_v = ops::zeros_dtype(&new_v_shape, values.dtype())?;

            if self.keys.ndim() != 0 {
                // Trim existing buffer to actual content if needed
                let old_k = std::mem::replace(&mut self.keys, Array::from_f32(0.0));
                let old_v = std::mem::replace(&mut self.values, Array::from_f32(0.0));
                let old_cap = old_k.shape()[old_k.ndim() - 2];
                let trimmed_k = if prev != old_cap {
                    old_k.index((Ellipsis, ..prev, ..))
                } else {
                    old_k
                };
                let trimmed_v = if prev != old_cap {
                    old_v.index((Ellipsis, ..prev, ..))
                } else {
                    old_v
                };
                self.keys = concatenate_axis(&[trimmed_k, new_k], -2)?;
                self.values = concatenate_axis(&[trimmed_v, new_v], -2)?;
            } else {
                self.keys = new_k;
                self.values = new_v;
            }
        }

        // Write new keys/values into the buffer at [prev..prev+new_seq_len]
        self.offset = prev + new_seq_len;
        let end = self.offset;

        self.keys.index_mut(
            (Ellipsis, prev..end, ..),
            &keys,
        );
        self.values.index_mut(
            (Ellipsis, prev..end, ..),
            &values,
        );

        // Return the filled portion
        Ok((
            self.keys.index((Ellipsis, ..end, ..)),
            self.values.index((Ellipsis, ..end, ..)),
        ))
    }
}
