use mlx_rs::error::Exception;
use mlx_rs::ops::{self, concatenate_axis, indexing::{Ellipsis, IndexMutOp, IndexOp}};
use mlx_rs::Array;

/// Trait for key-value cache implementations used in attention layers.
///
/// This trait mirrors the Python mlx-lm's `KVCache` interface.
pub trait KeyValueCache {
    fn offset(&self) -> i32;

    fn max_size(&self) -> Option<i32>;

    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<(Array, Array), Exception>;
}

impl<T> KeyValueCache for &'_ mut T
where
    T: KeyValueCache,
{
    fn offset(&self) -> i32 {
        T::offset(self)
    }

    fn max_size(&self) -> Option<i32> {
        T::max_size(self)
    }

    fn update_and_fetch(&mut self, keys: Array, values: Array) -> Result<(Array, Array), Exception> {
        T::update_and_fetch(self, keys, values)
    }
}

/// Pre-allocated KV Cache that grows in fixed-size steps.
///
/// Mirrors Python mlx-lm's `KVCache`: pre-allocates buffers in chunks of `STEP`
/// (256) slots along the sequence axis, then writes new keys/values via slice
/// updates. This avoids the O(n²) copying of `ConcatKeyValueCache`.
const STEP: i32 = 256;

#[derive(Debug, Clone)]
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
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
            keys: None,
            values: None,
            offset: 0,
        }
    }

    pub fn keys(&self) -> Option<&Array> {
        self.keys.as_ref()
    }

    pub fn values(&self) -> Option<&Array> {
        self.values.as_ref()
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
        let need_grow = match &self.keys {
            None => true,
            Some(k) => (prev + new_seq_len) > k.shape()[k.ndim() - 2],
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

            if let (Some(old_k), Some(old_v)) = (self.keys.take(), self.values.take()) {
                // Trim existing buffer to actual content if needed
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
                self.keys = Some(concatenate_axis(&[trimmed_k, new_k], -2)?);
                self.values = Some(concatenate_axis(&[trimmed_v, new_v], -2)?);
            } else {
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
        }

        // Write new keys/values into the buffer at [prev..prev+new_seq_len]
        self.offset = prev + new_seq_len;
        let end = self.offset;

        self.keys.as_mut().unwrap().index_mut(
            (Ellipsis, prev..end, ..),
            &keys,
        );
        self.values.as_mut().unwrap().index_mut(
            (Ellipsis, prev..end, ..),
            &values,
        );

        // Return the filled portion
        Ok((
            self.keys.as_ref().unwrap().index((Ellipsis, ..end, ..)),
            self.values.as_ref().unwrap().index((Ellipsis, ..end, ..)),
        ))
    }
}
