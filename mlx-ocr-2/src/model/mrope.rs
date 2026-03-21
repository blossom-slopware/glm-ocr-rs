use mlx_rs::{error::Exception, ops, ops::indexing::{IndexOp, IntoStrideBy, Ellipsis}, Array};

use crate::config::TextConfig;

/// Multimodal Rotary Position Embedding (M-RoPE).
/// No learned parameters — just precomputed inv_freq.
#[derive(Debug, Clone)]
pub struct GlmOcrRotaryEmbedding {
    inv_freq: Array,
    mrope_section: Vec<i32>,
    attention_scaling: f32,
}

impl GlmOcrRotaryEmbedding {
    pub fn new(config: &TextConfig) -> Result<Self, Exception> {
        let base = config.rope_theta();
        let partial_rotary_factor = config.partial_rotary_factor();
        let head_dim = config.head_dim;
        let dim = (head_dim as f32 * partial_rotary_factor) as i32;

        // inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))
        let arange = ops::arange::<_, f32>(0, dim, 2)?;
        let inv_freq = arange.divide(Array::from_f32(dim as f32))?.power(Array::from_f32(1.0))?;
        // base ** x = exp(x * ln(base))
        let ln_base = Array::from_f32(base.ln());
        let inv_freq = ops::negative(inv_freq.multiply(ln_base)?)?;
        let inv_freq = ops::exp(inv_freq)?;

        let mrope_section = config.mrope_section();
        let attention_scaling = 1.0;

        Ok(Self {
            inv_freq,
            mrope_section,
            attention_scaling,
        })
    }

    /// position_ids: shape [3, batch, seq_len]
    /// Returns (cos, sin) each shape [batch, seq_len, head_dim]
    pub fn forward(&self, x: &Array, position_ids: &Array) -> Result<(Array, Array), Exception> {
        let _batch = position_ids.dim(1);
        let d = self.inv_freq.dim(0); // head_dim / 2

        // inv_freq_expanded: [1, 1, D, 1] -> broadcast to [3, batch, D, 1]
        let inv_freq_expanded = self
            .inv_freq
            .reshape(&[1, 1, d, 1])?;

        // position_ids_expanded: [3, batch, 1, seq_len]
        let position_ids_f32 = position_ids.as_dtype(mlx_rs::Dtype::Float32)?;
        let position_ids_expanded = ops::expand_dims(position_ids_f32, 2)?;

        // freqs = inv_freq_expanded @ position_ids_expanded -> [3, batch, D, seq_len]
        // then transpose to [3, batch, seq_len, D]
        let freqs = ops::matmul(inv_freq_expanded, position_ids_expanded)?
            .transpose_axes(&[0, 1, 3, 2])?;

        // Apply M-RoPE section selection
        let freqs = self.apply_mrope(&freqs)?;

        // emb = [freqs, freqs] along last axis -> [batch, seq_len, 2*D] = [batch, seq_len, head_dim]
        let emb = ops::concatenate_axis(&[&freqs, &freqs], -1)?;
        let cos = ops::cos(&emb)?.multiply(Array::from_f32(self.attention_scaling))?;
        let sin = ops::sin(&emb)?.multiply(Array::from_f32(self.attention_scaling))?;

        let dtype = x.dtype();
        Ok((cos.as_dtype(dtype)?, sin.as_dtype(dtype)?))
    }

    /// Apply M-RoPE by selecting different dimensions for T, H, W.
    /// freqs shape: [3, batch, seq_len, D]
    /// mrope_section = [16, 24, 24], sum = 64 = D
    /// Result shape: [batch, seq_len, D]
    fn apply_mrope(&self, freqs: &Array) -> Result<Array, Exception> {
        // Compute split indices (cumsum of section without last)
        let mut split_indices = Vec::new();
        let mut cumsum = 0i32;
        for (i, &s) in self.mrope_section.iter().enumerate() {
            cumsum += s;
            if i < self.mrope_section.len() - 1 {
                split_indices.push(cumsum);
            }
        }

        // Split freqs along last axis
        let chunks = freqs.split_axis(&split_indices, -1)?;

        // For each chunk i, select chunk[i % 3] along dim 0
        let mut selected = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let idx = (i % 3) as i32;
            // chunk shape: [3, batch, seq_len, section_size]
            // Select idx along dim 0: chunk[idx]
            let s = chunk.index(idx);
            selected.push(s);
        }

        // Concatenate along last axis -> [batch, seq_len, D]
        let refs: Vec<&Array> = selected.iter().collect();
        ops::concatenate_axis(&refs, -1)
    }
}

/// Rotate half using the interleaved LLM style.
/// x shape: [..., head_dim]
/// x1 = x[..., 0::2], x2 = x[..., 1::2]
/// return interleave([-x2, x1])
pub fn rotate_half_llm(x: &Array) -> Result<Array, Exception> {
    let shape = x.shape().to_vec();
    let last_dim = *shape.last().unwrap();

    let x1 = x.index((Ellipsis, (..).stride_by(2)));
    let x2 = x.index((Ellipsis, (1..).stride_by(2)));

    let neg_x2 = ops::negative(&x2)?;

    // stack [-x2, x1] on new last axis -> [..., half_dim, 2], then reshape to [..., head_dim]
    let stacked = ops::stack_axis(&[&neg_x2, &x1], -1)?;
    let mut new_shape = shape[..shape.len() - 1].to_vec();
    new_shape.push(last_dim);
    stacked.reshape(&new_shape)
}

/// repeat_interleave: [a,b,c] with repeats=2 -> [a,a,b,b,c,c] along last axis.
/// Uses stack+reshape (2 ops) instead of expand_dims+tile+reshape (3 ops).
pub fn repeat_interleave_last(x: &Array) -> Result<Array, Exception> {
    let shape = x.shape().to_vec();
    let ndim = shape.len();
    // stack x with itself on a new last axis: [..., n] -> [..., n, 2]
    let stacked = ops::stack_axis(&[x, x], -1)?;
    // reshape [..., n, 2] -> [..., n*2]
    let mut new_shape = shape[..ndim - 1].to_vec();
    new_shape.push(shape[ndim - 1] * 2);
    stacked.reshape(&new_shape)
}

/// Apply rotary position embeddings to queries and keys.
/// q: [batch, n_heads, seq_len, head_dim]
/// k: [batch, n_kv_heads, seq_len, head_dim]
/// cos, sin: [batch, seq_len, head_dim]
pub fn apply_rotary_pos_emb(
    q: &Array,
    k: &Array,
    cos: &Array,
    sin: &Array,
) -> Result<(Array, Array), Exception> {
    // cos, sin: [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
    let cos = ops::expand_dims(cos, 1)?;
    let sin = ops::expand_dims(sin, 1)?;

    let half = cos.dim(-1) / 2;
    // Take first half, then repeat_interleave with 2
    let cos_half = cos.index((Ellipsis, ..half));
    let cos_full = repeat_interleave_last(&cos_half)?;
    let sin_half = sin.index((Ellipsis, ..half));
    let sin_full = repeat_interleave_last(&sin_half)?;

    let rotary_dim = cos_full.dim(-1);

    // Split q, k into rotary and passthrough parts
    let q_rot = q.index((Ellipsis, ..rotary_dim));
    let q_pass = q.index((Ellipsis, rotary_dim..));
    let k_rot = k.index((Ellipsis, ..rotary_dim));
    let k_pass = k.index((Ellipsis, rotary_dim..));

    // q_embed = q_rot * cos + rotate_half_llm(q_rot) * sin
    let q_embed = q_rot
        .multiply(&cos_full)?
        .add(rotate_half_llm(&q_rot)?.multiply(&sin_full)?)?;
    let k_embed = k_rot
        .multiply(&cos_full)?
        .add(rotate_half_llm(&k_rot)?.multiply(&sin_full)?)?;

    let q_out = ops::concatenate_axis(&[&q_embed, &q_pass], -1)?;
    let k_out = ops::concatenate_axis(&[&k_embed, &k_pass], -1)?;

    Ok((q_out, k_out))
}
