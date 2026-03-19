use mlx_rs::{
    error::Exception,
    ops,
    ops::indexing::{IndexOp, Ellipsis},
    Array, Dtype,
};

/// Vision-specific rotary position embedding.
/// dim = head_dim // 2, theta = 10000.0
/// No learned parameters.
pub struct GlmOcrVisionRotaryEmbedding {
    pub dim: i32,
    pub theta: f32,
}

impl GlmOcrVisionRotaryEmbedding {
    pub fn new(head_dim: i32) -> Self {
        Self {
            dim: head_dim / 2,
            theta: 10000.0,
        }
    }

    /// Returns frequency table of shape [seqlen, dim/2]
    pub fn forward(&self, seqlen: i32) -> Result<Array, Exception> {
        // inv_freq = 1.0 / (theta ^ (arange(0, dim, 2) / dim))
        let half_dim = self.dim / 2;
        let exponents = Array::arange::<_, f32>(0, half_dim, 1)?
            .multiply(Array::from_f32(2.0 / self.dim as f32))?;
        let inv_freq = Array::from_f32(self.theta)
            .power(&exponents)?
            .reciprocal()?;

        let seq = Array::arange::<_, f32>(0, seqlen, 1)?;
        ops::outer(&seq, &inv_freq)
    }
}

/// Rotate half: [-x2, x1] where x1 = x[..., :d/2], x2 = x[..., d/2:]
pub fn rotate_half(x: &Array) -> Result<Array, Exception> {
    let d = x.shape()[x.ndim() - 1] as i32;
    let x1 = x.index((Ellipsis, ..(d / 2)));
    let x2 = x.index((Ellipsis, (d / 2)..));
    let neg_x2 = ops::negative(&x2)?;
    ops::concatenate_axis(&[&neg_x2, &x1], -1)
}

/// Apply rotary position embeddings to q and k for vision.
/// q, k: [seq, num_heads, head_dim]
/// cos, sin: [seq, head_dim] -> expand to [seq, 1, head_dim]
pub fn apply_rotary_pos_emb_vision(
    q: &Array,
    k: &Array,
    cos: &Array,
    sin: &Array,
) -> Result<(Array, Array), Exception> {
    let cos = cos.expand_dims(1)?;
    let sin = sin.expand_dims(1)?;

    let q_f32 = q.as_dtype(Dtype::Float32)?;
    let k_f32 = k.as_dtype(Dtype::Float32)?;

    let q_embed = q_f32.multiply(&cos)?.add(&rotate_half(&q_f32)?.multiply(&sin)?)?;
    let k_embed = k_f32.multiply(&cos)?.add(&rotate_half(&k_f32)?.multiply(&sin)?)?;

    let q_embed = q_embed.as_dtype(q.dtype())?;
    let k_embed = k_embed.as_dtype(k.dtype())?;

    Ok((q_embed, k_embed))
}
