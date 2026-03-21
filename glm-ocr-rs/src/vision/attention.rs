use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    ops,
    Array,
};

use super::config::VisionConfig;
use super::rotary::apply_rotary_pos_emb_vision;

#[derive(Debug, Clone, ModuleParameters)]
pub struct GlmOcrVisionAttention {
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[param]
    pub qkv: nn::Linear,
    #[param]
    pub proj: nn::Linear,
    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,
}

impl GlmOcrVisionAttention {
    pub fn new(config: &VisionConfig) -> Result<Self, Exception> {
        let head_dim = config.head_dim();
        let num_heads = config.num_heads;
        let hidden_size = config.hidden_size;
        let scale = (head_dim as f32).sqrt().recip();

        let qkv = nn::LinearBuilder::new(hidden_size, hidden_size * 3)
            .bias(config.attention_bias)
            .build()?;
        let proj = nn::LinearBuilder::new(hidden_size, hidden_size)
            .bias(config.attention_bias)
            .build()?;
        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            num_heads,
            head_dim,
            scale,
            qkv,
            proj,
            q_norm,
            k_norm,
        })
    }
}

pub struct VisionAttentionInput<'a> {
    pub x: &'a Array,
    pub cu_seqlens: &'a [i32],
    pub cos: &'a Array,
    pub sin: &'a Array,
}

impl Module<VisionAttentionInput<'_>> for GlmOcrVisionAttention {
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: VisionAttentionInput<'_>) -> Result<Array, Exception> {
        let VisionAttentionInput { x, cu_seqlens, cos, sin } = input;

        let seq_total = x.shape()[0];

        // QKV projection: [N, hidden] -> [N, 3*hidden]
        let qkv = self.qkv.forward(x)?;

        // Reshape: [N, 3, num_heads, head_dim]
        let qkv = qkv.reshape(&[seq_total, 3, self.num_heads, self.head_dim])?;
        // Transpose: [3, N, num_heads, head_dim]
        let qkv = qkv.transpose_axes(&[1, 0, 2, 3])?;
        // Split into q, k, v each [1, N, num_heads, head_dim]
        let parts = qkv.split(3, 0)?;
        let mut q = parts[0].squeeze_axes(&[0])?; // [N, num_heads, head_dim]
        let mut k = parts[1].squeeze_axes(&[0])?;
        let v = parts[2].squeeze_axes(&[0])?;

        // Per-head RMSNorm
        q = self.q_norm.forward(&q)?;
        k = self.k_norm.forward(&k)?;

        // Apply RoPE
        let (q, k) = apply_rotary_pos_emb_vision(&q, &k, cos, sin)?;

        // Add batch dim and transpose to [1, num_heads, N, head_dim]
        let q = q.expand_dims(0)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.expand_dims(0)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.expand_dims(0)?.transpose_axes(&[0, 2, 1, 3])?;

        // Split by cu_seqlens and run per-image attention
        let num_segments = cu_seqlens.len() - 1;
        let mut attn_outputs = Vec::with_capacity(num_segments);

        // Compute split indices (lengths of each segment)
        let mut split_indices = Vec::with_capacity(num_segments - 1);
        let mut cumulative = 0i32;
        for i in 0..num_segments {
            let len = cu_seqlens[i + 1] - cu_seqlens[i];
            cumulative += len;
            if i < num_segments - 1 {
                split_indices.push(cumulative);
            }
        }

        if num_segments == 1 {
            // Single image - no split needed
            let attn = mlx_rs::fast::scaled_dot_product_attention(
                &q, &k, &v, self.scale, None, None,
            )?;
            attn_outputs.push(attn);
        } else {
            // Split along seq dimension (axis=2)
            let q_parts = ops::split_sections(&q, &split_indices, 2)?;
            let k_parts = ops::split_sections(&k, &split_indices, 2)?;
            let v_parts = ops::split_sections(&v, &split_indices, 2)?;

            for i in 0..num_segments {
                let attn = mlx_rs::fast::scaled_dot_product_attention(
                    &q_parts[i], &k_parts[i], &v_parts[i], self.scale, None, None,
                )?;
                attn_outputs.push(attn);
            }
        }

        // Concat back: [1, num_heads, N, head_dim]
        let output = if attn_outputs.len() == 1 {
            attn_outputs.into_iter().next().unwrap()
        } else {
            let refs: Vec<&Array> = attn_outputs.iter().collect();
            ops::concatenate_axis(&refs, 2)?
        };

        // [1, num_heads, N, head_dim] -> [N, num_heads, head_dim] -> [N, hidden]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[seq_total, -1])?;

        self.proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.qkv.training_mode(mode);
        self.proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        self.k_norm.training_mode(mode);
    }
}
