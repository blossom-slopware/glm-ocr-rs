use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn, Array,
};

use crate::cache::KVCache;
use crate::config::TextConfig;
use super::AttentionMask;
use super::mrope::apply_rotary_pos_emb;

#[derive(Debug, Clone, ModuleParameters)]
pub struct GlmOcrAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub o_proj: nn::Linear,
}

impl GlmOcrAttention {
    pub fn new(config: &TextConfig) -> Result<Self, Exception> {
        let dim = config.hidden_size;
        let n_heads = config.num_attention_heads;
        let n_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(config.attention_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(config.attention_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(config.attention_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(false)
            .build()?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }
}

pub struct AttentionInput<'a> {
    pub x: &'a Array,
    pub mask: &'a AttentionMask,
    pub cache: &'a mut KVCache,
    pub position_embeddings: (&'a Array, &'a Array), // (cos, sin)
}

impl Module<AttentionInput<'_>> for GlmOcrAttention {
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_>) -> Result<Array, Exception> {
        let AttentionInput {
            x,
            mask,
            cache,
            position_embeddings: (cos, sin),
        } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape and transpose to [B, heads, L, head_dim]
        let queries = queries
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let keys = keys
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply rotary embeddings
        let (queries, mut keys) = apply_rotary_pos_emb(&queries, &keys, cos, sin)?;

        // Update KV cache
        let (k, v) = cache.update_and_fetch(keys, values)?;
        keys = k;
        values = v;

        // Scaled dot-product attention
        let output = mlx_rs::fast::scaled_dot_product_attention(
            queries,
            keys,
            values,
            self.scale,
            mask.as_sdpa_mask(),
            None,
        )?;

        // Reshape back: [B, heads, L, head_dim] -> [B, L, heads * head_dim]
        let output = output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
    }
}
