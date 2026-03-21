use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn, Array,
};
use mlx_lm::cache::KeyValueCache;

use crate::config::TextConfig;
use super::attention::{AttentionInput, GlmOcrAttention};
use super::mlp::GlmOcrMlp;
use super::AttentionMask;

#[derive(Debug, Clone, ModuleParameters)]
pub struct GlmOcrDecoderLayer {
    #[param]
    pub self_attn: GlmOcrAttention,
    #[param]
    pub mlp: GlmOcrMlp,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_self_attn_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub post_mlp_layernorm: nn::RmsNorm,
}

impl GlmOcrDecoderLayer {
    pub fn new(config: &TextConfig) -> Result<Self, Exception> {
        let self_attn = GlmOcrAttention::new(config)?;
        let mlp = GlmOcrMlp::new(config)?;
        let input_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_self_attn_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let post_mlp_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_self_attn_layernorm,
            post_attention_layernorm,
            post_mlp_layernorm,
        })
    }
}

pub struct DecoderLayerInput<'a, C> {
    pub x: &'a Array,
    pub mask: &'a AttentionMask,
    pub cache: &'a mut C,
    pub position_embeddings: (&'a Array, &'a Array),
}

impl<C: KeyValueCache> Module<DecoderLayerInput<'_, C>> for GlmOcrDecoderLayer {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: DecoderLayerInput<'_, C>) -> Result<Array, Exception> {
        let DecoderLayerInput {
            x,
            mask,
            cache,
            position_embeddings,
        } = input;

        // Sandwich norm: pre-attn norm -> attn -> post-attn norm -> residual
        let r = x.clone();
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(AttentionInput {
            x: &normed,
            mask,
            cache,
            position_embeddings,
        })?;
        let attn_out = self.post_self_attn_layernorm.forward(&attn_out)?;
        let h = r.add(attn_out)?;

        // Pre-MLP norm -> MLP -> post-MLP norm -> residual
        let r = h.clone();
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let mlp_out = self.post_mlp_layernorm.forward(&mlp_out)?;
        r.add(mlp_out)
    }

    fn training_mode(&mut self, _mode: bool) {}
}
