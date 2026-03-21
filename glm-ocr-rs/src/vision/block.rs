use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    Array,
};

use super::config::VisionConfig;
use super::attention::{GlmOcrVisionAttention, VisionAttentionInput};
use super::mlp::GlmOcrVisionMlp;

#[derive(Debug, Clone, ModuleParameters)]
pub struct GlmOcrVisionBlock {
    #[param]
    pub norm1: nn::RmsNorm,
    #[param]
    pub norm2: nn::RmsNorm,
    #[param]
    pub attn: GlmOcrVisionAttention,
    #[param]
    pub mlp: GlmOcrVisionMlp,
}

impl GlmOcrVisionBlock {
    pub fn new(config: &VisionConfig) -> Result<Self, Exception> {
        let norm1 = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let norm2 = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let attn = GlmOcrVisionAttention::new(config)?;
        let mlp = GlmOcrVisionMlp::new(config)?;

        Ok(Self { norm1, norm2, attn, mlp })
    }
}

pub struct VisionBlockInput<'a> {
    pub x: &'a Array,
    pub cu_seqlens: &'a [i32],
    pub cos: &'a Array,
    pub sin: &'a Array,
}

impl Module<VisionBlockInput<'_>> for GlmOcrVisionBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: VisionBlockInput<'_>) -> Result<Array, Exception> {
        let VisionBlockInput { x, cu_seqlens, cos, sin } = input;

        // Attention with pre-norm + residual
        let normed = self.norm1.forward(x)?;
        let attn_out = self.attn.forward(VisionAttentionInput {
            x: &normed,
            cu_seqlens,
            cos,
            sin,
        })?;
        let x = x.add(&attn_out)?;

        // MLP with pre-norm + residual
        let normed = self.norm2.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        x.add(&mlp_out)
    }

    fn training_mode(&mut self, mode: bool) {
        self.norm1.training_mode(mode);
        self.norm2.training_mode(mode);
        self.attn.training_mode(mode);
        self.mlp.training_mode(mode);
    }
}
