use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn, Array,
};
use mlx_lm::cache::KeyValueCache;

use crate::config::TextConfig;
use super::decoder::{DecoderLayerInput, GlmOcrDecoderLayer};
use super::mrope::GlmOcrRotaryEmbedding;
use super::AttentionMask;

#[derive(Debug, Clone, ModuleParameters)]
pub struct GlmOcrTextModel {
    pub vocab_size: i32,
    pub num_layers: i32,

    #[param]
    pub embed_tokens: nn::Embedding,
    #[param]
    pub layers: Vec<GlmOcrDecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,

    pub rotary_emb: GlmOcrRotaryEmbedding,
}

impl GlmOcrTextModel {
    pub fn new(config: &TextConfig) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(config.vocab_size, config.hidden_size)?;
        let layers = (0..config.num_hidden_layers)
            .map(|_| GlmOcrDecoderLayer::new(config))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;
        let rotary_emb = GlmOcrRotaryEmbedding::new(config)?;

        Ok(Self {
            vocab_size: config.vocab_size,
            num_layers: config.num_hidden_layers,
            embed_tokens,
            layers,
            norm,
            rotary_emb,
        })
    }

    /// Forward pass with optional KV cache.
    /// inputs: [batch, seq_len] token ids
    /// position_ids: [3, batch, seq_len]
    /// cache: per-layer KV caches
    pub fn forward_with_positions<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        cache: &mut [Option<C>],
    ) -> Result<Array, Exception> {
        let mut h = self.embed_tokens.forward(inputs)?;

        let (cos, sin) = self.rotary_emb.forward(&h, position_ids)?;

        let owned_mask;
        let mask = match mask {
            Some(m) => {
                owned_mask = AttentionMask::Array(m.clone());
                &owned_mask
            }
            None => {
                let seq_len = h.dim(1);
                if seq_len <= 1 {
                    owned_mask = AttentionMask::None;
                    &owned_mask
                } else {
                    // Match Python mlx-vlm: rely on SDPA's built-in causal path for
                    // cached decode instead of materializing a sliced additive mask.
                    owned_mask = AttentionMask::Causal;
                    &owned_mask
                }
            }
        };

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(DecoderLayerInput {
                x: &h,
                mask,
                cache: c.as_mut(),
                position_embeddings: (&cos, &sin),
            })?;
        }

        self.norm.forward(&h)
    }

    /// Forward pass with pre-computed embeddings (skips embed_tokens).
    /// inputs_embeds: [batch, seq_len, hidden_size]
    /// position_ids: [3, batch, seq_len]
    pub fn forward_with_embeds<C: KeyValueCache>(
        &mut self,
        inputs_embeds: &Array,
        position_ids: &Array,
        mask: Option<&Array>,
        cache: &mut [Option<C>],
    ) -> Result<Array, Exception> {
        let mut h = inputs_embeds.clone();

        let (cos, sin) = self.rotary_emb.forward(&h, position_ids)?;

        let owned_mask;
        let mask = match mask {
            Some(m) => {
                owned_mask = AttentionMask::Array(m.clone());
                &owned_mask
            }
            None => {
                let seq_len = h.dim(1);
                if seq_len <= 1 {
                    owned_mask = AttentionMask::None;
                    &owned_mask
                } else {
                    // Match Python mlx-vlm: rely on SDPA's built-in causal path for
                    // cached decode instead of materializing a sliced additive mask.
                    owned_mask = AttentionMask::Causal;
                    &owned_mask
                }
            }
        };

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(DecoderLayerInput {
                x: &h,
                mask,
                cache: c.as_mut(),
                position_embeddings: (&cos, &sin),
            })?;
        }

        self.norm.forward(&h)
    }
}
