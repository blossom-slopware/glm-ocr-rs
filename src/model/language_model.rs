use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn, Array,
};
use crate::cache::KeyValueCache;

use crate::cache::KVCache;
use crate::config::TextConfig;
use super::text_model::GlmOcrTextModel;

/// The language model head: text_model + lm_head projection.
#[derive(Debug, Clone, ModuleParameters)]
pub struct LanguageModel {
    #[param]
    pub model: GlmOcrTextModel,
    #[param]
    pub lm_head: nn::Linear,
}

impl LanguageModel {
    pub fn new(config: &TextConfig) -> Result<Self, Exception> {
        let model = GlmOcrTextModel::new(config)?;
        let lm_head = nn::LinearBuilder::new(config.hidden_size, config.vocab_size)
            .bias(false)
            .build()?;

        Ok(Self { model, lm_head })
    }

    /// Forward pass returning logits.
    /// inputs: [batch, seq_len]
    /// position_ids: [3, batch, seq_len]
    /// cache: per-layer KV caches (empty slice = no cache)
    pub fn forward_with_positions<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        position_ids: &Array,
        cache: &mut [Option<C>],
    ) -> Result<Array, Exception> {
        let hidden = self.model.forward_with_positions(inputs, position_ids, None, cache)?;
        self.lm_head.forward(&hidden)
    }

    /// Forward pass with pre-computed embeddings.
    pub fn forward_with_embeds<C: KeyValueCache>(
        &mut self,
        inputs_embeds: &Array,
        position_ids: &Array,
        cache: &mut [Option<C>],
    ) -> Result<Array, Exception> {
        let hidden = self.model.forward_with_embeds(inputs_embeds, position_ids, None, cache)?;
        self.lm_head.forward(&hidden)
    }
}

/// Outer wrapper that matches safetensors key prefix `language_model.`
#[derive(Debug, Clone, ModuleParameters)]
pub struct GlmOcrModel {
    #[param]
    pub language_model: LanguageModel,
    pub num_layers: i32,
}

impl GlmOcrModel {
    pub fn new(config: &TextConfig) -> Result<Self, Exception> {
        let language_model = LanguageModel::new(config)?;
        Ok(Self {
            language_model,
            num_layers: config.num_hidden_layers,
        })
    }

    pub fn forward<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        position_ids: &Array,
        cache: &mut [Option<C>],
    ) -> Result<Array, Exception> {
        self.language_model.forward_with_positions(inputs, position_ids, cache)
    }

    pub fn forward_with_embeds<C: KeyValueCache>(
        &mut self,
        inputs_embeds: &Array,
        position_ids: &Array,
        cache: &mut [Option<C>],
    ) -> Result<Array, Exception> {
        self.language_model.forward_with_embeds(inputs_embeds, position_ids, cache)
    }

    /// Get text token embeddings without running the transformer.
    pub fn embed_tokens(&mut self, input_ids: &Array) -> Result<Array, Exception> {
        self.language_model.model.embed_tokens.forward(input_ids)
    }

    /// Create a cache vector for this model with initialized entries.
    pub fn new_cache(&self) -> Vec<Option<KVCache>> {
        (0..self.num_layers).map(|_| Some(KVCache::new())).collect()
    }
}
