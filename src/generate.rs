use mlx_rs::{error::Exception, Array, Dtype};
use mlx_rs::ops::{self, indexing::IndexOp};
use mlx_lm::cache::{ConcatKeyValueCache, KeyValueCache};

use crate::full_model::Model;
use crate::sampler;

/// Configuration for text generation.
pub struct GenerateConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub repetition_context_size: usize,
    pub prefill_step_size: i32,
    pub stop_tokens: Vec<i32>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            repetition_context_size: 20,
            prefill_step_size: 2048,
            stop_tokens: vec![],
        }
    }
}

/// Result yielded per generation step.
pub struct GenerateStepResult {
    pub token_id: i32,
    pub logprobs: Array,
}

/// Internal step: takes logits, applies processors, samples token.
fn step(
    logits: &Array,
    tokens: &[i32],
    config: &GenerateConfig,
) -> Result<(Array, Array), Exception> {
    // logits: [B, seq_len, vocab] → take last token
    let logits_last = logits.index((.., -1, ..));

    // Apply repetition penalty on raw logits
    let logits_processed = if config.repetition_penalty != 1.0 && !tokens.is_empty() {
        sampler::apply_repetition_penalty(
            &logits_last,
            tokens,
            config.repetition_penalty,
            config.repetition_context_size,
        )?
    } else {
        logits_last
    };

    // Compute logprobs: logits - logsumexp(logits)
    let logprobs = {
        let lse = ops::logsumexp_axis(&logits_processed, -1, Some(true))?;
        ops::subtract(&logits_processed, &lse)?
    };

    // Sample token
    let y = sampler::sample_token(
        &logprobs,
        config.temperature,
        config.top_p,
        config.top_k,
        config.min_p,
    )?;

    Ok((y, logprobs))
}

/// Run generation: prefill + decode loop.
/// Returns a vector of generated token IDs.
pub fn generate(
    model: &mut Model,
    input_ids: &Array,               // [1, seq_len]
    pixel_values: Option<&Array>,     // [N, C*T*H*W]
    image_grid_thw: Option<&[(i32, i32, i32)]>,
    mask: Option<&Array>,
    config: &GenerateConfig,
) -> Result<Vec<i32>, Exception> {
    let mut cache = model.new_cache();
    let mut generated_tokens: Vec<i32> = Vec::new();

    // Phase 1: Get input embeddings (vision + merge if needed)
    let inputs_embeds = model.get_input_embeddings(input_ids, pixel_values, image_grid_thw, mask)?;
    let position_ids = model.compute_position_ids(input_ids, 0)?;

    // Phase 2: Chunked prefill
    let seq_len = inputs_embeds.shape()[1];
    let mut processed = 0i32;

    while processed < seq_len - 1 {
        let chunk_end = std::cmp::min(processed + config.prefill_step_size, seq_len - 1);
        let chunk_embeds = inputs_embeds.index((.., processed..chunk_end, ..));
        let chunk_pos = position_ids.index((.., .., processed..chunk_end));

        let _logits = model.language_model.forward_with_embeds(&chunk_embeds, &chunk_pos, &mut cache)?;
        _logits.eval()?;
        processed = chunk_end;
    }

    // Last chunk (or full if short enough) — get first token
    let last_embeds = inputs_embeds.index((.., processed.., ..));
    let last_pos = position_ids.index((.., .., processed..));
    let logits = model.language_model.forward_with_embeds(&last_embeds, &last_pos, &mut cache)?;

    // Collect initial tokens for repetition penalty context
    {
        input_ids.eval()?;
        let input_len = input_ids.shape()[1];
        for j in 0..input_len {
            let tok = input_ids.index((0, j));
            tok.eval()?;
            generated_tokens.push(tok.item::<i32>());
        }
    }

    let (mut y, mut _logprobs) = step(&logits, &generated_tokens, config)?;
    y.eval()?;
    let token_id: i32 = y.item();
    generated_tokens.push(token_id);

    if config.stop_tokens.contains(&token_id) {
        return Ok(vec![token_id]);
    }

    let mut result_tokens = vec![token_id];

    // Phase 3: Decode loop
    for _ in 1..config.max_tokens {
        let cache_offset = cache[0].as_ref().map(|c| c.offset()).unwrap_or(0);
        let current_token = *result_tokens.last().unwrap();

        // Build token input [1, 1]
        let y_input = Array::from_int(current_token).reshape(&[1, 1])?;
        let decode_pos = model.compute_position_ids(&y_input, cache_offset)?;

        let logits = model.language_model.forward(&y_input, &decode_pos, &mut cache)?;
        let (next_y, _next_logprobs) = step(&logits, &generated_tokens, config)?;
        next_y.eval()?;
        let next_token_id: i32 = next_y.item();

        generated_tokens.push(next_token_id);
        result_tokens.push(next_token_id);

        if config.stop_tokens.contains(&next_token_id) {
            break;
        }
    }

    Ok(result_tokens)
}
