use mlx_rs::{error::Exception, transforms, with_new_default_stream, Array, Stream};
use mlx_rs::ops::{self, indexing::IndexOp};
use mlx_lm::cache::{ConcatKeyValueCache, KeyValueCache};

use crate::full_model::Model;
use crate::ocr::abort::AbortSignal;
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

/// Summary of a generation run.
pub struct GenerateSummary {
    pub generated_tokens: usize,
    pub stopped_by_stop_token: bool,
}

pub struct GenerateState {
    pub cache: Vec<Option<ConcatKeyValueCache>>,
    pub generated_tokens: Vec<i32>,
    pub stream: Stream,
}

/// Error type for generation — supports abort and MLX errors.
#[derive(Debug)]
pub enum GenerateError {
    Aborted,
    Mlx(Exception),
}

impl From<Exception> for GenerateError {
    fn from(e: Exception) -> Self {
        GenerateError::Mlx(e)
    }
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerateError::Aborted => write!(f, "Generation aborted"),
            GenerateError::Mlx(e) => write!(f, "MLX error: {}", e),
        }
    }
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

fn eval_cache(cache: &[Option<ConcatKeyValueCache>]) -> Result<(), Exception> {
    let mut arrays = Vec::new();
    for layer_cache in cache {
        if let Some(layer_cache) = layer_cache {
            if let Some(keys) = layer_cache.keys() {
                arrays.push(keys);
            }
            if let Some(values) = layer_cache.values() {
                arrays.push(values);
            }
        }
    }
    if arrays.is_empty() {
        Ok(())
    } else {
        transforms::eval(arrays)
    }
}

pub fn prefill(
    model: &mut Model,
    input_ids: &Array,
    pixel_values: Option<&Array>,
    image_grid_thw: Option<&[(i32, i32, i32)]>,
    mask: Option<&Array>,
    config: &GenerateConfig,
    abort: &AbortSignal,
) -> Result<(GenerateState, i32), GenerateError> {
    let mut cache = model.new_cache();
    let mut generated_tokens = Vec::new();

    let stream = Stream::new();
    let (first_token, generated_tokens) = with_new_default_stream(stream.clone(), || {
        let inputs_embeds = model.get_input_embeddings(input_ids, pixel_values, image_grid_thw, mask)?;
        let position_ids = model.compute_position_ids(input_ids, 0)?;

        let seq_len = inputs_embeds.shape()[1];
        let mut processed = 0i32;

        while processed < seq_len - 1 {
            if abort.is_set() {
                return Err(GenerateError::Aborted);
            }

            let chunk_end = std::cmp::min(processed + config.prefill_step_size, seq_len - 1);
            let chunk_embeds = inputs_embeds.index((.., processed..chunk_end, ..));
            let chunk_pos = position_ids.index((.., .., processed..chunk_end));

            let logits = model.language_model.forward_with_embeds(&chunk_embeds, &chunk_pos, &mut cache)?;
            logits.eval()?;
            eval_cache(&cache)?;
            processed = chunk_end;

            if processed % (config.prefill_step_size * 2) == 0 {
                mlx_rs::transforms::compile::clear_cache();
            }
        }

        let last_embeds = inputs_embeds.index((.., processed.., ..));
        let last_pos = position_ids.index((.., .., processed..));
        let logits = model.language_model.forward_with_embeds(&last_embeds, &last_pos, &mut cache)?;

        input_ids.eval()?;
        let input_len = input_ids.shape()[1];
        for j in 0..input_len {
            let tok = input_ids.index((0, j));
            tok.eval()?;
            generated_tokens.push(tok.item::<i32>());
        }

        let (y, _logprobs) = step(&logits, &generated_tokens, config)?;
        transforms::async_eval([&y])?;
        y.eval()?;
        let token_id = y.item::<i32>();
        generated_tokens.push(token_id);
        Ok::<_, GenerateError>((token_id, generated_tokens))
    })?;

    Ok((
        GenerateState {
            cache,
            generated_tokens,
            stream,
        },
        first_token,
    ))
}

pub fn decode_next(
    model: &mut Model,
    state: &mut GenerateState,
    config: &GenerateConfig,
    abort: &AbortSignal,
) -> Result<i32, GenerateError> {
    if abort.is_set() {
        return Err(GenerateError::Aborted);
    }

    with_new_default_stream(state.stream.clone(), || {
        let cache_offset = state.cache[0].as_ref().map(|c| c.offset()).unwrap_or(0);
        let current_token = *state.generated_tokens.last().unwrap();

        let y_input = Array::from_int(current_token).reshape(&[1, 1])?;
        let decode_pos = model.compute_position_ids(&y_input, cache_offset)?;

        let logits = model.language_model.forward(&y_input, &decode_pos, &mut state.cache)?;
        let (next_y, _next_logprobs) = step(&logits, &state.generated_tokens, config)?;
        transforms::async_eval([&next_y])?;
        next_y.eval()?;
        eval_cache(&state.cache)?;
        let next_token_id = next_y.item::<i32>();
        state.generated_tokens.push(next_token_id);
        Ok::<_, GenerateError>(next_token_id)
    })
}

/// Run generation incrementally: prefill + decode loop.
/// Calls `on_token` for each sampled token as soon as it is available.
/// The callback can return `Err(GenerateError::Aborted)` to stop generation.
pub fn generate_stream<F>(
    model: &mut Model,
    input_ids: &Array,               // [1, seq_len]
    pixel_values: Option<&Array>,     // [N, C*T*H*W]
    image_grid_thw: Option<&[(i32, i32, i32)]>,
    mask: Option<&Array>,
    config: &GenerateConfig,
    abort: &AbortSignal,
    mut on_token: F,
) -> Result<GenerateSummary, GenerateError>
where
    F: FnMut(i32) -> Result<(), GenerateError>,
{
    if config.max_tokens == 0 {
        return Ok(GenerateSummary {
            generated_tokens: 0,
            stopped_by_stop_token: false,
        });
    }

    let (mut state, token_id) = prefill(
        model,
        input_ids,
        pixel_values,
        image_grid_thw,
        mask,
        config,
        abort,
    )?;
    on_token(token_id)?;

    let mut emitted_tokens = 1usize;
    let mut stopped_by_stop_token = config.stop_tokens.contains(&token_id);

    if stopped_by_stop_token {
        return Ok(GenerateSummary {
            generated_tokens: emitted_tokens,
            stopped_by_stop_token,
        });
    }

    // Phase 3: Decode loop
    for _ in 1..config.max_tokens {
        let next_token_id = decode_next(model, &mut state, config, abort)?;
        on_token(next_token_id)?;
        emitted_tokens += 1;

        if config.stop_tokens.contains(&next_token_id) {
            stopped_by_stop_token = true;
            break;
        }
    }

    Ok(GenerateSummary {
        generated_tokens: emitted_tokens,
        stopped_by_stop_token,
    })
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
) -> Result<Vec<i32>, GenerateError> {
    let mut result_tokens = Vec::new();
    generate_stream(
        model,
        input_ids,
        pixel_values,
        image_grid_thw,
        mask,
        config,
        &AbortSignal::none(),
        |token_id| { result_tokens.push(token_id); Ok(()) },
    )?;
    Ok(result_tokens)
}
