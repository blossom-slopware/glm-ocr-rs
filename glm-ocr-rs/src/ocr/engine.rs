use std::time::{Duration, Instant};

use mlx_rs::Array;

use crate::full_model::Model;
use crate::generate::{self, GenerateConfig, GenerateError};
use crate::image_processor::{self, ImageProcessor};
use crate::tokenizer::{self, GlmTokenizer};

use super::abort::AbortSignal;
use super::error::EngineError;
use super::request::{ImageSource, OcrRequest, OcrRunResult, StopReason};

/// Default OCR instruction prompt.
pub const DEFAULT_OCR_PROMPT: &str =
    "Recognize the text in the image and output in Markdown format. \
     Preserve the original layout (headings/paragraphs/tables/formulas). \
     Do not fabricate content that does not exist in the image.";

/// Tracks token generation speed, logging periodically and at finish.
struct TpsLogger {
    start: Instant,
    last_log: Instant,
    total_tokens: usize,
    interval_tokens: usize,
    log_interval: Duration,
}

impl TpsLogger {
    fn new(log_interval: Duration) -> Self {
        let now = Instant::now();
        TpsLogger {
            start: now,
            last_log: now,
            total_tokens: 0,
            interval_tokens: 0,
            log_interval,
        }
    }

    /// Record one token. Logs at info level if the interval has elapsed.
    fn tick(&mut self) {
        self.total_tokens += 1;
        self.interval_tokens += 1;

        let now = Instant::now();
        let elapsed = now.duration_since(self.last_log);
        if elapsed >= self.log_interval {
            let overall_tps = self.total_tokens as f64 / now.duration_since(self.start).as_secs_f64();
            let interval_tps = self.interval_tokens as f64 / elapsed.as_secs_f64();
            log::info!(
                "Generating: {} tokens so far ({:.1} tok/s overall, {:.1} tok/s recent)",
                self.total_tokens,
                overall_tps,
                interval_tps
            );
            self.last_log = now;
            self.interval_tokens = 0;
        }
    }

    /// Log final summary at info level.
    fn finish(&self, stop_reason: &StopReason) {
        let elapsed = self.start.elapsed();
        let ms = elapsed.as_millis();
        let tps = if ms > 0 {
            self.total_tokens as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        log::info!(
            "Generation completed: {} tokens in {}ms ({:.1} tok/s) stop_reason={:?}",
            self.total_tokens,
            ms,
            tps,
            stop_reason
        );
    }
}

/// The OCR engine owns model, tokenizer, image processor, and chat template.
/// All inference happens synchronously through `run()`.
pub struct OcrEngine {
    pub model: Model,
    pub tokenizer: GlmTokenizer,
    pub image_processor: ImageProcessor,
    pub template_str: String,
}

impl OcrEngine {
    /// Run the full OCR pipeline: load image → preprocess → tokenize → generate → decode.
    ///
    /// `on_text_chunk` is called with each incrementally decoded text delta.
    /// This method is blocking and should be called from `spawn_blocking`.
    pub fn run(
        &mut self,
        req: &OcrRequest,
        abort: &AbortSignal,
        mut on_text_chunk: impl FnMut(String),
    ) -> Result<OcrRunResult, EngineError> {
        let effective_max_tokens = req.validate()?;
        let total_start = Instant::now();

        // ── 1. Load image bytes ──
        let load_start = Instant::now();
        let img_bytes = match &req.image {
            ImageSource::Url { url } => {
                log::debug!(
                    "Loading image from: {}",
                    if url.starts_with("data:") {
                        format!("data:...({} bytes)", url.len())
                    } else if url.len() > 120 {
                        format!("{}...", &url[..120])
                    } else {
                        url.clone()
                    }
                );
                image_processor::load_image_bytes(url).map_err(|source| EngineError::ImageLoad {
                    source: source.context("failed while loading image source"),
                })?
            }
            ImageSource::Bytes(bytes) => {
                log::debug!("Using provided image bytes: {} bytes", bytes.len());
                bytes.clone()
            }
        };
        let load_ms = load_start.elapsed().as_millis();
        log::debug!("Image loaded: {} bytes in {}ms", img_bytes.len(), load_ms);

        // ── 2. Preprocess image ──
        let preprocess_start = Instant::now();
        let (pixel_values, grid_thw) = self
            .image_processor
            .preprocess(&img_bytes)
            .map_err(|source| {
                let message = format!("{source:#}");
                if message.contains("failed to decode image bytes") {
                    EngineError::ImageDecode { source }
                } else {
                    EngineError::Preprocess { source }
                }
            })?;
        let preprocess_ms = preprocess_start.elapsed().as_millis();
        log::info!(
            "Image preprocessed: grid_thw=({},{},{}) pixel_values shape={:?} in {}ms",
            grid_thw.0,
            grid_thw.1,
            grid_thw.2,
            pixel_values.shape(),
            preprocess_ms
        );

        // ── 3. Build prompt via chat template ──
        let tokenize_start = Instant::now();
        let user_prompt = match req.prompt.as_deref() {
            Some(s) if !s.trim().is_empty() => s,
            _ => DEFAULT_OCR_PROMPT,
        };
        let messages = vec![serde_json::json!({
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": user_prompt}
            ]
        })];
        let prompt_text = tokenizer::render_chat_template(
            &self.template_str,
            &messages,
            true,  // add_generation_prompt
            false, // enable_thinking = false for OCR
        )
        .map_err(|source| EngineError::PromptRender {
            source: source.context("failed to render OCR prompt template"),
        })?;
        log::debug!("Rendered chat template ({} chars)", prompt_text.len());
        log::trace!("Template output:\n{}", prompt_text);

        // ── 4. Compute image token count and encode ──
        let merge_size = self.image_processor.merge_size;
        let num_image_tokens = (grid_thw.0 as usize * grid_thw.1 as usize * grid_thw.2 as usize)
            / (merge_size as usize * merge_size as usize);
        log::debug!(
            "Image token expansion: grid=({},{},{}) merge_size={} → {} tokens",
            grid_thw.0,
            grid_thw.1,
            grid_thw.2,
            merge_size,
            num_image_tokens
        );

        let input_ids = self
            .tokenizer
            .encode(&prompt_text, &[num_image_tokens])
            .map_err(|source| EngineError::Tokenization {
                source: source.context("failed to tokenize OCR prompt"),
            })?;
        let prompt_tokens = input_ids.len();
        let tokenize_ms = tokenize_start.elapsed().as_millis();
        log::info!(
            "Tokenization completed in {}ms → {} prompt tokens",
            tokenize_ms,
            prompt_tokens
        );

        // ── 5. Build MLX arrays ──
        let input_ids_arr = Array::from_slice(&input_ids, &[1, input_ids.len() as i32]);
        let grid_thw_vec = vec![grid_thw];

        let eos_tokens = self.model.config.eos_token_id.clone();
        log::debug!("EOS tokens: {:?}", eos_tokens);

        let config = GenerateConfig {
            max_tokens: effective_max_tokens,
            temperature: req.temperature,
            stop_tokens: eos_tokens.clone(),
            ..Default::default()
        };
        log::debug!(
            "GenerateConfig: max_tokens={} temp={} top_p={} top_k={} min_p={} rep_penalty={} prefill_step={}",
            config.max_tokens,
            config.temperature,
            config.top_p,
            config.top_k,
            config.min_p,
            config.repetition_penalty,
            config.prefill_step_size
        );

        // ── 6. Generate with streaming decode ──
        log::info!("Starting generation...");
        let mut tps = TpsLogger::new(Duration::from_secs(2));
        let mut decode_stream = self.tokenizer.inner.decode_stream(true);
        let mut generated_token_ids: Vec<i32> = Vec::new();
        let mut emitted_text = String::new();
        let mut chunk_count = 0usize;

        let abort_clone = abort.clone();
        let result = generate::generate_stream(
            &mut self.model,
            &input_ids_arr,
            Some(&pixel_values),
            Some(grid_thw_vec.as_slice()),
            None,
            &config,
            abort,
            |token_id| {
                if abort_clone.is_set() {
                    return Err(GenerateError::Aborted);
                }

                if eos_tokens.contains(&token_id) {
                    log::debug!("Hit EOS token {}", token_id);
                    return Ok(());
                }

                generated_token_ids.push(token_id);
                tps.tick();

                match decode_stream.step(token_id as u32) {
                    Ok(Some(text_chunk)) if !text_chunk.is_empty() => {
                        emitted_text.push_str(&text_chunk);
                        chunk_count += 1;
                        log::trace!(
                            "Chunk #{} token_id={} text={:?}",
                            chunk_count,
                            token_id,
                            text_chunk
                        );
                        on_text_chunk(text_chunk);
                    }
                    Ok(Some(_)) | Ok(None) => {}
                    Err(e) => {
                        return Err(GenerateError::Other(anyhow::anyhow!(
                            "stream decode failed: {}",
                            e
                        )));
                    }
                }

                Ok(())
            },
        );

        // ── 7. Determine stop reason and handle result ──
        let (summary_tokens, stop_reason) = match result {
            Ok(summary) => {
                let reason = if summary.stopped_by_stop_token {
                    StopReason::StopToken
                } else {
                    StopReason::MaxTokens
                };
                (summary.generated_tokens, reason)
            }
            Err(GenerateError::Aborted) => {
                log::info!(
                    "Generation aborted after {} tokens for image: {}",
                    generated_token_ids.len(),
                    req.image_description()
                );
                (generated_token_ids.len(), StopReason::Aborted)
            }
            Err(GenerateError::Mlx(source)) => {
                return Err(EngineError::Generation {
                    source: anyhow::Error::new(source).context("MLX generation error"),
                });
            }
            Err(GenerateError::Other(source)) => {
                return Err(EngineError::StreamDecode {
                    source: source.context("stream decoder callback failed"),
                });
            }
            Err(GenerateError::StateInvariant(message)) => {
                return Err(EngineError::StateInvariant {
                    code: "generation_state_invariant",
                    message,
                });
            }
        };

        tps.finish(&stop_reason);

        // ── 8. Final flush — emit any remaining decoded text ──
        let final_text = self
            .tokenizer
            .decode(&generated_token_ids, true)
            .map_err(|source| EngineError::Tokenization {
                source: source.context("failed to decode final generated text"),
            })?;
        if !final_text.starts_with(&emitted_text) {
            return Err(EngineError::StateInvariant {
                code: "stream_decode_prefix_mismatch",
                message: "stream decode output is not a prefix of final decode".to_string(),
            });
        }
        let remaining_text = &final_text[emitted_text.len()..];
        if !remaining_text.is_empty() {
            chunk_count += 1;
            log::trace!("Final flush chunk #{} text={:?}", chunk_count, remaining_text);
            on_text_chunk(remaining_text.to_string());
        }

        let total_ms = total_start.elapsed().as_millis();
        log::info!(
            "OCR request completed: {} prompt + {} gen tokens, {} chunks in {}ms (preprocess={}ms, tokenize={}ms)",
            prompt_tokens,
            summary_tokens,
            chunk_count,
            total_ms,
            preprocess_ms,
            tokenize_ms
        );

        Ok(OcrRunResult {
            text: final_text,
            generated_tokens: summary_tokens,
            stop_reason,
        })
    }
}
