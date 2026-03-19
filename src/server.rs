use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json as AxumJson};
use axum::routing::{get, post};
use axum::Router;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;

use crate::full_model::Model;
use crate::generate::{self, GenerateConfig};
use crate::image_processor::{self, ImageProcessor};
use crate::tokenizer::{self, GlmTokenizer};

use mlx_rs::Array;

/// Shared application state.
pub struct AppState {
    pub model: tokio::sync::Mutex<Model>,
    pub tokenizer: GlmTokenizer,
    pub image_processor: ImageProcessor,
    pub template_str: String,
    pub model_name: String,
}

// ─── Request types ───

#[derive(Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<serde_json::Value>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> usize {
    4096
}
fn default_temperature() -> f32 {
    0.01
}
fn default_top_p() -> f32 {
    1.0
}

// ─── Response types ───

#[derive(Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: AssistantMessage,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct AssistantMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ─── Streaming response types ───

#[derive(Serialize)]
struct StreamChunk {
    id: String,
    object: String,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Serialize)]
struct StreamChoice {
    index: usize,
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

// ─── Routes ───

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/models", get(models_handler))
        .route("/v1/models", get(models_handler))
        .route("/chat/completions", post(chat_completions_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    log::debug!("GET /health");
    AxumJson(serde_json::json!({
        "status": "ok",
        "model": state.model_name,
    }))
}

async fn models_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    log::debug!("GET /models");
    AxumJson(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "owned_by": "local",
        }]
    }))
}

/// Extract image URLs and text content from OpenAI-format messages.
fn extract_images_and_text(messages: &[serde_json::Value]) -> (Vec<String>, String) {
    let mut image_urls = Vec::new();
    let mut text_parts = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let content = &msg["content"];

        if role == "user" {
            if let Some(s) = content.as_str() {
                text_parts.push(s.to_string());
            } else if let Some(arr) = content.as_array() {
                for item in arr {
                    let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    match item_type {
                        "text" => {
                            if let Some(t) = item.get("text").and_then(|t| t.as_str()) {
                                text_parts.push(t.to_string());
                            }
                        }
                        "image_url" => {
                            if let Some(url) = item.get("image_url")
                                .and_then(|u| u.get("url"))
                                .and_then(|u| u.as_str())
                            {
                                image_urls.push(url.to_string());
                            }
                        }
                        "input_image" => {
                            if let Some(url) = item.get("image_url").and_then(|u| u.as_str()) {
                                image_urls.push(url.to_string());
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    (image_urls, text_parts.join("\n"))
}

/// Build template-compatible messages from the request.
/// Converts image_url content items to the format the chat template expects.
fn build_template_messages(messages: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut result = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("").to_string();
        let content = &msg["content"];

        if let Some(arr) = content.as_array() {
            // Convert content array: image_url -> image format for template
            let mut new_content = Vec::new();
            for item in arr {
                let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match item_type {
                    "image_url" | "input_image" => {
                        new_content.push(serde_json::json!({"type": "image", "image": "placeholder"}));
                    }
                    "text" => {
                        new_content.push(item.clone());
                    }
                    _ => {
                        new_content.push(item.clone());
                    }
                }
            }
            result.push(serde_json::json!({
                "role": role,
                "content": new_content,
            }));
        } else {
            result.push(msg.clone());
        }
    }

    result
}

async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    AxumJson(req): AxumJson<ChatRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    log::info!(
        "[{}] POST /chat/completions model={:?} stream={} max_tokens={} temp={} top_p={} messages={}",
        request_id, req.model, req.stream, req.max_tokens, req.temperature, req.top_p,
        req.messages.len()
    );

    if req.stream {
        let sse = handle_streaming(state, req, request_id).await
            .map_err(|e| {
                log::error!("Streaming request failed: {}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, e)
            })?;
        Ok(sse.into_response())
    } else {
        let response = handle_non_streaming(state, req, request_id).await
            .map_err(|e| {
                log::error!("Non-streaming request failed: {}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, e)
            })?;
        Ok(AxumJson(response).into_response())
    }
}

async fn handle_non_streaming(
    state: Arc<AppState>,
    req: ChatRequest,
    request_id: String,
) -> Result<ChatResponse, String> {
    let total_start = Instant::now();
    let model_name = if req.model.is_empty() {
        state.model_name.clone()
    } else {
        req.model.clone()
    };

    // ── 1. Extract images and text ──
    let (image_urls, _user_text) = extract_images_and_text(&req.messages);
    log::info!(
        "[{}] Extracted {} image(s) from request",
        request_id, image_urls.len()
    );
    for (i, url) in image_urls.iter().enumerate() {
        let display_url = if url.starts_with("data:") {
            format!("data:...({} bytes base64)", url.len())
        } else if url.len() > 120 {
            format!("{}...", &url[..120])
        } else {
            url.clone()
        };
        log::debug!("[{}]   image[{}]: {}", request_id, i, display_url);
    }

    // ── 2. Preprocess images ──
    let preprocess_start = Instant::now();
    let state_clone = state.clone();
    let image_urls_clone = image_urls.clone();
    let rid = request_id.clone();
    let image_data: Vec<(Array, (i32, i32, i32))> = tokio::task::spawn_blocking(move || {
        image_urls_clone.iter().enumerate().map(|(i, url)| {
            let load_start = Instant::now();
            let bytes = image_processor::load_image_bytes(url);
            let load_ms = load_start.elapsed().as_millis();
            log::debug!("[{}]   image[{}]: loaded {} bytes in {}ms", rid, i, bytes.len(), load_ms);

            let proc_start = Instant::now();
            let result = state_clone.image_processor.preprocess(&bytes)
                .unwrap_or_else(|e| panic!("Image preprocessing failed: {}", e));
            let proc_ms = proc_start.elapsed().as_millis();
            log::debug!(
                "[{}]   image[{}]: preprocessed to grid_thw=({},{},{}) pixel_values shape=[{}, 1176] in {}ms",
                rid, i, result.1.0, result.1.1, result.1.2,
                result.1.0 * result.1.1 * result.1.2, proc_ms
            );
            result
        }).collect()
    }).await.map_err(|e| format!("Image preprocessing task failed: {}", e))?;
    let preprocess_ms = preprocess_start.elapsed().as_millis();
    log::info!("[{}] Image preprocessing completed in {}ms", request_id, preprocess_ms);

    // ── 3. Build template and tokenize ──
    let tokenize_start = Instant::now();
    let template_messages = build_template_messages(&req.messages);
    let prompt_text = tokenizer::render_chat_template(
        &state.template_str,
        &template_messages,
        true,  // add_generation_prompt
        false, // enable_thinking = false for OCR
    );
    log::debug!(
        "[{}] Rendered chat template ({} chars)",
        request_id, prompt_text.len()
    );
    log::trace!("[{}] Template output:\n{}", request_id, prompt_text);

    // Compute image token counts and encode
    let merge_size = state.image_processor.merge_size;
    let num_image_tokens: Vec<usize> = image_data.iter().map(|(_, (t, h, w))| {
        let n = (*t as usize * *h as usize * *w as usize) / (merge_size as usize * merge_size as usize);
        log::debug!(
            "[{}] Image token expansion: grid=({},{},{}) merge_size={} → {} tokens",
            request_id, t, h, w, merge_size, n
        );
        n
    }).collect();

    let input_ids = state.tokenizer.encode(&prompt_text, &num_image_tokens);
    let prompt_tokens = input_ids.len();
    let tokenize_ms = tokenize_start.elapsed().as_millis();
    log::info!(
        "[{}] Tokenization completed in {}ms → {} prompt tokens",
        request_id, tokenize_ms, prompt_tokens
    );

    // ── 4. Generate ──
    let generate_start = Instant::now();
    let state_clone = state.clone();
    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let top_p = req.top_p;
    let top_k = req.top_k.unwrap_or(0);
    let min_p = req.min_p.unwrap_or(0.0);
    let repetition_penalty = req.repetition_penalty.unwrap_or(1.0);
    let rid = request_id.clone();

    let generated_tokens: Vec<i32> = tokio::task::spawn_blocking(move || {
        log::debug!("[{}] Acquiring model lock...", rid);
        let lock_start = Instant::now();
        let mut model = state_clone.model.blocking_lock();
        let lock_ms = lock_start.elapsed().as_millis();
        if lock_ms > 10 {
            log::info!("[{}] Model lock acquired in {}ms (was contended)", rid, lock_ms);
        } else {
            log::debug!("[{}] Model lock acquired in {}ms", rid, lock_ms);
        }

        let input_ids_arr = Array::from_slice(
            &input_ids,
            &[1, input_ids.len() as i32],
        );

        // Combine all image pixel values and grid_thw
        let (pixel_values, grid_thw) = if !image_data.is_empty() {
            let (pv, gt) = &image_data[0];
            log::debug!(
                "[{}] Passing pixel_values shape={:?} grid_thw={:?} to model",
                rid, pv.shape(), gt
            );
            (Some(pv.clone()), Some(vec![*gt]))
        } else {
            log::debug!("[{}] Text-only request (no pixel_values)", rid);
            (None, None)
        };

        let eos_tokens = model.config.eos_token_id.clone();
        log::debug!("[{}] EOS tokens: {:?}", rid, eos_tokens);

        let config = GenerateConfig {
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            stop_tokens: eos_tokens,
            ..Default::default()
        };
        log::debug!(
            "[{}] GenerateConfig: max_tokens={} temp={} top_p={} top_k={} min_p={} rep_penalty={} prefill_step={}",
            rid, config.max_tokens, config.temperature, config.top_p,
            config.top_k, config.min_p, config.repetition_penalty, config.prefill_step_size
        );

        log::info!("[{}] Starting generation...", rid);
        let gen_start = Instant::now();
        let result = generate::generate(
            &mut model,
            &input_ids_arr,
            pixel_values.as_ref(),
            grid_thw.as_deref(),
            None,
            &config,
        ).unwrap_or_else(|e| panic!("Generation failed: {}", e));

        let gen_ms = gen_start.elapsed().as_millis();
        let tok_per_sec = if gen_ms > 0 {
            result.len() as f64 / (gen_ms as f64 / 1000.0)
        } else {
            0.0
        };
        log::info!(
            "[{}] Generation completed: {} tokens in {}ms ({:.1} tok/s)",
            rid, result.len(), gen_ms, tok_per_sec
        );
        result
    }).await.map_err(|e| format!("Generation task failed: {}", e))?;
    let generate_ms = generate_start.elapsed().as_millis();

    // ── 5. Decode ──
    let decode_start = Instant::now();
    let completion_tokens = generated_tokens.len();
    let decoded = state.tokenizer.decode(&generated_tokens, true);
    let decode_ms = decode_start.elapsed().as_millis();
    log::debug!(
        "[{}] Decoded {} tokens → {} chars in {}ms",
        request_id, completion_tokens, decoded.len(), decode_ms
    );
    log::trace!("[{}] Decoded text:\n{}", request_id, decoded);

    let total_ms = total_start.elapsed().as_millis();
    log::info!(
        "[{}] Request completed: {} prompt + {} completion = {} total tokens in {}ms (preprocess={}ms, tokenize={}ms, generate={}ms, decode={}ms)",
        request_id, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
        total_ms, preprocess_ms, tokenize_ms, generate_ms, decode_ms
    );

    Ok(ChatResponse {
        id: request_id,
        object: "chat.completion".to_string(),
        model: model_name,
        choices: vec![Choice {
            index: 0,
            message: AssistantMessage {
                role: "assistant".to_string(),
                content: decoded,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    })
}

async fn handle_streaming(
    state: Arc<AppState>,
    req: ChatRequest,
    request_id: String,
) -> Result<Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>>, String> {
    let total_start = Instant::now();
    let model_name = if req.model.is_empty() {
        state.model_name.clone()
    } else {
        req.model.clone()
    };

    // ── 1. Extract images and text ──
    let (image_urls, _user_text) = extract_images_and_text(&req.messages);
    log::info!(
        "[{}] Streaming: extracted {} image(s)",
        request_id, image_urls.len()
    );

    // ── 2. Preprocess images ──
    let preprocess_start = Instant::now();
    let state_clone = state.clone();
    let image_urls_clone = image_urls.clone();
    let rid = request_id.clone();
    let image_data: Vec<(Array, (i32, i32, i32))> = tokio::task::spawn_blocking(move || {
        image_urls_clone.iter().enumerate().map(|(i, url)| {
            let load_start = Instant::now();
            let bytes = image_processor::load_image_bytes(url);
            let load_ms = load_start.elapsed().as_millis();
            log::debug!("[{}]   image[{}]: loaded {} bytes in {}ms", rid, i, bytes.len(), load_ms);

            let proc_start = Instant::now();
            let result = state_clone.image_processor.preprocess(&bytes)
                .unwrap_or_else(|e| panic!("Image preprocessing failed: {}", e));
            let proc_ms = proc_start.elapsed().as_millis();
            log::debug!(
                "[{}]   image[{}]: preprocessed grid_thw=({},{},{}) in {}ms",
                rid, i, result.1.0, result.1.1, result.1.2, proc_ms
            );
            result
        }).collect()
    }).await.map_err(|e| format!("Image preprocessing task failed: {}", e))?;
    let preprocess_ms = preprocess_start.elapsed().as_millis();
    log::info!("[{}] Streaming: image preprocessing {}ms", request_id, preprocess_ms);

    // ── 3. Build template and tokenize ──
    let tokenize_start = Instant::now();
    let template_messages = build_template_messages(&req.messages);
    let prompt_text = tokenizer::render_chat_template(
        &state.template_str,
        &template_messages,
        true,
        false,
    );

    let merge_size = state.image_processor.merge_size;
    let num_image_tokens: Vec<usize> = image_data.iter().map(|(_, (t, h, w))| {
        (*t as usize * *h as usize * *w as usize) / (merge_size as usize * merge_size as usize)
    }).collect();

    let input_ids = state.tokenizer.encode(&prompt_text, &num_image_tokens);
    let prompt_tokens = input_ids.len();
    let tokenize_ms = tokenize_start.elapsed().as_millis();
    log::info!(
        "[{}] Streaming: tokenization {}ms → {} prompt tokens",
        request_id, tokenize_ms, prompt_tokens
    );

    // ── 4. Build MLX arrays ──
    let (pixel_values, grid_thw) = if !image_data.is_empty() {
        let (pv, gt) = &image_data[0];
        (Some(pv.clone()), Some(vec![*gt]))
    } else {
        (None, None)
    };

    let max_tokens = req.max_tokens;
    let temperature = req.temperature;
    let top_p = req.top_p;
    let top_k = req.top_k.unwrap_or(0);
    let min_p = req.min_p.unwrap_or(0.0);
    let repetition_penalty = req.repetition_penalty.unwrap_or(1.0);

    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(128);

    let req_id = request_id.clone();
    let model_nm = model_name.clone();
    let state_clone = state.clone();

    // ── 5. Spawn generation task ──
    tokio::task::spawn_blocking(move || {
        let rid = req_id.clone();
        log::debug!("[{}] Streaming: acquiring model lock...", rid);
        let lock_start = Instant::now();
        let mut model = state_clone.model.blocking_lock();
        let lock_ms = lock_start.elapsed().as_millis();
        if lock_ms > 10 {
            log::info!("[{}] Streaming: model lock acquired in {}ms (contended)", rid, lock_ms);
        } else {
            log::debug!("[{}] Streaming: model lock acquired in {}ms", rid, lock_ms);
        }

        let input_ids_arr = Array::from_slice(
            &input_ids,
            &[1, input_ids.len() as i32],
        );

        let eos_tokens = model.config.eos_token_id.clone();

        let config = GenerateConfig {
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            stop_tokens: eos_tokens.clone(),
            ..Default::default()
        };

        // Send role delta first
        log::debug!("[{}] Streaming: sending role delta", rid);
        let role_chunk = StreamChunk {
            id: req_id.clone(),
            object: "chat.completion.chunk".to_string(),
            model: model_nm.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        let _ = tx.blocking_send(Ok(
            Event::default().data(serde_json::to_string(&role_chunk).unwrap())
        ));

        // Run generation
        log::info!("[{}] Streaming: starting generation...", rid);
        let gen_start = Instant::now();
        let tokens = generate::generate(
            &mut model,
            &input_ids_arr,
            pixel_values.as_ref(),
            grid_thw.as_deref(),
            None,
            &config,
        ).unwrap_or_else(|e| panic!("Generation failed: {}", e));

        let gen_ms = gen_start.elapsed().as_millis();
        let tok_per_sec = if gen_ms > 0 {
            tokens.len() as f64 / (gen_ms as f64 / 1000.0)
        } else {
            0.0
        };
        log::info!(
            "[{}] Streaming: generation completed: {} tokens in {}ms ({:.1} tok/s)",
            rid, tokens.len(), gen_ms, tok_per_sec
        );

        // Decode all tokens using streaming decoder
        log::debug!("[{}] Streaming: decoding and sending chunks...", rid);
        let decode_start = Instant::now();
        let mut decode_stream = state_clone.tokenizer.inner.decode_stream(true);
        let mut chunk_count = 0usize;
        let mut total_text_len = 0usize;
        for &token_id in &tokens {
            if eos_tokens.contains(&token_id) {
                log::debug!("[{}] Streaming: hit EOS token {}", rid, token_id);
                break;
            }
            if let Ok(Some(text_chunk)) = decode_stream.step(token_id as u32) {
                if !text_chunk.is_empty() {
                    total_text_len += text_chunk.len();
                    chunk_count += 1;
                    log::trace!(
                        "[{}] Streaming: chunk #{} token_id={} text={:?}",
                        rid, chunk_count, token_id, text_chunk
                    );
                    let chunk = StreamChunk {
                        id: req_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        model: model_nm.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: Some(text_chunk),
                            },
                            finish_reason: None,
                        }],
                    };
                    let _ = tx.blocking_send(Ok(
                        Event::default().data(serde_json::to_string(&chunk).unwrap())
                    ));
                }
            }
        }
        let decode_ms = decode_start.elapsed().as_millis();
        log::debug!(
            "[{}] Streaming: sent {} chunks ({} chars) in {}ms",
            rid, chunk_count, total_text_len, decode_ms
        );

        // Final chunk with finish_reason
        let final_chunk = StreamChunk {
            id: req_id.clone(),
            object: "chat.completion.chunk".to_string(),
            model: model_nm.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        };
        let _ = tx.blocking_send(Ok(
            Event::default().data(serde_json::to_string(&final_chunk).unwrap())
        ));
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));

        let total_ms = total_start.elapsed().as_millis();
        log::info!(
            "[{}] Streaming: request completed: {} prompt + {} completion tokens, {} SSE chunks in {}ms total (preprocess={}ms, tokenize={}ms, generate={}ms)",
            rid, prompt_tokens, tokens.len(), chunk_count, total_ms,
            preprocess_ms, tokenize_ms, gen_ms
        );
    });

    let stream = ReceiverStream::new(rx);
    Ok(Sse::new(stream))
}
