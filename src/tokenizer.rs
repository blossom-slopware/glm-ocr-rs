use minijinja::Environment;

use tokenizers::Tokenizer;

// Special token IDs
pub const TOKEN_IMAGE: u32 = 59280;
pub const TOKEN_BEGIN_IMAGE: u32 = 59256;
pub const TOKEN_END_IMAGE: u32 = 59257;
pub const TOKEN_USER: u32 = 59253;
pub const TOKEN_ASSISTANT: u32 = 59254;
pub const TOKEN_ENDOFTEXT: u32 = 59246;
pub const TOKEN_GMASK: u32 = 59248;
pub const TOKEN_SOP: u32 = 59250;

pub struct GlmTokenizer {
    pub inner: Tokenizer,
}

impl GlmTokenizer {
    pub fn from_file(path: &str) -> Self {
        let inner = Tokenizer::from_file(path)
            .unwrap_or_else(|e| panic!("Failed to load tokenizer from {}: {}", path, e));
        Self { inner }
    }

    /// Encode text with image token expansion.
    /// `num_image_tokens_per_image` is a slice with one entry per image,
    /// each being (T*H*W) / merge_size^2.
    pub fn encode(&self, text: &str, num_image_tokens_per_image: &[usize]) -> Vec<i32> {
        let expanded = expand_image_tokens(text, num_image_tokens_per_image);
        let encoding = self.inner.encode(expanded.as_str(), false)
            .unwrap_or_else(|e| panic!("Failed to encode text: {}", e));
        encoding.get_ids().iter().map(|&id| id as i32).collect()
    }

    /// Decode token IDs to string.
    pub fn decode(&self, ids: &[i32], skip_special_tokens: bool) -> String {
        let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner.decode(&ids_u32, skip_special_tokens)
            .unwrap_or_else(|e| panic!("Failed to decode tokens: {}", e))
    }
}

/// Expand <|image|> placeholders in text.
/// Python's GlmOcrProcessor does:
///   1. Replace each <|image|> with N copies of <|placeholder|> (one at a time)
///   2. Replace all <|placeholder|> back to <|image|>
fn expand_image_tokens(text: &str, num_tokens_per_image: &[usize]) -> String {
    let image_marker = "<|image|>";
    let placeholder = "<|placeholder|>";

    let mut result = text.to_string();
    let mut index = 0;

    // Stage 1: replace each <|image|> with N placeholders
    while let Some(pos) = result.find(image_marker) {
        assert!(
            index < num_tokens_per_image.len(),
            "More <|image|> markers in text than image token counts provided"
        );
        let n = num_tokens_per_image[index];
        let replacement = placeholder.repeat(n);
        // Replace only the first occurrence
        result = format!(
            "{}{}{}",
            &result[..pos],
            replacement,
            &result[pos + image_marker.len()..]
        );
        index += 1;
    }

    // Stage 2: replace all placeholders back to image markers
    result = result.replace(placeholder, image_marker);

    result
}

/// Render the chat template using minijinja.
/// Falls back to Rust-native rendering if minijinja fails.
pub fn render_chat_template(
    template_str: &str,
    messages: &[serde_json::Value],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    // Try minijinja first
    match render_with_minijinja(template_str, messages, add_generation_prompt, enable_thinking) {
        Ok(result) => result,
        Err(e) => {
            log::warn!("minijinja template rendering failed: {}; using native renderer", e);
            render_native(messages, add_generation_prompt, enable_thinking)
        }
    }
}

fn render_with_minijinja(
    template_str: &str,
    messages: &[serde_json::Value],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Result<String, minijinja::Error> {
    let mut env = Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.add_template("chat", template_str)?;

    let tmpl = env.get_template("chat")?;

    let ctx = minijinja::context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
        enable_thinking => enable_thinking,
    };

    tmpl.render(ctx)
}

/// Native Rust implementation of the chat template.
/// Faithfully reproduces the Jinja2 template logic for all cases.
fn render_native(
    messages: &[serde_json::Value],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    let mut output = String::new();

    // Start with [gMASK]<sop>
    output.push_str("[gMASK]<sop>");

    // Find last user message index
    let mut last_user_index: i64 = -1;
    for (i, m) in messages.iter().enumerate() {
        if m.get("role").and_then(|r| r.as_str()) == Some("user") {
            last_user_index = i as i64;
        }
    }

    // Process each message
    for (i, m) in messages.iter().enumerate() {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let content = &m["content"];

        match role {
            "user" => {
                output.push_str("<|user|>\n");
                if let Some(s) = content.as_str() {
                    output.push_str(s);
                } else if let Some(arr) = content.as_array() {
                    for item in arr {
                        let item_type = item.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if item_type == "video" || item.get("video").is_some() {
                            output.push_str("\n<|begin_of_video|><|video|><|end_of_video|>");
                        } else if item_type == "image" || item.get("image").is_some() {
                            output.push_str("\n<|begin_of_image|><|image|><|end_of_image|>");
                        } else if item_type == "text" {
                            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                                output.push_str("\n");
                                output.push_str(text);
                            }
                        }
                    }
                }
                // Add /nothink if enable_thinking is false
                if !enable_thinking {
                    let visible = visible_text(content);
                    if !visible.ends_with("/nothink") {
                        output.push_str("/nothink");
                    }
                }
            }
            "assistant" => {
                output.push_str("<|assistant|>");
                let visible = visible_text(content);
                let mut reasoning_content = String::new();
                let mut content_text = visible.clone();

                if let Some(rc) = m.get("reasoning_content").and_then(|r| r.as_str()) {
                    reasoning_content = rc.to_string();
                } else if content_text.contains("</think>") {
                    let parts: Vec<&str> = content_text.splitn(2, "</think>").collect();
                    if parts.len() == 2 {
                        let before_think = parts[0];
                        let after_parts: Vec<&str> = before_think.rsplitn(2, "<think>").collect();
                        if after_parts.len() == 2 {
                            reasoning_content = after_parts[0].trim_start_matches('\n')
                                .trim_end_matches('\n').to_string();
                        }
                        content_text = parts[1].trim_start_matches('\n').to_string();
                    }
                }

                if (i as i64) > last_user_index && !reasoning_content.is_empty() {
                    output.push_str(&format!("\n<think>{}</think>", reasoning_content.trim()));
                } else {
                    output.push_str("\n<think></think>");
                }

                let stripped = content_text.trim();
                if !stripped.is_empty() {
                    output.push_str(&format!("\n{}", stripped));
                }

                // Handle tool_calls
                if let Some(tool_calls) = m.get("tool_calls").and_then(|t| t.as_array()) {
                    output.push_str("\n");
                    for tc in tool_calls {
                        let tc_obj = if let Some(func) = tc.get("function") {
                            func
                        } else {
                            tc
                        };
                        if let Some(name) = tc_obj.get("name").and_then(|n| n.as_str()) {
                            output.push_str(&format!("\n<tool_call>{}\n", name));
                            if let Some(args) = tc_obj.get("arguments").and_then(|a| a.as_object()) {
                                for (k, v) in args {
                                    let v_str = if let Some(s) = v.as_str() {
                                        s.to_string()
                                    } else {
                                        serde_json::to_string(v).unwrap_or_default()
                                    };
                                    output.push_str(&format!("<arg_key>{}</arg_key>\n", k));
                                    output.push_str(&format!("<arg_value>{}</arg_value>\n", v_str));
                                }
                            }
                            output.push_str("</tool_call>");
                        }
                    }
                    output.push_str("\n");
                }
            }
            "tool" => {
                // Check if previous message was also tool
                let is_first_tool = i == 0 || messages.get(i - 1)
                    .and_then(|m| m.get("role"))
                    .and_then(|r| r.as_str()) != Some("tool");

                if let Some(s) = content.as_str() {
                    if is_first_tool {
                        output.push_str("<|observation|>");
                    }
                    output.push_str(&format!("\n<tool_response>\n{}\n</tool_response>\n", s));
                } else if let Some(arr) = content.as_array() {
                    if is_first_tool {
                        output.push_str("<|observation|>");
                    }
                    output.push_str("\n<tool_response>\n");
                    for tr in arr {
                        if let Some(obj) = tr.as_object() {
                            if let Some(t) = obj.get("type").and_then(|t| t.as_str()) {
                                let t_lower = t.to_lowercase();
                                if t_lower == "text" {
                                    if let Some(text) = obj.get("text").and_then(|t| t.as_str()) {
                                        output.push_str(text);
                                    }
                                } else if t_lower == "image" || t_lower == "image_url" {
                                    output.push_str("<|begin_of_image|><|image|><|end_of_image|>");
                                } else if t_lower == "video" || t_lower == "video_url" {
                                    output.push_str("<|begin_of_video|><|video|><|end_of_video|>");
                                } else {
                                    output.push_str(&serde_json::to_string(tr).unwrap_or_default());
                                }
                            } else if let Some(out) = obj.get("output").and_then(|o| o.as_str()) {
                                output.push_str(out);
                            } else {
                                output.push_str(&serde_json::to_string(tr).unwrap_or_default());
                            }
                        }
                    }
                    output.push_str("\n</tool_response>");
                }
            }
            "system" => {
                output.push_str("<|system|>\n");
                output.push_str(&visible_text(content));
            }
            _ => {}
        }
    }

    // Add generation prompt
    if add_generation_prompt {
        output.push_str("<|assistant|>\n");
        if !enable_thinking {
            output.push_str("<think></think>\n");
        }
    }

    output
}

/// Extract visible text from content (string or array of content parts).
fn visible_text(content: &serde_json::Value) -> String {
    if let Some(s) = content.as_str() {
        return s.to_string();
    }
    if let Some(arr) = content.as_array() {
        let mut result = String::new();
        for item in arr {
            if let Some(obj) = item.as_object() {
                let item_type = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
                if item_type == "text" {
                    if let Some(text) = obj.get("text").and_then(|t| t.as_str()) {
                        result.push_str(text);
                    }
                } else if item_type == "image" || obj.contains_key("image") {
                    result.push_str("<|begin_of_image|><|image|><|end_of_image|>");
                } else if item_type == "video" || obj.contains_key("video") {
                    result.push_str("<|begin_of_video|><|video|><|end_of_video|>");
                }
            } else if let Some(s) = item.as_str() {
                result.push_str(s);
            }
        }
        return result;
    }
    content.to_string()
}
