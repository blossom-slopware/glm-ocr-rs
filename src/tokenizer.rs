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
pub fn render_chat_template(
    template_str: &str,
    messages: &[serde_json::Value],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> String {
    let mut env = Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template("chat", template_str)
        .unwrap_or_else(|e| panic!("Failed to add chat template: {}", e));

    let tmpl = env.get_template("chat")
        .unwrap_or_else(|e| panic!("Failed to get chat template: {}", e));

    let ctx = minijinja::context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
        enable_thinking => enable_thinking,
    };

    tmpl.render(ctx)
        .unwrap_or_else(|e| panic!("Chat template rendering failed: {}", e))
}
