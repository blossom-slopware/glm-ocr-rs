use anyhow::{Context, anyhow, bail};
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
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow!(e))
            .with_context(|| format!("failed to load tokenizer from {path}"))?;
        Ok(Self { inner })
    }

    /// Encode text with image token expansion.
    /// `num_image_tokens_per_image` is a slice with one entry per image,
    /// each being (T*H*W) / merge_size^2.
    pub fn encode(&self, text: &str, num_image_tokens_per_image: &[usize]) -> anyhow::Result<Vec<i32>> {
        let expanded = expand_image_tokens(text, num_image_tokens_per_image)?;
        let encoding = self.inner.encode(expanded.as_str(), false)
            .map_err(|e| anyhow!(e))
            .context("failed to encode prompt text")?;
        Ok(encoding.get_ids().iter().map(|&id| id as i32).collect())
    }

    /// Decode token IDs to string.
    pub fn decode(&self, ids: &[i32], skip_special_tokens: bool) -> anyhow::Result<String> {
        let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner.decode(&ids_u32, skip_special_tokens)
            .map_err(|e| anyhow!(e))
            .context("failed to decode generated tokens")
    }
}

/// Expand <|image|> placeholders in text.
/// Python's GlmOcrProcessor does:
///   1. Replace each <|image|> with N copies of <|placeholder|> (one at a time)
///   2. Replace all <|placeholder|> back to <|image|>
fn expand_image_tokens(text: &str, num_tokens_per_image: &[usize]) -> anyhow::Result<String> {
    let image_marker = "<|image|>";
    let placeholder = "<|placeholder|>";

    let mut result = text.to_string();
    let mut index = 0usize;

    // Stage 1: replace each <|image|> with N placeholders
    while let Some(pos) = result.find(image_marker) {
        if index >= num_tokens_per_image.len() {
            bail!("more <|image|> markers in text than image token counts provided");
        }
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

    if index != num_tokens_per_image.len() {
        bail!(
            "unused image token counts: expected {} markers, found {}",
            num_tokens_per_image.len(),
            index
        );
    }

    // Stage 2: replace all placeholders back to image markers
    result = result.replace(placeholder, image_marker);

    Ok(result)
}

/// Render the chat template using minijinja.
pub fn render_chat_template(
    template_str: &str,
    messages: &[serde_json::Value],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> anyhow::Result<String> {
    let mut env = Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template("chat", template_str)
        .context("failed to register chat template")?;

    let tmpl = env.get_template("chat")
        .context("failed to load registered chat template")?;

    let ctx = minijinja::context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
        enable_thinking => enable_thinking,
    };

    tmpl.render(ctx)
        .context("chat template rendering failed")
}
