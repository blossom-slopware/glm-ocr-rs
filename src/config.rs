use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
pub struct FullConfig {
    pub text_config: TextConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub model_type: String,
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: i32,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
    pub rope_parameters: HashMap<String, Value>,
}

impl TextConfig {
    pub fn mrope_section(&self) -> Vec<i32> {
        self.rope_parameters
            .get("mrope_section")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(|v| v.as_i64().unwrap() as i32).collect())
            .unwrap_or_else(|| vec![16, 24, 24])
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(10000.0)
    }

    pub fn partial_rotary_factor(&self) -> f32 {
        self.rope_parameters
            .get("partial_rotary_factor")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(1.0)
    }
}
