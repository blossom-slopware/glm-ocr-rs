use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_heads: i32,
    pub patch_size: i32,
    #[serde(default = "default_in_channels")]
    pub in_channels: i32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    pub out_hidden_size: i32,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: i32,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: i32,
}

fn default_rms_norm_eps() -> f32 { 1e-5 }
fn default_attention_bias() -> bool { true }
fn default_spatial_merge_size() -> i32 { 2 }
fn default_temporal_patch_size() -> i32 { 2 }
fn default_in_channels() -> i32 { 3 }

impl VisionConfig {
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_heads
    }
}
