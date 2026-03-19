pub mod mrope;
pub mod attention;
pub mod mlp;
pub mod decoder;
pub mod text_model;
pub mod language_model;

use mlx_rs::Array;

/// Mask type for attention: None (no mask), Causal (built-in), or explicit Array.
pub enum AttentionMask {
    None,
    Causal,
    Array(Array),
}

impl AttentionMask {
    pub fn as_sdpa_mask(&self) -> Option<mlx_rs::fast::ScaledDotProductAttentionMask<'_>> {
        match self {
            AttentionMask::None => None,
            AttentionMask::Causal => Some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal),
            AttentionMask::Array(a) => Some(mlx_rs::fast::ScaledDotProductAttentionMask::Array(a)),
        }
    }
}
