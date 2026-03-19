use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    Array,
};

use super::config::VisionConfig;

#[derive(Debug, Clone, ModuleParameters)]
pub struct GlmOcrVisionPatchEmbed {
    #[param]
    pub proj: nn::Conv3d,
    pub temporal_patch_size: i32,
    pub patch_size: i32,
    pub in_channels: i32,
}

impl GlmOcrVisionPatchEmbed {
    pub fn new(config: &VisionConfig) -> Result<Self, Exception> {
        let proj = nn::Conv3dBuilder::new(
            config.in_channels,
            config.hidden_size,
            (config.temporal_patch_size, config.patch_size, config.patch_size),
        )
        .stride((config.temporal_patch_size, config.patch_size, config.patch_size))
        .bias(true)
        .build()?;

        Ok(Self {
            proj,
            temporal_patch_size: config.temporal_patch_size,
            patch_size: config.patch_size,
            in_channels: config.in_channels,
        })
    }
}

impl Module<&Array> for GlmOcrVisionPatchEmbed {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, hidden_states: &Array) -> Result<Array, Exception> {
        // Input: [N, C*T*H*W] = [N, 1176] where 1176 = 3*2*14*14
        // Reshape to PyTorch layout: [N, C, T, H, W]
        let x = hidden_states.reshape(&[
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ])?;
        // moveaxis(1, 4): [N, C, T, H, W] -> [N, T, H, W, C] (channels-last for MLX Conv3d)
        let x = x.transpose_axes(&[0, 2, 3, 4, 1])?;
        // Conv3d (channels-last): [N, T, H, W, C] -> [N, 1, 1, 1, hidden_size]
        let x = self.proj.forward(&x)?;
        // Flatten to [N, hidden_size]
        let shape = x.shape();
        let hidden_size = shape[shape.len() - 1];
        x.reshape(&[-1, hidden_size])
    }

    fn training_mode(&mut self, mode: bool) {
        self.proj.training_mode(mode);
    }
}
