use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    Array,
};

use crate::config::TextConfig;

#[derive(Debug, Clone, ModuleParameters)]
pub struct GlmOcrMlp {
    #[param]
    pub gate_up_proj: nn::Linear,
    #[param]
    pub down_proj: nn::Linear,
}

impl GlmOcrMlp {
    pub fn new(config: &TextConfig) -> Result<Self, Exception> {
        let gate_up_proj = nn::LinearBuilder::new(config.hidden_size, config.intermediate_size * 2)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(config.intermediate_size, config.hidden_size)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_up_proj,
            down_proj,
        })
    }
}

impl Module<&Array> for GlmOcrMlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let x = self.gate_up_proj.forward(x)?;
        // Split into gate and up, each of size intermediate_size
        let parts = x.split(2, -1)?;
        let gate = &parts[0];
        let up = &parts[1];
        let h = nn::silu(gate.clone())?.multiply(up)?;
        self.down_proj.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}
