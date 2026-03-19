pub mod config;
pub mod model;
pub mod vision;
pub mod loader;
pub mod py_bindings;
pub mod full_model;
pub mod sampler;
pub mod generate;

use pyo3::prelude::*;

#[pymodule]
fn mlx_ocr_2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    env_logger::try_init().ok();
    py_bindings::register_module(m)?;
    Ok(())
}
