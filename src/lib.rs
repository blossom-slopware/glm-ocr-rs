pub mod config;
pub mod model;
pub mod loader;
pub mod py_bindings;

use pyo3::prelude::*;

#[pymodule]
fn mlx_ocr_2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_bindings::register_module(m)?;
    Ok(())
}
