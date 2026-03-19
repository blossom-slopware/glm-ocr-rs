use std::path::Path;

use mlx_rs::{Array, Dtype};
use mlx_lm::cache::{ConcatKeyValueCache, KeyValueCache};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use crate::loader::load_glm_ocr_model;
use crate::model::language_model::GlmOcrModel;

#[pyclass(unsendable)]
pub struct PyGlmOcrModel {
    inner: GlmOcrModel,
    cache: Vec<Option<ConcatKeyValueCache>>,
}

#[pymethods]
impl PyGlmOcrModel {
    #[staticmethod]
    fn load(model_dir: &str) -> PyResult<Self> {
        let model = load_glm_ocr_model(Path::new(model_dir))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let cache = model.new_cache();
        Ok(Self { inner: model, cache })
    }

    /// Reset KV cache (call before a new sequence).
    fn reset_cache(&mut self) {
        self.cache = self.inner.new_cache();
    }

    /// Forward pass without cache (full sequence, no state).
    fn forward_with_shape<'py>(
        &mut self,
        py: Python<'py>,
        token_ids: PyReadonlyArray1<i32>,
        token_ids_shape: Vec<i32>,
        position_ids: PyReadonlyArray1<i32>,
        position_ids_shape: Vec<i32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<i32>)> {
        let token_arr = Array::from_slice(
            token_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &token_ids_shape,
        );
        let pos_arr = Array::from_slice(
            position_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &position_ids_shape,
        );

        // No cache
        let mut no_cache: Vec<Option<ConcatKeyValueCache>> =
            (0..self.inner.num_layers).map(|_| None).collect();
        let logits = self.inner.forward(&token_arr, &pos_arr, &mut no_cache)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        array_to_py(py, &logits)
    }

    /// Prefill: process prompt tokens, populate KV cache, return logits.
    fn prefill<'py>(
        &mut self,
        py: Python<'py>,
        token_ids: PyReadonlyArray1<i32>,
        token_ids_shape: Vec<i32>,
        position_ids: PyReadonlyArray1<i32>,
        position_ids_shape: Vec<i32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<i32>)> {
        self.reset_cache();

        let token_arr = Array::from_slice(
            token_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &token_ids_shape,
        );
        let pos_arr = Array::from_slice(
            position_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &position_ids_shape,
        );

        let logits = self.inner.forward(&token_arr, &pos_arr, &mut self.cache)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        array_to_py(py, &logits)
    }

    /// Decode: process a single token using cached KV, return logits.
    fn decode<'py>(
        &mut self,
        py: Python<'py>,
        token_ids: PyReadonlyArray1<i32>,
        token_ids_shape: Vec<i32>,
        position_ids: PyReadonlyArray1<i32>,
        position_ids_shape: Vec<i32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<i32>)> {
        let token_arr = Array::from_slice(
            token_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &token_ids_shape,
        );
        let pos_arr = Array::from_slice(
            position_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &position_ids_shape,
        );

        let logits = self.inner.forward(&token_arr, &pos_arr, &mut self.cache)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        array_to_py(py, &logits)
    }

    /// Prefill with pre-computed embeddings (from vision encoder).
    fn prefill_embeds<'py>(
        &mut self,
        py: Python<'py>,
        embeds: PyReadonlyArray1<f32>,
        embeds_shape: Vec<i32>,
        position_ids: PyReadonlyArray1<i32>,
        position_ids_shape: Vec<i32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<i32>)> {
        self.reset_cache();

        let embeds_arr = Array::from_slice(
            embeds.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &embeds_shape,
        );
        // Cast to bfloat16 to match model weights
        let embeds_arr = embeds_arr.as_dtype(Dtype::Bfloat16)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let pos_arr = Array::from_slice(
            position_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &position_ids_shape,
        );

        let logits = self.inner.forward_with_embeds(&embeds_arr, &pos_arr, &mut self.cache)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        array_to_py(py, &logits)
    }

    /// Continue prefill with embeddings (does NOT reset cache).
    fn continue_prefill_embeds<'py>(
        &mut self,
        py: Python<'py>,
        embeds: PyReadonlyArray1<f32>,
        embeds_shape: Vec<i32>,
        position_ids: PyReadonlyArray1<i32>,
        position_ids_shape: Vec<i32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<i32>)> {
        let embeds_arr = Array::from_slice(
            embeds.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &embeds_shape,
        );
        let embeds_arr = embeds_arr.as_dtype(Dtype::Bfloat16)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let pos_arr = Array::from_slice(
            position_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &position_ids_shape,
        );

        let logits = self.inner.forward_with_embeds(&embeds_arr, &pos_arr, &mut self.cache)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        array_to_py(py, &logits)
    }

    /// Get current cache offset (number of tokens processed).
    fn cache_offset(&self) -> i32 {
        self.cache.first()
            .and_then(|c| c.as_ref())
            .map(|c| c.offset())
            .unwrap_or(0)
    }

    /// Get number of layers.
    fn num_layers(&self) -> i32 {
        self.inner.num_layers
    }
}

fn array_to_py<'py>(py: Python<'py>, arr: &Array) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<i32>)> {
    let logits = arr.as_dtype(Dtype::Float32)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    logits.eval().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let shape = logits.shape().to_vec();
    let data: &[f32] = logits.as_slice();
    let result = PyArray1::from_slice(py, data);
    Ok((result, shape))
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGlmOcrModel>()?;
    Ok(())
}
