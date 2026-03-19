use mlx_rs::{Array, Dtype};
use mlx_lm::cache::{ConcatKeyValueCache, KeyValueCache};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

use crate::full_model::Model;
use crate::generate::{self, GenerateConfig};

#[pyclass(unsendable)]
pub struct PyFullModel {
    inner: Model,
    cache: Vec<Option<ConcatKeyValueCache>>,
}

#[pymethods]
impl PyFullModel {
    #[staticmethod]
    fn load(model_dir: &str) -> PyResult<Self> {
        let model = Model::load(model_dir)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let cache = model.new_cache();
        Ok(Self { inner: model, cache })
    }

    /// Reset KV cache (call before a new sequence).
    fn reset_cache(&mut self) {
        self.cache = self.inner.new_cache();
        self.inner.position_ids = None;
        self.inner.rope_deltas = None;
    }

    /// Get current cache offset.
    fn cache_offset(&self) -> i32 {
        self.cache.first()
            .and_then(|c| c.as_ref())
            .map(|c| c.offset())
            .unwrap_or(0)
    }

    /// Get EOS token IDs from config.
    fn eos_token_ids(&self) -> Vec<i32> {
        self.inner.config.eos_token_id.clone()
    }

    /// Full forward pass (prefill with vision or decode).
    /// Returns (logits_flat, logits_shape).
    fn forward<'py>(
        &mut self,
        py: Python<'py>,
        input_ids: PyReadonlyArray1<i32>,
        input_ids_shape: Vec<i32>,
        pixel_values: Option<PyReadonlyArray1<f32>>,
        pixel_values_shape: Option<Vec<i32>>,
        grid_thw: Option<PyReadonlyArray1<i32>>,
        grid_thw_shape: Option<Vec<i32>>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<i32>)> {
        let input_ids_arr = Array::from_slice(
            input_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &input_ids_shape,
        );

        let pv_arr = pixel_values.map(|pv| -> PyResult<Array> {
            let arr = Array::from_slice(
                pv.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                pixel_values_shape.as_ref().unwrap(),
            );
            arr.as_dtype(Dtype::Bfloat16).map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }).transpose()?;

        let grid: Option<Vec<(i32, i32, i32)>> = grid_thw.map(|g| -> PyResult<Vec<(i32, i32, i32)>> {
            let slice = g.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let shape = grid_thw_shape.as_ref().unwrap();
            let num_images = shape[0] as usize;
            Ok((0..num_images)
                .map(|i| (slice[i * 3], slice[i * 3 + 1], slice[i * 3 + 2]))
                .collect())
        }).transpose()?;

        let cache_offset = self.cache_offset();

        let logits = self.inner.forward(
            &input_ids_arr,
            pv_arr.as_ref(),
            grid.as_deref(),
            None,
            &mut self.cache,
            cache_offset,
        ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        array_to_py(py, &logits)
    }

    /// Full generation: vision + prefill + decode loop.
    /// Returns list of generated token IDs (not including prompt tokens).
    #[pyo3(signature = (
        input_ids, input_ids_shape,
        pixel_values=None, pixel_values_shape=None,
        grid_thw=None, grid_thw_shape=None,
        max_tokens=4096,
        temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
        repetition_penalty=1.0, repetition_context_size=20,
        prefill_step_size=2048,
        stop_tokens=None
    ))]
    fn generate(
        &mut self,
        input_ids: PyReadonlyArray1<i32>,
        input_ids_shape: Vec<i32>,
        pixel_values: Option<PyReadonlyArray1<f32>>,
        pixel_values_shape: Option<Vec<i32>>,
        grid_thw: Option<PyReadonlyArray1<i32>>,
        grid_thw_shape: Option<Vec<i32>>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        min_p: f32,
        repetition_penalty: f32,
        repetition_context_size: usize,
        prefill_step_size: i32,
        stop_tokens: Option<Vec<i32>>,
    ) -> PyResult<Vec<i32>> {
        self.reset_cache();

        let input_ids_arr = Array::from_slice(
            input_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &input_ids_shape,
        );

        let pv_arr = pixel_values.map(|pv| -> PyResult<Array> {
            let arr = Array::from_slice(
                pv.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                pixel_values_shape.as_ref().unwrap(),
            );
            arr.as_dtype(Dtype::Bfloat16).map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }).transpose()?;

        let grid: Option<Vec<(i32, i32, i32)>> = grid_thw.map(|g| -> PyResult<Vec<(i32, i32, i32)>> {
            let slice = g.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let shape = grid_thw_shape.as_ref().unwrap();
            let num_images = shape[0] as usize;
            Ok((0..num_images)
                .map(|i| (slice[i * 3], slice[i * 3 + 1], slice[i * 3 + 2]))
                .collect())
        }).transpose()?;

        let mut stop = stop_tokens.unwrap_or_default();
        // Add EOS tokens from config
        for eos in &self.inner.config.eos_token_id {
            if !stop.contains(eos) {
                stop.push(*eos);
            }
        }

        let config = GenerateConfig {
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            repetition_context_size,
            prefill_step_size,
            stop_tokens: stop,
        };

        let result = generate::generate(
            &mut self.inner,
            &input_ids_arr,
            pv_arr.as_ref(),
            grid.as_deref(),
            None,
            &config,
        ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(result)
    }
    /// Get text-only embeddings (no vision). Returns (embeds_flat, embeds_shape).
    fn get_text_embeddings<'py>(
        &mut self,
        py: Python<'py>,
        input_ids: PyReadonlyArray1<i32>,
        input_ids_shape: Vec<i32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<i32>)> {
        let input_ids_arr = Array::from_slice(
            input_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &input_ids_shape,
        );
        let embeds = self.inner.language_model.embed_tokens(&input_ids_arr)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        array_to_py(py, &embeds)
    }

    /// Run vision encoder + embedding merge in Rust. Returns (embeds_flat, embeds_shape).
    fn get_input_embeddings<'py>(
        &mut self,
        py: Python<'py>,
        input_ids: PyReadonlyArray1<i32>,
        input_ids_shape: Vec<i32>,
        pixel_values: PyReadonlyArray1<f32>,
        pixel_values_shape: Vec<i32>,
        grid_thw: PyReadonlyArray1<i32>,
        grid_thw_shape: Vec<i32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<i32>)> {
        let input_ids_arr = Array::from_slice(
            input_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &input_ids_shape,
        );
        let pv_arr = Array::from_slice(
            pixel_values.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &pixel_values_shape,
        );
        let grid_slice = grid_thw.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let num_images = grid_thw_shape[0] as usize;
        let grid: Vec<(i32, i32, i32)> = (0..num_images)
            .map(|i| (grid_slice[i * 3], grid_slice[i * 3 + 1], grid_slice[i * 3 + 2]))
            .collect();

        let embeds = self.inner.get_input_embeddings(
            &input_ids_arr,
            Some(&pv_arr),
            Some(&grid),
            None,
        ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        array_to_py(py, &embeds)
    }

    /// Get number of decoder layers.
    fn num_layers(&self) -> i32 {
        self.inner.language_model.num_layers
    }

    /// Prefill with token IDs (resets cache).
    fn prefill<'py>(
        &mut self,
        py: Python<'py>,
        token_ids: PyReadonlyArray1<i32>,
        token_ids_shape: Vec<i32>,
        position_ids: PyReadonlyArray1<i32>,
        position_ids_shape: Vec<i32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<i32>)> {
        self.cache = self.inner.new_cache();
        let token_arr = Array::from_slice(
            token_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &token_ids_shape,
        );
        let pos_arr = Array::from_slice(
            position_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &position_ids_shape,
        );
        let logits = self.inner.language_model.forward(&token_arr, &pos_arr, &mut self.cache)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        array_to_py(py, &logits)
    }

    /// Decode with token IDs (uses existing cache).
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
        let logits = self.inner.language_model.forward(&token_arr, &pos_arr, &mut self.cache)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        array_to_py(py, &logits)
    }

    /// Prefill with pre-computed embeddings (resets cache).
    fn prefill_embeds<'py>(
        &mut self,
        py: Python<'py>,
        embeds: PyReadonlyArray1<f32>,
        embeds_shape: Vec<i32>,
        position_ids: PyReadonlyArray1<i32>,
        position_ids_shape: Vec<i32>,
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<i32>)> {
        self.cache = self.inner.new_cache();
        let embeds_arr = Array::from_slice(
            embeds.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &embeds_shape,
        ).as_dtype(Dtype::Bfloat16).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let pos_arr = Array::from_slice(
            position_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &position_ids_shape,
        );
        let logits = self.inner.language_model.forward_with_embeds(&embeds_arr, &pos_arr, &mut self.cache)
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
        ).as_dtype(Dtype::Bfloat16).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let pos_arr = Array::from_slice(
            position_ids.as_slice().map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            &position_ids_shape,
        );
        let logits = self.inner.language_model.forward_with_embeds(&embeds_arr, &pos_arr, &mut self.cache)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        array_to_py(py, &logits)
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
    m.add_class::<PyFullModel>()?;
    Ok(())
}
