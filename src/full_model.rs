use std::path::Path;

use mlx_rs::{error::Exception, Array, Dtype};
use mlx_rs::module::Module;
use mlx_rs::ops::{self, indexing::IndexOp};
use crate::cache::KVCache;
use crate::config::FullConfig;
use crate::model::language_model::GlmOcrModel;
use crate::vision::vision_model::{VisionTower, VisionModelInput};
use crate::loader::{load_full_config, load_glm_ocr_model, load_vision_model};

/// Full GLM-OCR model: vision encoder + language model + embedding merge.
pub struct Model {
    pub vision_tower: VisionTower,
    pub language_model: GlmOcrModel,
    pub config: FullConfig,

    // Stateful for decode: cached from prefill
    pub position_ids: Option<Array>,
    pub rope_deltas: Option<Array>,
}

impl Model {
    fn scalar_i32(array: &Array) -> Result<i32, Exception> {
        let value = array.as_dtype(Dtype::Int32)?;
        value.eval()?;
        Ok(value.item::<i32>())
    }

    fn vec_i32(array: &Array) -> Result<Vec<i32>, Exception> {
        let value = array.as_dtype(Dtype::Int32)?;
        value.eval()?;
        Ok(value.as_slice::<i32>().to_vec())
    }

    pub fn load(model_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let path = Path::new(model_dir);
        let config = load_full_config(path)?;
        let language_model = load_glm_ocr_model(path)?;
        let vision_tower = load_vision_model(path)?;
        Ok(Self {
            vision_tower,
            language_model,
            config,
            position_ids: None,
            rope_deltas: None,
        })
    }

    pub fn new_cache(&self) -> Vec<KVCache> {
        self.language_model.new_cache()
    }

    /// Merge image features into text embeddings at image token positions.
    pub fn merge_input_ids_with_image_features(
        image_token_id: i32,
        video_token_id: i32,
        image_features: &Array,    // [total_image_tokens, hidden]
        inputs_embeds: &Array,      // [B, seq_len, hidden]
        input_ids: &Array,          // [B, seq_len]
    ) -> Result<Array, Exception> {
        let image_token = Array::from_int(image_token_id);
        let mut image_positions = input_ids.eq(&image_token)?;

        let sum_positions = ops::sum(&image_positions, None)?;
        sum_positions.eval()?;
        let sum_val: i32 = sum_positions.item();
        if sum_val == 0 {
            let video_token = Array::from_int(video_token_id);
            image_positions = input_ids.eq(&video_token)?;
        }

        let batch_size = input_ids.shape()[0];
        let mut batch_outputs = Vec::with_capacity(batch_size as usize);
        let mut feature_start_idx: i32 = 0;

        for batch_idx in 0..batch_size {
            let image_mask = image_positions.index(batch_idx); // [seq_len]
            let num_positions_arr = ops::sum(&image_mask, None)?;
            num_positions_arr.eval()?;
            let num_positions: i32 = num_positions_arr.item();

            if num_positions > 0 {
                let batch_features = image_features.index(
                    feature_start_idx..(feature_start_idx + num_positions)
                );

                assert_eq!(
                    batch_features.shape()[0], num_positions,
                    "Number of image token positions ({}) does not match number of image features ({}) for batch {}",
                    num_positions, batch_features.shape()[0], batch_idx
                );

                let cumsum = ops::cumsum(
                    &image_mask.as_dtype(Dtype::Int32)?,
                    Some(0), None, None,
                )?;
                let ones = Array::from_int(1);
                let feature_indices = ops::which(
                    &image_mask,
                    &ops::subtract(&cumsum, &ones)?,
                    &Array::from_int(0),
                )?;

                let gathered_features = batch_features.index(&feature_indices);
                let image_mask_expanded = image_mask.reshape(&[-1, 1])?;
                let batch_embed = inputs_embeds.index(batch_idx);
                let batch_output = ops::which(
                    &image_mask_expanded,
                    &gathered_features,
                    &batch_embed,
                )?;

                feature_start_idx += num_positions;
                batch_outputs.push(batch_output);
            } else {
                batch_outputs.push(inputs_embeds.index(batch_idx));
            }
        }

        let refs: Vec<&Array> = batch_outputs.iter().collect();
        ops::stack_axis(&refs, 0)
    }

    /// Compute M-RoPE position_ids and rope_deltas from input_ids and image grid info.
    pub fn get_rope_index(
        &self,
        input_ids: &Array,
        image_grid_thw: Option<&[(i32, i32, i32)]>,
        video_grid_thw: Option<&[(i32, i32, i32)]>,
        attention_mask: Option<&Array>,
    ) -> Result<(Array, Array), Exception> {
        let batch_size = input_ids.shape()[0];
        let seq_length = input_ids.shape()[1];
        let mut position_ids = ops::arange::<_, i32>(0, seq_length, None)?;
        position_ids = position_ids.reshape(&[1, seq_length])?;
        position_ids = ops::broadcast_to(&position_ids, &[batch_size, seq_length])?;

        let spatial_merge_size = self.config.vision_config.spatial_merge_size;
        let image_token_id = self.config.image_token_id;
        let video_token_id = self.config.video_token_id;
        let image_start_token_id = self.config.image_start_token_id;

        if image_grid_thw.is_some() || video_grid_thw.is_some() {
            let total_input_ids = input_ids;
            let attention_mask = match attention_mask {
                Some(m) if m.shape().last().copied() == Some(seq_length) => m.clone(),
                _ => ops::ones::<i32>(&[batch_size, seq_length])?,
            };

            position_ids = ops::ones::<i32>(&[3, batch_size, seq_length])?;
            let mut image_index: usize = 0;
            let mut video_index: usize = 0;
            let mut mrope_position_deltas: Vec<i32> = Vec::new();

            for i in 0..batch_size {
                let input_ids_i = total_input_ids.index(i);
                let attn_mask_i = attention_mask.index(i);
                let input_ids_masked = ops::which(
                    &attn_mask_i.eq(&Array::from_int(1))?,
                    &input_ids_i,
                    &ops::zeros::<i32>(&[seq_length])?,
                )?;

                let start_positions = ops::which(
                    &input_ids_masked.eq(&Array::from_int(image_start_token_id))?,
                    &ops::arange::<_, i32>(0, seq_length, None)?,
                    &ops::zeros::<i32>(&[seq_length])?,
                )?;
                let vision_start_indices = ops::sum(&start_positions, None)?;
                let vision_start_index = Self::scalar_i32(&vision_start_indices)?;
                let vision_tokens = input_ids_masked.index(vision_start_index + 1);
                let image_nums = Self::scalar_i32(
                    &ops::sum(&vision_tokens.eq(&Array::from_int(image_token_id))?, None)?,
                )?;
                let video_nums = Self::scalar_i32(
                    &ops::sum(&vision_tokens.eq(&Array::from_int(video_token_id))?, None)?,
                )?;

                let input_tokens = Self::vec_i32(&input_ids_masked)?;

                let mut llm_pos_ids_list: Vec<Array> = Vec::new();
                let mut st: usize = 0;
                let mut remain_images = image_nums;
                let mut remain_videos = video_nums;

                for _ in 0..(image_nums + video_nums) {
                    let ed_image = if remain_images > 0 {
                        input_tokens[st..].iter().position(|&t| t == image_token_id)
                            .map(|p| p + st)
                            .unwrap_or(input_tokens.len() + 1)
                    } else {
                        input_tokens.len() + 1
                    };
                    let ed_video = if remain_videos > 0 {
                        input_tokens[st..].iter().position(|&t| t == video_token_id)
                            .map(|p| p + st)
                            .unwrap_or(input_tokens.len() + 1)
                    } else {
                        input_tokens.len() + 1
                    };

                    let (t, h, w, ed);
                    if ed_image < ed_video {
                        let grid = image_grid_thw.unwrap()[image_index];
                        t = grid.0;
                        h = grid.1;
                        w = grid.2;
                        image_index += 1;
                        remain_images -= 1;
                        ed = ed_image;
                    } else {
                        let grid = video_grid_thw.unwrap()[video_index];
                        t = grid.0;
                        h = grid.1;
                        w = grid.2;
                        video_index += 1;
                        remain_videos -= 1;
                        ed = ed_video;
                    }

                    let llm_grid_t = t;
                    let llm_grid_h = h / spatial_merge_size;
                    let llm_grid_w = w / spatial_merge_size;

                    let text_len = (ed - st) as i32;
                    let st_idx = if !llm_pos_ids_list.is_empty() {
                        let last = llm_pos_ids_list.last().unwrap();
                        Self::scalar_i32(&ops::max(last, None)?)? + 1
                    } else {
                        0
                    };

                    let text_index = ops::arange::<_, i32>(0, text_len, None)?.reshape(&[1, text_len])?;
                    let text_index = ops::broadcast_to(&text_index, &[3, text_len])?;
                    let text_index = ops::add(&text_index, &Array::from_int(st_idx))?;
                    llm_pos_ids_list.push(text_index);

                    // Vision positions: T/H/W grids
                    let num_patches = llm_grid_t * llm_grid_h * llm_grid_w;

                    let t_index = ops::arange::<_, i32>(0, llm_grid_t, None)?
                        .reshape(&[llm_grid_t, 1])?;
                    let t_index = ops::broadcast_to(&t_index, &[llm_grid_t, llm_grid_h * llm_grid_w])?;
                    let t_index = ops::flatten(&t_index, None, None)?;

                    let h_index = ops::arange::<_, i32>(0, llm_grid_h, None)?
                        .reshape(&[1, llm_grid_h, 1])?;
                    let h_index = ops::broadcast_to(&h_index, &[llm_grid_t, llm_grid_h, llm_grid_w])?;
                    let h_index = ops::flatten(&h_index, None, None)?;

                    let w_index = ops::arange::<_, i32>(0, llm_grid_w, None)?
                        .reshape(&[1, 1, llm_grid_w])?;
                    let w_index = ops::broadcast_to(&w_index, &[llm_grid_t, llm_grid_h, llm_grid_w])?;
                    let w_index = ops::flatten(&w_index, None, None)?;

                    let vision_pos = ops::stack_axis(&[&t_index, &h_index, &w_index], 0)?;
                    let offset = Array::from_int(text_len + st_idx);
                    let vision_pos = ops::add(&vision_pos, &offset)?;
                    llm_pos_ids_list.push(vision_pos);

                    st = ed + (num_patches as usize);
                }

                if st < input_tokens.len() {
                    let st_idx = if !llm_pos_ids_list.is_empty() {
                        let last = llm_pos_ids_list.last().unwrap();
                        Self::scalar_i32(&ops::max(last, None)?)? + 1
                    } else {
                        0
                    };
                    let text_len = (input_tokens.len() - st) as i32;
                    let t_index = ops::arange::<_, i32>(0, text_len, None)?.reshape(&[1, text_len])?;
                    let t_index = ops::broadcast_to(&t_index, &[3, text_len])?;
                    let t_index = ops::add(&t_index, &Array::from_int(st_idx))?;
                    llm_pos_ids_list.push(t_index);
                }

                let refs: Vec<&Array> = llm_pos_ids_list.iter().collect();
                let llm_positions = ops::concatenate_axis(&refs, 1)?
                    .reshape(&[3, -1])?;

                let mask_i = attention_mask.index(i).eq(&Array::from_int(1))?;
                let expanded_mask = mask_i.reshape(&[1, 1, seq_length])?;
                let expanded_mask = ops::broadcast_to(&expanded_mask, &[3, 1, seq_length])?;
                let expanded_positions = llm_positions.reshape(&[3, 1, -1])?;

                let current_pos = position_ids.index((.., i..i+1, ..));
                let new_positions = ops::which(&expanded_mask, &expanded_positions, &current_pos)?;

                if batch_size == 1 {
                    position_ids = new_positions;
                } else {
                    let before = position_ids.index((.., ..i, ..));
                    let after = position_ids.index((.., (i+1).., ..));
                    let parts: Vec<&Array> = vec![&before, &new_positions, &after];
                    position_ids = ops::concatenate_axis(&parts, 1)?;
                }

                let max_pos_val = Self::scalar_i32(&ops::max(&llm_positions, None)?)?;
                let total_len = total_input_ids.index(i).shape()[0];
                let delta = max_pos_val + 1 - total_len;
                mrope_position_deltas.push(delta);
            }

            let delta_arr = Array::from_slice(&mrope_position_deltas, &[mrope_position_deltas.len() as i32]);
            let delta_arr = delta_arr.index(0);
            Ok((position_ids, delta_arr))
        } else {
            if let Some(attn_mask) = attention_mask {
                let cumsum = ops::cumsum(&attn_mask.as_dtype(Dtype::Int64)?, Some(-1), None, None)?;
                let ones = Array::from_int(1);
                let pos = ops::subtract(&cumsum, &ones)?;
                let zero_mask = attn_mask.eq(&Array::from_int(0))?;
                let pos = ops::which(&zero_mask, &ops::ones::<i32>(&pos.shape())?, &pos)?;
                let pos_slice = pos.index(0);
                let pos_reshaped = pos_slice.reshape(&[1, 1, -1])?;
                let pos = ops::broadcast_to(&pos_reshaped, &[3, batch_size, seq_length])?;

                let max_pos = ops::max(&pos.index((0, 0, ..)), None)?;
                let max_val = Self::scalar_i32(&max_pos)?;
                let delta = max_val + 1 - seq_length;
                Ok((pos, Array::from_int(delta)))
            } else {
                let pos = ops::arange::<_, i32>(0, seq_length, None)?.reshape(&[1, 1, -1])?;
                let pos = ops::broadcast_to(&pos, &[3, batch_size, seq_length])?;
                let delta = ops::zeros::<i32>(&[batch_size, 1])?;
                Ok((pos, delta))
            }
        }
    }

    /// Get input embeddings, optionally merging vision features.
    pub fn get_input_embeddings(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
        image_grid_thw: Option<&[(i32, i32, i32)]>,
        mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        if pixel_values.is_none() {
            self.position_ids = None;
            self.rope_deltas = None;
            return self.language_model.embed_tokens(input_ids);
        }

        let pixel_values = pixel_values.unwrap();
        let inputs_embeds = self.language_model.embed_tokens(input_ids)?;

        let pixel_values_bf16 = pixel_values.as_dtype(Dtype::Bfloat16)?;
        let hidden_states = self.vision_tower.vision_tower.forward(VisionModelInput {
            pixel_values: &pixel_values_bf16,
            grid_thw: image_grid_thw.unwrap(),
        })?;

        let final_inputs_embeds = Self::merge_input_ids_with_image_features(
            self.config.image_token_id,
            self.config.video_token_id,
            &hidden_states,
            &inputs_embeds,
            input_ids,
        )?;

        // Pre-calculate position_ids for chunked prefill
        if image_grid_thw.is_some() {
            let (position_ids, rope_deltas) = self.get_rope_index(
                input_ids, image_grid_thw, None, mask,
            )?;
            self.position_ids = Some(position_ids);
            self.rope_deltas = Some(rope_deltas);
        }

        Ok(final_inputs_embeds)
    }

    /// Compute position_ids for a forward pass given current state.
    pub fn compute_position_ids(
        &mut self,
        input_ids: &Array,
        cache_offset: i32,
    ) -> Result<Array, Exception> {
        let batch_size = input_ids.shape()[0];
        let seq_length = input_ids.shape()[1];

        if cache_offset == 0 || self.rope_deltas.is_none() {
            // Prefill: use cached position_ids
            if let Some(ref pos) = self.position_ids {
                let pos = pos.index((.., .., cache_offset..(cache_offset + seq_length)));
                return Ok(pos);
            }
            // Fallback: simple sequential
            let (pos, deltas) = self.get_rope_index(input_ids, None, None, None)?;
            self.rope_deltas = Some(deltas);
            self.position_ids = Some(pos.clone());
            Ok(pos)
        } else {
            // Decode: compute from cache_offset + rope_deltas
            let delta = self.rope_deltas.as_ref().unwrap();
            let delta_val = ops::add(&Array::from_int(cache_offset), delta)?;

            let pos = ops::arange::<_, i32>(0, seq_length, None)?.reshape(&[1, -1])?;
            let pos = ops::broadcast_to(&pos, &[batch_size, seq_length])?;

            let delta_expanded = if delta_val.ndim() == 0 {
                delta_val.reshape(&[1])?
            } else {
                delta_val.clone()
            };
            let delta_expanded = if delta_expanded.shape()[0] < batch_size {
                ops::broadcast_to(&delta_expanded, &[batch_size, 1])?
            } else {
                delta_expanded.index(..batch_size)
            };

            let pos = ops::add(&pos, &delta_expanded)?;
            let pos = pos.reshape(&[1, batch_size, seq_length])?;
            ops::broadcast_to(&pos, &[3, batch_size, seq_length])
        }
    }

    /// Full forward pass: vision (if pixel_values) + merge + LM.
    /// Returns logits [B, seq_len, vocab_size].
    pub fn forward(
        &mut self,
        input_ids: &Array,
        pixel_values: Option<&Array>,
        image_grid_thw: Option<&[(i32, i32, i32)]>,
        mask: Option<&Array>,
        cache: &mut [KVCache],
        cache_offset: i32,
    ) -> Result<Array, Exception> {
        if pixel_values.is_some() {
            // Prefill with vision
            let embeds = self.get_input_embeddings(input_ids, pixel_values, image_grid_thw, mask)?;
            let position_ids = self.compute_position_ids(input_ids, cache_offset)?;
            self.language_model.forward_with_embeds(&embeds, &position_ids, cache)
        } else {
            // Text-only (decode or text-only prefill)
            let position_ids = self.compute_position_ids(input_ids, cache_offset)?;
            self.language_model.forward(input_ids, &position_ids, cache)
        }
    }
}
