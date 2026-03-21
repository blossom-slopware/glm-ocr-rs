use mlx_rs::{error::Exception, Array, Dtype};
use mlx_rs::ops::{
    self, indexing::{take_along_axis, put_along_axis, argmax_axis, IndexOp},
};

/// Apply top-p (nucleus) filtering on log-probabilities.
/// Tokens whose cumulative probability (in ascending sort order) falls below `1 - top_p` are masked to -inf.
pub fn apply_top_p(logprobs: &Array, top_p: f32) -> Result<Array, Exception> {
    let probs = ops::exp(logprobs)?;
    let sorted_indices = ops::argsort_axis(logprobs, -1)?;

    let sorted_probs = take_along_axis(&probs, &sorted_indices, -1)?;
    let cumulative_probs = ops::cumsum(&sorted_probs, Some(-1), None, None)?;

    // Compute inverse permutation to map cumulative probs back to original order
    let num_classes = logprobs.shape().last().copied().expect("logprobs must have at least 1 dimension");
    let arange = ops::arange::<_, i32>(0, num_classes, None)?;
    let inverse_indices = put_along_axis(&arange.reshape(&[-1])?, &sorted_indices, &arange.reshape(&[-1])?, -1)?;
    // Broadcast: expand arange to match batch dims
    let inverse_indices = {
        let batch_shape = &logprobs.shape()[..logprobs.shape().len() - 1];
        let mut shape = vec![1i32; batch_shape.len()];
        shape.push(num_classes);
        let inv = inverse_indices.reshape(&shape)?;
        let mut expand_shape: Vec<i32> = batch_shape.iter().map(|&s| s).collect();
        expand_shape.push(num_classes);
        ops::broadcast_to(&inv, &expand_shape)?
    };
    let cumulative_probs_orig = take_along_axis(&cumulative_probs, &inverse_indices, -1)?;

    let neg_inf = Array::from_f32(f32::NEG_INFINITY).as_dtype(Dtype::Bfloat16)?;
    let threshold = Array::from_f32(1.0 - top_p).as_dtype(Dtype::Bfloat16)?;
    let mask = cumulative_probs_orig.gt(&threshold)?;
    ops::which(&mask, logprobs, &neg_inf)
}

/// Apply min-p filtering on log-probabilities.
/// Removes tokens whose probability is below `min_p * max_prob`.
pub fn apply_min_p(logprobs: &Array, min_p: f32, min_tokens_to_keep: i32) -> Result<Array, Exception> {
    // Sort descending
    let sorted_indices = ops::argsort_axis(&ops::negative(logprobs)?, -1)?;
    let sorted_logprobs = take_along_axis(logprobs, &sorted_indices, -1)?;

    // Top logprob (index 0 of sorted)
    let top_logprobs = sorted_logprobs.index((.., 0..1));
    let scaled_min_p = &top_logprobs + Array::from_f32(min_p.ln()).as_dtype(Dtype::Bfloat16)?;

    // Mask tokens below threshold
    let tokens_to_remove = sorted_logprobs.lt(&scaled_min_p)?;

    // Always keep at least min_tokens_to_keep
    let neg_inf = Array::from_f32(f32::NEG_INFINITY).as_dtype(Dtype::Bfloat16)?;
    let num_classes = logprobs.shape().last().copied().expect("logprobs must have at least 1 dimension");

    // Build mask: set first min_tokens_to_keep positions to false (keep them)
    let keep_mask = if min_tokens_to_keep > 0 {
        let keep_indices = ops::arange::<_, i32>(0, num_classes, None)?;
        let keep_threshold = Array::from_int(min_tokens_to_keep);
        keep_indices.lt(&keep_threshold)?
    } else {
        Array::from_bool(false)
    };
    // tokens_to_remove AND NOT keep_mask
    let final_remove = ops::logical_and(&tokens_to_remove, &ops::logical_not(&keep_mask)?)?;

    let filtered_sorted = ops::which(&final_remove, &neg_inf, &sorted_logprobs)?;

    // Restore original order via inverse permutation
    let arange = ops::arange::<_, i32>(0, num_classes, None)?;
    let inverse_indices = put_along_axis(&arange.reshape(&[-1])?, &sorted_indices, &arange.reshape(&[-1])?, -1)?;
    let inverse_indices = {
        let batch_shape = &logprobs.shape()[..logprobs.shape().len() - 1];
        let mut shape = vec![1i32; batch_shape.len()];
        shape.push(num_classes);
        let inv = inverse_indices.reshape(&shape)?;
        let mut expand_shape: Vec<i32> = batch_shape.iter().map(|&s| s).collect();
        expand_shape.push(num_classes);
        ops::broadcast_to(&inv, &expand_shape)?
    };

    take_along_axis(&filtered_sorted, &inverse_indices, -1)
}

/// Apply top-k filtering on log-probabilities.
/// Uses argpartition to efficiently mask out all but the top-k tokens.
pub fn apply_top_k(logprobs: &Array, top_k: i32) -> Result<Array, Exception> {
    let neg_logprobs = ops::negative(logprobs)?;
    let partitioned_indices = ops::argpartition_axis(&neg_logprobs, top_k - 1, -1)?;

    // Indices beyond top_k are the ones to mask
    let num_classes = logprobs.shape().last().copied().expect("logprobs must have at least 1 dimension");
    let mask_indices = partitioned_indices.index((.., top_k..num_classes));

    let neg_inf = Array::from_f32(f32::NEG_INFINITY).as_dtype(Dtype::Bfloat16)?;
    put_along_axis(logprobs, &mask_indices, &neg_inf, -1)
}

/// Apply repetition penalty to raw logits given recent token context.
pub fn apply_repetition_penalty(
    logits: &Array,
    tokens: &[i32],
    penalty: f32,
    context_size: usize,
) -> Result<Array, Exception> {
    if tokens.is_empty() || penalty == 1.0 {
        return Ok(logits.clone());
    }

    let start = if tokens.len() > context_size { tokens.len() - context_size } else { 0 };
    let context_tokens = &tokens[start..];

    // Create index array of recent tokens
    let token_indices = Array::from_slice(context_tokens, &[context_tokens.len() as i32]);
    let token_indices = token_indices.as_dtype(Dtype::Int32)?;

    // Gather logits at token positions
    let selected_logits = take_along_axis(logits, &token_indices.reshape(&[1, -1])?, -1)?;

    let penalty_arr = Array::from_f32(penalty).as_dtype(Dtype::Bfloat16)?;
    let zero = Array::from_f32(0.0).as_dtype(Dtype::Bfloat16)?;
    let is_negative = selected_logits.lt(&zero)?;

    // Negative logits get multiplied by penalty (more negative), positive get divided
    let penalized = ops::which(
        &is_negative,
        &ops::multiply(&selected_logits, &penalty_arr)?,
        &ops::divide(&selected_logits, &penalty_arr)?,
    )?;

    // Scatter penalized values back
    put_along_axis(logits, &token_indices.reshape(&[1, -1])?, &penalized, -1)
}

/// Sample a token from logprobs.
/// If temperature == 0, uses argmax (greedy). Otherwise applies temperature and categorical sampling.
pub fn sample_token(
    logprobs: &Array,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    min_p: f32,
) -> Result<Array, Exception> {
    if temperature == 0.0 {
        return argmax_axis(logprobs, -1, None);
    }

    let mut lp = logprobs.clone();

    // Apply filters in order: top_p → min_p → top_k
    if top_p > 0.0 && top_p < 1.0 {
        lp = apply_top_p(&lp, top_p)?;
    }
    if min_p > 0.0 {
        lp = apply_min_p(&lp, min_p, 1)?;
    }
    if top_k > 0 {
        lp = apply_top_k(&lp, top_k)?;
    }

    // Scale by temperature and sample
    let inv_temp = Array::from_f32(1.0 / temperature).as_dtype(Dtype::Bfloat16)?;
    let scaled = ops::multiply(&lp, &inv_temp)?;
    mlx_rs::random::categorical(&scaled, None, None, None)
}
