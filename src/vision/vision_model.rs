use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::Module,
    nn,
    ops,
    ops::indexing::IndexOp,
    Array, Dtype,
};

use super::config::VisionConfig;
use super::patch_embed::GlmOcrVisionPatchEmbed;
use super::rotary::GlmOcrVisionRotaryEmbedding;
use super::block::{GlmOcrVisionBlock, VisionBlockInput};
use super::merger::GlmOcrVisionPatchMerger;

/// Top-level wrapper matching safetensors key prefix `vision_tower.`
#[derive(Debug, Clone, ModuleParameters)]
pub struct VisionTower {
    #[param]
    pub vision_tower: VisionModel,
}

impl VisionTower {
    pub fn new(config: &VisionConfig) -> Result<Self, Exception> {
        Ok(Self {
            vision_tower: VisionModel::new(config)?,
        })
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct VisionModel {
    #[param]
    pub patch_embed: GlmOcrVisionPatchEmbed,
    #[param]
    pub blocks: Vec<GlmOcrVisionBlock>,
    #[param]
    pub post_layernorm: nn::RmsNorm,
    #[param]
    pub downsample: nn::Conv2d,
    #[param]
    pub merger: GlmOcrVisionPatchMerger,

    pub spatial_merge_size: i32,
    pub num_heads: i32,
}

impl VisionModel {
    pub fn new(config: &VisionConfig) -> Result<Self, Exception> {
        let patch_embed = GlmOcrVisionPatchEmbed::new(config)?;

        let mut blocks = Vec::with_capacity(config.depth);
        for _ in 0..config.depth {
            blocks.push(GlmOcrVisionBlock::new(config)?);
        }

        let post_layernorm = nn::RmsNormBuilder::new(config.hidden_size)
            .eps(config.rms_norm_eps)
            .build()?;

        let downsample = nn::Conv2dBuilder::new(
            config.hidden_size,
            config.out_hidden_size,
            (config.spatial_merge_size, config.spatial_merge_size),
        )
        .stride((config.spatial_merge_size, config.spatial_merge_size))
        .bias(true)
        .build()?;

        let merger = GlmOcrVisionPatchMerger::new(config)?;

        Ok(Self {
            patch_embed,
            blocks,
            post_layernorm,
            downsample,
            merger,
            spatial_merge_size: config.spatial_merge_size,
            num_heads: config.num_heads,
        })
    }

    /// Sanitize Conv weights from PyTorch layout to MLX channels-last layout.
    /// Call after load_safetensors.
    pub fn sanitize_weights(&mut self) -> Result<(), Exception> {
        // Conv3d: patch_embed.proj.weight
        // PyTorch: [O, I, kT, kH, kW] -> MLX: [O, kT, kH, kW, I]
        {
            let w = &*self.patch_embed.proj.weight;
            let shape = w.shape();
            if shape.len() == 5 && shape[1] == 3 {
                // shape[1] == in_channels (3) means PyTorch layout [O, I=3, kT, kH, kW]
                *self.patch_embed.proj.weight = w.transpose_axes(&[0, 2, 3, 4, 1])?;
            }
        }

        // Conv2d: downsample.weight
        // PyTorch: [O, I, kH, kW] -> MLX: [O, kH, kW, I]
        {
            let w = &*self.downsample.weight;
            let shape = w.shape();
            if shape.len() == 4 && shape[1] > shape[2] {
                // shape[1] > shape[2] means PyTorch layout [O, I=1024, kH=2, kW=2]
                *self.downsample.weight = w.transpose_axes(&[0, 2, 3, 1])?;
            }
        }

        Ok(())
    }

    /// Build rotary position embeddings for vision tokens.
    /// grid_thw: &[(t, h, w)] for each image
    /// Returns (cos, sin) each [N_total, head_dim]
    fn rot_pos_emb(
        &self,
        grid_thw: &[(i32, i32, i32)],
        head_dim: i32,
        spatial_merge_size: i32,
    ) -> Result<(Array, Array), Exception> {
        let rotary = GlmOcrVisionRotaryEmbedding::new(head_dim);

        // Find max grid size for frequency table
        let max_grid = grid_thw.iter()
            .flat_map(|&(_, h, w)| [h, w])
            .max()
            .unwrap_or(1);

        let freq_table = rotary.forward(max_grid)?; // [max_grid, dim/2]

        let mut all_pos = Vec::new();

        for &(t, h, w) in grid_thw {
            // Build h position ids with merge-size interleaving
            let hpos = Array::arange::<_, f32>(0, h, 1)?;
            let hpos = hpos.reshape(&[h, 1])?;
            let hpos = ops::broadcast_to(&hpos, &[h, w])?;
            let hm = h / spatial_merge_size;
            let wm = w / spatial_merge_size;
            let hpos = hpos.reshape(&[hm, spatial_merge_size, wm, spatial_merge_size])?;
            let hpos = hpos.transpose_axes(&[0, 2, 1, 3])?;
            let hpos = hpos.reshape(&[-1])?.as_dtype(Dtype::Int32)?;

            // wpos: same but for columns
            let wpos = Array::arange::<_, f32>(0, w, 1)?;
            let wpos = wpos.reshape(&[1, w])?;
            let wpos = ops::broadcast_to(&wpos, &[h, w])?;
            let wpos = wpos.reshape(&[hm, spatial_merge_size, wm, spatial_merge_size])?;
            let wpos = wpos.transpose_axes(&[0, 2, 1, 3])?;
            let wpos = wpos.reshape(&[-1])?.as_dtype(Dtype::Int32)?;

            // Stack: [h*w, 2]
            let pos = ops::stack_axis(&[&hpos, &wpos], -1)?;

            // Tile for temporal: [t*h*w, 2]
            let pos = ops::tile(&pos, &[t, 1])?;

            all_pos.push(pos);
        }

        // Concatenate all images: [N_total, 2]
        let pos_refs: Vec<&Array> = all_pos.iter().collect();
        let pos_ids = ops::concatenate_axis(&pos_refs, 0)?;

        // Index into frequency table: [N_total, 2, dim/2]
        let emb = freq_table.index(&pos_ids);

        // Reshape [N_total, 2 * dim/2] = [N_total, dim]
        let n = emb.shape()[0];
        let emb = emb.reshape(&[n, -1])?;

        // Duplicate for full head_dim: [N_total, 2*dim] = [N_total, head_dim]
        let emb = ops::concatenate_axis(&[&emb, &emb], -1)?;

        let cos = ops::cos(&emb)?;
        let sin = ops::sin(&emb)?;

        Ok((cos, sin))
    }

    /// Build cu_seqlens from grid_thw
    fn build_cu_seqlens(grid_thw: &[(i32, i32, i32)]) -> Vec<i32> {
        let mut cu_seqlens = vec![0i32];
        let mut cumulative = 0i32;
        for &(t, h, w) in grid_thw {
            let seq_len = h * w;
            for _ in 0..t {
                cumulative += seq_len;
                cu_seqlens.push(cumulative);
            }
        }
        cu_seqlens
    }
}

pub struct VisionModelInput<'a> {
    pub pixel_values: &'a Array,  // [N_total, C*T*H*W]
    pub grid_thw: &'a [(i32, i32, i32)],
}

impl Module<VisionModelInput<'_>> for VisionModel {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: VisionModelInput<'_>) -> Result<Array, Exception> {
        let VisionModelInput { pixel_values, grid_thw } = input;

        log::info!("VisionModel forward: pixel_values shape {:?}, {} image(s), grid_thw {:?}",
            pixel_values.shape(), grid_thw.len(), grid_thw);

        // 1. Patch embedding
        let mut hidden_states = self.patch_embed.forward(pixel_values)?;
        log::info!("  patch_embed -> {:?}", hidden_states.shape());

        // 2. Rotary position embeddings
        let hidden_size = hidden_states.shape()[1];
        let head_dim = hidden_size as i32 / self.num_heads;
        let (cos, sin) = self.rot_pos_emb(grid_thw, head_dim, self.spatial_merge_size)?;
        log::info!("  rot_pos_emb -> cos {:?}, sin {:?}", cos.shape(), sin.shape());

        // 3. Build cu_seqlens
        let cu_seqlens = Self::build_cu_seqlens(grid_thw);
        log::info!("  cu_seqlens: {:?} ({} segments)", cu_seqlens, cu_seqlens.len() - 1);

        // 4. Transformer blocks
        let num_blocks = self.blocks.len();
        for (i, block) in self.blocks.iter_mut().enumerate() {
            hidden_states = block.forward(VisionBlockInput {
                x: &hidden_states,
                cu_seqlens: &cu_seqlens,
                cos: &cos,
                sin: &sin,
            })?;
            if i == 0 || i == num_blocks - 1 {
                log::info!("  block {}/{} -> {:?}", i, num_blocks, hidden_states.shape());
            }
        }

        // 5. Post layer norm
        hidden_states = self.post_layernorm.forward(&hidden_states)?;
        log::info!("  post_layernorm -> {:?}", hidden_states.shape());

        // 6. Reshape for spatial downsampling: [N, hidden] -> [N/4, 2, 2, hidden]
        let n = hidden_states.shape()[0] as i32;
        let h = hidden_states.shape()[1] as i32;
        let merge = self.spatial_merge_size;
        hidden_states = hidden_states.reshape(&[n / (merge * merge), merge, merge, h])?;

        // 7. Downsample Conv2d (channels-last): [N/4, 2, 2, hidden] -> [N/4, 1, 1, out_hidden]
        hidden_states = self.downsample.forward(&hidden_states)?;
        let out_h = hidden_states.shape()[hidden_states.ndim() - 1] as i32;
        hidden_states = hidden_states.reshape(&[-1, out_h])?;
        log::info!("  downsample -> {:?}", hidden_states.shape());

        // 8. Patch merger
        hidden_states = self.merger.forward(&hidden_states)?;
        log::info!("  merger -> {:?} (final output)", hidden_states.shape());

        Ok(hidden_states)
    }

    fn training_mode(&mut self, mode: bool) {
        self.patch_embed.training_mode(mode);
        for block in &mut self.blocks {
            block.training_mode(mode);
        }
        self.post_layernorm.training_mode(mode);
        self.downsample.training_mode(mode);
        self.merger.training_mode(mode);
    }
}
