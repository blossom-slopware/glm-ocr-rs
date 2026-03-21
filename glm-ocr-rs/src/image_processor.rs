use mlx_rs::error::Exception;
use mlx_rs::ops;
use mlx_rs::Array;
use serde::Deserialize;

/// Image processor config from preprocessor_config.json.
pub struct ImageProcessor {
    pub patch_size: i32,
    pub temporal_patch_size: i32,
    pub merge_size: i32,
    pub min_pixels: u64,
    pub max_pixels: u64,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

#[derive(Deserialize)]
struct SizeConfig {
    shortest_edge: u64,
    longest_edge: u64,
}

#[derive(Deserialize)]
struct PreprocessorConfig {
    size: SizeConfig,
    patch_size: i32,
    temporal_patch_size: i32,
    merge_size: i32,
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
}

impl ImageProcessor {
    pub fn from_config(model_dir: &str) -> Self {
        let path = format!("{}/preprocessor_config.json", model_dir);
        let f = std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("Failed to open {}: {}", path, e));
        let cfg: PreprocessorConfig = serde_json::from_reader(f)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {}", path, e));
        Self {
            patch_size: cfg.patch_size,
            temporal_patch_size: cfg.temporal_patch_size,
            merge_size: cfg.merge_size,
            min_pixels: cfg.size.shortest_edge,
            max_pixels: cfg.size.longest_edge,
            image_mean: [cfg.image_mean[0], cfg.image_mean[1], cfg.image_mean[2]],
            image_std: [cfg.image_std[0], cfg.image_std[1], cfg.image_std[2]],
        }
    }

    /// Preprocess an image from raw bytes.
    /// Returns (pixel_values [N, 1176], (grid_t, grid_h, grid_w)).
    pub fn preprocess(&self, img_bytes: &[u8]) -> Result<(Array, (i32, i32, i32)), Exception> {
        // Load image, convert to RGB
        let img = image::load_from_memory(img_bytes)
            .unwrap_or_else(|e| panic!("Failed to load image: {}", e))
            .to_rgb8();

        let (orig_w, orig_h) = (img.width(), img.height());

        // smart_resize
        let (resized_h, resized_w) = smart_resize(
            self.temporal_patch_size as u32,
            orig_h,
            orig_w,
            self.temporal_patch_size as u32,
            (self.patch_size * self.merge_size) as u32,
            self.min_pixels,
            self.max_pixels,
        );

        // Resize with BICUBIC (CatmullRom)
        let resized = image::imageops::resize(
            &img,
            resized_w,
            resized_h,
            image::imageops::FilterType::CatmullRom,
        );

        // Convert to f32 Vec in HWC layout
        let h = resized_h as i32;
        let w = resized_w as i32;
        let raw = resized.into_raw();
        let pixels_f32: Vec<f32> = raw.iter().map(|&v| v as f32).collect();

        // Create mlx Array [H, W, 3] (HWC)
        let arr = Array::from_slice(&pixels_f32, &[h, w, 3]);

        // Rescale: / 255.0
        let arr = ops::divide(&arr, &Array::from_f32(255.0))?;

        // Normalize per channel: (pixel - mean) / std
        // mean and std as [1, 1, 3] for broadcasting
        let mean = Array::from_slice(&self.image_mean, &[1, 1, 3]);
        let std = Array::from_slice(&self.image_std, &[1, 1, 3]);
        let arr = ops::subtract(&arr, &mean)?;
        let arr = ops::divide(&arr, &std)?;

        // HWC -> CHW: transpose to [C, H, W]
        let arr = arr.transpose_axes(&[2, 0, 1])?;

        // Temporal duplicate: stack to [2, C, H, W]
        let patches = ops::stack_axis(&[&arr, &arr], 0)?;

        // Now patches shape: [temporal_patch_size, C, H, W] = [2, 3, h, w]
        let channel = 3i32;
        let grid_t = self.temporal_patch_size / self.temporal_patch_size; // = 1
        let grid_h = h / self.patch_size;
        let grid_w = w / self.patch_size;
        let ps = self.patch_size;
        let ms = self.merge_size;
        let tp = self.temporal_patch_size;

        // Reshape to 9D: (grid_t, temporal_patch_size, C, grid_h//merge_size, merge_size, patch_size, grid_w//merge_size, merge_size, patch_size)
        let patches = patches.reshape(&[
            grid_t,
            tp,
            channel,
            grid_h / ms,
            ms,
            ps,
            grid_w / ms,
            ms,
            ps,
        ])?;

        // Transpose (0, 3, 6, 4, 7, 2, 1, 5, 8)
        let patches = patches.transpose_axes(&[0, 3, 6, 4, 7, 2, 1, 5, 8])?;

        // Flatten to (grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
        let n_patches = grid_t * grid_h * grid_w;
        let feat_dim = channel * tp * ps * ps; // 3 * 2 * 14 * 14 = 1176
        let flatten_patches = patches.reshape(&[n_patches, feat_dim])?;

        Ok((flatten_patches, (grid_t, grid_h, grid_w)))
    }
}

/// Port of Python smart_resize.
/// Returns (resized_height, resized_width) as multiples of factor.
pub fn smart_resize(
    num_frames: u32,
    height: u32,
    width: u32,
    temporal_factor: u32,
    factor: u32,
    min_pixels: u64,
    max_pixels: u64,
) -> (u32, u32) {
    assert!(
        num_frames >= temporal_factor,
        "t:{} must be larger than temporal_factor:{}",
        num_frames,
        temporal_factor
    );

    let mut height_f = height as f64;
    let mut width_f = width as f64;
    let factor_f = factor as f64;

    if (height_f) < factor_f || (width_f) < factor_f {
        let scale = f64::max(factor_f / height_f, factor_f / width_f);
        height_f = (height_f * scale).floor();
        width_f = (width_f * scale).floor();
    }

    assert!(
        f64::max(height_f, width_f) / f64::min(height_f, width_f) <= 200.0,
        "absolute aspect ratio must be smaller than 200, got {}",
        f64::max(height_f, width_f) / f64::min(height_f, width_f)
    );

    let mut h_bar = (height_f / factor_f).round() as u32 * factor;
    let mut w_bar = (width_f / factor_f).round() as u32 * factor;
    let t_bar = (num_frames as f64 / temporal_factor as f64).round() as u32 * temporal_factor;

    if (t_bar as u64) * (h_bar as u64) * (w_bar as u64) > max_pixels {
        let beta = f64::sqrt((num_frames as f64 * height_f * width_f) / max_pixels as f64);
        h_bar = u32::max(factor, (height_f / beta / factor_f).floor() as u32 * factor);
        w_bar = u32::max(factor, (width_f / beta / factor_f).floor() as u32 * factor);
    } else if (t_bar as u64) * (h_bar as u64) * (w_bar as u64) < min_pixels {
        let beta = f64::sqrt(min_pixels as f64 / (num_frames as f64 * height_f * width_f));
        h_bar = (height_f * beta / factor_f).ceil() as u32 * factor;
        w_bar = (width_f * beta / factor_f).ceil() as u32 * factor;
    }

    (h_bar, w_bar)
}

/// Load image bytes from various URL formats.
pub fn load_image_bytes(url: &str) -> Vec<u8> {
    if url.starts_with("data:") {
        // data:image/png;base64,<data>
        let comma_pos = url.find(',')
            .unwrap_or_else(|| panic!("Invalid data URL: no comma found"));
        let b64_data = &url[comma_pos + 1..];
        use base64::Engine;
        base64::engine::general_purpose::STANDARD
            .decode(b64_data)
            .unwrap_or_else(|e| panic!("Failed to decode base64 image: {}", e))
    } else if url.starts_with("http://") || url.starts_with("https://") {
        let resp = reqwest::blocking::get(url)
            .unwrap_or_else(|e| panic!("Failed to fetch image from {}: {}", url, e));
        resp.bytes()
            .unwrap_or_else(|e| panic!("Failed to read image bytes from {}: {}", url, e))
            .to_vec()
    } else {
        // Local file path
        let path = if url.starts_with("file://") {
            &url[7..]
        } else {
            url
        };
        std::fs::read(path)
            .unwrap_or_else(|e| panic!("Failed to read image file {}: {}", path, e))
    }
}
