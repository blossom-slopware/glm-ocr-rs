use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::module::ModuleParametersExt;
use serde::Deserialize;

use crate::config::{FullConfig, TextConfig};
use crate::model::language_model::GlmOcrModel;
use crate::vision::vision_model::VisionTower;

#[derive(Debug, Deserialize)]
struct WeightIndex {
    weight_map: HashMap<String, String>,
}

pub fn load_full_config(model_dir: &Path) -> Result<FullConfig, Box<dyn std::error::Error>> {
    let config_path = model_dir.join("config.json");
    log::info!("Loading config from {:?}", config_path);
    let file = std::fs::File::open(config_path)?;
    let full: FullConfig = serde_json::from_reader(file)?;
    Ok(full)
}

pub fn load_config(model_dir: &Path) -> Result<TextConfig, Box<dyn std::error::Error>> {
    Ok(load_full_config(model_dir)?.text_config)
}

fn load_safetensors_sharded(
    model: &mut impl ModuleParametersExt,
    model_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let json = std::fs::read_to_string(&index_path)?;
        let index: WeightIndex = serde_json::from_str(&json)?;
        let files: HashSet<&String> = index.weight_map.values().collect();
        log::info!("Loading weights from {} sharded file(s)", files.len());
        for file in &files {
            let path = model_dir.join(file);
            log::info!("  loading shard: {}", file);
            model.load_safetensors(path)?;
        }
    } else {
        let path = model_dir.join("model.safetensors");
        log::info!("Loading weights from single file: model.safetensors");
        model.load_safetensors(path)?;
    }
    Ok(())
}

pub fn load_glm_ocr_model(
    model_dir: &Path,
) -> Result<GlmOcrModel, Box<dyn std::error::Error>> {
    log::info!("Loading language model from {:?}", model_dir);
    let config = load_config(model_dir)?;
    let mut model = GlmOcrModel::new(&config)?;
    load_safetensors_sharded(&mut model, model_dir)?;
    log::info!("Language model loaded ({} layers)", config.num_hidden_layers);
    Ok(model)
}

pub fn load_vision_model(
    model_dir: &Path,
) -> Result<VisionTower, Box<dyn std::error::Error>> {
    log::info!("Loading vision model from {:?}", model_dir);
    let full_config = load_full_config(model_dir)?;
    let vc = &full_config.vision_config;
    log::info!("  vision config: depth={}, hidden_size={}, num_heads={}, out_hidden_size={}",
        vc.depth, vc.hidden_size, vc.num_heads, vc.out_hidden_size);
    let mut tower = VisionTower::new(vc)?;
    load_safetensors_sharded(&mut tower, model_dir)?;
    tower.vision_tower.sanitize_weights()?;
    log::info!("Vision model loaded ({} blocks)", vc.depth);
    Ok(tower)
}
