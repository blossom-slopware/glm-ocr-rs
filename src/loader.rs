use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::module::ModuleParametersExt;
use serde::Deserialize;
use serde_json::Value;

use crate::config::{FullConfig, TextConfig};
use crate::model::language_model::GlmOcrModel;

#[derive(Debug, Deserialize)]
struct WeightIndex {
    weight_map: HashMap<String, String>,
}

pub fn load_config(model_dir: &Path) -> Result<TextConfig, Box<dyn std::error::Error>> {
    let config_path = model_dir.join("config.json");
    let file = std::fs::File::open(config_path)?;
    let full: FullConfig = serde_json::from_reader(file)?;
    Ok(full.text_config)
}

pub fn load_glm_ocr_model(
    model_dir: &Path,
) -> Result<GlmOcrModel, Box<dyn std::error::Error>> {
    let config = load_config(model_dir)?;
    let mut model = GlmOcrModel::new(&config)?;

    // Try sharded weights first, then single file
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let json = std::fs::read_to_string(&index_path)?;
        let index: WeightIndex = serde_json::from_str(&json)?;
        let files: HashSet<&String> = index.weight_map.values().collect();
        for file in files {
            let path = model_dir.join(file);
            model.load_safetensors(path)?;
        }
    } else {
        let path = model_dir.join("model.safetensors");
        model.load_safetensors(path)?;
    }

    Ok(model)
}
