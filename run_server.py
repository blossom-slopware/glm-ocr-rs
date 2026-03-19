"""Launch mlx_vlm server with full Rust model (vision + LM)."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mlx-vlm"))

MODEL_DIR = "/Users/daisy/develop/GLM-OCR/models/GLM-OCR-bf16"

import mlx_ocr_2
from rust_lm_wrapper import RustFullModel

import mlx_vlm.server as server

_original_load = server.load_model_resources

def _patched_load(model_path, adapter_path):
    model, processor, config = _original_load(model_path, adapter_path)

    print("Loading Rust full model (vision + LM)...")
    rust_model = mlx_ocr_2.PyFullModel.load(MODEL_DIR)

    print("Replacing ENTIRE Python model with Rust backend...")
    model = RustFullModel(rust_model, model)

    print("Rust full model active (vision + merge + LM all in Rust)!")
    return model, processor, config

server.load_model_resources = _patched_load

os.environ["PRELOAD_MODEL"] = MODEL_DIR

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "mlx_vlm.server:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
        reload=False,
    )
