"""Launch mlx_vlm server with Rust language model."""

import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mlx-vlm"))

MODEL_DIR = "/Users/daisy/develop/GLM-OCR/models/GLM-OCR-bf16"

import mlx_ocr_2
from rust_lm_wrapper import RustLanguageModel

# Monkey-patch the server's model loading to swap in Rust LM
import mlx_vlm.server as server

_original_load = server.load_model_resources

def _patched_load(model_path, adapter_path):
    model, processor, config = _original_load(model_path, adapter_path)

    print("Loading Rust language model...")
    rust_model = mlx_ocr_2.PyGlmOcrModel.load(MODEL_DIR)

    print("Replacing Python language model with Rust backend...")
    rust_lm = RustLanguageModel(rust_model, model.language_model)
    model.language_model = rust_lm

    print("Rust language model active!")
    return model, processor, config

server.load_model_resources = _patched_load

# Set model path and start server
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
