# glm-ocr-rs

Inference only Rust port of GLM-OCR.

## Get and Build

### Prerequisites

- Rust (latest stable)
- Xcode Command Line Tools (for macOS)
- CMake (for building MLX C bindings)

### Clone

This repository uses nested submodules. A plain `git clone` will not pull the required dependencies.

```bash
# HTTPS
git clone --recurse-submodules https://github.com/blossom-slopware/glm-ocr-rs.git

# SSH
git clone --recurse-submodules git@github.com:blossom-slopware/glm-ocr-rs.git
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### Build

```bash
cargo build --release -p glm-ocr-rs
```

The build process will:
1. Compile the MLX C bindings (mlx-c) via CMake
2. Compile the Rust MLX bindings (mlx-rs)
3. Build the glm-ocr-rs library and server binary

The compiled server binary will be at:
```
./glm-ocr-rs/target/release/glm-ocr-server
```

## Run

### Start the Server

```bash
RUST_LOG=info cargo run -r --bin glm-ocr-server -- \
    --model-dir /Users/daisy/develop/GLM-OCR/models/GLM-OCR-bf16 \
    --port 8080
```

The server will:
1. Load model weights from `--model-dir` (requires config.json, tokenizer.json, model.safetensors)
2. Start HTTP server on specified host:port
3. Log initialization progress


### Test with Script

```bash
python /Users/daisy/develop/GLM-OCR/ocr-inference/test_ocr.py
```

Image URLs can be:
- Local file paths: `file:///absolute/path/to/image.png`
- HTTP/HTTPS: `http://example.com/image.jpg`
- Data URIs: `data:image/png;base64,...`


## Profile

### 1. Build with Metal debug support


```bash
cargo clean && cargo build --release -p glm-ocr-rs --features gpu-capture
```

### 3. Run with capture enabled

Run:

```bash
METAL_CAPTURE_ENABLED=1 RUST_LOG=info ./glm-ocr-rs/target/release/glm-ocr-server \
  --model-dir /path/to/your/model
```

Send a request to trigger the decode loop, then open the trace in Xcode:

```bash
open /tmp/decode.gputrace
```

### 4. Analyze in Xcode

1. Check **"Profile after replay"**
2. Click **"Replay"**
3. Open the **Performance** view (and wait for it to replay) to see per-kernel GPU execution time
4. Check **Timeline**, **Shaders** or other tabs.

