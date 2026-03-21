# glm-ocr-rs

Inference only Rust port of GLM-OCR.

> This repo is assisted by Claude, GPT, GLM.

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

## Run

### Start the Server

```bash
RUST_LOG=info cargo run -r --bin glm-ocr-server -- \
    --model-dir /path/to/model \
    --port 8080
```

### Test with Script

```bash
python test_ocr.py
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

### 2. Run with capture enabled

Run:

```bash
METAL_CAPTURE_ENABLED=1 RUST_LOG=info ./glm-ocr-rs/target/release/glm-ocr-server \
  --model-dir /path/to/your/model
```

Send a request to trigger the decode loop, then open the trace in Xcode:

```bash
open /tmp/decode.gputrace
```

### 3. Analyze in Xcode

1. Check **"Profile after replay"**
2. Click **"Replay"**
3. Open the **Performance** view (and wait for it to replay) to see per-kernel GPU execution time
4. Check **Timeline**, **Shaders** or other tabs.

