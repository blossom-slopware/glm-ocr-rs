# Metal GPU Profiling for MLX-OCR (Rust)

## 前置条件

### 1. 启用 MLX Metal Debug 编译

修改 `mlx-rs/mlx-sys/build.rs`，在 CMake config 中添加：

```rust
config.define("MLX_METAL_DEBUG", "ON");
```

这会让 Metal shader 带上调试信息（`-gline-tables-only -frecord-sources`）和对象标签，Xcode Performance 视图才能显示性能数据。

添加位置示例：
```rust
#[cfg(feature = "accelerate")]
{
    config.define("MLX_BUILD_ACCELERATE", "ON");
}

config.define("MLX_METAL_DEBUG", "ON");  // 添加这一行

// build the mlx-c project
let dst = config.build();
```

### 2. 重新编译

```bash
cd glm-ocr-rs
cargo clean
cargo build --release
```

## GPU Trace 捕获

### 1. 在代码中插入捕获点

在 `generate.rs` 的 decode 循环中：

```rust
// 需要在 Cargo.toml 中添加: mlx-sys = { path = "../mlx-rs/mlx-sys" }

// 开始捕获（在某个 token 位置）
if emitted_tokens == 10 {
    let path = std::ffi::CString::new("/tmp/decode.gputrace").unwrap();
    let ret = unsafe { mlx_sys::mlx_metal_start_capture(path.as_ptr()) };
    log::info!("GPU capture started (ret={})", ret);
}

// 停止捕获（几个 token 之后）
if emitted_tokens == 12 {
    let ret = unsafe { mlx_sys::mlx_metal_stop_capture() };
    log::info!("GPU capture stopped (ret={}). Trace at /tmp/decode.gputrace", ret);
}
```

### 2. 运行时设置环境变量

必须设置 `METAL_CAPTURE_ENABLED=1`，否则 capture 会返回 ret=1 失败：

```bash
METAL_CAPTURE_ENABLED=1 RUST_LOG=info ./target/release/glm-ocr-server --model-dir /path/to/model
```

### 3. 触发捕获

运行 `test_ocr.py` 发送请求，server 在 decode 到指定 token 数时会自动捕获。

日志中看到以下内容表示成功：
```
GPU capture started (ret=0), recording decode step...
GPU capture stopped (ret=0). Trace at /tmp/decode.gputrace
```

## 在 Xcode 中分析

### 1. 打开 trace

```bash
open /tmp/decode.gputrace
```

### 2. 操作步骤

1. Xcode 打开后会显示一个初始界面，包含设备信息和 "Replay" 按钮
2. **勾选 "Profile after replay"**
3. 点击 **"Replay"** — Xcode 会在 GPU 上重放捕获的 Metal 命令并收集性能计数
4. 重放完成后，点击 **"Performance"** 视图查看每个 Metal kernel 的 GPU 执行时间

### 3. 可用视图

- **Performance** — GPU kernel 执行时间（需要先 Replay）
- **Memory** — GPU 内存使用
- **Dependencies** — command buffer / compute encoder 结构
- **Summary** — 概览

## 注意事项

- `/tmp/decode.gputrace` 路径不能已存在，否则捕获会失败。每次重新捕获前需要先删除：`rm -rf /tmp/decode.gputrace`
- `MLX_METAL_DEBUG=ON` 会略微影响 GPU 性能，profiling 完成后可以关闭
- 捕获期间 TPS 会下降，这是正常的
