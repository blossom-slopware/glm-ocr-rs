use std::net::SocketAddr;
use std::sync::Arc;

use mlx_ocr_2::full_model::Model;
use mlx_ocr_2::image_processor::ImageProcessor;
use mlx_ocr_2::server::{build_router, AppState};
use mlx_ocr_2::tokenizer::GlmTokenizer;

struct Args {
    model_dir: String,
    port: u16,
    host: String,
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1);
    let mut model_dir = String::new();
    let mut port = 8080u16;
    let mut host = "127.0.0.1".to_string();

    while let Some(key) = args.next() {
        match key.as_str() {
            "--model-dir" | "--model" => {
                model_dir = args.next().unwrap_or_else(|| panic!("--model-dir requires a value"));
            }
            "--port" => {
                port = args.next()
                    .unwrap_or_else(|| panic!("--port requires a value"))
                    .parse()
                    .unwrap_or_else(|e| panic!("Invalid port: {}", e));
            }
            "--host" => {
                host = args.next().unwrap_or_else(|| panic!("--host requires a value"));
            }
            _ => {
                panic!("Unknown argument: {}. Usage: glm-ocr-server --model-dir <path> [--port <port>] [--host <host>]", key);
            }
        }
    }

    if model_dir.is_empty() {
        panic!("--model-dir is required. Usage: glm-ocr-server --model-dir <path> [--port <port>] [--host <host>]");
    }

    Args { model_dir, port, host }
}

#[tokio::main]
async fn main() {
    env_logger::init();

    let args = parse_args();

    log::info!("Loading model from {}", args.model_dir);
    let model = Model::load(&args.model_dir)
        .unwrap_or_else(|e| panic!("Failed to load model: {}", e));

    let tokenizer_path = format!("{}/tokenizer.json", args.model_dir);
    log::info!("Loading tokenizer from {}", tokenizer_path);
    let tokenizer = GlmTokenizer::from_file(&tokenizer_path);

    let template_path = format!("{}/chat_template.jinja", args.model_dir);
    log::info!("Loading chat template from {}", template_path);
    let template_str = std::fs::read_to_string(&template_path)
        .unwrap_or_else(|e| panic!("Failed to read chat template {}: {}", template_path, e));

    log::info!("Loading image processor config");
    let image_processor = ImageProcessor::from_config(&args.model_dir);

    let model_name = args.model_dir.clone();

    let state = Arc::new(AppState {
        model: tokio::sync::Mutex::new(model),
        tokenizer,
        image_processor,
        template_str,
        model_name,
    });

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()
        .unwrap_or_else(|e| panic!("Invalid address {}:{}: {}", args.host, args.port, e));
    log::info!("Starting GLM-OCR server on {}", addr);

    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await
        .unwrap_or_else(|e| panic!("Failed to bind to {}: {}", addr, e));

    log::info!("Server ready. Listening on http://{}", addr);
    axum::serve(listener, app).await
        .unwrap_or_else(|e| panic!("Server error: {}", e));
}
