#!/usr/bin/env bash
set -euo pipefail

BINARY="./target/release/glm-ocr-server"
MODEL_DIR="/Users/daisy/develop/GLM-OCR/models/GLM-OCR-bf16"
OUTPUT_TRACE="/tmp/glm_ocr_trace.trace"
DURATION=60
TEMPLATE="${XCTRACE_TEMPLATE:-Time Profiler}"

# Clean up any previous trace
rm -rf "$OUTPUT_TRACE"

echo "=== Starting ${TEMPLATE} for ${DURATION}s ==="
echo "Binary : $BINARY"
echo "Output : $OUTPUT_TRACE"
echo "Template: $TEMPLATE"
echo ""

# Time Profiler is the template that shows sampled CPU call stacks in Instruments.
# Use XCTRACE_TEMPLATE='Metal System Trace' when you need GPU timelines instead.
xcrun xctrace record \
    --template "$TEMPLATE" \
    --output "$OUTPUT_TRACE" \
    --time-limit "${DURATION}s" \
    --env "RUST_LOG=info" \
    --launch -- "$BINARY" \
        --model-dir "$MODEL_DIR" \
        --port 8080

echo ""
echo "=== Trace saved to: $OUTPUT_TRACE ==="
echo "Opening in Instruments..."
open "$OUTPUT_TRACE"
