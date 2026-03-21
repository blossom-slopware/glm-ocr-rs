"""OCR test using GLM-OCR local server."""

from pathlib import Path
import requests
from datetime import datetime
import json

API_URL = "http://127.0.0.1:8080/ocr/stream"
IMAGE_PATH = Path(__file__).parent.joinpath("test_images").joinpath("exam2.png")
OUTPUT_DIR = Path(__file__).parent.joinpath("results")
OUTPUT_DIR.mkdir(exist_ok=True)

payload = {
    "image": {"type": "url", "url": str(IMAGE_PATH.resolve())},
    "prompt": "",
    "max_tokens": 4096,
    "temperature": 0.01,
}

print(f"Sending OCR request for {IMAGE_PATH.name} ...\n")
print("=" * 60)

resp = requests.post(API_URL, json=payload, timeout=300, stream=True)
resp.raise_for_status()

content_parts = []
for line in resp.iter_lines(delimiter=b"\n"):
    if not line:
        continue
    if line.startswith(b"data: "):
        event = json.loads(line[6:])
        if "delta" in event:
            text = event["delta"]
            print(text, end="", flush=True)
            content_parts.append(text)
        elif event.get("done"):
            print(f"\n\n[stop_reason: {event.get('stop_reason')}, tokens: {event.get('generated_tokens')}]")
            break
        elif "error" in event:
            print(f"\nError: {event['error']}")
            break

content = "".join(content_parts)
print("=" * 60)

# Save markdown result
instant = datetime.now().strftime("%Y%m%d_%H_%M_%S")
out_md = OUTPUT_DIR / f"exam_result_{instant}.md"
out_md.write_text(content, encoding="utf-8")

print(f"\nSaved to {out_md}")
