#!/usr/bin/env python3
"""
DeepSeek-OCR example for Apple Silicon Macs (CPU mode)
See README.md for installation instructions and resolution modes.
Run patch-for-cpu.py after first download to enable CPU compatibility.
"""

from transformers import AutoModel, AutoTokenizer
import torch
import time
import os
from PIL import Image

model_name = "deepseek-ai/DeepSeek-OCR"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    attn_implementation="eager",  # Eager mode for Apple Silicon MPS compatibility
    trust_remote_code=True,
    use_safetensors=True
)

# Use CPU for compatibility (slower but more reliable)
model = model.eval().to(torch.float32).to("cpu")

prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = "image.jpg"
output_path = "output/"

# Resolution modes (see README.md for details):
# - Tiny:   base_size=512,  image_size=512,  crop_mode=False  (64 tokens)
# - Small:  base_size=640,  image_size=640,  crop_mode=False  (100 tokens)
# - Base:   base_size=1024, image_size=1024, crop_mode=False  (256 tokens)
# - Large:  base_size=1280, image_size=1280, crop_mode=False  (400 tokens)
# - Gundam: base_size=1024, image_size=640,  crop_mode=True   (256-1056+ tokens)

# Configuration
base_size = 1024      # Using Gundam mode (recommended for complex documents)
image_size = 640
crop_mode = True

# Determine mode name
if crop_mode:
    if base_size == 1024 and image_size == 640:
        mode_name = "Gundam"
    else:
        mode_name = "Dynamic (custom crop)"
else:
    if base_size == 512:
        mode_name = "Tiny"
    elif base_size == 640:
        mode_name = "Small"
    elif base_size == 1024:
        mode_name = "Base"
    elif base_size == 1280:
        mode_name = "Large"
    else:
        mode_name = "Custom"

# Get image dimensions
try:
    img = Image.open(image_file)
    img_width, img_height = img.size
    print(f"\n{'='*60}")
    print(f"DeepSeek-OCR Run Metadata")
    print(f"{'='*60}")
    print(f"Image: {image_file}")
    print(f"Image size: {img_width}x{img_height} pixels")
    print(f"Mode: {mode_name}")
    print(f"Settings: base_size={base_size}, image_size={image_size}, crop_mode={crop_mode}")
except Exception as e:
    print(f"Warning: Could not read image metadata: {e}")

print(f"Processing...")
start_time = time.time()

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=base_size,
    image_size=image_size,
    crop_mode=crop_mode,
    save_results=True,
    test_compress=True
)

end_time = time.time()
elapsed_time = end_time - start_time

# Read the result from the output file if model.infer() didn't return it
result_file = os.path.join(output_path, "result.md")
if not res and os.path.exists(result_file):
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            res = f.read()
    except Exception as e:
        print(f"Warning: Could not read result file: {e}")

# Calculate tokens and compression ratio
if res:
    # Count text tokens in the result
    text_tokens = len(tokenizer.encode(res, add_special_tokens=False))
    
    # Estimate vision tokens based on mode
    # This is approximate - the actual number comes from the model's processing
    if crop_mode:
        # Gundam mode: variable based on image, estimate from base mode minimum
        vision_tokens_estimate = 256  # Minimum, could be up to 1056+
        vision_tokens_label = "~256-1056+"
    else:
        if base_size == 512:
            vision_tokens_estimate = 64
            vision_tokens_label = "64"
        elif base_size == 640:
            vision_tokens_estimate = 100
            vision_tokens_label = "100"
        elif base_size == 1024:
            vision_tokens_estimate = 256
            vision_tokens_label = "256"
        elif base_size == 1280:
            vision_tokens_estimate = 400
            vision_tokens_label = "400"
        else:
            vision_tokens_estimate = (base_size // 16) ** 2
            vision_tokens_label = str(vision_tokens_estimate)
    
    compression_ratio = text_tokens / vision_tokens_estimate if vision_tokens_estimate > 0 else 0

    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Vision tokens: {vision_tokens_label}")
    print(f"Text tokens generated: {text_tokens}")

    # Format compression ratio message based on mode
    if crop_mode and base_size == 1024 and image_size == 640:
        # Gundam mode has variable vision tokens
        print(f"Compression ratio: ~{compression_ratio:.2f}× (based on estimated minimum vision tokens)")
    else:
        print(f"Compression ratio: {compression_ratio:.2f}×")

    # Provide performance guidance based on compression ratio (estimates)
    if compression_ratio < 10:
        performance_estimate = "~97% accuracy (estimate)"
    elif compression_ratio < 12:
        performance_estimate = "~90-97% accuracy (estimate)"
    elif compression_ratio < 15:
        performance_estimate = "~85-90% accuracy (estimate)"
    elif compression_ratio < 20:
        performance_estimate = "~60-85% accuracy (estimate)"
    else:
        performance_estimate = "~60% or less accuracy (estimate, high compression)"

    print(f"Expected accuracy: {performance_estimate}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}\n")
else:
    print(f"\n{'='*60}")
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"No result returned - check output directory")
    print(f"{'='*60}\n")
