#!/usr/bin/env python3
"""
Patch DeepSeek-OCR model for CPU/MPS compatibility on Apple Silicon Macs.
Run this once after downloading the model for the first time.
"""

import os
import glob

def find_model_file():
    """Find the modeling_deepseekocr.py file in Hugging Face cache."""
    cache_paths = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-OCR/*/modeling_deepseekocr.py"
        )
    )
    if not cache_paths:
        raise FileNotFoundError(
            "Model file not found. Please run the model once to download it first."
        )
    return cache_paths[0]


def patch_model(model_path):
    """Apply patches to make the model device-agnostic."""
    print(f"Patching model at: {model_path}")

    with open(model_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    patches_applied = 0

    # Patch 1: Add device detection in infer method
    if "device = next(self.parameters()).device" not in content:
        content = content.replace(
            "    def infer(self, tokenizer, prompt='', image_file='', output_path = '', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False):\n        self.disable_torch_init()",
            """    def infer(self, tokenizer, prompt='', image_file='', output_path = '', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False, eval_mode=False):
        self.disable_torch_init()

        # Get device from model parameters (supports cuda, mps, cpu)
        device = next(self.parameters()).device
        device_type = 'cpu' if device.type == 'mps' else device.type
        # Use appropriate dtype for each device: float32 for CPU, float16 for MPS, bfloat16 for CUDA
        if device.type == 'cpu':
            autocast_dtype = torch.float32
        elif device.type == 'mps':
            autocast_dtype = torch.float16
        else:
            autocast_dtype = torch.bfloat16"""
        )
        patches_applied += 1

    # Patch 2: Replace .cuda() calls with .to(device)
    replacements = [
        (".cuda()", ".to(device)"),
        ("torch.autocast(\"cuda\", dtype=torch.bfloat16)", "torch.autocast(device_type, dtype=autocast_dtype)"),
        (".to(torch.bfloat16)", ".to(autocast_dtype)"),
        ("images_seq_mask[idx].unsqueeze(-1).cuda()", "images_seq_mask[idx].unsqueeze(-1).to(inputs_embeds.device)"),
    ]

    for old, new in replacements:
        count = content.count(old)
        if count > 0:
            content = content.replace(old, new)
            patches_applied += count

    # Patch 3: Change output file extension from .mmd to .md
    if "result.mmd" in content:
        content = content.replace("result.mmd", "result.md")
        patches_applied += 1

    if content == original_content:
        print("✓ Model already patched!")
        return

    # Write patched content
    with open(model_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Successfully applied {patches_applied} patches!")
    print("✓ Model is now compatible with CPU/MPS backends")


if __name__ == "__main__":
    try:
        model_path = find_model_file()
        patch_model(model_path)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
