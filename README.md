# DeepSeek-OCR for Apple Silicon (CPU)

Setup guide for testing DeepSeek-OCR on Apple Silicon Macs using CPU inference.

## Prerequisites

### Install Miniconda (if not already installed)

```bash
# Download Miniconda for Apple Silicon
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Install
bash Miniconda3-latest-MacOSX-arm64.sh

# Follow prompts, then restart terminal or run:
source ~/.zshrc  # or ~/.bash_profile if using bash

# Verify installation
conda --version
```

## Quick Start

```bash
# 1. Setup environment
conda create -n deepseek-ocr python=3.12 -y
conda activate deepseek-ocr
pip install transformers==4.46.3 tokenizers==0.20.3 torch torchvision Pillow einops

# 2. Run once to download model (~17GB)
python run-ocr.py  # Will take a few minutes

# 3. Patch for CPU compatibility (run once)
python patch-for-cpu.py

# 4. Run OCR (now works correctly)
python run-ocr.py
```

Check `output/result.md` for extracted text.

---

## Installation

### 1. Create conda environment
```bash
conda create -n deepseek-ocr python=3.12 -y
conda activate deepseek-ocr
```

### 2. Install dependencies
```bash
pip install transformers==4.46.3 tokenizers==0.20.3 torch torchvision Pillow einops
```

### 3. Download and patch the model

First run will download the model (~17GB):
```bash
python run-ocr.py
```

Then patch for CPU compatibility:
```bash
python patch-for-cpu.py
```

Then run `python run-ocr.py` again. See below for details on the optional parameters.

## Usage

### Basic OCR
```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = "deepseek-ai/DeepSeek-OCR"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    attn_implementation="eager",
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().to(torch.float32).to("cpu")

# Run OCR
prompt = "<image>\n<|grounding|>Convert the document to markdown."
result = model.infer(
    tokenizer,
    prompt=prompt,
    image_file="your-image.jpg",
    output_path="output/",
    base_size=1024,    # See resolution modes below
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True  # Optional: enables compression ratio analysis
)
```

### Output Files
- `output/result.md` - Extracted text in markdown format
- `output/result_with_boxes.jpg` - Image with bounding boxes
- `output/images/` - Cropped regions (if applicable)

### Performance Metrics
When `test_compress=True`, the script displays detailed performance metrics:
- **Processing time** - Total time taken for OCR
- **Vision tokens** - Number of tokens used to encode the image
- **Text tokens generated** - Number of tokens in the output text
- **Compression ratio** - Text tokens / Vision tokens (higher = more compression)
- **Expected accuracy** - Estimated accuracy based on compression ratio:
  - < 10×: ~97% accuracy
  - 10-12×: ~90-97% accuracy
  - 12-15×: ~85-90% accuracy
  - 15-20×: ~60-85% accuracy
  - > 20×: ~60% or less accuracy (high compression)

## Customizing the Script

The provided `run-ocr.py` script has several configurable parameters. Edit the following lines to customize:

### Input/Output Settings
```python
image_file = "image.jpg"       # Change to your image file path
output_path = "output/"        # Change output directory
prompt = "<image>\n<|grounding|>Convert the document to markdown."  # Change prompt
```

### Resolution Settings
```python
base_size = 1024      # Image base resolution (512, 640, 1024, 1280)
image_size = 640      # Processing resolution
crop_mode = True      # Enable/disable Gundam mode (dynamic cropping)
```

See "Resolution Modes" below for recommended configurations.

## Resolution Modes

Choose based on your image size and quality needs:

### Native Modes (crop_mode=False)
| Mode | Config | Tokens | Use Case |
|------|--------|--------|----------|
| **Tiny** | `base_size=512, image_size=512` | 64 | Quick tests, small images |
| **Small** | `base_size=640, image_size=640` | 100 | Business cards, receipts |
| **Base** | `base_size=1024, image_size=1024` | 256 | Standard documents (recommended) |
| **Large** | `base_size=1280, image_size=1280` | 400 | High-resolution scans |

### Dynamic Mode (crop_mode=True)
| Mode | Config | Tokens | Use Case |
|------|--------|--------|----------|
| **Gundam** | `base_size=1024, image_size=640, crop_mode=True` | 256-1056+ | Large/complex documents, newspapers |

**Gundam mode** splits images into tiles and adds a global view for better handling of large or wide documents.

## Prompts

```python
# Document with layout preservation
"<image>\n<|grounding|>Convert the document to markdown."

# Simple OCR without layout
"<image>\nFree OCR."

# Figure/diagram parsing
"<image>\nParse the figure."

# Text localization (provides coordinates)
"<image>\nLocate <|ref|>specific text<|/ref|> in the image."
```

## Performance Notes

- **CPU mode**: Slower but reliable on Apple Silicon
- **Processing time**: ~1-2 minutes per page (e.g. 30 sec. for 3551×4686 jpeg on M4 with 24GB, Gundam mode)
- **Memory**: Requires ~8-16GB RAM depending on resolution mode
- **MPS (Apple GPU)**: Currently not recommended due to output quality issues

## Supported Image Formats

- `.jpg`, `.jpeg` - Images (recommended)
- `.png` - Images
- `.pdf` - Use the vLLM scripts in `DeepSeek-OCR-master/DeepSeek-OCR-vllm/`

## Example

```bash
# Using the provided script
python run-ocr.py
```

Check `output/` directory for results.

## Troubleshooting

**Model not found error**: Run `run-ocr.py` once to download the model before patching.

**Out of memory**: Try a smaller resolution mode (Tiny or Small).

**Poor quality output**: Use Base or Gundam mode for better results.

**Repetitive text**: Model not patched correctly, re-run `patch-for-cpu.py`.

## Credits

Model by DeepSeek-AI: https://github.com/deepseek-ai/DeepSeek-OCR. Original Apple Silicon installation instructions: https://readmultiplex.com/2025/10/20/an-ai-model-just-compressed-an-entire-encyclopedia-into-a-single-high-resolution-image/ and https://www.lvtao.net/tool/deepseek-ocr-macos-arm-deployment.html. Subsequent revisions by Claude Code.
