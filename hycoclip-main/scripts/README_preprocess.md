# ImageNet Preprocessing Script

This script preprocesses the ImageNet dataset to create a hierarchical tree structure with embeddings.

## Overview

The script reads `synsets.csv` and creates a new folder structure:
- `ImageNet/tree1/`, `tree2/`, etc.
- Each tree folder contains:
  - A `.pkl` file with keys: `parent_image`, `parent_text`, `child_image`, `child_text`
  - Renamed images moved from the original ID-based folders
  - Embeddings generated using HyCoCLIP or CLIP models

## Usage

### Basic Usage (testing with limited synsets)
```bash
python preprocess_imagenet.py \
    --base_path ./hycoclip-main \
    --checkpoint_path ./checkpoints/hycoclip_vit_b.pth \
    --train_config ./configs/train_hycoclip_vit_b.py \
    --limit 10
```

### Full Processing (all synsets)
```bash
python preprocess_imagenet.py \
    --base_path ./hycoclip-main \
    --checkpoint_path ./checkpoints/hycoclip_vit_b.pth \
    --train_config ./configs/train_hycoclip_vit_b.py
```

## Arguments

- `--base_path`: Base path containing the imagenet folder and synsets.csv (default: `./hycoclip-main`)
- `--checkpoint_path`: Path to HyCoCLIP checkpoint (required)
- `--train_config`: Path to HyCoCLIP train config (required)
- `--limit`: Limit number of synsets to process for testing (optional)

## Input Structure

The script expects:
```
hycoclip-main/
├── imagenet/
│   ├── synsets.csv
│   ├── n01440764/          # ID-based folders
│   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   ├── ILSVRC2012_val_00002138.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ...
```

## Output Structure

The script creates:
```
hycoclip-main/
├── ImageNet/
│   ├── tree1/
│   │   ├── tree1.pkl
│   │   ├── tench_001.JPEG
│   │   ├── tench_002.JPEG
│   │   └── ...
│   ├── tree2/
│   │   ├── tree2.pkl
│   │   ├── goldfish_001.JPEG
│   │   └── ...
│   └── ...
```

## Pickle File Structure

Each `treeX.pkl` file contains:
```python
{
    'parent_image': {},  # Empty in this case
    'parent_text': {
        'text': 'synset_name',
        'embedding': numpy_array
    },
    'child_image': {
        'child_image_1': {
            'name': 'synset_name_001.JPEG',
            'embedding': numpy_array
        },
        'child_image_2': {
            'name': 'synset_name_002.JPEG', 
            'embedding': numpy_array
        },
        # ... more images
    },
    'child_text': {
        'text': 'definition',
        'embedding': numpy_array
    }
}
```

## Testing

Run a quick test with 5 synsets:
```bash
python test_preprocess.py
```

## Requirements

- PyTorch
- PIL/Pillow
- tqdm
- numpy
- HyCoCLIP dependencies

## Notes

- Images are automatically renamed to match synset names (with sanitized filenames)
- The script handles special characters in synset names
- Progress bars show processing status
- Large datasets may take significant time and storage space
- GPU is recommended for embedding generation 