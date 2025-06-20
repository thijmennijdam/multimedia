# Dataset Preprocessing Script

This script preprocesses ImageNet and GRIT datasets to create a hierarchical tree structure with embeddings.

## Overview

The script supports two datasets:
- **ImageNet**: Reads `synsets.csv` and creates tree structure from ID-based folders
- **GRIT**: Reads TAR files and creates tree structure from child/parent image-text pairs

Both datasets create the same output structure with consistent folder organization:
- `child_images/`: Contains processed child images
- `parent_images/`: Contains processed parent images (empty for ImageNet)
- `child_texts/`: Contains child text data as txt files for inspection
- `parent_texts/`: Contains parent text data as txt files for inspection
- `tree.pkl`: Contains all embeddings and structured data

## Usage

### ImageNet Processing

#### Basic Usage (testing with limited synsets)
```bash
python preprocess_datasets.py \
    --dataset imagenet \
    --base_path ./hycoclip-main \
    --checkpoint_path ./checkpoints/hycoclip_vit_b.pth \
    --train_config ./configs/train_hycoclip_vit_b.py \
    --limit 10
```

#### Full Processing (all synsets)
```bash
python preprocess_datasets.py \
    --dataset imagenet \
    --base_path ./hycoclip-main \
    --checkpoint_path ./checkpoints/hycoclip_vit_b.pth \
    --train_config ./configs/train_hycoclip_vit_b.py
```

### GRIT Processing

#### Basic Usage (testing with limited samples)
```bash
python preprocess_datasets.py \
    --dataset grit \
    --grit_path /scratch-shared/grit/processed \
    --output_dir ./hycoclip-main \
    --checkpoint_path ./checkpoints/hycoclip_vit_b.pth \
    --train_config ./configs/train_hycoclip_vit_b.py \
    --limit 100
```

#### Full Processing
```bash
python preprocess_datasets.py \
    --dataset grit \
    --grit_path /scratch-shared/grit/processed \
    --output_dir ./hycoclip-main \
    --checkpoint_path ./checkpoints/hycoclip_vit_b.pth \
    --train_config ./configs/train_hycoclip_vit_b.py
```

## Arguments

- `--dataset`: Dataset to preprocess (`imagenet` or `grit`) (required)
- `--base_path`: Base path for ImageNet containing imagenet folder and synsets.csv (default: `./hycoclip-main`)
- `--grit_path`: Path to GRIT TAR files (default: `/scratch-shared/grit/processed`)
- `--output_dir`: Output directory for processed data (default: `./hycoclip-main`)
- `--checkpoint_path`: Path to HyCoCLIP checkpoint (required)
- `--train_config`: Path to HyCoCLIP train config (required)
- `--limit`: Limit number of samples to process for testing (optional)

## Input Structures

### ImageNet Input
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

### GRIT Input
```
/scratch-shared/grit/processed/
├── grit_001.tar
├── grit_002.tar
└── ...
```

Each TAR file contains samples with:
- `child.jpg`: Child image
- `child.txt`: Child caption
- `parent000.jpg`, `parent001.jpg`, etc.: Parent images
- `parent000.txt`, `parent001.txt`, etc.: Parent captions
- `numparents.txt`: Number of parent images

## Output Structures

### ImageNet Output
```
hycoclip-main/
├── ImageNet/
│   ├── tree1/
│   │   ├── child_images/
│   │   │   ├── tench_001.JPEG
│   │   │   ├── tench_002.JPEG
│   │   │   └── ...
│   │   ├── parent_images/      # Empty for ImageNet
│   │   ├── child_texts/
│   │   │   └── child_text.txt
│   │   ├── parent_texts/
│   │   │   └── parent_text.txt
│   │   └── tree1.pkl
│   ├── tree2/
│   └── ...
```

### GRIT Output
```
hycoclip-main/
├── GRIT/
│   ├── tree1/
│   │   ├── child_images/
│   │   │   └── sample_key_child.jpg
│   │   ├── parent_images/
│   │   │   ├── sample_key_parent_1.jpg
│   │   │   ├── sample_key_parent_2.jpg
│   │   │   └── ...
│   │   ├── child_texts/
│   │   │   └── child_text.txt
│   │   ├── parent_texts/
│   │   │   └── parent_text.txt
│   │   └── tree1.pkl
│   ├── tree2/
│   └── ...
```

## Pickle File Structure

Each `treeX.pkl` file contains the same structure for both datasets:

### ImageNet
```python
{
    'parent_image': {},  # Empty for ImageNet
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

### GRIT
```python
{
    'parent_image': {
        'parent_image_1': {
            'name': 'sample_key_parent_1.jpg',
            'embedding': numpy_array
        },
        'parent_image_2': {
            'name': 'sample_key_parent_2.jpg',
            'embedding': numpy_array
        },
        # ... more parent images
    },
    'parent_text': {
        'text': 'parent_caption',
        'embedding': numpy_array
    },
    'child_image': {
        'child_image_1': {
            'name': 'sample_key_child.jpg',
            'embedding': numpy_array
        }
    },
    'child_text': {
        'text': 'child_caption',
        'embedding': numpy_array
    }
}
```

## Text Files for Inspection

### ImageNet Text Files
- `parent_texts/parent_text.txt`: Contains synset ID, synset name, and tree number
- `child_texts/child_text.txt`: Contains synset ID, definition, and tree number

### GRIT Text Files
- `parent_texts/parent_text.txt`: Contains sample key, number of parents, and all parent captions
- `child_texts/child_text.txt`: Contains sample key, child caption, and tree number

## Requirements

- PyTorch
- PIL/Pillow
- tqdm
- numpy
- torchvision
- HyCoCLIP dependencies

## Notes

- Images are automatically renamed and organized in separate folders
- The script handles special characters in filenames
- Progress bars show processing status
- Large datasets may take significant time and storage space
- GPU is recommended for embedding generation
- Both datasets produce the same consistent folder structure for easy analysis
- Text files allow easy inspection of raw data without loading pickle files 