#!/usr/bin/env python3
"""
Dataset Preprocessing Script

This script processes ImageNet and GRIT datasets to create a hierarchical tree structure.

For ImageNet:
- Reads synsets.csv and creates tree folders with pickle files containing:
  - parent_text: synset name with embeddings
  - child_text: definition with embeddings  
  - child_image: image names with embeddings
  - parent_image: (empty in this case)

For GRIT:
- Reads TAR files and creates tree folders with pickle files containing:
  - parent_text: parent captions with embeddings
  - child_text: child captions with embeddings  
  - child_image: child image names with embeddings
  - parent_image: parent image names with embeddings

Both datasets create the same folder structure:
- child_images/: Contains processed child images
- parent_images/: Contains processed parent images (empty for ImageNet)
- child_texts/: Contains child text data as txt files
- parent_texts/: Contains parent text data as txt files
- tree.pkl: Contains all embeddings and structured data
"""

import os
import csv
import pickle
import shutil
import glob
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from torchvision import transforms as T

# HyCoCLIP imports
from hycoclip.config import LazyConfig, LazyFactory
from hycoclip.utils.checkpointing import CheckpointManager
from hycoclip.tokenizer import Tokenizer

# Add path imports for GRIT processing
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from prepare_GRIT_webdataset import ImageTextWebDataset

def load_hycoclip_model(checkpoint_path, train_config_path):
    """Load HyCoCLIP model for generating embeddings"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not checkpoint_path or not train_config_path:
        raise ValueError("Both checkpoint_path and train_config_path are required for HyCoCLIP model")
    
    # Load HyCoCLIP model
    train_config = LazyConfig.load(train_config_path)
    model = LazyFactory.build_model(train_config, device).eval()
    CheckpointManager(model=model).load(checkpoint_path)
    
    # Create image preprocessing transform
    preprocess = T.Compose([
        T.Resize(224, T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    
    return model, preprocess, device

def generate_text_embedding(model, text, device):
    """Generate embedding for text using HyCoCLIP model"""
    tokenizer = Tokenizer()
    text_tokens = tokenizer([text])
    with torch.no_grad():
        text_features = model.encode_text(text_tokens, project=True)
    return text_features.cpu().numpy().flatten()

def generate_image_embedding(model, preprocess, image_path, device):
    """Generate embedding for image using HyCoCLIP model"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor, project=True)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def generate_image_embedding_from_pil(model, image_pil, device):
    """Generate embedding for PIL image using HyCoCLIP model"""
    try:
        # Convert PIL image to tensor (same preprocessing as in visualize_embeddings.py)
        image = image_pil.convert('RGB').resize((224, 224))
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(img_tensor, project=True)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing PIL image: {e}")
        return None

def print_pickle_structure(data, indent=0):
    """Print pickle structure without embeddings for readability"""
    spaces = "  " * indent
    if isinstance(data, dict):
        print(f"{spaces}{{")
        for key, value in data.items():
            if key == 'embedding':
                print(f"{spaces}  '{key}': <numpy_array shape={value.shape if hasattr(value, 'shape') else 'unknown'}>")
            else:
                print(f"{spaces}  '{key}':", end="")
                if isinstance(value, (dict, list)):
                    print()
                    print_pickle_structure(value, indent + 2)
                else:
                    print(f" {repr(value)}")
        print(f"{spaces}}}")
    elif isinstance(data, list):
        print(f"{spaces}[")
        for i, item in enumerate(data):
            print(f"{spaces}  [{i}]:", end="")
            if isinstance(item, (dict, list)):
                print()
                print_pickle_structure(item, indent + 2)
            else:
                print(f" {repr(item)}")
        print(f"{spaces}]")
    else:
        print(f"{spaces}{repr(data)}")

def read_synsets_csv(csv_path):
    """Read the synsets.csv file and return list of synset data"""
    synsets = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            synsets.append({
                'synset_id': row['synset_id'],
                'synset_name': row['synset_name'],
                'definition': row['definition']
            })
    return synsets

def create_imagenet_tree_structure(base_path, synsets, model, preprocess, device):
    """Create the ImageNet tree structure with pickle files"""
    
    # Create ImageNet base directory
    imagenet_path = Path(base_path) / "ImageNet"
    imagenet_path.mkdir(exist_ok=True)
    
    print(f"Processing {len(synsets)} ImageNet synsets...")
    
    for i, synset in enumerate(tqdm(synsets, desc="Creating ImageNet tree structure")):
        tree_folder = imagenet_path / f"tree{i+1}"
        tree_folder.mkdir(exist_ok=True)
        
        # Create child_images and parent_images folders
        child_images_folder = tree_folder / "child_images"
        parent_images_folder = tree_folder / "parent_images"
        child_images_folder.mkdir(exist_ok=True)
        parent_images_folder.mkdir(exist_ok=True)
        
        # Create text folders for raw data inspection
        parent_texts_folder = tree_folder / "parent_texts"
        child_texts_folder = tree_folder / "child_texts"
        parent_texts_folder.mkdir(exist_ok=True)
        child_texts_folder.mkdir(exist_ok=True)
        
        # Generate text embeddings
        synset_name_embedding = generate_text_embedding(model, synset['synset_name'], device)
        definition_embedding = generate_text_embedding(model, synset['definition'], device)
        
        # Save raw text data to txt files for easy inspection
        parent_text_file = parent_texts_folder / "parent_text.txt"
        child_text_file = child_texts_folder / "child_text.txt"
        
        with open(parent_text_file, 'w', encoding='utf-8') as f:
            f.write(f"Synset ID: {synset['synset_id']}\n")
            f.write(f"Synset Name: {synset['synset_name']}\n")
            f.write(f"Tree: tree{i+1}\n")
        
        with open(child_text_file, 'w', encoding='utf-8') as f:
            f.write(f"Synset ID: {synset['synset_id']}\n")
            f.write(f"Definition: {synset['definition']}\n")
            f.write(f"Tree: tree{i+1}\n")
        
        # Process images from the original synset folder
        original_folder = Path(base_path) / "imagenet" / synset['synset_id']
        child_images = {}
        
        if original_folder.exists():
            image_files = list(original_folder.glob("*.JPEG"))
            print(f"\nProcessing {len(image_files)} images for {synset['synset_name']}...")
            
            for j, image_path in enumerate(tqdm(image_files, desc=f"Processing images for {synset['synset_name']}", leave=False)):
                # Generate new image name (sanitize synset name for filename)
                safe_synset_name = synset['synset_name'].replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                new_image_name = f"{safe_synset_name}_{j+1:03d}.JPEG"
                new_image_path = child_images_folder / new_image_name  # Save to child_images folder
                
                # Copy and rename image
                shutil.copy2(image_path, new_image_path)
                
                # Generate image embedding
                image_embedding = generate_image_embedding(model, preprocess, new_image_path, device)
                
                if image_embedding is not None:
                    child_images[f"child_image_{j+1}"] = {
                        'name': new_image_name,
                        'embedding': image_embedding
                    }
        
        # Create the data structure for pickle file
        tree_data = {
            'parent_image': {},  # Empty as specified for ImageNet
            'parent_text': {
                'text': synset['synset_name'],
                'embedding': synset_name_embedding
            },
            'child_image': child_images,
            'child_text': {
                'text': synset['definition'],
                'embedding': definition_embedding
            }
        }
        
        # Save pickle file
        pickle_path = tree_folder / f"tree{i+1}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(tree_data, f)
        
        # Print pickle structure without embeddings
        print(f"\nCreated {tree_folder} with {len(child_images)} images")
        print("Pickle structure (without embeddings):")
        print_pickle_structure(tree_data)

def create_grit_tree_structure(grit_path, model, device, output_dir, max_samples=None):
    """Create the GRIT tree structure with pickle files"""
    
    # Find GRIT TAR files
    tar_files = glob.glob(os.path.join(grit_path, "*.tar"))
    
    if not tar_files:
        print(f"No TAR files found in {grit_path}")
        return
    
    # Sort TAR files for consistent processing order
    tar_files.sort()
    print(f"Found {len(tar_files)} TAR files in {grit_path}")
    
    # Create GRIT base directory
    grit_output_path = Path(output_dir) / "GRIT"
    grit_output_path.mkdir(exist_ok=True)
    
    sample_count = 0
    tree_count = 0
    
    # Process multiple TAR files if needed
    for tar_file in tar_files:
        if max_samples and sample_count >= max_samples:
            break
            
        print(f"Loading from: {tar_file}")
        dataset = ImageTextWebDataset(tarfiles=[tar_file], infinite_stream=False)
        
        for sample in dataset:
            if max_samples and sample_count >= max_samples:
                break
                
            try:
                sample_key = sample['__key__']
                tree_count += 1
                
                # Create tree folder
                tree_folder = grit_output_path / f"tree{tree_count}"
                tree_folder.mkdir(exist_ok=True)
                
                # Create child_images and parent_images folders
                child_images_folder = tree_folder / "child_images"
                parent_images_folder = tree_folder / "parent_images"
                child_images_folder.mkdir(exist_ok=True)
                parent_images_folder.mkdir(exist_ok=True)
                
                # Create text folders for raw data inspection
                parent_texts_folder = tree_folder / "parent_texts"
                child_texts_folder = tree_folder / "child_texts"
                parent_texts_folder.mkdir(exist_ok=True)
                child_texts_folder.mkdir(exist_ok=True)
                
                # Initialize data structures
                child_images = {}
                parent_images = {}
                child_texts = []
                parent_texts = []
                child_text_embedding = np.array([])
                parent_text_embedding = np.array([])
                
                # Process child image and text
                if 'child.jpg' in sample and 'child.txt' in sample:
                    child_image = sample['child.jpg']
                    child_caption = str(sample['child.txt'])
                    
                    if isinstance(child_image, Image.Image):
                        # Save child image
                        child_image_name = f"{sample_key}_child.jpg"
                        child_image_path = child_images_folder / child_image_name
                        child_image.convert('RGB').save(child_image_path, 'JPEG')
                        
                        # Generate child image embedding
                        child_image_embedding = generate_image_embedding_from_pil(model, child_image, device)
                        
                        # Generate child text embedding
                        child_text_embedding = generate_text_embedding(model, child_caption, device)
                        
                        if child_image_embedding is not None:
                            child_images['child_image_1'] = {
                                'name': child_image_name,
                                'embedding': child_image_embedding
                            }
                        
                        child_texts.append(child_caption)
                        
                        # Save child text to file
                        child_text_file = child_texts_folder / "child_text.txt"
                        with open(child_text_file, 'w', encoding='utf-8') as f:
                            f.write(f"Sample Key: {sample_key}\n")
                            f.write(f"Child Caption: {child_caption}\n")
                            f.write(f"Tree: tree{tree_count}\n")
                
                # Process parent images and texts
                num_parents = int(sample.get('numparents.txt', 0))
                parent_image_count = 0
                
                for i in range(num_parents):
                    parent_key = f"parent{i:03d}"
                    if f"{parent_key}.jpg" in sample and f"{parent_key}.txt" in sample:
                        parent_image = sample[f"{parent_key}.jpg"]
                        parent_caption = str(sample[f"{parent_key}.txt"])
                        
                        if isinstance(parent_image, Image.Image):
                            parent_image_count += 1
                            
                            # Save parent image
                            parent_image_name = f"{sample_key}_parent_{parent_image_count}.jpg"
                            parent_image_path = parent_images_folder / parent_image_name
                            parent_image.convert('RGB').save(parent_image_path, 'JPEG')
                            
                            # Generate parent image embedding
                            parent_image_embedding = generate_image_embedding_from_pil(model, parent_image, device)
                            
                            if parent_image_embedding is not None:
                                parent_images[f'parent_image_{parent_image_count}'] = {
                                    'name': parent_image_name,
                                    'embedding': parent_image_embedding
                                }
                            
                            parent_texts.append(parent_caption)
                
                # Generate parent text embedding (use first parent text if available)
                if parent_texts:
                    parent_text_embedding = generate_text_embedding(model, parent_texts[0], device)
                
                # Save parent texts to file
                if parent_texts:
                    parent_text_file = parent_texts_folder / "parent_text.txt"
                    with open(parent_text_file, 'w', encoding='utf-8') as f:
                        f.write(f"Sample Key: {sample_key}\n")
                        f.write(f"Number of Parents: {len(parent_texts)}\n")
                        f.write(f"Tree: tree{tree_count}\n")
                        f.write("Parent Captions:\n")
                        for idx, caption in enumerate(parent_texts, 1):
                            f.write(f"  {idx}. {caption}\n")
                
                # Create the data structure for pickle file
                tree_data = {
                    'parent_image': parent_images,
                    'parent_text': {
                        'text': parent_texts[0] if parent_texts else "",
                        'embedding': parent_text_embedding
                    },
                    'child_image': child_images,
                    'child_text': {
                        'text': child_texts[0] if child_texts else "",
                        'embedding': child_text_embedding
                    }
                }
                
                # Save pickle file
                pickle_path = tree_folder / f"tree{tree_count}.pkl"
                with open(pickle_path, 'wb') as f:
                    pickle.dump(tree_data, f)
                
                # Print pickle structure without embeddings
                print(f"\nCreated {tree_folder}")
                print("Pickle structure (without embeddings):")
                print_pickle_structure(tree_data)
                
                sample_count += 1
                
                if sample_count % 100 == 0:
                    print(f"Processed {sample_count} GRIT samples...")
                    
            except Exception as e:
                print(f"Error processing GRIT sample {sample.get('__key__', 'unknown')}: {e}")
                continue
    
    print(f"GRIT processing complete! Created {tree_count} trees with {sample_count} samples")

def main():
    parser = argparse.ArgumentParser(description='Preprocess ImageNet or GRIT dataset into hierarchical tree structure')
    parser.add_argument('--dataset', choices=['imagenet', 'grit'], required=True,
                        help='Dataset to preprocess')
    parser.add_argument('--base_path', type=str, default='./', 
                        help='Base path for ImageNet (containing imagenet folder and synsets.csv)')
    parser.add_argument('--grit_path', type=str, default='/scratch-shared/grit/processed',
                        help='Path to GRIT TAR files')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Output directory for processed data')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to HyCoCLIP checkpoint')
    parser.add_argument('--train_config', type=str, required=True,
                        help='Path to HyCoCLIP train config')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model, preprocess, device = load_hycoclip_model(args.checkpoint_path, args.train_config)
    print(f"Model loaded on device: {device}")
    
    if args.dataset == 'imagenet':
        # Process ImageNet
        base_path = Path(args.base_path)
        csv_path = base_path / "imagenet" / "synsets.csv"
        
        if not csv_path.exists():
            print(f"Error: synsets.csv not found at {csv_path}")
            return
        
        # Read synsets
        print("Reading ImageNet synsets...")
        synsets = read_synsets_csv(csv_path)
        
        if args.limit:
            synsets = synsets[:args.limit]
            print(f"Limited to first {args.limit} synsets for testing")
        
        # Create ImageNet tree structure
        create_imagenet_tree_structure(args.output_dir, synsets, model, preprocess, device)
        
    elif args.dataset == 'grit':
        # Process GRIT
        if not os.path.exists(args.grit_path):
            print(f"Error: GRIT path not found at {args.grit_path}")
            return
        
        # Create GRIT tree structure
        create_grit_tree_structure(args.grit_path, model, device, args.output_dir, max_samples=args.limit)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main() 