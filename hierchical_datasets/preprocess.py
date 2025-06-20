#!/usr/bin/env python3
"""
Dataset Preprocessing Script

This script processes ImageNet and GRIT datasets to create a hierarchical tree structure.

For ImageNet:
- Reads synsets.csv and creates tree folders with JSON files containing:
  - meta_data_tree.json: Tree structure with metadata and file paths
  - embeddings.json: All embeddings indexed by ID
  - data/treeN/: Actual data files organized by tree

For GRIT:
- Reads TAR files and creates tree folders with JSON files containing:
  - meta_data_tree.json: Tree structure with metadata and file paths
  - embeddings.json: All embeddings indexed by ID
  - data/treeN/: Actual data files organized by tree

Both datasets create the same folder structure:
- data/treeN/child_images/: Contains processed child images
- data/treeN/parent_images/: Contains processed parent images (empty for ImageNet)
- data/treeN/child_texts/: Contains child text data as txt files
- data/treeN/parent_texts/: Contains parent text data as txt files
- meta_data_tree.json: Contains tree structure and file paths
- embeddings.json: Contains all embeddings indexed by ID
"""

import os
import csv
import json
import shutil
import glob
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from torchvision import transforms as T
from datetime import datetime

# HyCoCLIP imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hycoclip'))
from hycoclip.config import LazyConfig, LazyFactory
from hycoclip.utils.checkpointing import CheckpointManager
from hycoclip.tokenizer import Tokenizer
from utils.prepare_GRIT_webdataset import ImageTextWebDataset

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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

def print_json_structure(data, indent=0):
    """Print JSON structure without embeddings for readability"""
    spaces = "  " * indent
    if isinstance(data, dict):
        print(f"{spaces}{{")
        for key, value in data.items():
            if key == 'embeddings':
                print(f"{spaces}  '{key}': <embeddings_dict with {len(value)} entries>")
            else:
                print(f"{spaces}  '{key}':", end="")
                if isinstance(value, (dict, list)):
                    print()
                    print_json_structure(value, indent + 2)
                else:
                    print(f" {repr(value)}")
        print(f"{spaces}}}")
    elif isinstance(data, list):
        print(f"{spaces}[")
        for i, item in enumerate(data):
            print(f"{spaces}  [{i}]:", end="")
            if isinstance(item, (dict, list)):
                print()
                print_json_structure(item, indent + 2)
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

def create_imagenet_tree_structure(base_path, output_dir, synsets, model, preprocess, device):
    """Create the ImageNet tree structure with JSON files"""
    
    # Create ImageNet base directory
    imagenet_path = Path(output_dir) / "ImageNet"
    imagenet_path.mkdir(exist_ok=True)
    
    # Create data directory
    data_path = imagenet_path / "data"
    data_path.mkdir(exist_ok=True)
    
    print(f"Processing {len(synsets)} ImageNet synsets...")
    
    # Initialize metadata and embeddings structures
    meta_data = {
        "dataset_info": {
            "name": "ImageNet",
            "total_trees": len(synsets),
            "created_at": datetime.now().isoformat(),
            "model": "HyCoCLIP"
        },
        "trees": {}
    }
    
    embeddings_data = {
        "embedding_info": {
            "model": "HyCoCLIP",
            "dimension": None,  # Will be set after first embedding
            "created_at": datetime.now().isoformat()
        },
        "embeddings": {}
    }
    
    for i, synset in enumerate(tqdm(synsets, desc="Creating ImageNet tree structure")):
        tree_id = f"tree{i+1}"
        tree_folder = data_path / tree_id
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
        
        # Set embedding dimension if not set
        if embeddings_data["embedding_info"]["dimension"] is None:
            embeddings_data["embedding_info"]["dimension"] = len(synset_name_embedding)
        
        # Generate unique IDs for embeddings
        parent_text_id = f"pt_{tree_id}"
        child_text_id = f"ct_{tree_id}"
        
        # Store embeddings
        embeddings_data["embeddings"][parent_text_id] = synset_name_embedding
        embeddings_data["embeddings"][child_text_id] = definition_embedding
        
        # Save raw text data to txt files for easy inspection
        parent_text_file = parent_texts_folder / "parent_text.txt"
        child_text_file = child_texts_folder / "child_text.txt"
        
        with open(parent_text_file, 'w', encoding='utf-8') as f:
            f.write(f"Synset ID: {synset['synset_id']}\n")
            f.write(f"Synset Name: {synset['synset_name']}\n")
            f.write(f"Tree: {tree_id}\n")
        
        with open(child_text_file, 'w', encoding='utf-8') as f:
            f.write(f"Synset ID: {synset['synset_id']}\n")
            f.write(f"Definition: {synset['definition']}\n")
            f.write(f"Tree: {tree_id}\n")
        
        # Process images from the original synset folder
        original_folder = Path(base_path) / synset['synset_id']
        child_images_list = []
        
        if original_folder.exists():
            image_files = list(original_folder.glob("*.JPEG"))
            print(f"Processing {len(image_files)} images for {synset['synset_name']}...")
            
            for j, image_path in enumerate(tqdm(image_files, desc=f"Processing images for {synset['synset_name']}", leave=False)):
                # Generate new image name (sanitize synset name for filename)
                safe_synset_name = synset['synset_name'].replace(' ', '_').replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                new_image_name = f"{safe_synset_name}_{j+1:03d}.JPEG"
                new_image_path = child_images_folder / new_image_name
                
                # Copy and rename image
                shutil.copy2(image_path, new_image_path)
                
                # Generate image embedding
                image_embedding = generate_image_embedding(model, preprocess, new_image_path, device)
                
                if image_embedding is not None:
                    # Generate unique ID for image embedding
                    child_image_id = f"ci_{tree_id}_{j+1:03d}"
                    
                    # Store embedding
                    embeddings_data["embeddings"][child_image_id] = image_embedding
                    
                    # Add to child images list
                    child_images_list.append({
                        "id": child_image_id,
                        "name": new_image_name,
                        "path": f"hierchical_datasets/ImageNet/data/{tree_id}/child_images/{new_image_name}"
                    })
        
        # Create tree metadata
        tree_metadata = {
            "tree_id": tree_id,
            "synset_id": synset['synset_id'],
            "parent_text": {
                "id": parent_text_id,
                "text": synset['synset_name'],
                "path": f"hierchical_datasets/ImageNet/data/{tree_id}/parent_texts/parent_text.txt"
            },
            "child_text": {
                "id": child_text_id,
                "text": synset['definition'],
                "path": f"hierchical_datasets/ImageNet/data/{tree_id}/child_texts/child_text.txt"
            },
            "parent_images": [],  # Empty for ImageNet
            "child_images": child_images_list
        }
        
        meta_data["trees"][tree_id] = tree_metadata
        
        # Save JSON files after each tree (incremental saving)
        meta_data_path = imagenet_path / "meta_data_tree.json"
        embeddings_path = imagenet_path / "embeddings.json"
        
        # Update total trees count
        meta_data["dataset_info"]["total_trees"] = i + 1
        
        with open(meta_data_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
        
        with open(embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, indent=2, cls=NumpyEncoder)
        
        print(f"Created {tree_folder} with {len(child_images_list)} images")
    
    print(f"\nCompleted processing {len(synsets)} ImageNet synsets")
    print(f"Saved metadata to: {meta_data_path}")
    print(f"Saved embeddings to: {embeddings_path}")

def create_grit_tree_structure(grit_path, model, device, output_dir, max_samples=None):
    """Create the GRIT tree structure with JSON files"""
    
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
    
    # Create data directory
    data_path = grit_output_path / "data"
    data_path.mkdir(exist_ok=True)
    
    # Initialize metadata and embeddings structures
    meta_data = {
        "dataset_info": {
            "name": "GRIT",
            "total_trees": 0,  # Will be updated as we process
            "created_at": datetime.now().isoformat(),
            "model": "HyCoCLIP"
        },
        "trees": {}
    }
    
    embeddings_data = {
        "embedding_info": {
            "model": "HyCoCLIP",
            "dimension": None,  # Will be set after first embedding
            "created_at": datetime.now().isoformat()
        },
        "embeddings": {}
    }
    
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
                tree_id = f"tree{tree_count}"
                
                # Create tree folder
                tree_folder = data_path / tree_id
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
                child_images_list = []
                parent_images_list = []
                child_texts = []
                parent_texts = []
                
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
                        
                        # Set embedding dimension if not set
                        if embeddings_data["embedding_info"]["dimension"] is None:
                            embeddings_data["embedding_info"]["dimension"] = len(child_text_embedding)
                        
                        # Generate unique IDs
                        child_image_id = f"ci_{tree_id}_001"
                        child_text_id = f"ct_{tree_id}"
                        
                        if child_image_embedding is not None:
                            # Store embeddings
                            embeddings_data["embeddings"][child_image_id] = child_image_embedding
                            embeddings_data["embeddings"][child_text_id] = child_text_embedding
                            
                            child_images_list.append({
                                "id": child_image_id,
                                "name": child_image_name,
                                "path": f"hierchical_datasets/GRIT/data/{tree_id}/child_images/{child_image_name}"
                            })
                        
                        child_texts.append(child_caption)
                        
                        # Save child text to file
                        child_text_file = child_texts_folder / "child_text.txt"
                        with open(child_text_file, 'w', encoding='utf-8') as f:
                            f.write(f"Sample Key: {sample_key}\n")
                            f.write(f"Child Caption: {child_caption}\n")
                            f.write(f"Tree: {tree_id}\n")
                
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
                                # Generate unique ID
                                parent_image_id = f"pi_{tree_id}_{parent_image_count:03d}"
                                
                                # Store embedding
                                embeddings_data["embeddings"][parent_image_id] = parent_image_embedding
                                
                                parent_images_list.append({
                                    "id": parent_image_id,
                                    "name": parent_image_name,
                                    "path": f"hierchical_datasets/GRIT/data/{tree_id}/parent_images/{parent_image_name}"
                                })
                            
                            parent_texts.append(parent_caption)
                
                # Generate parent text embedding (use first parent text if available)
                parent_text_id = f"pt_{tree_id}"
                if parent_texts:
                    parent_text_embedding = generate_text_embedding(model, parent_texts[0], device)
                    embeddings_data["embeddings"][parent_text_id] = parent_text_embedding
                
                # Save parent texts to file
                if parent_texts:
                    parent_text_file = parent_texts_folder / "parent_text.txt"
                    with open(parent_text_file, 'w', encoding='utf-8') as f:
                        f.write(f"Sample Key: {sample_key}\n")
                        f.write(f"Number of Parents: {len(parent_texts)}\n")
                        f.write(f"Tree: {tree_id}\n")
                        f.write("Parent Captions:\n")
                        for idx, caption in enumerate(parent_texts, 1):
                            f.write(f"  {idx}. {caption}\n")
                
                # Create tree metadata
                tree_metadata = {
                    "tree_id": tree_id,
                    "sample_key": sample_key,
                    "parent_text": {
                        "id": parent_text_id,
                        "text": parent_texts[0] if parent_texts else "",
                        "path": f"hierchical_datasets/GRIT/data/{tree_id}/parent_texts/parent_text.txt"
                    },
                    "child_text": {
                        "id": child_text_id if child_texts else "",
                        "text": child_texts[0] if child_texts else "",
                        "path": f"hierchical_datasets/GRIT/data/{tree_id}/child_texts/child_text.txt"
                    },
                    "parent_images": parent_images_list,
                    "child_images": child_images_list
                }
                
                meta_data["trees"][tree_id] = tree_metadata
                
                # Save JSON files after each tree (incremental saving)
                meta_data_path = grit_output_path / "meta_data_tree.json"
                embeddings_path = grit_output_path / "embeddings.json"
                
                # Update total trees count
                meta_data["dataset_info"]["total_trees"] = tree_count
                
                with open(meta_data_path, 'w', encoding='utf-8') as f:
                    json.dump(meta_data, f, indent=2, ensure_ascii=False)
                
                with open(embeddings_path, 'w', encoding='utf-8') as f:
                    json.dump(embeddings_data, f, indent=2, cls=NumpyEncoder)
                
                sample_count += 1
                
                if sample_count % 100 == 0:
                    print(f"Processed {sample_count} GRIT samples...")
                    
            except Exception as e:
                print(f"Error processing GRIT sample {sample.get('__key__', 'unknown')}: {e}")
                continue
    
    print(f"\nGRIT processing complete! Created {tree_count} trees with {sample_count} samples")
    print(f"Saved metadata to: {meta_data_path}")
    print(f"Saved embeddings to: {embeddings_path}")

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
        csv_path = base_path / "synsets.csv"
        
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
        create_imagenet_tree_structure(args.base_path, args.output_dir, synsets, model, preprocess, device)
        
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