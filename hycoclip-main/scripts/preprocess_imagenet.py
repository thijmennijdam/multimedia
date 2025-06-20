#!/usr/bin/env python3
"""
ImageNet Preprocessing Script

This script processes the ImageNet dataset to create a hierarchical tree structure.
It reads synsets.csv and creates tree folders with pickle files containing:
- parent_text: synset name with embeddings
- child_text: definition with embeddings  
- child_image: image names with embeddings
- parent_image: (empty in this case)

The script also moves and renames images to match the new structure.
"""

import os
import csv
import pickle
import shutil
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

def create_tree_structure(base_path, synsets, model, preprocess, device):
    """Create the tree structure with pickle files"""
    
    # Create ImageNet base directory
    imagenet_path = Path(base_path) / "ImageNet"
    imagenet_path.mkdir(exist_ok=True)
    
    print(f"Processing {len(synsets)} synsets...")
    
    for i, synset in enumerate(tqdm(synsets, desc="Creating tree structure")):
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
            'parent_image': {},  # Empty as specified
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
        print(tree_data)
        # Save pickle file
        pickle_path = tree_folder / f"tree{i+1}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(tree_data, f)
        
        print(f"Created {tree_folder} with {len(child_images)} images")

def main():
    parser = argparse.ArgumentParser(description='Preprocess ImageNet dataset into hierarchical tree structure')
    parser.add_argument('--base_path', type=str, default='./hycoclip-main', 
                        help='Base path containing the imagenet folder and synsets.csv')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to HyCoCLIP checkpoint')
    parser.add_argument('--train_config', type=str, required=True,
                        help='Path to HyCoCLIP train config')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of synsets to process (for testing)')
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    csv_path = base_path / "imagenet" / "synsets.csv"
    
    if not csv_path.exists():
        print(f"Error: synsets.csv not found at {csv_path}")
        return
    
    # Load model
    print("Loading model...")
    model, preprocess, device = load_hycoclip_model(args.checkpoint_path, args.train_config)
    print(f"Model loaded on device: {device}")
    
    # Read synsets
    print("Reading synsets...")
    synsets = read_synsets_csv(csv_path)
    
    if args.limit:
        synsets = synsets[:args.limit]
        print(f"Limited to first {args.limit} synsets for testing")
    
    # Create tree structure
    create_tree_structure(base_path, synsets, model, preprocess, device)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main() 