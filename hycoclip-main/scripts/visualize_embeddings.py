import torch
from hycoclip.config import LazyConfig, LazyFactory
from hycoclip.utils.checkpointing import CheckpointManager
from hycoclip.tokenizer import Tokenizer
import argparse
import sys
import os
import glob
import numpy as np
from PIL import Image
import time


# Add path imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hycoclip', 'CO-SNE'))
from main import run_TSNE, plot_low_dims

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hycoclip', 'HoroPCA'))

from learning.frechet import Frechet
from learning.pca import HoroPCA
import geom.hyperboloid as hyperboloid
import geom.poincare as poincare

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from prepare_GRIT_webdataset import ImageTextWebDataset

# Import plotting utilities
from plotting_utils import (plot_norm_histograms, plot_poincare_disk, plot_euclidean_2d, 
                           plot_comparison_poincare_euclidean, plot_cosne_results, plot_sample_overview)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    _AA = parser.add_argument
    _AA("--checkpoint-path", help="Path to checkpoint of a trained HyCoCLIP/MERU/CLIP model.")
    _AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")
    _AA("--task", choices=['norms', 'horopca', 'cosne', 'all'], default='all', 
        help="Which analysis task to run")
    _AA("--output-dir", default="analysis_results", help="Output directory for plots and results")
    _AA("--norm-samples-k", type=int, default=10, 
        help="Number of samples (in thousands) to use for norm computation task (default: 10k)")
    return parser.parse_args()


def load_model(checkpoint_path, train_config_path, device):
    """Load and initialize the model from checkpoint."""
    train_config = LazyConfig.load(train_config_path)
    model = LazyFactory.build_model(train_config, device).eval()
    CheckpointManager(model=model).load(checkpoint_path)
    return model


def load_grit_samples_comprehensive(max_samples=9000, batch_size=32, category_limits=None):
    """
    Load comprehensive GRIT dataset samples including both child and parent images/text.
    
    Args:
        max_samples: Maximum number of samples to load
        batch_size: Batch size for processing
        category_limits: Dict with limits per category (e.g., {'child_image': 200, 'parent_image': 200})
        
    Returns:
        Dictionary containing all loaded data
    """
    # Find GRIT TAR files
    grit_path = "/scratch-shared/grit/processed"
    tar_files = glob.glob(os.path.join(grit_path, "*.tar"))
    
    if not tar_files:
        print(f"No TAR files found in {grit_path}")
        return None
    
    # Sort TAR files for consistent processing order
    tar_files.sort()
    print(f"Found {len(tar_files)} TAR files in {grit_path}")
    
    # Storage for all data
    data = {
        'child_images': [],
        'child_texts': [],
        'parent_images': [],
        'parent_texts': [],
        'sample_keys': []
    }
    
    # Track counts per category if limits are specified
    if category_limits is None:
        category_limits = {}
    
    category_counts = {
        'child_image': 0,
        'parent_image': 0,
        'child_text': 0,
        'parent_text': 0
    }
    
    sample_count = 0
    
    # Process multiple TAR files if needed
    for tar_file in tar_files:
        if sample_count >= max_samples:
            break
            
        print(f"Loading from: {tar_file}")
        dataset = ImageTextWebDataset(tarfiles=[tar_file], infinite_stream=False)
        
        for sample in dataset:
            if sample_count >= max_samples:
                break
                
            try:
                sample_key = sample['__key__']
                
                # Process child image and text
                if 'child.jpg' in sample and 'child.txt' in sample:
                    child_image = sample['child.jpg']
                    child_caption = sample['child.txt']
                    
                    # Check if we can add child image and text
                    can_add_child_image = category_counts['child_image'] < category_limits.get('child_image', float('inf'))
                    can_add_child_text = category_counts['child_text'] < category_limits.get('child_text', float('inf'))
                    
                    # Convert PIL image to tensor
                    if isinstance(child_image, Image.Image):
                        child_image = child_image.convert('RGB').resize((224, 224))
                        img_array = np.array(child_image).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                        
                        # Add child image if under limit
                        if can_add_child_image:
                            data['child_images'].append(img_tensor)
                            category_counts['child_image'] += 1
                            
                        # Add child text if under limit
                        if can_add_child_text:
                            data['child_texts'].append(str(child_caption))
                            category_counts['child_text'] += 1
                            
                        # Always add sample key if we added anything
                        if can_add_child_image or can_add_child_text:
                            data['sample_keys'].append(sample_key)
                        
                        # Process parent images and texts
                        num_parents = int(sample.get('numparents.txt', 0))
                        for i in range(num_parents):
                            parent_key = f"parent{i:03d}"
                            if f"{parent_key}.jpg" in sample and f"{parent_key}.txt" in sample:
                                parent_image = sample[f"{parent_key}.jpg"]
                                parent_caption = sample[f"{parent_key}.txt"]
                                
                                # Check if we can add parent image and text
                                can_add_parent_image = category_counts['parent_image'] < category_limits.get('parent_image', float('inf'))
                                can_add_parent_text = category_counts['parent_text'] < category_limits.get('parent_text', float('inf'))
                                
                                if isinstance(parent_image, Image.Image) and (can_add_parent_image or can_add_parent_text):
                                    parent_image = parent_image.convert('RGB').resize((224, 224))
                                    parent_array = np.array(parent_image).astype(np.float32) / 255.0
                                    parent_tensor = torch.from_numpy(parent_array).permute(2, 0, 1)
                                    
                                    # Add parent image if under limit
                                    if can_add_parent_image:
                                        data['parent_images'].append(parent_tensor)
                                        category_counts['parent_image'] += 1
                                        
                                    # Add parent text if under limit
                                    if can_add_parent_text:
                                        data['parent_texts'].append(str(parent_caption))
                                        category_counts['parent_text'] += 1
                        
                        sample_count += 1
                        
                        if sample_count % 1000 == 0:
                            print(f"Loaded {sample_count} samples...")
                            
                        # Check if all category limits are reached
                        if category_limits:
                            all_limits_reached = True
                            for cat, limit in category_limits.items():
                                if category_counts[cat] < limit:
                                    all_limits_reached = False
                                    break
                            if all_limits_reached:
                                print("All category limits reached, stopping data loading...")
                                break
                            
            except Exception as e:
                print(f"Error processing sample {sample.get('__key__', 'unknown')}: {e}")
                continue
    
    # Convert lists to tensors where appropriate
    if len(data['child_images']) > 0:
        data['child_images'] = torch.stack(data['child_images'])
    else:
        data['child_images'] = torch.empty(0)
        
    if len(data['parent_images']) > 0:
        data['parent_images'] = torch.stack(data['parent_images'])
    else:
        data['parent_images'] = torch.empty(0)
    
    print(f"\nData loading complete:")
    print(f"- Child images: {len(data['child_images'])}")
    print(f"- Child texts: {len(data['child_texts'])}")
    print(f"- Parent images: {len(data['parent_images'])}")
    print(f"- Parent texts: {len(data['parent_texts'])}")
    
    if category_limits:
        print(f"\nCategory counts vs limits:")
        for cat in ['child_image', 'parent_image', 'child_text', 'parent_text']:
            limit = category_limits.get(cat, 'no limit')
            print(f"- {cat}: {category_counts[cat]} / {limit}")
    
    return data


def compute_embeddings_batch(model, images, texts, device, batch_size=32, desc=""):
    """Compute embeddings for images and texts in batches."""
    print(f"Computing embeddings for {len(images) if hasattr(images, '__len__') else 'unknown'} {desc}...")
    
    all_image_feats = None
    all_text_feats = None
    
    # Initialize tokenizer for text processing
    tokenizer = Tokenizer()
    
    # Process images if provided
    if images is not None and len(images) > 0:
        all_image_feats = []
        num_batches = (len(images) + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_end = min(i + batch_size, len(images))
                batch_images = images[i:batch_end].to(device)
                
                image_feats = model.encode_image(batch_images, project=True)
                all_image_feats.append(image_feats.cpu())
                
                if (i // batch_size + 1) % 50 == 0:
                    print(f"  Processed {i // batch_size + 1}/{num_batches} image batches")
        
        all_image_feats = torch.cat(all_image_feats, dim=0)
    
    # Process texts if provided using proper tokenization
    if texts is not None and len(texts) > 0:
        all_text_feats = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_end = min(i + batch_size, len(texts))
                batch_texts = texts[i:batch_end]
                
                # Proper tokenization using the HyCoCLIP tokenizer
                try:
                    batch_tokens = tokenizer(batch_texts)
                    text_feats = model.encode_text(batch_tokens, project=True)
                    all_text_feats.append(text_feats.cpu())
                    
                    if (i // batch_size + 1) % 50 == 0:
                        print(f"  Processed {i // batch_size + 1}/{num_batches} text batches")
                        
                except Exception as e:
                    print(f"Error tokenizing text batch {i//batch_size + 1}: {e}")
                    print(f"Batch texts: {batch_texts[:3]}...")  # Show first 3 texts for debugging
                    raise RuntimeError(f"Failed to tokenize text batch. Error: {e}")
        
        if len(all_text_feats) > 0:
            all_text_feats = torch.cat(all_text_feats, dim=0)
        else:
            print("Warning: No text features were successfully computed!")
            all_text_feats = None
    
    return all_image_feats, all_text_feats


def task_1_compute_norms(model, device, output_dir, max_samples=10000):
    """Task 1: Load samples and compute norms with histograms."""
    print("\n" + "="*80)
    print(f"TASK 1: Computing embedding norms for {max_samples:,} samples")
    print("="*80)
    
    # Load comprehensive dataset
    data = load_grit_samples_comprehensive(max_samples=max_samples)
    if not data:
        print("Failed to load data for Task 1")
        return
    
    # Compute embeddings for all categories
    child_img_feats, child_txt_feats = compute_embeddings_batch(
        model, data['child_images'], data['child_texts'], device, desc="child samples"
    )
    
    parent_img_feats, parent_txt_feats = compute_embeddings_batch(
        model, data['parent_images'], data['parent_texts'], device, desc="parent samples"
    )
    
    # Compute L2 norms
    norms_dict = {}
    
    if child_img_feats is not None:
        norms_dict['child_image'] = torch.norm(child_img_feats, dim=1).numpy()
    if parent_img_feats is not None:
        norms_dict['parent_image'] = torch.norm(parent_img_feats, dim=1).numpy()
    if child_txt_feats is not None:
        norms_dict['child_text'] = torch.norm(child_txt_feats, dim=1).numpy()
    if parent_txt_feats is not None:
        norms_dict['parent_text'] = torch.norm(parent_txt_feats, dim=1).numpy()
    
    # Plot histograms
    plot_norm_histograms(norms_dict, save_path=os.path.join(output_dir, "task1_norm_histograms.png"))
    
    # Save statistics
    stats_file = os.path.join(output_dir, "task1_norm_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write("Embedding Norm Statistics\n")
        f.write("="*50 + "\n")
        for category, norms in norms_dict.items():
            f.write(f"\n{category.replace('_', ' ').title()}:\n")
            f.write(f"  Count: {len(norms)}\n")
            f.write(f"  Mean: {np.mean(norms):.6f}\n")
            f.write(f"  Std: {np.std(norms):.6f}\n")
            f.write(f"  Min: {np.min(norms):.6f}\n")
            f.write(f"  Max: {np.max(norms):.6f}\n")
    
    print(f"Task 1 complete! Results saved to {output_dir}")


def task_2_horopca_poincare(model, device, output_dir, max_samples=200):
    """Task 2: Load 200 samples, apply HoroPCA to 2D, plot on Poincare disk."""
    print("\n" + "="*80)
    print("TASK 2: HoroPCA reduction to 2D with Poincaré disk visualization")
    print("="*80)
    
    # Load smaller dataset with category limits
    category_limits = {
        'child_image': 200,
        'parent_image': 200,
        'child_text': 200,
        'parent_text': 200
    }
    data = load_grit_samples_comprehensive(max_samples=max_samples, category_limits=category_limits)
    if not data:
        print("Failed to load data for Task 2")
        return
    
    # Compute embeddings
    child_img_feats, child_txt_feats = compute_embeddings_batch(
        model, data['child_images'], data['child_texts'], device, desc="child samples"
    )
    
    parent_img_feats, parent_txt_feats = compute_embeddings_batch(
        model, data['parent_images'], data['parent_texts'], device, desc="parent samples"
    )
    
    # Combine embeddings and create labels
    all_embeddings = []
    labels = []
    
    if child_img_feats is not None:
        all_embeddings.append(child_img_feats)
        labels.extend(['child_image'] * len(child_img_feats))
    
    if parent_img_feats is not None:
        all_embeddings.append(parent_img_feats)
        labels.extend(['parent_image'] * len(parent_img_feats))
    
    if child_txt_feats is not None:
        all_embeddings.append(child_txt_feats)
        labels.extend(['child_text'] * len(child_txt_feats))
    
    if parent_txt_feats is not None:
        all_embeddings.append(parent_txt_feats)
        labels.extend(['parent_text'] * len(parent_txt_feats))
    
    if not all_embeddings:
        print("No embeddings to process for Task 2")
        return
    
    combined_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Combined embeddings shape: {combined_embeddings.shape}")
    
    # Apply HoroPCA reduction to 2D
    print("Applying HoroPCA reduction to 2D...")
    torch.set_default_dtype(torch.float64)
    embeddings = combined_embeddings.double()
    
    if torch.cuda.is_available():
        embeddings = embeddings.cuda()
    
    embeddings = hyperboloid.to_poincare(embeddings)
    
    # Compute Frechet mean
    frechet = Frechet(lr=1e-2, eps=1e-5, max_steps=5000)
    mu_ref, has_converged = frechet.mean(embeddings, return_converged=True)
    print(f"Frechet mean computation converged: {has_converged}")
    # x = poincare.reflect_at_zero(embeddings, mu_ref)
    x = embeddings
    # Apply HoroPCA to 2D
    original_dim = embeddings.shape[1]
    horopca = HoroPCA(dim=original_dim, n_components=2, lr=1e-2, max_steps=500)
    if torch.cuda.is_available():
        horopca.cuda()
    
    horopca.fit(x, iterative=False, optim=True)
    # x = hyperboloid.to_poincare(x)
    embeddings_2d = horopca.map_to_ball(x).detach().cpu().float()
    
    print(f"HoroPCA reduction complete! Shape: {embeddings_2d.shape}")
    
    # Create multiple visualizations
    print("Creating visualization plots...")
    
    # 1. Poincaré disk (respects hyperbolic geometry)
    plot_poincare_disk(embeddings_2d, labels, save_path=os.path.join(output_dir, "task2_poincare_disk.png"))
    
    # 2. Euclidean view (standard scatter plot)
    plot_euclidean_2d(embeddings_2d, labels, save_path=os.path.join(output_dir, "task2_euclidean_view.png"))
    
    # 3. Side-by-side comparison
    plot_comparison_poincare_euclidean(embeddings_2d, labels, 
                                      save_path=os.path.join(output_dir, "task2_comparison.png"))
    
    print(f"Task 2 complete! Results saved to {output_dir}")
    print("Created 3 visualization plots:")
    print("  - Poincaré disk view (hyperbolic geometry)")
    print("  - Euclidean view (standard scatter)")
    print("  - Side-by-side comparison")


def task_3_horopca_cosne(model, device, output_dir, max_samples=200):
    """Task 3: Load 200 samples, apply HoroPCA to 50D, then CO-SNE to 2D."""
    print("\n" + "="*80)
    print("TASK 3: HoroPCA + CO-SNE dimensionality reduction")
    print("="*80)
    
    # Load dataset with category limits
    category_limits = {
        'child_image': 200,
        'parent_image': 200,
        'child_text': 200,
        'parent_text': 200
    }
    data = load_grit_samples_comprehensive(max_samples=max_samples, category_limits=category_limits)
    if not data:
        print("Failed to load data for Task 3")
        return
    
    # Compute embeddings
    child_img_feats, child_txt_feats = compute_embeddings_batch(
        model, data['child_images'], data['child_texts'], device, desc="child samples"
    )
    
    parent_img_feats, parent_txt_feats = compute_embeddings_batch(
        model, data['parent_images'], data['parent_texts'], device, desc="parent samples"
    )
    
    # Combine embeddings and create labels
    all_embeddings = []
    labels = []
    
    if child_img_feats is not None:
        all_embeddings.append(child_img_feats)
        labels.extend(['child_image'] * len(child_img_feats))
    
    if parent_img_feats is not None:
        all_embeddings.append(parent_img_feats)
        labels.extend(['parent_image'] * len(parent_img_feats))
    
    if child_txt_feats is not None:
        all_embeddings.append(child_txt_feats)
        labels.extend(['child_text'] * len(child_txt_feats))
    
    if parent_txt_feats is not None:
        all_embeddings.append(parent_txt_feats)
        labels.extend(['parent_text'] * len(parent_txt_feats))
    
    if not all_embeddings:
        print("No embeddings to process for Task 3")
        return
    
    combined_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"Combined embeddings shape: {combined_embeddings.shape}")
    
    # Apply HoroPCA reduction to 50D
    print("Applying HoroPCA reduction to 50 dimensions...")
    torch.set_default_dtype(torch.float64)
    embeddings = combined_embeddings.double()
    
    if torch.cuda.is_available():
        embeddings = embeddings.cuda()
    
    # embeddings = hyperboloid.to_poincare(embeddings)
    
    # Compute Frechet mean
    frechet = Frechet(lr=1e-2, eps=1e-5, max_steps=5000)
    mu_ref, has_converged = frechet.mean(embeddings, return_converged=True)
    print(f"Frechet mean computation converged: {has_converged}")
    x = embeddings
    
    # Apply HoroPCA to 50D
    original_dim = embeddings.shape[1]
    horopca = HoroPCA(dim=original_dim, n_components=50, lr=1e-2, max_steps=500)
    if torch.cuda.is_available():
        horopca.cuda()
    
    horopca.fit(x, iterative=False, optim=True)
    embeddings_50d = horopca.map_to_ball(x).detach().cpu().float()
    
    print(f"HoroPCA reduction complete! Shape: {embeddings_50d.shape}")
    
    # Apply CO-SNE reduction to 2D
    print("Applying CO-SNE reduction to 2D...")
    # learning_rate = 5.0
    # learning_rate_for_h_loss = 0.1
    # perplexity = 20
    # early_exaggeration = 1
    # student_t_gamma = 0.1

    learning_rate = 0.5
    learning_rate_for_h_loss = 0.01
    perplexity = 250
    early_exaggeration = 10.0
    student_t_gamma = 0.1


    # Create colors list for CO-SNE (maps each data point to its color)
    color_map = {
        'child_image': 'tab:blue',
        'parent_image': 'tab:orange', 
        'child_text': 'tab:green',
        'parent_text': 'tab:red'
    }
    
    
    tsne_embeddings, ht_sne_embeddings, co_sne_embeddings = run_TSNE(
        embeddings_50d, learning_rate, learning_rate_for_h_loss, 
        perplexity, early_exaggeration, student_t_gamma
    )
    
    print("CO-SNE reduction complete!")
    colors = [color_map.get(label, 'gray') for label in labels]
    plot_low_dims(tsne_embeddings, ht_sne_embeddings, co_sne_embeddings, colors, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma)
    
    
    # Plot results
    plot_cosne_results(tsne_embeddings, ht_sne_embeddings, co_sne_embeddings, 
                      labels, save_path=os.path.join(output_dir, "task3_cosne_comparison.png"))
    
    print(f"Task 3 complete! Results saved to {output_dir}")


def main():
    """Main function to orchestrate all analysis tasks."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint_path, args.train_config, device)
    print("Model loaded successfully!")
    
    # Run selected tasks
    start_time = time.time()
    
    if args.task in ['norms', 'all']:
        task_1_compute_norms(model, device, args.output_dir, max_samples=args.norm_samples_k * 1000)
    
    if args.task in ['horopca', 'all']:
        task_2_horopca_poincare(model, device, args.output_dir)
    
    if args.task in ['cosne', 'all']:
        task_3_horopca_cosne(model, device, args.output_dir)
    
    total_time = time.time() - start_time
    print(f"\nAll tasks completed in {total_time:.2f} seconds!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
    
    # Example usage:
    # python scripts/visualize_embeddings.py --checkpoint-path checkpoints/hycoclip_vit_s.pth \
    #     --train-config configs/train_hycoclip_vit_s.py --task all --output-dir analysis_results