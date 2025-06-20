#!/usr/bin/env python3
"""
Projection Methods Analysis Script

Generates 5k GRIT embeddings and compares projection methods:
- HoroPCA (hyperbolic PCA)
- CO-SNE (hyperbolic t-SNE)  
- UMAP (with hyperbolic metric)

Usage:
    python projection_methods_analysis.py --checkpoint-path model.pth --train-config config.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import pickle
import time
from PIL import Image
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), 'HoroPCA'))

# HoroPCA imports
from learning.frechet import Frechet
from learning.pca import HoroPCA
import geom.hyperboloid as hyperboloid
import geom.poincare as poincare

sys.path.append(os.path.join(os.path.dirname(__file__), 'CO-SNE'))

# CO-SNE imports
import hyptorch.pmath as pmath
from htsne_impl import TSNE as hTSNE

# Add path imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hycoclip'))

# Import required modules
from hycoclip.config import LazyConfig, LazyFactory
from hycoclip.utils.checkpointing import CheckpointManager
from hycoclip.tokenizer import Tokenizer
from hycoclip_utils.prepare_GRIT_webdataset import ImageTextWebDataset

# UMAP import
import umap
import numba


@numba.njit(fastmath=True)
def hyperboloid_distance_grad(x, y):
    """
    Custom hyperboloid distance function for UMAP.
    
    Hyperboloid distance formula:
    d(x, y) = arccosh(-⟨x, y⟩_L)
    
    where ⟨x, y⟩_L = -x₀y₀ + x₁y₁ + ... + xₙyₙ (Lorentzian inner product)
    """
    # Compute Lorentzian inner product: -x₀y₀ + x₁y₁ + ... + xₙyₙ
    lorentz_product = -x[0] * y[0]  # First component with negative sign
    for i in range(1, x.shape[0]):
        lorentz_product += x[i] * y[i]
    
    # Clamp to avoid numerical issues with arccosh
    lorentz_product = max(lorentz_product, -1.0001)  # Ensure ≤ -1 for valid arccosh
    
    # Distance: arccosh(-⟨x, y⟩_L)
    distance = np.arccosh(-lorentz_product)
    
    # Gradient computation
    g = np.zeros_like(x)
    if distance > 1e-8:  # Avoid division by zero
        # d/dx arccosh(-⟨x, y⟩_L) = -1/√(⟨x, y⟩_L² - 1) * d/dx(-⟨x, y⟩_L)
        grad_factor = 1.0 / np.sqrt(lorentz_product * lorentz_product - 1.0)
        
        # Gradient of Lorentzian inner product
        g[0] = grad_factor * y[0]  # First component (negative in Lorentzian)
        for i in range(1, x.shape[0]):
            g[i] = -grad_factor * y[i]  # Other components (positive in Lorentzian)
    
    return distance, g


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Model and data args
    parser.add_argument("--checkpoint-path", required=True, help="Path to checkpoint")
    parser.add_argument("--train-config", required=True, help="Path to train config")
    parser.add_argument("--output-dir", default="projection_methods", help="Output directory")
    parser.add_argument("--embedding-file", default="grit_5k_embeddings.pkl", help="Embedding file name")
    
    # Data generation args
    parser.add_argument("--generate-embeddings", action="store_true", help="Generate new embeddings")
    parser.add_argument("--n-embed", type=int, default=5000, help="Number of samples to embed")
    parser.add_argument("--n-project", type=int, default=1000, help="Number of samples to project")
    
    # Method selection
    parser.add_argument("--methods", nargs="+", default=["horopca", "cosne", "umap"], 
                       choices=["horopca", "cosne", "umap"], help="Methods to run")
    
    # HoroPCA args
    parser.add_argument("--horopca-components", type=int, default=2, help="HoroPCA components")
    parser.add_argument("--horopca-lr", type=float, default=5e-2, help="HoroPCA learning rate")
    parser.add_argument("--horopca-steps", type=int, default=500, help="HoroPCA max steps")
    
    # CO-SNE args
    parser.add_argument("--cosne-reduce-method", choices=["none", "horopca", "umap"], default="none",
                       help="Reduce embeddings before CO-SNE")
    parser.add_argument("--cosne-reduce-dim", type=int, default=50, help="Reduction dimension for CO-SNE")
    parser.add_argument("--cosne-lr", type=float, default=0.5, help="CO-SNE learning rate")
    parser.add_argument("--cosne-lr-h", type=float, default=0.01, help="CO-SNE hyperbolic learning rate")
    parser.add_argument("--cosne-perplexity", type=float, default=250, help="CO-SNE perplexity")
    parser.add_argument("--cosne-exaggeration", type=float, default=10.0, help="CO-SNE early exaggeration")
    parser.add_argument("--cosne-gamma", type=float, default=0.1, help="CO-SNE student-t gamma")
    
    # UMAP args
    parser.add_argument("--umap-components", type=int, default=2, help="UMAP components")
    parser.add_argument("--umap-neighbors", type=int, default=100, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.2, help="UMAP min_dist")
    
    return parser.parse_args()


class DataManager:
    """Handles GRIT data loading and embedding generation."""
    
    def __init__(self, model, device, output_dir):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.tokenizer = Tokenizer() if model is not None else None
        
    def load_grit_samples(self, max_samples=5000):
        """Load GRIT samples."""
        print(f"Loading {max_samples:,} GRIT samples...")
        
        grit_path = "/scratch-shared/grit/processed"
        tar_files = glob.glob(os.path.join(grit_path, "*.tar"))
        
        if not tar_files:
            raise FileNotFoundError(f"No TAR files found in {grit_path}")
        
        tar_files.sort()
        
        data = {'child_images': [], 'child_texts': [], 'parent_images': [], 'parent_texts': []}
        sample_count = 0
        
        for tar_file in tar_files:
            if sample_count >= max_samples:
                break
                
            dataset = ImageTextWebDataset(tarfiles=[tar_file], infinite_stream=False)
            
            for sample in dataset:
                if sample_count >= max_samples:
                    break
                    
                try:
                    if 'child.jpg' in sample and 'child.txt' in sample:
                        child_image = sample['child.jpg']
                        child_caption = sample['child.txt']
                        
                        if isinstance(child_image, Image.Image):
                            child_image = child_image.convert('RGB').resize((224, 224))
                            img_array = np.array(child_image).astype(np.float32) / 255.0
                            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                            
                            data['child_images'].append(img_tensor)
                            data['child_texts'].append(str(child_caption))
                            
                            # Process parent data
                            num_parents = int(sample.get('numparents.txt', 0))
                            for i in range(min(num_parents, 1)):  # Take first parent only
                                parent_key = f"parent{i:03d}"
                                if f"{parent_key}.jpg" in sample and f"{parent_key}.txt" in sample:
                                    parent_image = sample[f"{parent_key}.jpg"]
                                    parent_caption = sample[f"{parent_key}.txt"]
                                    
                                    if isinstance(parent_image, Image.Image):
                                        parent_image = parent_image.convert('RGB').resize((224, 224))
                                        parent_array = np.array(parent_image).astype(np.float32) / 255.0
                                        parent_tensor = torch.from_numpy(parent_array).permute(2, 0, 1)
                                        
                                        data['parent_images'].append(parent_tensor)
                                        data['parent_texts'].append(str(parent_caption))
                            
                            sample_count += 1
                            if sample_count % 1000 == 0:
                                print(f"  Loaded {sample_count:,} samples...")
                                
                except Exception as e:
                    continue
        
        # Convert to tensors
        if data['child_images']:
            data['child_images'] = torch.stack(data['child_images'])
            data['parent_images'] = torch.stack(data['parent_images'])
        
        print(f"✓ Loaded {len(data['child_images'])} image-text pairs")
        return data
    
    def compute_embeddings(self, data, batch_size=32):
        """Compute embeddings for loaded data."""
        print("Computing embeddings...")
        
        def embed_batch(images, texts):
            with torch.no_grad():
                if images is not None and len(images) > 0:
                    image_feats = self.model.encode_image(images.to(self.device), project=True)
                else:
                    image_feats = None
                
                if texts is not None and len(texts) > 0:
                    text_tokens = self.tokenizer(texts)
                    text_feats = self.model.encode_text(text_tokens, project=True)
                else:
                    text_feats = None
            
            return image_feats, text_feats
        
        all_embeddings = []
        all_labels = []
        
        # Process in batches
        for i in range(0, len(data['child_images']), batch_size):
            end_idx = min(i + batch_size, len(data['child_images']))
            
            child_imgs = data['child_images'][i:end_idx]
            child_txts = data['child_texts'][i:end_idx]
            parent_imgs = data['parent_images'][i:end_idx] if i < len(data['parent_images']) else None
            parent_txts = data['parent_texts'][i:end_idx] if i < len(data['parent_texts']) else None
            
            # Embed child data
            child_img_feats, child_txt_feats = embed_batch(child_imgs, child_txts)
            
            if child_img_feats is not None:
                all_embeddings.append(child_img_feats.cpu())
                all_labels.extend(['child_image'] * len(child_img_feats))
            
            if child_txt_feats is not None:
                all_embeddings.append(child_txt_feats.cpu())
                all_labels.extend(['child_text'] * len(child_txt_feats))
            
            # Embed parent data
            if parent_imgs is not None and parent_txts is not None:
                parent_img_feats, parent_txt_feats = embed_batch(parent_imgs, parent_txts)
                
                if parent_img_feats is not None:
                    all_embeddings.append(parent_img_feats.cpu())
                    all_labels.extend(['parent_image'] * len(parent_img_feats))
                
                if parent_txt_feats is not None:
                    all_embeddings.append(parent_txt_feats.cpu())
                    all_labels.extend(['parent_text'] * len(parent_txt_feats))
        
        embeddings = torch.cat(all_embeddings, dim=0)
        print(f"✓ Generated embeddings: {embeddings.shape}")
        
        return {'embeddings': embeddings, 'labels': all_labels}
    
    def save_embeddings(self, embed_data, filename):
        """Save embeddings to file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(embed_data, f)
        print(f"✓ Saved embeddings to {filepath}")
    
    def load_embeddings(self, filename):
        """Load embeddings from file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'rb') as f:
            embed_data = pickle.load(f)
        print(f"✓ Loaded embeddings from {filepath}")
        return embed_data


class ProjectionMethods:
    """Handles different projection methods."""
    
    def __init__(self, device):
        self.device = device
    
    def apply_horopca(self, embeddings, n_components=2, lr=5e-2, max_steps=500):
        """Apply HoroPCA reduction."""
        print(f"Applying HoroPCA (dim: {n_components})...")
        
        torch.set_default_dtype(torch.float64)
        embeddings = embeddings.double().to(self.device)
        embeddings = hyperboloid.to_poincare(embeddings)
        
        # Compute Frechet mean
        frechet = Frechet(lr=1e-2, eps=1e-5, max_steps=5000)
        mu_ref, _ = frechet.mean(embeddings, return_converged=True)
        x = embeddings
        
        # Apply HoroPCA
        horopca = HoroPCA(dim=embeddings.shape[1], n_components=n_components, lr=lr, max_steps=max_steps)
        horopca.to(self.device)
        horopca.fit(x, iterative=False, optim=True)
        
        reduced = horopca.map_to_ball(x).detach().cpu().float()
        print(f"✓ HoroPCA complete: {embeddings.shape} → {reduced.shape}")
        
        return reduced
    
    def apply_cosne(self, embeddings, lr=0.5, lr_h=0.01, perplexity=30, 
                   exaggeration=12.0, gamma=0.1):
        """Apply CO-SNE reduction."""
        print("Applying CO-SNE...")
        
        co_sne = hTSNE(
            n_components=2, verbose=0, method='exact', square_distances=True,
            metric='precomputed', learning_rate_for_h_loss=lr_h,
            student_t_gamma=gamma, learning_rate=lr, n_iter=1000,
            perplexity=perplexity, early_exaggeration=exaggeration
        )
        
        dists = pmath.dist_matrix(embeddings, embeddings, c=1).numpy()
        reduced = co_sne.fit_transform(dists, embeddings)
        
        print(f"✓ CO-SNE complete: {embeddings.shape} → {reduced.shape}")
        return torch.tensor(reduced, dtype=torch.float32)
    
    def apply_umap(self, embeddings, n_components=3, n_neighbors=15, min_dist=0.1):
        """Apply UMAP with hyperbolic metric."""
        print(f"Applying UMAP (dim: {n_components})...")
        
        # Keep embeddings in hyperboloid coordinates for hyperboloid metric
        embeddings_np = embeddings.detach().cpu().numpy()
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=hyperboloid_distance_grad,
            output_metric='hyperboloid',
            random_state=42
        )
        reduced = reducer.fit_transform(embeddings_np)
        
        # Convert from hyperboloid to Poincare coordinates
        if n_components == 3:
            # Convert 3D hyperboloid to 2D Poincare
            reduced_torch = torch.tensor(reduced, dtype=torch.float64)
            reduced_poincare = hyperboloid.to_poincare(reduced_torch)
            reduced = reduced_poincare.numpy().astype(np.float32)
            print(f"✓ UMAP complete: {embeddings.shape} → hyperboloid {reduced_torch.shape} → Poincaré {reduced.shape}")
        else:
            print(f"✓ UMAP complete: {embeddings.shape} → {reduced.shape}")
        
        return torch.tensor(reduced, dtype=torch.float32)


class Visualizer:
    """Handles visualization of projection results."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.colors = {
            'child_image': '#1f77b4',
            'parent_image': '#ff7f0e', 
            'child_text': '#2ca02c',
            'parent_text': '#d62728'
        }
    
    def plot_projection(self, embeddings, labels, method_name, save_name):
        """Plot 2D projection results."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Calculate max radius for circle
        max_radius = np.max(np.linalg.norm(embeddings, axis=1))
        circle_radius = 1.2 * max_radius
        
        # Draw background circle
        circle = plt.Circle((0, 0), circle_radius, facecolor=(0.94, 0.94, 0.94), 
                           fill=True, linewidth=None, zorder=-3)
        ax.add_patch(circle)
        
        # Draw circle boundary
        circle_boundary = plt.Circle((0, 0), circle_radius, color='black', 
                                   fill=False, linewidth=1.5, alpha=0.7)
        ax.add_patch(circle_boundary)
        
        # Add origin point
        ax.plot([0], [0], 'o', markersize=3, color='black', zorder=10)
        ax.text(circle_radius*0.02, 0, "O", fontsize=10, ha='left', va='center')
        
        # Plot each category
        for label in set(labels):
            mask = np.array(labels) == label
            points = embeddings[mask]
            if len(points) > 0:
                ax.scatter(points[:, 0], points[:, 1], 
                          c=self.colors.get(label, 'gray'),
                          label=label.replace('_', ' ').title(),
                          alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
        
        # Set plot limits to show the full circle
        lim = circle_radius * 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        
        ax.set_title(f'{method_name} Projection', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        ax.axis('off')  # Remove axes for cleaner look
        
        save_path = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved {method_name} plot: {save_path}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("PROJECTION METHODS ANALYSIS")
    print("="*60)
    print(f"Device: {device}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Samples to embed: {args.n_embed:,}")
    print(f"Samples to project: {args.n_project:,}")
    print("="*60)
    
    # Initialize components
    if args.generate_embeddings or not os.path.exists(os.path.join(args.output_dir, args.embedding_file)):
        print("\n📡 Loading model...")
        train_config = LazyConfig.load(args.train_config)
        model = LazyFactory.build_model(train_config, device).eval()
        CheckpointManager(model=model).load(args.checkpoint_path)
        
        data_manager = DataManager(model, device, args.output_dir)
        
        # Generate embeddings
        print("\n📊 Generating embeddings...")
        grit_data = data_manager.load_grit_samples(args.n_embed)
        embed_data = data_manager.compute_embeddings(grit_data)
        data_manager.save_embeddings(embed_data, args.embedding_file)
    else:
        print("\n📁 Loading existing embeddings...")
        data_manager = DataManager(None, device, args.output_dir)
        embed_data = data_manager.load_embeddings(args.embedding_file)
    
    # Sample subset for projection
    embeddings = embed_data['embeddings']
    labels = embed_data['labels']
    
    if len(embeddings) > args.n_project:
        indices = torch.randperm(len(embeddings))[:args.n_project]
        embeddings = embeddings[indices]
        labels = [labels[i] for i in indices]
        print(f"\n🎯 Using {args.n_project:,} samples for projection")
    
    # Initialize projection methods and visualizer
    projector = ProjectionMethods(device)
    visualizer = Visualizer(args.output_dir)
    
    # Apply projection methods
    print(f"\n🔬 Applying projection methods...")
    
    if "horopca" in args.methods:
        horopca_result = projector.apply_horopca(
            embeddings, args.horopca_components, args.horopca_lr, args.horopca_steps
        )
        visualizer.plot_projection(horopca_result.numpy(), labels, "HoroPCA", "horopca_projection")
    
    if "umap" in args.methods:
        umap_result = projector.apply_umap(
            embeddings, args.umap_components, args.umap_neighbors, args.umap_min_dist
        )
        visualizer.plot_projection(umap_result.numpy(), labels, "UMAP", "umap_projection")
    
    if "cosne" in args.methods:
        # Optionally reduce first
        cosne_input = embeddings
        
        if args.cosne_reduce_method == "horopca":
            print(f"  Pre-reducing with HoroPCA to {args.cosne_reduce_dim}D...")
            cosne_input = projector.apply_horopca(
                embeddings, args.cosne_reduce_dim, args.horopca_lr, args.horopca_steps
            )
        elif args.cosne_reduce_method == "umap":
            print(f"  Pre-reducing with UMAP to {args.cosne_reduce_dim}D...")
            cosne_input = projector.apply_umap(
                embeddings, args.cosne_reduce_dim, args.umap_neighbors, args.umap_min_dist
            )
        
        cosne_result = projector.apply_cosne(
            cosne_input, args.cosne_lr, args.cosne_lr_h, args.cosne_perplexity,
            args.cosne_exaggeration, args.cosne_gamma
        )
        
        method_name = f"CO-SNE"
        if args.cosne_reduce_method != "none":
            method_name += f" ({args.cosne_reduce_method.upper()} → CO-SNE)"
        
        visualizer.plot_projection(cosne_result.numpy(), labels, method_name, "cosne_projection")
    
    print(f"\n✅ Analysis complete! Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main() 