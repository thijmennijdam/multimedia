#!/usr/bin/env python3
"""
Create Projections Script

Loads embeddings from hierarchical dataset folders (created by preprocess.py) and applies
projection methods (HoroPCA, CO-SNE, UMAP) to create reduced embeddings.

Saves projected embeddings as {method_name}_embeddings.pkl in the same folder.

Usage:
    python create_projections.py --dataset-path hierchical_datasets/ImageNet --methods horopca umap
    python create_projections.py --dataset-path hierchical_datasets/GRIT --methods horopca cosne umap
"""

import torch
import numpy as np
import argparse
import os
import sys
import pickle
from pathlib import Path

# Add paths for projection methods
sys.path.append(os.path.join(os.path.dirname(__file__), 'HoroPCA'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'CO-SNE'))

# HoroPCA imports
from learning.frechet import Frechet
from learning.pca import HoroPCA
import geom.hyperboloid as hyperboloid
import geom.poincare as poincare

# CO-SNE imports
import hyptorch.pmath as pmath
from htsne_impl import TSNE as hTSNE

# Remove paths to avoid conflicts
sys.path.remove(os.path.join(os.path.dirname(__file__), 'HoroPCA'))
sys.path.remove(os.path.join(os.path.dirname(__file__), 'CO-SNE'))

# UMAP import
import umap
import numba

# Import plotting utilities
from plotting_utils import plot_poincare_disk


@numba.njit(fastmath=True)
def hyperboloid_distance_grad(x, y):
    """Custom hyperboloid distance function for UMAP."""
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
        grad_factor = 1.0 / np.sqrt(lorentz_product * lorentz_product - 1.0)
        
        # Gradient of Lorentzian inner product
        g[0] = grad_factor * y[0]  # First component (negative in Lorentzian)
        for i in range(1, x.shape[0]):
            g[i] = -grad_factor * y[i]  # Other components (positive in Lorentzian)
    
    return distance, g


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Data args
    parser.add_argument("--dataset-path", required=True, help="Path to dataset folder (e.g., hierchical_datasets/ImageNet)")
    parser.add_argument("--n-project", type=int, default=0, help="Number of samples to project (0 = all)")
    
    # Plotting args
    parser.add_argument("--plot", action="store_true", help="Generate and save plots")
    
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
    parser.add_argument("--cosne-perplexity", type=float, default=30, help="CO-SNE perplexity")
    parser.add_argument("--cosne-exaggeration", type=float, default=12.0, help="CO-SNE early exaggeration")
    parser.add_argument("--cosne-gamma", type=float, default=0.1, help="CO-SNE student-t gamma")
    
    # UMAP args
    parser.add_argument("--umap-components", type=int, default=3, help="UMAP components")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    
    return parser.parse_args()


class EmbeddingLoader:
    """Loads embeddings from hierarchical dataset folders."""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        
    def load_embeddings(self):
        """Load embeddings from embeddings.pkl file."""
        embeddings_path = self.dataset_path / "embeddings.pkl"
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        print(f"Loading embeddings from: {embeddings_path}")
        
        with open(embeddings_path, 'rb') as f:
            embed_data = pickle.load(f)
        
        # Extract embeddings and create labels
        embeddings_dict = embed_data['embeddings']
        embeddings = []
        labels = []
        
        for embed_id, embedding in embeddings_dict.items():
            embeddings.append(embedding)
            
            # Create label based on ID prefix
            if embed_id.startswith('pt_'):
                labels.append('parent_text')
            elif embed_id.startswith('ct_'):
                labels.append('child_text')
            elif embed_id.startswith('pi_'):
                labels.append('parent_image')
            elif embed_id.startswith('ci_'):
                labels.append('child_image')
            else:
                labels.append('unknown')
        
        embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
        
        print(f"✓ Loaded {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        return {'embeddings': embeddings, 'labels': labels}


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
        
        # Convert from hyperboloid to Poincare coordinates if 3D
        if n_components == 3:
            # Convert 3D hyperboloid to 2D Poincare
            reduced_torch = torch.tensor(reduced, dtype=torch.float64)
            reduced_poincare = hyperboloid.to_poincare(reduced_torch)
            reduced = reduced_poincare.numpy().astype(np.float32)
            print(f"✓ UMAP complete: {embeddings.shape} → hyperboloid {reduced_torch.shape} → Poincaré {reduced.shape}")
        else:
            print(f"✓ UMAP complete: {embeddings.shape} → {reduced.shape}")
        
        return torch.tensor(reduced, dtype=torch.float32)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("CREATE PROJECTIONS")
    print("="*60)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Device: {device}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Samples to project: {args.n_project if args.n_project > 0 else 'all'}")
    print("="*60)
    
    # Load embeddings
    print("\n📁 Loading embeddings...")
    loader = EmbeddingLoader(args.dataset_path)
    embed_data = loader.load_embeddings()
    
    # Sample subset for projection if specified
    embeddings = embed_data['embeddings']
    labels = embed_data['labels']
    
    if args.n_project > 0 and len(embeddings) > args.n_project:
        indices = torch.randperm(len(embeddings))[:args.n_project]
        embeddings = embeddings[indices]
        labels = [labels[i] for i in indices]
        print(f"\n🎯 Using {args.n_project:,} samples for projection")
    

    
    # Initialize projection methods
    projector = ProjectionMethods(device)
    
    # Apply projection methods
    print(f"\n🔬 Applying projection methods...")
    
    if "horopca" in args.methods:
        horopca_result = projector.apply_horopca(
            embeddings, args.horopca_components, args.horopca_lr, args.horopca_steps
        )
        
        # Save HoroPCA result
        horopca_data = {
            'embeddings': horopca_result.numpy(),
            'labels': labels,
            'method': 'HoroPCA',
            'parameters': {
                'n_components': args.horopca_components,
                'lr': args.horopca_lr,
                'max_steps': args.horopca_steps
            }
        }
        
        horopca_path = Path(args.dataset_path) / "horopca_embeddings.pkl"
        with open(horopca_path, 'wb') as f:
            pickle.dump(horopca_data, f)
        print(f"✓ Saved HoroPCA embeddings: {horopca_path}")
        
        # Generate plot if requested
        if args.plot:
            print("  📈 Generating HoroPCA plot...")
            poincare_path = Path(args.dataset_path) / "horopca_plot.png"
            plot_poincare_disk(horopca_result, labels, save_path=str(poincare_path))
    
    if "umap" in args.methods:
        umap_result = projector.apply_umap(
            embeddings, args.umap_components, args.umap_neighbors, args.umap_min_dist
        )
        
        # Save UMAP result
        umap_data = {
            'embeddings': umap_result.numpy(),
            'labels': labels,
            'method': 'UMAP',
            'parameters': {
                'n_components': args.umap_components,
                'n_neighbors': args.umap_neighbors,
                'min_dist': args.umap_min_dist
            }
        }
        
        umap_path = Path(args.dataset_path) / "umap_embeddings.pkl"
        with open(umap_path, 'wb') as f:
            pickle.dump(umap_data, f)
        print(f"✓ Saved UMAP embeddings: {umap_path}")
        
        # Generate plot if requested
        if args.plot and umap_result.shape[1] == 2:
            print("  📈 Generating UMAP plot...")
            umap_path = Path(args.dataset_path) / "umap_plot.png"
            plot_poincare_disk(umap_result, labels, save_path=str(umap_path))
    
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
        
        # Save CO-SNE result
        cosne_data = {
            'embeddings': cosne_result.numpy(),
            'labels': labels,
            'method': 'CO-SNE',
            'parameters': {
                'lr': args.cosne_lr,
                'lr_h': args.cosne_lr_h,
                'perplexity': args.cosne_perplexity,
                'exaggeration': args.cosne_exaggeration,
                'gamma': args.cosne_gamma,
                'reduce_method': args.cosne_reduce_method,
                'reduce_dim': args.cosne_reduce_dim if args.cosne_reduce_method != "none" else None
            }
        }
        
        cosne_path = Path(args.dataset_path) / "cosne_embeddings.pkl"
        with open(cosne_path, 'wb') as f:
            pickle.dump(cosne_data, f)
        print(f"✓ Saved CO-SNE embeddings: {cosne_path}")
        
        # Generate plot if requested
        if args.plot:
            print("  📈 Generating CO-SNE plot...")
            cosne_plot_path = Path(args.dataset_path) / "cosne_plot.png"
            plot_poincare_disk(cosne_result, labels, save_path=str(cosne_plot_path))
    
    print(f"\n✅ Projections complete! Results saved to: {args.dataset_path}")
    print("="*60)


if __name__ == "__main__":
    main()
