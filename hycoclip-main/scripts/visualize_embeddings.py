#!/usr/bin/env python3
#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

"""
Script to visualize Flickr30K embeddings using hyperbolic PCA and t-SNE.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import logging

from hycoclip.config import LazyConfig, LazyFactory
from hycoclip.utils.checkpointing import CheckpointManager
from hycoclip import lorentz as L

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# HoroPCA implementation (from the provided class)
class PCA:
    """Base PCA class"""
    def __init__(self, dim, n_components, lr=1e-3, max_steps=100, keep_orthogonal=True):
        self.dim = dim
        self.n_components = n_components
        self.lr = lr
        self.max_steps = max_steps
        self.keep_orthogonal = keep_orthogonal
        self.components = [nn.Parameter(torch.randn(1, dim)) for _ in range(n_components)]

class HoroPCA(PCA):
    """Hyperbolic PCA using horocycle projections (assumes data has Frechet mean zero)."""

    def __init__(self, dim, n_components, lr=1e-3, max_steps=100, frechet_variance=False, auc=False, hyperboloid=True):
        """
        Currently auc=True and frechet_variance=True are not simultaneously supported (need to track mean parameter for each component).
        """
        super(HoroPCA, self).__init__(dim, n_components, lr, max_steps, keep_orthogonal=True)
        self.hyperboloid = hyperboloid
        self.frechet_variance = frechet_variance
        self.auc = auc
        if self.frechet_variance:
            self.mean_weights = nn.Parameter(torch.zeros(n_components))

    def _project(self, x, Q):
        if self.n_components == 1:
            proj = project_kd(Q, x)[0]
        else:
            if self.hyperboloid:
                hyperboloid_ideals = hyperboloid.from_poincare(Q, ideal=True)
                hyperboloid_x = hyperboloid.from_poincare(x)
                hyperboloid_proj = hyperboloid.horo_projection(hyperboloid_ideals, hyperboloid_x)[0]
                proj = hyperboloid.to_poincare(hyperboloid_proj)
            else:
                proj = project_kd(Q, x)[0]
        return proj

    def compute_variance(self, x):
        """ x are projected points. """
        if self.frechet_variance:
            Q = [self.mean_weights[i] * self.components[i] for i in range(self.n_components)]
            mean = sum(Q).squeeze(0)
            distances = poincare.distance(mean, x)
            var = torch.mean(distances ** 2)
        else:
            distances = poincare.pairwise_distance(x)
            var = torch.mean(distances ** 2)
        return var

    def compute_loss(self, x, Q):
        if self.n_components == 1:
            bus = busemann(x, Q[0])
            return -torch.var(bus)
        else:
            auc = []
            if self.auc:
                for i in range(1, self.n_components):
                    Q_ = Q[:i, :]
                    proj = self._project(x, Q_)
                    var = self.compute_variance(proj)
                    auc.append(var)
                return -sum(auc)
            else:
                proj = self._project(x, Q)
                var = self.compute_variance(proj)
            return -var

    def fit_transform(self, x, n_components=None):
        """Fit HoroPCA and transform the data"""
        if n_components is None:
            n_components = self.n_components
            
        # Initialize components randomly on the Poincare ball
        for i in range(n_components):
            self.components[i].data = torch.randn(1, self.dim) * 0.1
            # Ensure they're in the Poincare ball
            norm = torch.norm(self.components[i].data)
            if norm >= 1:
                self.components[i].data = self.components[i].data / (norm * 1.1)
        
        # Optimization loop
        optimizer = torch.optim.Adam(self.components, lr=self.lr)
        
        for step in range(self.max_steps):
            optimizer.zero_grad()
            Q = torch.stack([comp.squeeze(0) for comp in self.components], dim=0)
            loss = self.compute_loss(x, Q)
            loss.backward()
            optimizer.step()
            
            # Project components back to Poincare ball
            for comp in self.components:
                norm = torch.norm(comp.data)
                if norm >= 1:
                    comp.data = comp.data / (norm * 1.1)
            
            if step % 20 == 0:
                logger.info(f"HoroPCA step {step}, loss: {loss.item():.4f}")
        
        # Transform the data
        Q = torch.stack([comp.squeeze(0) for comp in self.components], dim=0)
        return self._project(x, Q)

# Helper functions for HoroPCA (simplified versions)
def project_kd(Q, x):
    """Simple projection for 1D case"""
    # For simplicity, use tangent space projection
    tangent_x = L.log_map0(x, 1.0)
    tangent_Q = L.log_map0(Q, 1.0)
    
    # Project onto Q direction
    proj_tangent = torch.sum(tangent_x * tangent_Q, dim=-1, keepdim=True) * tangent_Q
    proj = L.exp_map0(proj_tangent, 1.0)
    return proj, None

def busemann(x, q):
    """Busemann function (simplified)"""
    return L.pairwise_dist(x, q.unsqueeze(0), 1.0).squeeze()

# Simplified poincare module functions
class poincare:
    @staticmethod
    def distance(x, y):
        return L.pairwise_dist(x.unsqueeze(0), y, 1.0).squeeze()
    
    @staticmethod
    def pairwise_distance(x):
        return L.pairwise_dist(x, x, 1.0)

# Simplified hyperboloid module (placeholder)
class hyperboloid:
    @staticmethod
    def from_poincare(x, ideal=False):
        return x  # Placeholder
    
    @staticmethod
    def to_poincare(x):
        return x  # Placeholder
    
    @staticmethod
    def horo_projection(ideals, x):
        return x, None  # Placeholder

def apply_horopca(embeddings: torch.Tensor, n_components: int, curv: float) -> torch.Tensor:
    """
    Apply HoroPCA to reduce dimensionality of hyperbolic embeddings.
    
    Args:
        embeddings: Tensor of shape (N, D) containing hyperbolic embeddings
        n_components: Number of components to reduce to
        curv: Curvature of the hyperbolic space (unused in this simplified version)
        
    Returns:
        Reduced embeddings of shape (N, n_components)
    """
    input_dim = embeddings.shape[1]
    
    # Initialize HoroPCA
    horopca = HoroPCA(
        dim=input_dim,
        n_components=n_components,
        lr=1e-3,
        max_steps=50,  # Reduced for faster computation
        hyperboloid=False  # Use Poincare ball version for simplicity
    )
    
    # Apply HoroPCA
    logger.info(f"Applying HoroPCA to reduce from {input_dim}D to {n_components}D...")
    reduced_embeddings = horopca.fit_transform(embeddings)
    
    return reduced_embeddings

def co_sne(embeddings: torch.Tensor, n_components: int = 2, curv: float = 1.0, 
           perplexity: float = 30.0, learning_rate: float = 200.0, 
           n_iter: int = 1000, gamma: float = 0.1, lambda1: float = 10.0, 
           lambda2: float = 0.01) -> torch.Tensor:
    """
    CO-SNE: Hyperbolic t-SNE using Cauchy distribution for low-dimensional similarities.
    
    Args:
        embeddings: High-dimensional hyperbolic embeddings (N, D)
        n_components: Number of dimensions for output (typically 2)
        curv: Curvature parameter
        perplexity: Perplexity for high-dimensional similarities
        learning_rate: Learning rate for optimization
        n_iter: Number of optimization iterations
        gamma: Scale parameter for hyperbolic Cauchy distribution
        lambda1: Weight for KL divergence loss
        lambda2: Weight for distance preservation loss
        
    Returns:
        Low-dimensional hyperbolic embeddings (N, n_components)
    """
    n_samples = embeddings.shape[0]
    
    # Compute high-dimensional similarities using hyperbolic normal distribution
    # First compute pairwise distances
    distances = L.pairwise_dist(embeddings, embeddings, curv)
    
    # Convert to similarities using binary search for perplexity
    def binary_search_sigma(distances_i, target_perplexity, max_iter=50, tol=1e-5):
        """Binary search for sigma that gives target perplexity"""
        sigma_min, sigma_max = 1e-20, 1e20
        sigma = 1.0
        
        for _ in range(max_iter):
            # Compute conditional probabilities
            exp_neg_dist = torch.exp(-distances_i**2 / (2 * sigma**2))
            exp_neg_dist[torch.arange(len(distances_i)), torch.arange(len(distances_i))] = 0
            sum_exp = exp_neg_dist.sum()
            
            if sum_exp == 0:
                sigma_min = sigma
                sigma *= 2
                continue
                
            p_i = exp_neg_dist / sum_exp
            
            # Compute perplexity
            entropy = -(p_i * torch.log(p_i + 1e-12)).sum()
            perplexity_i = 2**entropy
            
            if abs(perplexity_i - target_perplexity) < tol:
                break
                
            if perplexity_i > target_perplexity:
                sigma_max = sigma
                sigma = (sigma_min + sigma) / 2
            else:
                sigma_min = sigma
                sigma = (sigma + sigma_max) / 2
                
        return sigma
    
    # Compute high-dimensional joint probabilities
    P = torch.zeros(n_samples, n_samples)
    for i in range(n_samples):
        sigma_i = binary_search_sigma(distances[i], perplexity)
        exp_neg_dist = torch.exp(-distances[i]**2 / (2 * sigma_i**2))
        exp_neg_dist[i] = 0
        P[i] = exp_neg_dist / exp_neg_dist.sum()
    
    # Symmetrize
    P = (P + P.T) / (2 * n_samples)
    P = torch.clamp(P, min=1e-12)
    
    # Initialize low-dimensional embeddings
    Y = torch.randn(n_samples, n_components) * 0.01
    
    # Optimization loop
    for iteration in range(n_iter):
        # Compute low-dimensional distances
        Y_dist = L.pairwise_dist(Y, Y, curv)
        
        # Compute low-dimensional similarities using hyperbolic Cauchy distribution
        Q_num = gamma**2 / (Y_dist**2 + gamma**2)
        Q_num.fill_diagonal_(0)
        Q = Q_num / Q_num.sum()
        Q = torch.clamp(Q, min=1e-12)
        
        # Compute KL divergence gradient
        PQ_diff = P - Q
        
        # Compute gradients for each point
        grad_Y = torch.zeros_like(Y)
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    # Gradient of hyperbolic distance
                    alpha_i = 1 - torch.norm(Y[i])**2
                    alpha_j = 1 - torch.norm(Y[j])**2
                    
                    y_diff = Y[i] - Y[j]
                    norm_diff_sq = torch.norm(y_diff)**2
                    
                    gamma_ij = 1 + 2 * norm_diff_sq / (alpha_i * alpha_j)
                    
                    if gamma_ij > 1:
                        # Gradient of distance w.r.t. Y[i]
                        factor = 4 / (alpha_j * torch.sqrt(gamma_ij**2 - 1))
                        term1 = (torch.norm(Y[j])**2 - 2 * torch.dot(Y[i], Y[j]) + 1) / alpha_i**2 * Y[i]
                        term2 = Y[j] / alpha_i
                        grad_dist = factor * (term1 - term2)
                        
                        # KL divergence gradient
                        grad_Y[i] += 2 * PQ_diff[i, j] * (1 + Y_dist[i, j]**2)**(-1) * grad_dist
        
        # Distance preservation loss gradient
        Y_norms = torch.norm(Y, dim=1)
        X_norms = torch.norm(embeddings, dim=1)
        
        for i in range(n_samples):
            grad_Y[i] += -4 * lambda2 * (X_norms[i]**2 - Y_norms[i]**2) * Y[i]
        
        # Update Y using gradient descent
        Y = Y - learning_rate * grad_Y
        
        # Project back to Poincaré ball
        Y_norms = torch.norm(Y, dim=1)
        Y[Y_norms >= 1] = Y[Y_norms >= 1] / (Y_norms[Y_norms >= 1].unsqueeze(1) * 1.01)
        
        if iteration % 100 == 0:
            kl_div = (P * torch.log(P / Q)).sum()
            logger.info(f"CO-SNE iteration {iteration}, KL divergence: {kl_div:.4f}")
    
    return Y

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load configs
    train_config = LazyConfig.load(args.train_config)
    model = LazyFactory.build_model(train_config, device).eval()
    CheckpointManager(model=model).load(args.checkpoint_path)
    
    # For this example, we'll create dummy data since we don't have access to Flickr30K
    # In practice, you would load your actual dataset here
    logger.info("Creating dummy dataset for demonstration...")
    
    # Create dummy image and text data
    batch_size = min(args.batch_size, 100)  # Limit for demo
    n_samples = 500  # Total samples for visualization
    
    dummy_images = torch.randn(n_samples, 3, 224, 224).to(device)
    dummy_tokens = [torch.randint(0, 1000, (20,)) for _ in range(n_samples)]
    
    # Extract embeddings
    all_image_embeddings = []
    all_text_embeddings = []
    
    with torch.no_grad():
        # Process in batches
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_images = dummy_images[i:end_idx]
            batch_tokens = dummy_tokens[i:end_idx]
            
            # Get embeddings
            image_feats = model.encode_image(batch_images, project=True)
            text_feats = model.encode_text(batch_tokens, project=True)
            
            all_image_embeddings.append(image_feats.cpu())
            all_text_embeddings.append(text_feats.cpu())
    
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    # Perform hyperbolic PCA
    logger.info("Performing hyperbolic PCA...")
    reduced_image_embeddings = apply_horopca(
        all_image_embeddings, 
        n_components=50,
        curv=model.curv.exp().item()
    )
    reduced_text_embeddings = apply_horopca(
        all_text_embeddings,
        n_components=50,
        curv=model.curv.exp().item()
    )
    
    # Combine embeddings for CO-SNE
    combined_embeddings = torch.cat([reduced_image_embeddings, reduced_text_embeddings], dim=0)
    
    # Perform CO-SNE
    logger.info("Performing CO-SNE...")
    embeddings_2d = co_sne(
        combined_embeddings, 
        n_components=2, 
        curv=model.curv.exp().item(),
        n_iter=1000
    )
    
    # Split back into image and text embeddings
    n_samples = len(all_image_embeddings)
    image_embeddings_2d = embeddings_2d[:n_samples].numpy()
    text_embeddings_2d = embeddings_2d[n_samples:].numpy()
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(image_embeddings_2d[:, 0], image_embeddings_2d[:, 1], 
                c='blue', label='Images', alpha=0.6)
    plt.scatter(text_embeddings_2d[:, 0], text_embeddings_2d[:, 1], 
                c='red', label='Text', alpha=0.6)
    plt.legend()
    plt.title('CO-SNE visualization of HyCoCLIP embeddings (after HoroPCA)')
    
    # Save plot
    output_path = Path(args.output_dir) / "embeddings_visualization.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    logger.info(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", required=True, help="Path to training config")
    parser.add_argument("--checkpoint-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-dir", required=True, help="Path to Flickr30K dataset")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save visualizations")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers")
    
    args = parser.parse_args()
    main(args) 