import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from matplotlib.patches import Circle
import matplotlib.patches as mpatches


def plot_norm_histograms(norms_dict, save_path="norm_histograms.png"):
    import seaborn as sns
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    colors = {
        'child_image': 'tab:blue',
        'parent_image': 'tab:orange',
        'child_text': 'tab:green',
        'parent_text': 'tab:red'
    }
    
    for key, data in norms_dict.items():
        sns.kdeplot(data, ax=axes[0], label=key.replace('_', ' ').title(), color=colors[key], linewidth=2)
        axes[0].axvline(np.mean(data), linestyle='--', color=colors[key], alpha=0.6)

    axes[0].set_ylabel('% of Samples')
    axes[0].legend()
    axes[0].set_title('Norm Distribution (Full Range)', fontweight='bold')

    # Zoom-in subplot
    zoom_min, zoom_max = 0.59, 0.66
    for key, data in norms_dict.items():
        sns.kdeplot(data, ax=axes[1], label=key.replace('_', ' ').title(), color=colors[key], linewidth=2)
    axes[1].set_xlim(zoom_min, zoom_max)
    axes[1].set_title('Zoom-in on High Norms', fontweight='bold')
    axes[1].set_xlabel(r"$\|\phi\|$")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_poincare_disk(embeddings_2d, labels, colors=None, save_path="poincare_disk.png"):
    """
    Plot 2D embeddings on a Poincare disk.
    
    Args:
        embeddings_2d: 2D embeddings tensor/array of shape (n_samples, 2)
        labels: List of labels for each embedding
        colors: Optional list of colors for each label type
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Convert to numpy if tensor
    if torch.is_tensor(embeddings_2d):
        embeddings_2d = embeddings_2d.detach().cpu().numpy()
    
    # Draw the unit circle (Poincare disk boundary)
    circle = Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    
    # Define colors for different categories
    if colors is None:
        colors = {
            'child_image': 'tab:blue',
            'parent_image': 'tab:orange', 
            'child_text': 'tab:green',
            'parent_text': 'tab:red'
        }
    
    # Get unique labels
    unique_labels = list(set(labels))
    
    # Plot points for each category
    for label in unique_labels:
        mask = np.array(labels) == label
        points = embeddings_2d[mask]
        
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], 
                      c=colors.get(label, 'gray'), 
                      label=label.replace('_', ' ').title(), 
                      alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('2D Embeddings on Poincaré Disk (HoroPCA)', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Poincaré disk plot saved to: {save_path}")


def plot_euclidean_2d(embeddings_2d, labels, colors=None, save_path="euclidean_2d.png"):
    """
    Plot 2D embeddings as standard Euclidean scatter plot.
    
    Args:
        embeddings_2d: 2D embeddings tensor/array of shape (n_samples, 2)
        labels: List of labels for each embedding
        colors: Optional list of colors for each label type
        save_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Convert to numpy if tensor
    if torch.is_tensor(embeddings_2d):
        embeddings_2d = embeddings_2d.detach().cpu().numpy()
    
    # Define colors for different categories
    if colors is None:
        colors = {
            'child_image': 'tab:blue',
            'parent_image': 'tab:orange', 
            'child_text': 'tab:green',
            'parent_text': 'tab:red'
        }
    
    # Get unique labels
    unique_labels = list(set(labels))
    
    # Plot points for each category
    for label in unique_labels:
        mask = np.array(labels) == label
        points = embeddings_2d[mask]
        
        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], 
                      c=colors.get(label, 'gray'), 
                      label=label.replace('_', ' ').title(), 
                      alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('2D Embeddings (Euclidean View - HoroPCA)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Euclidean 2D plot saved to: {save_path}")


def plot_comparison_poincare_euclidean(embeddings_2d, labels, colors=None, save_path="poincare_vs_euclidean.png"):
    """
    Plot side-by-side comparison of Poincaré disk and Euclidean views.
    
    Args:
        embeddings_2d: 2D embeddings tensor/array of shape (n_samples, 2)
        labels: List of labels for each embedding
        colors: Optional list of colors for each label type
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Convert to numpy if tensor
    if torch.is_tensor(embeddings_2d):
        embeddings_2d = embeddings_2d.detach().cpu().numpy()
    
    # Define colors for different categories
    if colors is None:
        colors = {
            'child_image': 'tab:blue',
            'parent_image': 'tab:orange', 
            'child_text': 'tab:green',
            'parent_text': 'tab:red'
        }
    
    # Get unique labels
    unique_labels = list(set(labels))
    
    # Plot 1: Poincaré disk
    circle = Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax1.add_patch(circle)
    
    for label in unique_labels:
        mask = np.array(labels) == label
        points = embeddings_2d[mask]
        
        if len(points) > 0:
            ax1.scatter(points[:, 0], points[:, 1], 
                       c=colors.get(label, 'gray'), 
                       label=label.replace('_', ' ').title(), 
                       alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Poincaré Disk View\n(Respects Hyperbolic Geometry)', fontweight='bold')
    
    # Plot 2: Euclidean view
    for label in unique_labels:
        mask = np.array(labels) == label
        points = embeddings_2d[mask]
        
        if len(points) > 0:
            ax2.scatter(points[:, 0], points[:, 1], 
                       c=colors.get(label, 'gray'), 
                       label=label.replace('_', ' ').title(), 
                       alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Euclidean View\n(Standard Scatter Plot)', fontweight='bold')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    
    # Add legend to the right
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    fig.suptitle('HoroPCA 2D Embeddings: Comparison of Visualization Methods', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def plot_cosne_results(tsne_embeddings, ht_sne_embeddings, co_sne_embeddings, labels, save_path="cosne_comparison.png"):
    """
    Plot comparison of t-SNE, HT-SNE, and CO-SNE results.
    
    Args:
        tsne_embeddings: t-SNE 2D embeddings
        ht_sne_embeddings: HT-SNE 2D embeddings  
        co_sne_embeddings: CO-SNE 2D embeddings
        labels: Labels for each embedding
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Comparison of Dimensionality Reduction Methods', fontsize=16, fontweight='bold')
    
    # Convert to numpy if tensors
    embeddings_list = [tsne_embeddings, ht_sne_embeddings, co_sne_embeddings]
    titles = ['t-SNE', 'HT-SNE', 'CO-SNE']
    
    colors = {
        'child_image': 'tab:blue',
        'parent_image': 'tab:orange', 
        'child_text': 'tab:green',
        'parent_text': 'tab:red'
    }
    
    for i, (embeddings, title) in enumerate(zip(embeddings_list, titles)):
        if torch.is_tensor(embeddings):
            embeddings = embeddings.detach().cpu().numpy()
        
        # Get unique labels
        unique_labels = list(set(labels))
        
        # Plot points for each category
        for label in unique_labels:
            mask = np.array(labels) == label
            points = embeddings[mask]
            
            if len(points) > 0:
                axes[i].scatter(points[:, 0], points[:, 1], 
                              c=colors.get(label, 'gray'), 
                              label=label.replace('_', ' ').title(), 
                              alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
        
        axes[i].set_title(title, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal')
        
        if i == 0:  # Only add legend to first subplot
            axes[i].legend(bbox_to_anchor=(-0.1, 1), loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CO-SNE comparison plot saved to: {save_path}")


def plot_sample_overview(sample_counts, save_path="sample_overview.png"):
    """
    Plot overview of sample counts by category.
    
    Args:
        sample_counts: Dictionary with sample counts for each category
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of sample counts
    categories = list(sample_counts.keys())
    counts = list(sample_counts.values())
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    bars = ax1.bar(categories, counts, color=colors[:len(categories)], alpha=0.7, edgecolor='black')
    ax1.set_title('Sample Counts by Category', fontweight='bold')
    ax1.set_ylabel('Number of Samples')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart of proportions
    ax2.pie(counts, labels=[cat.replace('_', ' ').title() for cat in categories], 
            colors=colors[:len(categories)], autopct='%1.1f%%', startangle=90)
    ax2.set_title('Sample Distribution by Category', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample overview plot saved to: {save_path}")