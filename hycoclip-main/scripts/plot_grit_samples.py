import os
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from prepare_GRIT_webdataset import ImageTextWebDataset


def plot_sample(sample, sample_idx, save_dir):
    """Plot a single GRIT sample with child and parent images."""
    
    # Get sample info
    sample_key = sample['__key__']
    child_image = sample['child.jpg']
    child_caption = sample['child.txt']
    num_parents = int(sample['numparents.txt'])
    
    print(f"Sample {sample_idx}: Key={sample_key}, Parents={num_parents}")
    print(f"Child caption: {child_caption}")
    
    # Calculate grid size - child image + parent images
    total_images = 1 + num_parents
    cols = min(4, total_images)  # Max 4 columns
    rows = (total_images + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 5*rows))
    if total_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Plot child image
    axes[0].imshow(child_image)
    axes[0].set_title(f"CHILD (Key: {sample_key})", fontweight='bold', fontsize=12)
    axes[0].text(0.5, -0.15, f"Caption: {child_caption}", 
                transform=axes[0].transAxes, ha='center', va='top', 
                fontsize=10, wrap=True, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[0].axis('off')
    
    # Plot parent images
    for i in range(num_parents):
        parent_key = f"parent{i:03d}"
        parent_image = sample[f"{parent_key}.jpg"]
        parent_caption = sample[f"{parent_key}.txt"]
        
        ax_idx = i + 1
        axes[ax_idx].imshow(parent_image)
        axes[ax_idx].set_title(f"PARENT {i+1}", fontweight='bold', fontsize=12)
        axes[ax_idx].text(0.5, -0.15, f"Caption: {parent_caption}", 
                         transform=axes[ax_idx].transAxes, ha='center', va='top', 
                         fontsize=10, wrap=True, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        axes[ax_idx].axis('off')
        
        print(f"  Parent {i+1}: {parent_caption}")
    
    # Hide unused subplots
    for i in range(total_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, f"grit_sample_{sample_idx:03d}_{sample_key}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {save_path}")
    print("-" * 80)


def create_summary_plot(all_samples, save_dir):
    """Create a summary plot showing all samples in one figure."""
    
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))  # 5 samples x max 5 images per sample
    fig.suptitle("GRIT Dataset Samples - Child and Parent Images", fontsize=16, fontweight='bold')
    
    for sample_idx, sample in enumerate(all_samples):
        if sample_idx >= 5:
            break
            
        sample_key = sample['__key__']
        child_image = sample['child.jpg']
        child_caption = sample['child.txt']
        num_parents = int(sample['numparents.txt'])
        
        # Plot child image
        axes[sample_idx, 0].imshow(child_image)
        axes[sample_idx, 0].set_title(f"Sample {sample_idx+1} - CHILD\n{sample_key}", fontsize=10, fontweight='bold')
        axes[sample_idx, 0].text(0.5, -0.1, child_caption[:50] + "..." if len(child_caption) > 50 else child_caption,
                                transform=axes[sample_idx, 0].transAxes, ha='center', va='top', 
                                fontsize=8, wrap=True)
        axes[sample_idx, 0].axis('off')
        
        # Plot parent images (up to 4)
        for i in range(min(4, num_parents)):
            parent_key = f"parent{i:03d}"
            parent_image = sample[f"{parent_key}.jpg"]
            parent_caption = sample[f"{parent_key}.txt"]
            
            col_idx = i + 1
            axes[sample_idx, col_idx].imshow(parent_image)
            axes[sample_idx, col_idx].set_title(f"Parent {i+1}", fontsize=10)
            axes[sample_idx, col_idx].text(0.5, -0.1, parent_caption[:30] + "..." if len(parent_caption) > 30 else parent_caption,
                                          transform=axes[sample_idx, col_idx].transAxes, ha='center', va='top', 
                                          fontsize=8, wrap=True)
            axes[sample_idx, col_idx].axis('off')
        
        # Hide unused parent slots
        for i in range(min(4, num_parents) + 1, 5):
            axes[sample_idx, i].axis('off')
    
    plt.tight_layout()
    
    # Save summary plot
    summary_path = os.path.join(save_dir, "grit_samples_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary plot: {summary_path}")


def main():
    """Main function to load and plot GRIT samples."""
    
    # Find GRIT TAR files
    grit_path = "/scratch-shared/grit/processed"
    tar_files = glob.glob(os.path.join(grit_path, "*.tar"))
    
    if not tar_files:
        print(f"No TAR files found in {grit_path}")
        return
    
    # Sort TAR files for consistent processing order
    tar_files.sort()
    
    # Use the first TAR file
    tar_file = tar_files[0]
    print(f"Loading samples from: {tar_file}")
    
    # Create output directory
    save_dir = "grit_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset
    dataset = ImageTextWebDataset(tarfiles=[tar_file], infinite_stream=False)
    
    # Process first 5 samples
    samples_to_plot = []
    sample_count = 0
    
    print("Loading samples...")
    for sample in dataset:
        if sample_count >= 5:
            break
            
        try:
            # Check if sample has the required keys
            if 'child.jpg' in sample and 'child.txt' in sample and 'numparents.txt' in sample:
                samples_to_plot.append(sample)
                sample_count += 1
                print(f"Loaded sample {sample_count}: {sample['__key__']}")
            else:
                print(f"Skipping incomplete sample: {sample.get('__key__', 'unknown')}")
                
        except Exception as e:
            print(f"Error loading sample: {e}")
            continue
    
    if not samples_to_plot:
        print("No valid samples found!")
        return
    
    print(f"\nCreating plots for {len(samples_to_plot)} samples...")
    
    # Create individual plots for each sample
    for idx, sample in enumerate(samples_to_plot):
        plot_sample(sample, idx + 1, save_dir)
    
    # Create summary plot
    create_summary_plot(samples_to_plot, save_dir)
    
    print(f"\nAll plots saved to: {save_dir}")
    print(f"Created {len(samples_to_plot)} individual plots + 1 summary plot")


if __name__ == "__main__":
    main() 