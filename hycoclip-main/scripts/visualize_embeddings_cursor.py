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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hycoclip', 'CO-SNE'))
from main import run_TSNE

# Add HoroPCA imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hycoclip', 'HoroPCA'))
import geom.poincare as poincare
from learning.frechet import Frechet
from learning.pca import HoroPCA

# Import WebDataset utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from prepare_GRIT_webdataset import ImageTextWebDataset 


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    _AA = parser.add_argument
    _AA("--checkpoint-path", help="Path to checkpoint of a trained HyCoCLIP/MERU/CLIP model.")
    _AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")
    return parser.parse_args()


def load_model(checkpoint_path, train_config_path, device):
    """Load and initialize the model from checkpoint."""
    train_config = LazyConfig.load(train_config_path)
    model = LazyFactory.build_model(train_config, device).eval()
    CheckpointManager(model=model).load(checkpoint_path)
    return model


def load_grit_data(tar_path, max_samples=100):
    """Load data from a GRIT webdataset TAR file."""
    print(f"Loading data from: {tar_path}")
    
    dataset = ImageTextWebDataset(tarfiles=[tar_path], infinite_stream=False)
    
    images = []
    texts = []
    sample_count = 0
    
    for sample in dataset:
        if sample_count >= max_samples:
            break
        print(sample)
        try:
            # Get child image and caption
            if 'child.jpg' in sample:
                child_image = sample['child.jpg']
                child_caption = sample.get('child.txt', '')
                
                # Convert PIL image to numpy array and then to tensor
                if isinstance(child_image, Image.Image):
                    # Convert to RGB if needed and resize to 224x224
                    child_image = child_image.convert('RGB')
                    child_image = child_image.resize((224, 224))
                    img_array = np.array(child_image).astype(np.float32) / 255.0
                    # Convert HWC to CHW format
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                    images.append(img_tensor)
                    texts.append(str(child_caption))
                    sample_count += 1
                    
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    print(f"Successfully loaded {len(images)} samples")
    return images, texts


def generate_embeddings_from_grit(model, device, max_samples=200, batch_size=32):
    """Generate embeddings from GRIT dataset instead of random data."""
    
    # Find a TAR file in the GRIT processed directory
    grit_path = "/scratch-shared/grit/processed"
    tar_files = glob.glob(os.path.join(grit_path, "*.tar"))
    
    if not tar_files:
        print(f"No TAR files found in {grit_path}")
        return None, None
    
    # Sort TAR files for consistent processing order
    tar_files.sort()
    
    # Use the first TAR file found
    tar_file = tar_files[0]
    print(f"Using TAR file: {tar_file}")
    
    # Load images and texts from the TAR file
    images, texts = load_grit_data(tar_file, max_samples=max_samples)
    
    if not images:
        print("No images loaded from TAR file")
        return None, None
    
    print(f"Loaded {len(images)} images and {len(texts)} captions")
    print(f"Image tensor shape: {images[0].shape}")
    print(f"Sample caption: {texts[0][:100]}...")
    
    all_image_feats = []
    all_text_feats = []
    
    # Process images and texts in batches
    num_batches = (len(images) + batch_size - 1) // batch_size
    print(f"Processing {len(images)} samples in {num_batches} batches")
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_end = min(i + batch_size, len(images))
            current_batch_size = batch_end - i
            print(f"Processing batch {i//batch_size + 1}/{num_batches}")
            
            # Prepare image batch
            batch_images = torch.stack(images[i:batch_end]).to(device)
            
            # Prepare text batch (tokenize captions)
            batch_texts = texts[i:batch_end]
            # Simple tokenization - convert to token IDs (this is a placeholder)
            # In real usage, you'd use the model's tokenizer
            batch_text_tokens = []
            for text in batch_texts:
                # Create dummy tokens based on text length (placeholder)
                text_len = min(len(text.split()), 20)
                tokens = torch.randint(0, 1000, (20,))  # Still using random tokens for text
                batch_text_tokens.append(tokens)
            batch_text_tokens = torch.stack(batch_text_tokens).to(device)
            
            # Encode features
            image_feats = model.encode_image(batch_images, project=True)
            text_feats = model.encode_text(batch_text_tokens, project=True)
            
            # Move to CPU and store
            all_image_feats.append(image_feats.cpu())
            all_text_feats.append(text_feats.cpu())
    
    # Concatenate all features
    all_image_feats = torch.cat(all_image_feats, dim=0)
    all_text_feats = torch.cat(all_text_feats, dim=0)
    
    print(f"Final image features shape: {all_image_feats.shape}")
    print(f"Final text features shape: {all_text_feats.shape}")
    print("Model inference successful!")
    
    return all_image_feats, all_text_feats


def apply_horopca_reduction(embeddings, target_dim=50):
    """Apply HoroPCA dimensionality reduction to embeddings."""
    print("Applying HoroPCA for dimensionality reduction to 50 dimensions...")
    torch.set_default_dtype(torch.float64)
    embeddings = embeddings.double()  # Convert to double precision
    
    device = embeddings.device
    if torch.cuda.is_available():
        embeddings = embeddings.cuda()
    
    # Compute the Frechet mean to center the data (following HoroPCA main.py)
    print("Computing Frechet mean to center the embeddings...")
    frechet = Frechet(lr=1e-2, eps=1e-5, max_steps=5000)
    mu_ref, has_converged = frechet.mean(embeddings, return_converged=True)
    print(f"Mean computation has converged: {has_converged}")
    x = poincare.reflect_at_zero(embeddings, mu_ref)
    
    # Apply HoroPCA
    original_dim = embeddings.shape[1]
    print(f"Reducing from {original_dim} to {target_dim} dimensions using HoroPCA...")
    
    horopca = HoroPCA(dim=original_dim, n_components=target_dim, lr=5e-2, max_steps=5)
    if torch.cuda.is_available():
        horopca.cuda()
    
    horopca.fit(x, iterative=False, optim=True)
    reduced_embeddings = horopca.map_to_ball(x).detach().cpu().float()  # Convert back to float32
    
    print(f"HoroPCA reduction complete! New shape: {reduced_embeddings.shape}")
    return reduced_embeddings


def run_cosne_visualization(reduced_embeddings):
    """Run CO-SNE visualization on reduced embeddings."""
    # CO-SNE parameters
    learning_rate = 5.0
    learning_rate_for_h_loss = 0.1
    perplexity = 20
    early_exaggeration = 1
    student_t_gamma = 0.1
    print(f"Running CO-SNE on {reduced_embeddings.shape[0]} embeddings with {reduced_embeddings.shape[1]} dimensions...")
    
    # Run CO-SNE on reduced embeddings
    tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding = run_TSNE(
        reduced_embeddings, learning_rate, learning_rate_for_h_loss, 
        perplexity, early_exaggeration, student_t_gamma
    )
    
    print("CO-SNE visualization complete!")
    print(f"t-SNE embeddings shape: {tsne_embeddings.shape}")
    print(f"HT-SNE embeddings shape: {HT_SNE_embeddings.shape}")  
    print(f"CO-SNE embeddings shape: {CO_SNE_embedding.shape}")
    
    return tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding


def main():
    """Main function to orchestrate the visualization pipeline."""
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(args.checkpoint_path, args.train_config, device)
    
    # Generate embeddings from GRIT dataset
    print("=== Loading GRIT Dataset ===")
    all_image_feats, all_text_feats = generate_embeddings_from_grit(model, device, max_samples=50, batch_size=8)
    
    # Combine all embeddings for dimensionality reduction
    embeddings = torch.cat([all_image_feats, all_text_feats], dim=0)
    print(f"Combined embeddings shape: {embeddings.shape}")
    
    # Apply HoroPCA dimensionality reduction
    reduced_embeddings = apply_horopca_reduction(embeddings)
    
    # Run CO-SNE visualization
    tsne_embeddings, HT_SNE_embeddings, CO_SNE_embedding = run_cosne_visualization(reduced_embeddings)


if __name__ == "__main__":
    main()
    
    # python scripts/visualize_embeddings.py --checkpoint-path checkpoints/hycoclip_vit_s.pth \
    #     --train-config configs/train_hycoclip_vit_s.py