"""Run dimensionality reduction experiment."""

import argparse
import logging

import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import random
from tqdm import tqdm

import geom.hyperboloid as hyperboloid
import geom.poincare as poincare
from learning.frechet import Frechet
from learning.pca import TangentPCA, EucPCA, PGA, HoroPCA, BSA
from utils.data import load_graph, load_embeddings
from utils.metrics import avg_distortion_measures, compute_metrics, format_metrics, aggregate_metrics
from utils.sarkar import sarkar, pick_root

parser = argparse.ArgumentParser(
    description="Hyperbolic dimensionality reduction"
)
parser.add_argument('--embedding-file', type=str, help='MERU generated embedding file')
parser.add_argument('--model', type=str, help='which dimensionality reduction method to use', default="horopca",
                    choices=["pca", "tpca", "pga", "bsa", "hmds", "horopca"])
parser.add_argument('--metrics', nargs='+', help='which metrics to use', default=["frechet_var"])
parser.add_argument(
    "--dim", default=10, type=int, help="input embedding dimension to use"
)
parser.add_argument(
    "--n-components", default=2, type=int, help="number of principal components"
)

parser.add_argument(
    "--lr", default=5e-2, type=float, help="learning rate to use for optimization-based methods"
)
parser.add_argument(
    "--n-runs", default=5, type=int, help="number of runs for optimization-based methods"
)
parser.add_argument('--use-sarkar', default=False, action='store_true', help="use sarkar to embed the graphs")
parser.add_argument('--visualize', default=False, action='store_true', help="visualize the embeddings on the ball")
parser.add_argument(
    "--sarkar-scale", default=3.5, type=float, help="scale to use for embeddings computed with Sarkar's construction"
)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)

    pca_models = {
        'pca': {'class': EucPCA, 'optim': False, 'iterative': False, "n_runs": 1},
        'tpca': {'class': TangentPCA, 'optim': False, 'iterative': False, "n_runs": 1},
        'pga': {'class': PGA, 'optim': True, 'iterative': True, "n_runs": args.n_runs},
        'bsa': {'class': BSA, 'optim': True, 'iterative': False, "n_runs": args.n_runs},
        'horopca': {'class': HoroPCA, 'optim': True, 'iterative': False, "n_runs": args.n_runs},
    }
    metrics = {}
    embeddings = {}
    logging.info(f"Running experiments for {args.embedding_file} dataset.")

    # load pre-trained hyperbolic embeddings
    logging.info("Using optimization-based embeddings")
    assert args.dim in [2, 10, 16, 50, 64, 512], "pretrained embeddings are only for 2, 10 and 50 dimensions"

    with open(args.embedding_file, 'rb') as f:
        embed_dict = pickle.load(f)

    no_of_samples = embed_dict['image_feats'].shape[0]
    random_indices = random.sample(range(no_of_samples), 200)  # Randomly sample 1000 samples for training
    
    child_image_feats = embed_dict['image_feats']
    parent_image_feats = embed_dict['box_image_feats']
    child_text_feats = embed_dict['text_feats']
    parent_text_feats = embed_dict['box_text_feats']

    train_child_image_feats = child_image_feats[random_indices, :]
    train_parent_image_feats = parent_image_feats[random_indices, :]
    train_child_text_feats = child_text_feats[random_indices, :]
    train_parent_text_feats = parent_text_feats[random_indices, :]

    z = np.concatenate([train_child_image_feats, train_parent_image_feats, 
                        train_child_text_feats, train_parent_text_feats], axis=0)
    np.random.shuffle(z)
    
    logging.info(f"Loaded train embeddings of shape {z.shape}")
    z = torch.from_numpy(z).to(dtype=torch.float64)
    z = hyperboloid.to_poincare(z)
    z_dist = poincare.pairwise_distance(z)
    if torch.cuda.is_available():
        z = z.cuda()
        z_dist = z_dist.cuda()


    # Compute the mean and center the data
    logging.info("Computing the Frechet mean to center the embeddings")
    frechet = Frechet(lr=1e-2, eps=1e-5, max_steps=5000)
    mu_ref, has_converged = frechet.mean(z, return_converged=True)
    logging.info(f"Mean computation has converged: {has_converged}")
    # x = poincare.reflect_at_zero(z, mu_ref)
    x = z

    # Run dimensionality reduction methods
    logging.info(f"Running {args.model} for dimensionality reduction")
    metrics = []
    dist_orig = poincare.pairwise_distance(x)
    if args.model in pca_models.keys():
        model_params = pca_models[args.model]
        for _ in range(model_params["n_runs"]):
            model = model_params['class'](dim=args.dim, n_components=args.n_components, lr=args.lr, max_steps=500)
            if torch.cuda.is_available():
                model.cuda()
            model.fit(x, iterative=model_params['iterative'], optim=model_params['optim'])
            metrics.append(model.compute_metrics(x))
        metrics = aggregate_metrics(metrics)
    else:
        logging.error(f"Model {args.model} not found")
        exit(1)
    
    logging.info(f"Generating embeddings")

    batch_size = 128
    child_image_embeds = []
    parent_image_embeds = []
    child_text_embeds = []
    parent_text_embeds = []

    def embed_batch(z):
        z = torch.from_numpy(z).to(dtype=torch.float64)
        z = hyperboloid.to_poincare(z)
        if torch.cuda.is_available():
            z = z.cuda()
        embeddings = model.map_to_ball(z).detach().cpu().numpy()
        return embeddings
    
    embed_samples_count_per_array = 200
    
    for batch_indices in tqdm(range(0, embed_samples_count_per_array, batch_size)):
        batch_child_image_feats = child_image_feats[batch_indices:batch_indices+batch_size, :]
        child_image_embeds.append(embed_batch(batch_child_image_feats))
        batch_parent_image_feats = parent_image_feats[batch_indices:batch_indices+batch_size, :]
        parent_image_embeds.append(embed_batch(batch_parent_image_feats))
        batch_child_text_feats = child_text_feats[batch_indices:batch_indices+batch_size, :]
        child_text_embeds.append(embed_batch(batch_child_text_feats))
        batch_parent_text_feats = parent_text_feats[batch_indices:batch_indices+batch_size, :]
        parent_text_embeds.append(embed_batch(batch_parent_text_feats))
    
    child_image_embeds = np.concatenate(child_image_embeds, axis=0)
    parent_image_embeds = np.concatenate(parent_image_embeds, axis=0)
    child_text_embeds = np.concatenate(child_text_embeds, axis=0)
    parent_text_embeds = np.concatenate(parent_text_embeds, axis=0)

    embeddings = np.concatenate([child_image_embeds, parent_image_embeds, child_text_embeds, parent_text_embeds], axis=0)

    logging.info(f"Experiments for {args.embedding_file} dataset completed.")
    logging.info("Computing evaluation metrics")
    results = format_metrics(metrics, args.metrics)
    for line in results:
        logging.info(line)
    
    embedding_dict = {
        "child_image_feats": child_image_embeds,
        "parent_image_feats": parent_image_embeds,
        "child_text_feats": child_text_embeds,
        "parent_text_feats": parent_text_embeds
    }

    with open("compressed_embeddings.pkl", 'wb') as f:
        pickle.dump(embedding_dict, f)
    
    
    if args.visualize:
        max_radius = np.max(np.linalg.norm(embeddings, axis=1))
        max_lim = 1.05 * max_radius

        ## Plotting on the ball

        # plt.title("HoroPCA embeddings\non the Poincaré ball")
        plt.rcParams["legend.markerscale"] = 2.0
        plt.axis("equal")
        plt.axis("off")
        circle1 = plt.Circle((0, 0), max_lim, facecolor=(0.94, 0.94, 0.94), fill=True, linewidth=None, zorder=-3)
        plt.gca().add_patch(circle1)
        plt.plot([0], [0], 'o', markersize=2, color='black')
        plt.text(max_lim*0.02, 0, "O", fontsize=11, ha='left')
        plt.scatter(child_image_embeds[:,0], child_image_embeds[:,1], s=5, alpha=0.5, label="Images")
        plt.scatter(parent_image_embeds[:,0], parent_image_embeds[:,1], s=5, alpha=0.5, label="Image boxes")
        plt.scatter(child_text_embeds[:,0], child_text_embeds[:,1], s=5, alpha=0.5, label="Texts")
        plt.scatter(parent_text_embeds[:,0], parent_text_embeds[:,1], s=5, alpha=0.5, label="Text boxes")
        
        plt.legend(loc='upper center')
        plt.savefig("checks.png", bbox_inches='tight', dpi=300)
