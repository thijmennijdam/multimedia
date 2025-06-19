import argparse
import logging
import random

import numpy as np
import torch

import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pickle

from pvae.distributions.riemannian_normal import RiemannianNormal
from pvae.manifolds.poincareball import PoincareBall

import hyptorch.pmath as pmath
import hyptorch.nn as pnn


from htsne_impl import TSNE as hTSNE
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO)

MIN_NORM = 1e-15
c = 1.0
COLOR_MAP ={
    'parent_image' : '#1f77b4',
    'child_image' : '#ff7f0e',
    'parent_text' : '#2ca02c',
    'child_text' : '#d62728',
}

def random_ball(num_points, dimension, radius=1):
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = torch.normal(mean=0.0, std=1.0, size=(dimension, num_points))
    random_directions /= torch.norm(random_directions, dim=0)
    
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = torch.rand(num_points) ** (1/dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T


def generate_riemannian_distri(idx, batch=10, dim=2, scale=1., all_loc=[]):
    
    pball = PoincareBall(dim, c=1)

    loc = random_ball(1, dim, radius=0.999)

    if idx == 0:
        loc = torch.zeros_like(loc)

    distri = RiemannianNormal(loc, torch.ones((1,1)) * scale, pball)

    return distri, loc


def generate_riemannian_clusters(clusters=5, batch=20, dim=2, scale=1.):
    embs  = torch.zeros((0, dim))
    means = torch.zeros((0, dim))
    
    pball = PoincareBall(dim, c=1)


    all_loc = []

    labels= []
    

    for i in range(clusters):

        distri, mean = generate_riemannian_distri(idx = i, batch=batch, dim=dim, scale=scale, all_loc=all_loc)

        labels.extend([i] * batch)

        for _ in range(batch):
            embs = torch.cat((embs, distri.sample()[0]))

        means = torch.cat((means, mean))

    ###############################################

    return embs, labels, means


def generate_high_dims():

    embs, labels, means = generate_riemannian_clusters(clusters=5, batch=20, dim=5, scale=0.25)

    print("embs", embs.shape)
    
    fig = plt.figure(figsize=(5,5))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    seed_colors = ['black', 'red', 'b', 'g', 'c']

    colors = []
    for label in labels:
        colors.append(seed_colors[label])

    plt.scatter(embs[:,0], embs[:,1], c=colors, alpha=0.3)


    mcolors = []
    for i in range(means.shape[0]):
        mcolors.append(seed_colors[i])

    plt.scatter(means[:,0], means[:,1], c=mcolors, marker='x', s=50)

    circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    plt.gca().add_patch(circle1)
    #####################################################


    plt.savefig("./saved_figures/high_dim" + ".png", bbox_inches='tight', dpi=fig.dpi)

    return embs, colors


def to_poincare(x, ideal=False):
    """Convert from hyperboloid model to Poincare ball model
    Args:
        x: torch.tensor of shape (..., Minkowski_dim), where Minkowski_dim >= 3
        ideal: boolean. Should be True if the input vectors are ideal points, False otherwise

    Returns:
        torch.tensor of shape (..., Minkowski_dim - 1)
    """
    if ideal:
        return x[..., 1:] / (x[..., 0].unsqueeze(-1)).clamp_min(MIN_NORM)
    else:
        return x[..., 1:] / (1 + x[..., 0].unsqueeze(-1)).clamp_min(MIN_NORM)
    

def load_embeddings(embedding_pkl_file, n=200):
    with open(embedding_pkl_file, 'rb') as f:
        embed_dict = pickle.load(f)

    no_of_samples = embed_dict['child_image_feats'].shape[0]
    logging.info(f"Loaded embeddings of shape {no_of_samples}")
    
    random_indices = random.sample(range(no_of_samples), n)
    
    child_image_feats = embed_dict['child_image_feats']
    parent_image_feats = embed_dict['parent_image_feats']
    child_text_feats = embed_dict['child_text_feats']
    parent_text_feats = embed_dict['parent_text_feats']

    train_child_image_feats = child_image_feats[random_indices, :]
    train_parent_image_feats = parent_image_feats[random_indices, :]
    train_child_text_feats = child_text_feats[random_indices, :]
    train_parent_text_feats = parent_text_feats[random_indices, :]

    z = np.concatenate([train_child_image_feats, train_parent_image_feats, 
                        train_child_text_feats, train_parent_text_feats], axis=0)
    colors = [COLOR_MAP['child_image']] * n + [COLOR_MAP['parent_image']] * n + [COLOR_MAP['child_text']] * n + [COLOR_MAP['parent_text']] * n

    logging.info(f"Loaded train embeddings of shape {z.shape}")
    z = torch.from_numpy(z).to(dtype=torch.float64)
    # z = to_poincare(z)
    return z, colors


def run_TSNE(embeddings, learning_rate = 1.0, learning_rate_for_h_loss = 0.0, perplexity=5, early_exaggeration=1, student_t_gamma=1.0):

    # tsne = TSNE(n_components=2, method='exact', perplexity=perplexity, learning_rate=learning_rate, early_exaggeration=1)

    # tsne_embeddings = tsne.fit_transform(embeddings)

    # print ("\n\n")
    co_sne = hTSNE(n_components=2, verbose=0, method='exact', square_distances=True, 
                  metric='precomputed', learning_rate_for_h_loss=learning_rate_for_h_loss, student_t_gamma=student_t_gamma, learning_rate=learning_rate, n_iter=10000, perplexity=perplexity, early_exaggeration=early_exaggeration)

    dists = pmath.dist_matrix(embeddings, embeddings, c=1).numpy()


    CO_SNE_embedding = co_sne.fit_transform(dists, embeddings)


    # _htsne = hTSNE(n_components=2, verbose=0, method='exact', square_distances=True, 
    #               metric='precomputed', learning_rate_for_h_loss=0.0, student_t_gamma=1.0, learning_rate=learning_rate, n_iter=1000, perplexity=perplexity, early_exaggeration=early_exaggeration)

    # HT_SNE_embeddings = _htsne.fit_transform(dists, embeddings)


    return CO_SNE_embedding


def rotate_points(points, angle):
    angle = np.radians(angle)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.dot(points, rotation_matrix)


def plot_low_dims(CO_SNE_embedding, colors, n, hierarchy_index):



    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)

    # plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=colors, s=30)
    # ax.set_aspect('equal')
    # plt.axis('off')
    # plt.savefig("./saved_figures/" + "tsne.png", bbox_inches='tight', dpi=fig.dpi)



    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)

    # circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
    # plt.gca().add_patch(circle1)

    # plt.scatter(HT_SNE_embeddings[:,0], HT_SNE_embeddings[:,1], c=colors, s=30)
    # ax.set_aspect('equal')
    # plt.axis('off')
    # plt.savefig("./saved_figures/" + "HT-SNE.png", bbox_inches='tight', dpi=fig.dpi)

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)

    do_rotate = True
    if do_rotate:
        CO_SNE_embedding = rotate_points(CO_SNE_embedding, 180)


    max_radius = np.max(np.linalg.norm(CO_SNE_embedding, axis=1))
    max_lim = 1.05 * max_radius
    
    # circle1 = plt.Circle((0, 0), max_radius, color='black', fill=False)
    circle1 = plt.Circle((0, 0), max_lim, facecolor=(0.94, 0.94, 0.94), fill=True, linewidth=None, zorder=-3)
    plt.gca().add_patch(circle1)
    plt.plot([0], [0], 'o', markersize=2, color='black')
    plt.text(max_lim*0.02, 0, "O", fontsize=11, ha='left')

    child_image_embeds = []
    parent_image_embeds = []
    child_text_embeds = []
    parent_text_embeds = []
    for i in range(len(CO_SNE_embedding)):
        if colors[i] == COLOR_MAP['child_image']:
            child_image_embeds.append(CO_SNE_embedding[i])
        elif colors[i] == COLOR_MAP['parent_image']:
            parent_image_embeds.append(CO_SNE_embedding[i])
        elif colors[i] == COLOR_MAP['child_text']:
            child_text_embeds.append(CO_SNE_embedding[i])
        elif colors[i] == COLOR_MAP['parent_text']:
            parent_text_embeds.append(CO_SNE_embedding[i])
    
    child_image_embeds = np.array(child_image_embeds)
    parent_image_embeds = np.array(parent_image_embeds)
    child_text_embeds = np.array(child_text_embeds)
    parent_text_embeds = np.array(parent_text_embeds)

    plt.scatter(child_image_embeds[:,0], child_image_embeds[:,1], s=5, alpha=0.5, label="Images")
    plt.scatter(parent_image_embeds[:,0], parent_image_embeds[:,1], s=5, alpha=0.5, label="Image boxes")
    plt.scatter(child_text_embeds[:,0], child_text_embeds[:,1], s=5, alpha=0.5, label="Texts")
    plt.scatter(parent_text_embeds[:,0], parent_text_embeds[:,1], s=5, alpha=0.5, label="Text boxes")
    
    # plt.scatter(CO_SNE_embedding[:,0], CO_SNE_embedding[:,1], c=colors, s=5, alpha=0.3)
    
    # Plot hierarchy line
    # tracked_child_image = CO_SNE_embedding[hierarchy_index]
    # tracked_parent_image = CO_SNE_embedding[hierarchy_index + n]
    # tracked_child_text = CO_SNE_embedding[hierarchy_index + 2*n]
    # tracked_parent_text = CO_SNE_embedding[hierarchy_index + 3*n]

    # plt.plot([tracked_parent_text[0], tracked_child_text[0]],
    #          [tracked_parent_text[1], tracked_child_text[1]], color='lime', linestyle='dotted')
    # plt.plot([tracked_parent_text[0], tracked_parent_image[0]],
    #          [tracked_parent_text[1], tracked_parent_image[1]], color='cyan', linestyle='dotted')
    # plt.plot([tracked_child_text[0], tracked_child_image[0]],
    #          [tracked_child_text[1], tracked_child_image[1]], color='magenta', linestyle='dotted')
    # plt.plot([tracked_parent_image[0], tracked_child_image[0]],
    #          [tracked_parent_image[1], tracked_child_image[1]], color='black', linestyle='dotted')

    plt.axis('equal')
    plt.axis('off')
    plt.rcParams["legend.markerscale"] = 2.0
    plt.legend(loc='upper center')
    plt.savefig("./saved_figures/" + "CO-SNE.png", bbox_inches='tight', dpi=300)



if __name__ == "__main__":

    n = 200     # Randomly sample 100 samples for training
    generate_cosne_embeds = False

    if generate_cosne_embeds:
        embeddings, colors = load_embeddings("embeddings/compressed_embeddings_grit_50dim.pkl")

        learning_rate = 0.5
        learning_rate_for_h_loss = 0.01
        perplexity = 250
        early_exaggeration = 10.0
        student_t_gamma = 0.1


        CO_SNE_embedding  = run_TSNE(embeddings, learning_rate, learning_rate_for_h_loss, perplexity, early_exaggeration, student_t_gamma)
        all_parent_text_2d_embeddings = []

        for i, _emb in enumerate(CO_SNE_embedding):
            if colors[i] == COLOR_MAP['parent_text']:
                all_parent_text_2d_embeddings.append(_emb)
        
        all_parent_text_2d_embeddings = np.array(all_parent_text_2d_embeddings)
        avg_parent_text_2d_embeddings = np.mean(all_parent_text_2d_embeddings, axis=0)

        logging.info(f"Average parent text embedding in 2D space: {avg_parent_text_2d_embeddings}")

        # Centering all points to the average parent text embedding
        CO_SNE_embedding = CO_SNE_embedding - avg_parent_text_2d_embeddings

        cosne_embed_dict = {
            "embeds": CO_SNE_embedding,
            "colors": colors
        }

        with open("embeddings/generated_embeddings.pkl", 'wb') as f:
            pickle.dump(cosne_embed_dict, f)
    
    with open("embeddings/generated_embeddings.pkl", 'rb') as f:
        cosne_embed_dict = pickle.load(f)


    hierarchy_index = 50
    plot_low_dims(cosne_embed_dict["embeds"], cosne_embed_dict["colors"], n, hierarchy_index)

