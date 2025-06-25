import numpy as np
import umap

import torch
import lorentz as L

# ... existing code for projection helpers ... 

def _hyperboloid_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (xh, yh, zh) coordinates on the hyperboloid ℍ²."""
    emb2 = umap.UMAP(
        n_components=2,
        n_jobs=1,
        random_state=42,
        output_metric="hyperboloid",
    ).fit(x)
    xh, yh = emb2.embedding_[:, 0], emb2.embedding_[:, 1]
    zh = np.sqrt(1.0 + np.sum(emb2.embedding_**2, axis=1), dtype=np.float32)
    return xh, yh, zh


def _umap_hyperbolic(x: np.ndarray) -> np.ndarray:
    """3D hyperbolic (hyperboloid model) embedding."""
    xh, yh, zh = _hyperboloid_2d(x)
    return np.column_stack((xh, yh, zh))


PROJECTIONS = {
    "UMAP": _umap_hyperbolic,
}



def interpolate(model, feats: torch.Tensor, root_feat: torch.Tensor, steps: int):
    """
    Interpolate between given feature vector and `[ROOT]` depending on model type.
    """

    # Linear interpolation between root and image features. For HyCoCLIP and MERU,
    # this happens in the tangent space of the origin.
    # if isinstance(model, (HyCoCLIP, MERU)):
    #     feats = L.log_map0(feats, model.curv.exp())
    # Placeholder with curvature of 1
    if model == "tmp":
        feats = L.log_map0(feats)

    interp_feats = [
        torch.lerp(root_feat, feats, weight.item())
        for weight in torch.linspace(0.0, 1.0, steps=steps)
    ]
    interp_feats = torch.stack(interp_feats)

    # Lift on the Hyperboloid (for HyCoCLIP and MERU), or L2 normalize (for CLIP).
    # if isinstance(model, (HyCoCLIP, MERU)):
    #     feats = L.log_map0(feats, model.curv.exp())
    #     interp_feats = L.exp_map0(interp_feats, model.curv.exp())
    # Placeholder with curvature of 1
    if model == "tmp":
        feats = L.log_map0(feats)
        interp_feats = L.exp_map0(interp_feats)
    else:
        interp_feats = torch.nn.functional.normalize(interp_feats, dim=-1)

    # Reverse the traversal order: (image first, root last)
    return interp_feats.flip(0)



def calc_scores(
    model, image_feats: torch.Tensor, all_feats: torch.Tensor, has_root: bool
):
    """
    Calculate similarity scores between the input image and dataset features depending
    on model type.

    Args:
        has_root: Flag to indicate whether the last text embedding (at dim=0)
            is the `[ROOT]` embedding.
    """

    all_scores = []

    # if isinstance(model, (HyCoCLIP, MERU)):
    #     for feats_batch in all_feats.split(65536):
    #         scores = L.pairwise_inner(image_feats, feats_batch, model.curv.exp())
    #         all_scores.append(scores)
        
    #     all_scores = torch.cat(all_scores, dim=1)
    #     return all_scores

    # Placeholder with curvature of 1
    if model == "tmp":
        for feats_batch in all_feats.split(65536):
            scores = L.pairwise_inner(image_feats, feats_batch)
            all_scores.append(scores)
        
        all_scores = torch.cat(all_scores, dim=1)
        return all_scores
    else:
        # model is not needed here.
        return image_feats @ all_feats.T


def _interpolate_hyperbolic(feat1, feat2, all_feats, model, steps=10):
    """
    Traverse from feat1 to feat2 through all_feats using the model.
    Returns a list of integer indices into all_feats for each interpolated feature.
    """
    # Interpolate between feat1 and feat2
    feat1, feat2, all_feats = torch.tensor(feat1), torch.tensor(feat2), torch.tensor(all_feats)
    interp_feats = interpolate(model, feat1, feat2, steps=steps+2)
    
    # Calculate scores for the interpolated features
    nn_scores = calc_scores(model, interp_feats, all_feats, has_root=True)
    
    # Get the maximum score and corresponding indices
    _, nn_idxs = nn_scores.max(dim=-1)

    # Return the indices with duplicates removed, preserving order
    seen = set()
    ordered_unique = []
    for idx in nn_idxs.tolist():
        if idx not in seen:
            seen.add(idx)
            ordered_unique.append(idx)

    return ordered_unique