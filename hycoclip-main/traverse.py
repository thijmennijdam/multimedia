import pickle
import torch
from hycoclip.models import HyCoCLIP, MERU
from hycoclip import lorentz as L
import numpy as np

def interpolate(model, feats: torch.Tensor, root_feat: torch.Tensor, steps: int):
    """
    Interpolate between given feature vector and `[ROOT]` depending on model type.
    """

    # Linear interpolation between root and image features. For HyCoCLIP and MERU,
    # this happens in the tangent space of the origin.
    if isinstance(model, (HyCoCLIP, MERU)):
        feats = L.log_map0(feats, model.curv.exp())
    # Placeholder with curvature of 1
    elif model == "tmp":
        feats = L.log_map0(feats)

    interp_feats = [
        torch.lerp(root_feat, feats, weight.item())
        for weight in torch.linspace(0.0, 1.0, steps=steps)
    ]
    interp_feats = torch.stack(interp_feats)

    # Lift on the Hyperboloid (for HyCoCLIP and MERU), or L2 normalize (for CLIP).
    if isinstance(model, (HyCoCLIP, MERU)):
        feats = L.log_map0(feats, model.curv.exp())
        interp_feats = L.exp_map0(interp_feats, model.curv.exp())
    # Placeholder with curvature of 1
    elif model == "tmp":
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

    if isinstance(model, (HyCoCLIP, MERU)):
        for feats_batch in all_feats.split(65536):
            scores = L.pairwise_inner(image_feats, feats_batch, model.curv.exp())
            all_scores.append(scores)
        
        all_scores = torch.cat(all_scores, dim=1)
        return all_scores
    # Placeholder with curvature of 1
    elif model == "tmp":
        for feats_batch in all_feats.split(65536):
            scores = L.pairwise_inner(image_feats, feats_batch)
            all_scores.append(scores)
        
        all_scores = torch.cat(all_scores, dim=1)
        return all_scores
    else:
        # model is not needed here.
        return image_feats @ all_feats.T



def main():
    embed_path = "../multimedia/hycoclip-main/projection_analysis/grit_5k_embeddings.pkl"
    model_path = "checkpoints/hycoclip_vit_s.pth"
    
    with open(embed_path, 'rb') as f:
        feats = pickle.load(f)
    
    # Test interpolation with two embeddings
    feats1, feats2 = feats['embeddings'][0], feats['embeddings'][1984]
    
    # model = HyCoCLIP().load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    model = "tmp"  # Placeholder for the model, replace with actual model instance if needed
    
    # Interpolate and determine nearest neighbors
    interp_feats = interpolate(model, feats1, feats2, steps=10)
    nn1_scores = calc_scores(model, interp_feats, feats['embeddings'], has_root=True)
    
    # Determine the nearest neighbor for each interpolated feature
    nn1_scores, _nn1_idxs = nn1_scores.max(dim=-1)
    nn1_texts = [feats['embeddings'][_idx.item()] for _idx in _nn1_idxs]
    
    print("Steps before removing duplicates: ",
          np.array(nn1_texts).shape)
    
    # Remove duplicates
    nn1_texts = np.unique(nn1_texts, axis=0)
    
    print("Steps after removing duplicates: ",
          np.array(nn1_texts).shape)

if __name__ == "__main__":
    main()