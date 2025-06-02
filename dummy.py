from tqdm import tqdm
from functools import partial
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
import transformer_lens.utils as utils

import torch
import plotly.express as px
import mamba_lens

model = mamba_lens.HookedMamba.from_pretrained("state-spaces/mamba-130m", device='mps')

prompt_uncorrupted = 'Lately, Emma and Shelby had fun at school. Shelby gave an apple to'
prompt_corrupted = 'Lately, Emma and Shelby had fun at school. Emma gave an apple to'
uncorrupted_answer = ' Emma' # note the space in front is important
corrupted_answer = ' Shelby'

prompt_uncorrupted_tokens = model.to_tokens(prompt_uncorrupted)
prompt_corrupted_tokens = model.to_tokens(prompt_corrupted)

L = len(prompt_uncorrupted_tokens[0])
if len(prompt_corrupted_tokens[0]) != len(prompt_uncorrupted_tokens[0]):
    raise Exception("Prompts are not the same length") # feel free to comment this out, you can patch for different sized prompts its just a lil sus

# logits should be [B,L,V] 
def uncorrupted_logit_minus_corrupted_logit(logits, uncorrupted_answer, corrupted_answer):
    uncorrupted_index = model.to_single_token(uncorrupted_answer)
    corrupted_index = model.to_single_token(corrupted_answer)
    return logits[0, -1, uncorrupted_index] - logits[0, -1, corrupted_index]

# [B,L,V]
corrupted_logits, corrupted_activations = model.run_with_cache(prompt_corrupted_tokens)
corrupted_logit_diff = uncorrupted_logit_minus_corrupted_logit(logits=corrupted_logits, uncorrupted_answer=uncorrupted_answer, corrupted_answer=corrupted_answer)

# [B,L,V]
uncorrupted_logits = model(prompt_uncorrupted_tokens)
uncorrupted_logit_diff = uncorrupted_logit_minus_corrupted_logit(logits=uncorrupted_logits, uncorrupted_answer=uncorrupted_answer, corrupted_answer=corrupted_answer)

# diff is logit of uncorrupted_answer - logit of corrupted_answer
# we expect corrupted_diff to have a negative value (as corrupted should put high pr on corrupted_answer)
# we expect uncorrupted to have a positive value (as uncorrupted should put high pr on uncorrupted_answer)
# thus we can treat these as (rough) min and max possible values
min_logit_diff = corrupted_logit_diff
max_logit_diff = uncorrupted_logit_diff

# make token labels that describe the patch
corrupted_str_tokens = model.to_str_tokens(prompt_corrupted_tokens)
uncorrupted_str_tokens = model.to_str_tokens(prompt_uncorrupted_tokens)

# 'blocks.{layer}.hook_h.{pos}' is the recurrent state of that layer after processing tokens at and before pos position
def h_patching_hook(
    h: Float[torch.Tensor, "B E N"],
    hook: HookPoint,
    layer: int,
    position: int,
) -> Float[torch.Tensor, "B E N"]:
    return corrupted_activations[hook.name]
    
patching_result_logits = torch.zeros((model.cfg.n_layers, L), device=model.cfg.device)
for layer in tqdm(range(model.cfg.n_layers)[:2]):
    for position in range(L):
        patching_hook_name = f'blocks.{layer}.hook_h.{position}'
        patching_hook = partial(h_patching_hook, layer=layer, position=position)
        # [B,L,V]
        patched_logits = model.run_with_hooks(prompt_uncorrupted_tokens, fwd_hooks=[
            (patching_hook_name, patching_hook)
        ])
        
        patched_logit_diff = uncorrupted_logit_minus_corrupted_logit(logits=patched_logits,
                                                                     uncorrupted_answer=uncorrupted_answer,
                                                                     corrupted_answer=corrupted_answer)
        # normalize it so
        # 0 means min_logit_diff (so 0 means that it is acting like the corrupted model)
        # 1 means max_logit_diff (so 1 means that it is acting like the uncorrupted model)
        normalized_patched_logit_diff = (patched_logit_diff-min_logit_diff)/(max_logit_diff - min_logit_diff)
        # now flip them, since most interventions will do nothing and thus act like uncorrupted model, visually its better to have that at 0
        # so now
        # 0 means that it is acting like the uncorrupted model
        # 1 means that it is acting like the corrupted model
        normalized_patched_logit_diff = 1.0 - normalized_patched_logit_diff
        patching_result_logits[layer, position] = normalized_patched_logit_diff

# make token labels that describe the patch
corrupted_str_tokens = model.to_str_tokens(prompt_corrupted_tokens)
uncorrupted_str_tokens = model.to_str_tokens(prompt_uncorrupted_tokens)
token_labels = []
for index, (corrupted_token, uncorrupted_token) in enumerate(zip(corrupted_str_tokens, uncorrupted_str_tokens)):
    if corrupted_token == uncorrupted_token:
        token_labels.append(f"{corrupted_token}_{index}")
    else:
        token_labels.append(f"{uncorrupted_token}->{corrupted_token}_{index}")

# display outputs
px.imshow(utils.to_numpy(patching_result_logits), x=token_labels, color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":"Position", "y":"Layer"}, title='Normalized Logit Difference After Patching hook_h (hidden state)').show()