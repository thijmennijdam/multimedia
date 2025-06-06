from hysac.models import HySAC

print("Loading model...")

model_id = "aimagelab/hysac"

model = HySAC.from_pretrained(model_id, device="cpu")
print("Model loaded succesfully!")




# import torch
# from collections import OrderedDict

# # Load the original checkpoint
# state_dict = torch.load("/home/kt-47/.cache/huggingface/hub/models--aimagelab--hysac/snapshots/5166e4d6bd1a311dfe4131c4cc5768cc2f3d809e/hysac_model.pth", map_location="cpu")

# # Create a new state dict with 'base_model' removed from keys
# new_state_dict = OrderedDict()

# for k, v in state_dict.items():
#     if "base_model" in k:
#         new_key = k.replace("base_model.", "")
#     else:
#         new_key = k
#     new_state_dict[new_key] = v

# # Now load into your model
# # model = HySAC()  # or HySAC.from_config(...) if needed
# model.load_state_dict(new_state_dict, strict=False)  # or True if keys now match exactly