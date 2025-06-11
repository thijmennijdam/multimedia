import torch, numpy as np
from PIL import Image
from transformers import CLIPTokenizer, CLIPProcessor
import umap.umap_ as umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from hysac.models import HySAC

# ---------------------------------------------------------------------
# 1.   Input -------------------------------------------------------------------
# ---------------------------------------------------------------------
# images = [
#     Image.open("../dummy_dataset/pistol.jpg").convert("RGB"),
#     Image.open("../dummy_dataset/umbrella.jpg").convert("RGB")
# ]
# captions = [
#     "A man that is holding a pistol",
#     "A man is standing with an umbrella"
# ]
images = [
    Image.open("../dummy_dataset/nsfw_plane.png").convert("RGB"),
    Image.open("../dummy_dataset/plane.png").convert("RGB")
]
captions = [
    "The plane is taking off into the bloddy sky as it crashes into a group of innocent people.",
    "The plane is taking off into the sky."
]
# ---------------------------------------------------------------------
# 2.   Model & embeddings -------------------------------------------------------
# ---------------------------------------------------------------------
model = HySAC.from_pretrained("aimagelab/hysac", device="cpu").eval()
tokenizer   = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
processor   = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

with torch.no_grad():
    # text (batch)
    txt_ids   = tokenizer(captions, return_tensors="pt",
                          padding="max_length", truncation=True)["input_ids"]
    txt_embs  = model.encode_text(txt_ids, project=True)          # (2, D+1)

    # images (batch)
    img_tensor = processor(images=images, return_tensors="pt")["pixel_values"]
    img_embs   = model.encode_image(img_tensor, project=True)     # (2, D+1)

all_embs = torch.cat([txt_embs, img_embs], dim=0).cpu().numpy()   # (4, D+1)

# ---------------------------------------------------------------------
# 3.   2-D hyperbolic UMAP  -----------------------------------------------
# ---------------------------------------------------------------------
reducer = umap.UMAP(n_components=2, output_metric="hyperboloid",
                    random_state=42)
x_h, y_h = reducer.fit_transform(all_embs).T        # hyperboloid coords

# Hyperboloid → Poincaré disk
z_h = np.sqrt(1.0 + x_h**2 + y_h**2)
x_d = x_h / (1.0 + z_h)
y_d = y_h / (1.0 + z_h)

# ---------------------------------------------------------------------
# 4.   Plot  ---------------------------------------------------------------
# ---------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

fig  = plt.figure(figsize=(10, 5))
gs   = GridSpec(2, 2, width_ratios=[2, 1], hspace=0.25, wspace=0.05)
axD  = fig.add_subplot(gs[:, 0])        # disc spans both rows
axI1 = fig.add_subplot(gs[0, 1])
axI2 = fig.add_subplot(gs[1, 1])

# ---- Poincaré disc ----
R = 1.0  # radius of the unit Poincaré disc
axD.set_xlim(-R, R)
axD.set_ylim(-R, R)
axD.set_aspect('equal')
axD.axis('off')
axD.set_title("HySAC embeddings (Poincaré disc)", pad=12)

# Draw the outline *before* plotting points
circle = plt.Circle((0, 0), R, edgecolor="black",
                    linewidth=1.2, linestyle="--", fill=False, zorder=1)
axD.add_artist(circle)

# Colour / marker scheme:  text → ‘o’, image → ‘^’
col   = ["red", "blue", "red", "blue"]   # red = NSFW, blue = safe
mark  = ["o",  "o",    "^",    "^"]

for xx, yy, c, m in zip(x_d, y_d, col, mark):
    axD.scatter(xx, yy, c=c, marker=m, s=20, zorder=2)

marker_size = 10
# Legend
legend = [
    Line2D([0], [0], marker='o', color='w', label='NSFW text',
           markerfacecolor='red', markersize=marker_size),
    Line2D([0], [0], marker='^', color='w', label='NSFW image',
           markerfacecolor='red', markersize=marker_size),
    Line2D([0], [0], marker='o', color='w', label='Safe text',
           markerfacecolor='blue', markersize=marker_size),
    Line2D([0], [0], marker='^', color='w', label='Safe image',
           markerfacecolor='blue', markersize=marker_size),
]
axD.legend(handles=legend, loc='upper left',
           bbox_to_anchor=(0, 1), frameon=False, fontsize=9)

# ---- images with captions ----
for ax, img, cap in zip([axI1, axI2], images, captions):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(cap, fontsize=9, pad=6)

fig.tight_layout()
plt.show()
fig.savefig("hysac_embeddings.png", dpi=300, bbox_inches="tight")
