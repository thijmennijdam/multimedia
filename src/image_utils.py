import base64
import io
from pathlib import Path
import numpy as np
from PIL import Image
from dash import html

_IMAGENET_ROOT = Path("imagenet-subset")

def _encode_image(rel_path: str) -> str:
    """Return base64 data URI for *rel_path* (relative to *_IMAGENET_ROOT* or absolute path)."""
    # Handle both old imagenet-subset paths and new hierarchical dataset paths
    if rel_path.startswith("hierchical_datasets/"):
        img_path = Path(rel_path)
    else:
        img_path = _IMAGENET_ROOT / rel_path
    
    mime = {
        ".jpeg": "jpeg",
        ".jpg": "jpeg",
        ".png": "png",
    }.get(img_path.suffix.lower(), "jpeg")

    try:
        with open(img_path, "rb") as f_img:
            enc = base64.b64encode(f_img.read()).decode()
    except FileNotFoundError:
        # If image not found, return empty string to avoid broken UI
        return ""

    return f"data:image/{mime};base64,{enc}"


def _create_img_tag(idx: int, images: np.ndarray | None) -> html.Img | html.Span:
    """Creates a base64 encoded image tag from the IMAGES dataset."""
    if images is None:
        return html.Span()

    if isinstance(images, (list, np.ndarray)) and isinstance(images[idx], (list, np.ndarray)):
        images_np = np.asarray(images)
        pil = (
            Image.fromarray((images_np[idx] * 16).astype("uint8"), mode="L")
            .resize((64, 64), Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)
        )
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        return html.Img(
            src=uri, style={"marginRight": "0.5rem", "border": "1px solid #bbb"}
        )

    try:
        img_rel = images[idx]
        uri = _encode_image(str(img_rel))  # type: ignore[arg-type]
        return html.Img(
            src=uri, style={"marginRight": "0.5rem", "border": "1px solid #bbb"}
        )
    except Exception:
        return html.Span()


def _create_interpolated_img_tag(i: int, j: int, t_value: float, images: np.ndarray | None) -> html.Img | html.Span:
    """Creates an image tag for the interpolated point."""
    if images is None:
        return html.Span()
    images_np = np.asarray(images)
    img1 = images_np[i].astype(np.float32)
    img2 = images_np[j].astype(np.float32)
    interpolated_img_data = (1 - t_value) * img1 + t_value * img2
    pil = Image.fromarray(
        (interpolated_img_data * 16).astype("uint8"), mode="L"
    ).resize((64, 64), Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return html.Img(
        src=uri,
        style={"marginRight": "0.5rem", "border": "2px solid orange"},
    )


def _create_content_element(idx: int, images: np.ndarray | None, points: list | None = None, meta: dict | None = None) -> html.Div | html.Img | html.Span:
    """Creates either a text element or image element based on the embedding type."""
    # If we have points data, check the embedding type
    if points and idx < len(points):
        point = points[idx]
        embedding_type = point.get("embedding_type", "")
        
        # For text embeddings, display the actual text content
        if embedding_type in ["parent_text", "child_text"]:
            synset_id = point.get("synset_id", "")
            if meta and synset_id in meta:
                if embedding_type == "parent_text":
                    text_content = meta[synset_id].get("name", "No parent text available")
                else:  # child_text
                    text_content = meta[synset_id].get("description", "No child text available")
                
                return html.Div([
                    html.P(text_content, style={
                        "margin": "0", 
                        "padding": "0.5rem", 
                        "backgroundColor": "#f0f8ff",
                        "border": "1px solid #d0d0d0",
                        "borderRadius": "4px",
                        "fontStyle": "italic",
                        "maxWidth": "200px",
                        "wordWrap": "break-word"
                    })
                ], style={"marginRight": "0.5rem"})
            else:
                return html.Div([
                    html.P(f"Text content (type: {embedding_type})", style={
                        "margin": "0", 
                        "padding": "0.5rem", 
                        "backgroundColor": "#f0f8ff",
                        "border": "1px solid #d0d0d0",
                        "borderRadius": "4px",
                        "fontStyle": "italic",
                        "color": "#666"
                    })
                ], style={"marginRight": "0.5rem"})
    
    # For image embeddings or when we don't have points data, fall back to image display
    return _create_img_tag(idx, images)