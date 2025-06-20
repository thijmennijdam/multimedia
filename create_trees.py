#!/usr/bin/env python3
"""Utility to build *meta* and *points* JSON files from an ImageNet-style folder.

The folder structure is expected to be::

    imagenet-subset/
        n01440764/
            img_0.JPEG
            img_1.JPEG
            ...
        n01443537/
            ...
        ...

For every *synset* directory we create:

* one **word** node  –  ``{synset_id}_word``
* one **desc** node  –  ``{synset_id}_desc``
* one **image** node per image file inside the directory.

Each point entry follows the schema used in the earlier prototype::

    {
        "id": "n01440764_00",  # unique id
        "synset_id": "n01440764",
        "kind": "image" | "word" | "desc",
        "embedding": [x, y],          # here: random 2-D vector
        "image_path": "n01440764/ILSVRC2012_val_00012345.JPEG"  # only for kind=="image"
    }

In addition, a *meta* mapping is stored which contains human-readable information
about the synset and an *anchor image* (the first image encountered)::

    "n01440764": {
        "synset_id": "n01440764",
        "name": "tench",
        "description": "Tinca tinca, freshwater fish",
        "first_image_path": "n01440764/ILSVRC2012_val_00000293.JPEG"
    }

Both objects are serialised as pretty-printed JSON files:

* ``<output>/meta.json``
* ``<output>/points.json``

The script tries to fetch *name* and *description* from WordNet (via *nltk*).
If that fails (e.g. WordNet not available or lookup error) the synset id is used
as fallback for the *name* and an empty string for the *description*.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

# -----------------------------------------------------------------------------
# Synset information source
# -----------------------------------------------------------------------------
# We primarily rely on a CSV (id,name,description). If a record is missing, we
# optionally fall back to WordNet (if available) or to the raw id.
# -----------------------------------------------------------------------------

import csv


_CSV_LOOKUP: dict[str, tuple[str, str]] | None = None


def _load_synset_csv(path: Path | None) -> dict[str, tuple[str, str]]:
    """Load *synset_id → (name, description)* mapping from *path* if present."""
    cache: dict[str, tuple[str, str]] = {}
    if path and path.exists():
        with path.open(newline="", encoding="utf-8") as f_csv:
            reader = csv.DictReader(f_csv)
            for row in reader:
                sid = row.get("synset_id") or row.get("id")
                if not sid:
                    continue
                name = row.get("synset_name") or row.get("name") or sid
                desc = row.get("definition") or row.get("description") or ""
                cache[sid] = (name, desc)
    return cache


# Lazy WordNet (fallback only when lookup misses & import works)
def _lazy_wordnet(name: str):  # noqa: ANN001
    try:
        from nltk.corpus import wordnet as _wn  # type: ignore

        # Ensure corpus present
        try:
            _wn.synsets("dog")
        except LookupError:
            import nltk

            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        return _wn
    except Exception:  # noqa: BLE001
        return None


wn = None  # to be initialised lazily only if needed

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _get_synset_info(synset_id: str) -> tuple[str, str]:
    """Return *(name, description)* for *synset_id*.

    Order of precedence:
    1. entry found in CSV lookup
    2. WordNet (lazy import/download on first use)
    3. Fallback to *(synset_id, "")*
    """

    global _CSV_LOOKUP, wn  # noqa: PLW0603

    if _CSV_LOOKUP is None:
        _CSV_LOOKUP = _load_synset_csv(Path("synsets.csv"))

    # 1) CSV mapping
    if synset_id in _CSV_LOOKUP:
        return _CSV_LOOKUP[synset_id]

    # 2) WordNet fallback (only if available)
    if wn is None:
        wn = _lazy_wordnet("auto")

    if wn is not None:
        try:
            syn = wn.synset_from_pos_and_offset("n", int(synset_id[1:]))
            lemma_name = syn.lemmas()[0].name().replace("_", " ")
            desc = syn.definition()
            return lemma_name, desc
        except Exception:  # noqa: BLE001
            pass  # continue to default fallback

    # 3) final fallback
    return synset_id, ""


# -------------------------------------------------------------
# Embedding strategy (fake hyperbolic layout)
# -------------------------------------------------------------
# We arrange points on concentric circles:
#   • word      → inner circle (radius 0.25)
#   • desc      → middle circle (radius 0.60)
#   • images    → outer circle (radius 0.95) with slight angular jitter
# Each synset occupies a unique angle θ along the circle so its nodes align
# radially.  This gives a nice tree-like projection.
# -------------------------------------------------------------


def _circle_coord(angle: float, radius: float, dim: int = 2) -> list[float]:
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    if dim == 2:
        return [float(x), float(y)]
    # Pad higher dims with zeros
    return [float(x), float(y)] + [0.0] * (dim - 2)


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def build_dataset(root: Path, output: Path, emb_dim: int = 2, seed: int | None = 42) -> None:
    """Walk *root* folder and create ``meta.json`` & ``points.json`` in *output*."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    meta: dict[str, dict[str, Any]] = {}
    points: list[dict[str, Any]] = []

    syn_dirs = [path for path in root.iterdir() if path.is_dir()]
    syn_dirs_sorted = sorted(syn_dirs)
    total_syns = len(syn_dirs_sorted)

    for idx_syn, syn_dir in enumerate(syn_dirs_sorted):
        synset_id = syn_dir.name
        # Determine unique angle for this synset so its nodes lie on one radial line
        theta = 2 * np.pi * idx_syn / max(total_syns, 1)
        # ------------------------------------------------------------------
        # Collect metadata for this synset
        # ------------------------------------------------------------------
        name, description = _get_synset_info(synset_id)

        # Gather image files (common extensions)
        image_files = sorted(
            [p for p in syn_dir.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"}]
        )
        if not image_files:
            # Nothing to process → skip
            continue

        first_image_rel = str(image_files[0].relative_to(root))
        meta[synset_id] = {
            "synset_id": synset_id,
            "name": name,
            "description": description,
            "first_image_path": first_image_rel,
        }

        # ------------------------------------------------------------------
        # Word & description nodes (one each)
        # ------------------------------------------------------------------
        points.append(
            {
                "id": f"{synset_id}_word",
                "synset_id": synset_id,
                "kind": "word",
                "embedding": _circle_coord(theta, 0.25),
            }
        )
        points.append(
            {
                "id": f"{synset_id}_desc",
                "synset_id": synset_id,
                "kind": "desc",
                "embedding": _circle_coord(theta, 0.60),
            }
        )

        # ------------------------------------------------------------------
        # Image nodes – one per file
        # ------------------------------------------------------------------
        n_imgs = len(image_files)
        for idx, img_path in enumerate(image_files):
            # Spread images slightly around the main theta of the synset so they don't overlap exactly.
            jitter = (idx - n_imgs / 2) / n_imgs * (np.pi / 180 * 8)  # ±4° spread
            angle_img = theta + jitter
            img_rel = str(img_path.relative_to(root))
            points.append(
                {
                    "id": f"{synset_id}_{idx:04d}",
                    "synset_id": synset_id,
                    "kind": "image",
                    "embedding": _circle_coord(angle_img, 0.95),
                    "image_path": img_rel,
                }
            )

    # ----------------------------------------------------------------------
    # Serialise to disk
    # ----------------------------------------------------------------------
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "meta.json", "w", encoding="utf-8") as f_meta:
        json.dump(meta, f_meta, indent=2, ensure_ascii=False)
    with open(output / "points.json", "w", encoding="utf-8") as f_pts:
        json.dump(points, f_pts, indent=2, ensure_ascii=False)

    print(f"Processed {len(meta)} synsets → {output / 'meta.json'} & {output / 'points.json'}")


# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------

def cli() -> None:
    parser = argparse.ArgumentParser(description="Generate meta/points JSON from ImageNet-style folder")
    parser.add_argument("root", type=Path, nargs="?", default=Path("imagenet-subset"), help="Root directory containing synset sub-folders")
    parser.add_argument("--output", "-o", type=Path, default=Path("dataset"), help="Output directory for JSON files")
    parser.add_argument("--emb-dim", type=int, default=2, help="Dimensionality of random embeddings to generate")
    parser.add_argument("--serve", action="store_true", help="After building, launch a Dash scatter explorer.")
    args = parser.parse_args()

    # Always (re-)build the dataset to guarantee JSON freshness
    build_dataset(args.root, args.output, emb_dim=args.emb_dim)

    if args.serve:
        # ------------------------------------------------------------------
        # Lazy imports to keep base dependencies minimal when not serving
        # ------------------------------------------------------------------
        import base64
        from functools import lru_cache

        import dash  # type: ignore
        from dash import Input, Output, dcc, html, callback_context
        import plotly.graph_objs as go  # type: ignore

        # --------------------------------------------------------------
        # Load freshly generated JSON artefacts
        # --------------------------------------------------------------
        with open(args.output / "meta.json", "r", encoding="utf-8") as f_meta:
            meta = json.load(f_meta)
        with open(args.output / "points.json", "r", encoding="utf-8") as f_pts:
            points: list[dict[str, Any]] = json.load(f_pts)

        points_by_id: dict[str, dict[str, Any]] = {pt["id"]: pt for pt in points}

        # --------------------------------------------------------------
        # Dash helpers
        # --------------------------------------------------------------

        color_map = {"image": "#1f77b4", "word": "#2ca02c", "desc": "#d62728"}
        symbol_map = {"image": "circle", "word": "square", "desc": "diamond"}

        def _make_fig(selected_synset: str | None = None) -> go.Figure:
            """Return scatter figure, optionally with hierarchy lines for *selected_synset*."""
            xs = [pt["embedding"][0] for pt in points]
            ys = [pt["embedding"][1] for pt in points]
            kinds = [pt["kind"] for pt in points]
            ids = [pt["id"] for pt in points]

            base_scatter = go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    size=8,
                    color=[color_map[k] for k in kinds],
                    symbol=[symbol_map[k] for k in kinds],
                    opacity=0.8,
                ),
                customdata=ids,
                hovertemplate="%{customdata}<extra></extra>",
            )

            traces = [base_scatter]

            # ------------------------------------------------------
            # Optional tree edges (simple star: parent → children)
            # ------------------------------------------------------
            if selected_synset is not None:
                syn_pts = [pt for pt in points if pt["synset_id"] == selected_synset]

                word_node = next((p for p in syn_pts if p["kind"] == "word"), None)
                desc_node = next((p for p in syn_pts if p["kind"] == "desc"), None)
                img_nodes = [p for p in syn_pts if p["kind"] == "image"]

                lines_x: list[float | None] = []
                lines_y: list[float | None] = []

                # word  →  desc edge
                if word_node and desc_node:
                    wx, wy = word_node["embedding"][:2]
                    dx, dy = desc_node["embedding"][:2]
                    lines_x.extend([wx, dx, None])
                    lines_y.extend([wy, dy, None])
                else:
                    # degrade gracefully: treat desc as root if word missing
                    desc_node = desc_node or word_node

                # desc  →  images edges
                if desc_node is not None:
                    dx, dy = desc_node["embedding"][:2]
                    for img in img_nodes:
                        ix, iy = img["embedding"][:2]
                        lines_x.extend([dx, ix, None])
                        lines_y.extend([dy, iy, None])

                if lines_x:
                    traces.append(
                        go.Scatter(
                            x=lines_x,
                            y=lines_y,
                            mode="lines",
                            line=dict(color="#888", width=1),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

            fig = go.Figure(traces)
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                plot_bgcolor="white",
            )
            return fig

        # --------------------------------------------------------------
        # Image encoding helper (cached → avoids rereading on repeat clicks)
        # --------------------------------------------------------------

        @lru_cache(maxsize=512)
        def _encode_image(rel_path: str) -> str:
            img_path = args.root / rel_path
            mime = {
                ".jpeg": "jpeg",
                ".jpg": "jpeg",
                ".png": "png",
            }.get(img_path.suffix.lower(), "jpeg")
            with open(img_path, "rb") as f_img:
                enc = base64.b64encode(f_img.read()).decode()
            return f"data:image/{mime};base64,{enc}"

        # --------------------------------------------------------------
        # Dash application
        # --------------------------------------------------------------

        app = dash.Dash(__name__)

        app.layout = html.Div(
            [
                html.H2("Embedding Explorer", style={"textAlign": "center"}),
                dcc.Graph(id="scatter", figure=_make_fig(), style={"height": "75vh"}),
                html.Div(id="info", style={"padding": "1rem", "textAlign": "center"}),
            ],
            style={"maxWidth": "900px", "margin": "0 auto"},
        )

        @app.callback(Output("info", "children"), Input("scatter", "clickData"))
        def _display_info(click):  # noqa: ANN001
            if not click or "points" not in click:
                return "Click on a point to see details."

            pid = click["points"][0]["customdata"]
            pt = points_by_id[pid]
            meta_row = meta[pt["synset_id"]]

            if pt["kind"] == "image":
                img_rel = pt["image_path"]
            else:
                img_rel = meta_row["first_image_path"]

            img_src = _encode_image(img_rel)

            # Build a simple tree-like visualisation: Synset → Node info → Image
            return html.Div(
                [
                    # Synset node (root)
                    html.Div(
                        [html.Strong("Synset ID:"), html.Span(f" {pt['synset_id']}")],
                        style={
                            "padding": "0.25rem 0.5rem",
                            "backgroundColor": "#eef",
                            "borderRadius": "4px",
                            "marginBottom": "0.5rem",
                        },
                    ),
                    # Name/description node
                    html.Div(
                        [
                            html.H4(meta_row["name"], style={"margin": "0.25rem 0"}),
                            html.P(meta_row["description"] or "(no description)", style={"margin": 0}),
                        ],
                        style={
                            "padding": "0.5rem",
                            "borderLeft": "3px solid #99c",
                            "marginLeft": "1rem",
                            "marginBottom": "0.5rem",
                        },
                    ),
                    # Image leaf node
                    html.Div(
                        [
                            html.Img(src=img_src, style={"maxWidth": "220px", "border": "1px solid #ccc"}),
                            html.P(f"Image source: {img_rel}", style={"fontSize": "0.75rem", "color": "#666"}),
                        ],
                        style={
                            "padding": "0.5rem",
                            "borderLeft": "3px solid #c9c",
                            "marginLeft": "2rem",
                        },
                    ),
                    # Point specific info
                    html.P(
                        f"Point ID: {pid} | kind: {pt['kind']}",
                        style={"fontSize": "0.8rem", "color": "#666", "marginTop": "0.75rem"},
                    ),
                ]
            )

        # Update scatter to include tree edges when a point is selected
        @app.callback(Output("scatter", "figure"), Input("scatter", "clickData"))
        def _update_fig(click):  # noqa: ANN001
            if not click or "points" not in click:
                return _make_fig()

            pid = click["points"][0]["customdata"]
            synset = points_by_id[pid]["synset_id"]
            return _make_fig(synset)

        # Run the Dash dev server
        app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    cli()
