import dash
from dash import Input, Output, State, callback_context, html, dcc
import numpy as np
import plotly.graph_objs as go
from .projection import _interpolate_hyperbolic

from .image_utils import _encode_image, _create_img_tag, _create_interpolated_img_tag
from .layout import _tree_node
import json

# All callback functions and register_callbacks go here
# ... (move all @app.callback functions and register_callbacks from main.py) ... 

def register_callbacks(app: dash.Dash) -> None:
    # Dataset loading callback
    @app.callback(
        Output("data-store", "data"),
        Output("labels-store", "data"),
        Output("feature-names-store", "data"),
        Output("target-names-store", "data"),
        Output("images-store", "data"),
        Output("meta-store", "data"),
        Output("points-store", "data"),
        Output("sel", "data", allow_duplicate=True),
        Output("interpolated-point", "data", allow_duplicate=True),
        Output("emb", "data", allow_duplicate=True),
        Input("dataset-dropdown", "value"),
        Input("proj", "value"),
        prevent_initial_call=False,
    )
    def _update_dataset_stores(dataset_name, projection_method):
        if not dataset_name or not projection_method:
            return dash.no_update
        if dataset_name == "imagenet":
            try:
                import pickle
                with open("hierchical_datasets/ImageNet/meta_data_trees.json", "r", encoding="utf-8") as f_meta:
                    meta_data_trees = json.load(f_meta)
                
                # Load embeddings based on selected projection method
                emb_file = f"hierchical_datasets/ImageNet/{projection_method}_embeddings.pkl"
                with open(emb_file, "rb") as f_emb:
                    emb_data = pickle.load(f_emb)
            except FileNotFoundError as e:
                print(f"Error loading ImageNet files: {e}")
                return dash.no_update
            except Exception as e:
                print(f"ERROR: ImageNet loading failed: {e}")
                return dash.no_update
            
            # Extract embeddings
            embeddings = np.array(emb_data["embeddings"], dtype=np.float32)
            print(f"Loaded ImageNet with {projection_method}: {embeddings.shape} embeddings")
            
            # Create points list from meta_data_trees
            points = []
            synset_ids = []
            images_list = []
            
            for tree_id, tree_data in meta_data_trees["trees"].items():
                synset_id = tree_data["synset_id"]
                synset_ids.append(synset_id)
                
                # Get first child image path if available, but fix the path
                image_path = None
                if "child_images" in tree_data and tree_data["child_images"]:
                    # Fix the path: replace /data/ with /trees/
                    original_path = tree_data["child_images"][0]["path"]
                    image_path = original_path.replace("/data/", "/trees/")
                
                points.append({
                    "synset_id": synset_id,
                    "tree_id": tree_id,
                    "image_path": image_path,
                    "kind": "tree"
                })
                images_list.append(image_path)
            
            # Create meta dict for compatibility
            meta = {}
            for tree_id, tree_data in meta_data_trees["trees"].items():
                synset_id = tree_data["synset_id"]
                # Fix the image path here too
                first_image_path = None
                if tree_data.get("child_images"):
                    original_path = tree_data["child_images"][0]["path"]
                    first_image_path = original_path.replace("/data/", "/trees/")
                
                meta[synset_id] = {
                    "name": tree_data["parent_text"]["text"],
                    "description": tree_data["child_text"]["text"],
                    "first_image_path": first_image_path
                }
            
            unique_synsets = sorted({sid for sid in synset_ids})
            syn_to_int = {sid: i for i, sid in enumerate(unique_synsets)}
            labels = np.array([syn_to_int[s] for s in synset_ids], dtype=int)
            feature_names = [f"dim{i}" for i in range(embeddings.shape[1])]
            target_names = unique_synsets
            
            return (
                None,  # data-store (not needed anymore)
                labels.tolist(),
                feature_names,
                target_names,
                images_list,
                meta,
                points,
                [],
                None,
                embeddings.tolist(),  # emb store - this is what the scatter callback needs!
            )
        if dataset_name == "grit":
            try:
                import pickle
                with open("hierchical_datasets/GRIT/meta_data_trees.json", "r", encoding="utf-8") as f_meta:
                    meta_data_trees = json.load(f_meta)
                
                # Load embeddings based on selected projection method
                emb_file = f"hierchical_datasets/GRIT/{projection_method}_embeddings.pkl"
                with open(emb_file, "rb") as f_emb:
                    emb_data = pickle.load(f_emb)
            except FileNotFoundError as e:
                print(f"Error loading GRIT files: {e}")
                return dash.no_update
            except Exception as e:
                print(f"ERROR: GRIT loading failed: {e}")
                return dash.no_update
            
            # Extract embeddings
            embeddings = np.array(emb_data["embeddings"], dtype=np.float32)
            print(f"Loaded GRIT with {projection_method}: {embeddings.shape} embeddings")
            
            # Create points list from meta_data_trees
            points = []
            synset_ids = []
            images_list = []
            
            for tree_id, tree_data in meta_data_trees["trees"].items():
                # GRIT uses sample_key instead of synset_id
                sample_key = tree_data.get("sample_key", tree_id)
                synset_ids.append(sample_key)
                
                # Get first child image path if available, but fix the path
                image_path = None
                if "child_images" in tree_data and tree_data["child_images"]:
                    # Fix the path: replace /data/ with /trees/
                    original_path = tree_data["child_images"][0]["path"]
                    image_path = original_path.replace("/data/", "/trees/")
                # If no child images, try parent images
                elif "parent_images" in tree_data and tree_data["parent_images"]:
                    original_path = tree_data["parent_images"][0]["path"]
                    image_path = original_path.replace("/data/", "/trees/")
                
                points.append({
                    "synset_id": sample_key,
                    "tree_id": tree_id,
                    "image_path": image_path,
                    "kind": "tree"
                })
                images_list.append(image_path)
            
            # Create meta dict for compatibility
            meta = {}
            for tree_id, tree_data in meta_data_trees["trees"].items():
                sample_key = tree_data.get("sample_key", tree_id)
                # Fix the image path here too
                first_image_path = None
                if tree_data.get("child_images"):
                    original_path = tree_data["child_images"][0]["path"]
                    first_image_path = original_path.replace("/data/", "/trees/")
                elif tree_data.get("parent_images"):
                    original_path = tree_data["parent_images"][0]["path"]
                    first_image_path = original_path.replace("/data/", "/trees/")
                
                meta[sample_key] = {
                    "name": tree_data["parent_text"]["text"],
                    "description": tree_data["child_text"]["text"],
                    "first_image_path": first_image_path
                }
            
            unique_synsets = sorted({sid for sid in synset_ids})
            syn_to_int = {sid: i for i, sid in enumerate(unique_synsets)}
            labels = np.array([syn_to_int[s] for s in synset_ids], dtype=int)
            feature_names = [f"dim{i}" for i in range(embeddings.shape[1])]
            target_names = unique_synsets
            
            return (
                None,  # data-store (not needed anymore)
                labels.tolist(),
                feature_names,
                target_names,
                images_list,
                meta,
                points,
                [],
                None,
                embeddings.tolist(),  # emb store - this is what the scatter callback needs!
            )
        return dash.no_update

    @app.callback(
        Output("scatter-disk", "figure"),
        Input("emb", "data"),
        Input("sel", "data"),
        Input("proj", "value"),
        Input("interpolated-point", "data"),
        State("labels-store", "data"),
        State("target-names-store", "data"),
        Input("mode", "data"),
        Input("neighbors-slider", "value"),
        State("data-store", "data"),
        Input("dataset-dropdown", "value"),
    )
    def _scatter(edata, sel, proj, interpolated_point, labels_data, target_names, mode, k_neighbors, data_store, dataset_name):
        if edata is None or labels_data is None:
            print("Warning: No embedding or label data available for plotting")
            return {}
        emb = np.asarray(edata, dtype=np.float32)
        labels = np.asarray(labels_data, dtype=int)
        sel = sel or []
        interp_point = (
            np.asarray(interpolated_point, dtype=np.float32)
            if interpolated_point is not None
            else None
        )
        highlight = sel
        neighbor_indices = []
        selected_idx = sel
        if mode == "neighbors" and sel and len(sel) == 1:
            selected_idx = sel[:1]
            if data_store is not None:
                data_np = np.asarray(data_store, dtype=np.float32)
                dists = np.linalg.norm(data_np - data_np[sel[0]], axis=1)
                neighbor_indices = np.argsort(dists)
                neighbor_indices = neighbor_indices[neighbor_indices != sel[0]][:k_neighbors]
            else:
                neighbor_indices = []
        else:
            neighbor_indices = []

        # 2D disk plot with proper color coding and legend
        def _fig_disk(x, y, sel, labels, target_names, interpolated_point=None, neighbor_indices=None, emb_labels=None):
            # Define colors matching plotting_utils.py
            colors = {
                'child_image': '#1f77b4',    # tab:blue
                'parent_image': '#ff7f0e',   # tab:orange
                'child_text': '#2ca02c',     # tab:green  
                'parent_text': '#d62728'     # tab:red
            }
            
            traces = []
            
            # If we have emb_labels, create separate traces for each label type
            print(f"DEBUG: emb_labels length: {len(emb_labels) if emb_labels else 'None'}, x length: {len(x)}")
            if emb_labels and len(emb_labels) == len(x):
                unique_label_types = sorted(set(emb_labels))
                print(f"DEBUG: Using color-coded traces for: {unique_label_types}")
                
                for label_type in unique_label_types:
                    # Find indices for this label type
                    indices = [i for i, lbl in enumerate(emb_labels) if lbl == label_type]
                    
                    if indices:
                        x_coords = [x[i] for i in indices]
                        y_coords = [y[i] for i in indices]
                        hover_text = [
                            f"{i}: {target_names[labels[i]] if target_names is not None else labels[i]}"
                            for i in indices
                        ]
                        
                        trace = go.Scatter(
                            x=x_coords,
                            y=y_coords,
                            mode="markers",
                            text=hover_text,
                            hoverinfo="text",
                            customdata=indices,  # Store original indices directly
                            marker=dict(
                                size=8, 
                                opacity=0.7, 
                                color=colors.get(label_type, 'gray'),
                                line=dict(width=0.5, color='black')
                            ),
                            name=label_type.replace('_', ' ').title(),
                            showlegend=True,
                        )
                        traces.append(trace)
            else:
                # Fallback to single trace with colorscale
                print("DEBUG: Using fallback single trace with colorscale")
                base = go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    text=[
                        f"{i}: {target_names[labels[i]] if target_names is not None else labels[i]}"
                        for i in range(len(labels))
                    ],
                    hoverinfo="text",
                    marker=dict(size=8, opacity=0.7, color=labels, colorscale="Viridis"),
                    name="Data points",
                    showlegend=False,
                )
                traces = [base]
            if neighbor_indices is not None and len(neighbor_indices) > 0:
                traces.append(
                    go.Scatter(
                        x=x[neighbor_indices],
                        y=y[neighbor_indices],
                        mode="markers",
                        marker=dict(size=10, color="blue"),
                        name="Neighbors",
                    )
                )
            if sel:
                traces.append(
                    go.Scatter(
                        x=x[sel],
                        y=y[sel],
                        mode="markers",
                        marker=dict(size=12, color="red"),
                        name="Selected point",
                    )
                )
            if interpolated_point is not None:
                traces.append(
                    go.Scatter(
                        x=[interpolated_point[0]],
                        y=[interpolated_point[1]],
                        mode="markers",
                        marker=dict(size=12, color="orange", symbol="diamond"),
                        name="Interpolated point",
                        text=["Interpolated point"],
                        hoverinfo="text",
                    )
                )
            fig = go.Figure(data=traces)
            fig.update_layout(
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1),
                margin=dict(l=0, r=0, b=0, t=30),
                uirevision="embedding",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
            )
            return fig
        # Handle 2D embeddings for disk projection
        xh, yh = emb[:, 0], emb[:, 1]
        if emb.shape[1] > 2:
            zh = emb[:, 2]
        else:
            zh = np.zeros(emb.shape[0])  # For 2D embeddings, use z=0
        dx, dy = xh / (1.0 + zh), yh / (1.0 + zh)
        interp_transformed = None
        if interp_point is not None:
            interp_dx = interp_point[0] / (1.0 + interp_point[2])
            interp_dy = interp_point[1] / (1.0 + interp_point[2])
            interp_transformed = np.array([interp_dx, interp_dy])
        # Load the original label types for color coding
        emb_labels = None
        if dataset_name and proj:
            try:
                import pickle
                # Map dataset names to correct directory names
                dataset_dir = {"imagenet": "ImageNet", "grit": "GRIT"}.get(dataset_name, dataset_name)
                emb_file = f"hierchical_datasets/{dataset_dir}/{proj}_embeddings.pkl"
                print(f"DEBUG: Loading labels from {emb_file}")
                with open(emb_file, "rb") as f:
                    emb_data_loaded = pickle.load(f)
                emb_labels = emb_data_loaded.get("labels", [])
                print(f"DEBUG: Loaded {len(emb_labels)} labels: {set(emb_labels[:5]) if emb_labels else 'None'}")
            except Exception as e:
                print(f"DEBUG: Error loading labels: {e}")
                emb_labels = []
            
        fig_disk = _fig_disk(dx, dy, selected_idx if mode == "neighbors" else highlight, labels, target_names, interp_transformed, neighbor_indices, emb_labels)
        return fig_disk



    @app.callback(
        Output("sel", "data"),
        [
            Input("scatter-disk", "clickData"),
            Input({"type": "close-button", "index": dash.ALL}, "n_clicks")
        ],
        State("sel", "data"),
        State("mode", "data"),
        prevent_initial_call=True,
    )
    def _select(click_disk, close_clicks, sel, mode):
        ctx = callback_context
        if not ctx.triggered or not ctx.triggered_id:
            return dash.no_update
        triggered_id = ctx.triggered_id
        current_sel = sel or []
        def _clicked(click):
            try:
                pt = click["points"][0]
                curve_number = pt.get("curveNumber", 0)
                point_index = int(pt.get("pointIndex", pt["pointNumber"]))
                
                print(f"DEBUG CLICK: curve_number={curve_number}, point_index={point_index}")
                print(f"DEBUG CLICK: point data keys: {list(pt.keys())}")
                
                # Check if we have customdata with original indices
                if "customdata" in pt and pt["customdata"] is not None:
                    result = int(pt["customdata"])
                    print(f"DEBUG CLICK: using customdata, returning: {result}")
                    return result
                
                # Fallback to original logic for single trace
                print(f"DEBUG CLICK: using fallback logic")
                if curve_number != 0:
                    return None
                return point_index
            except (TypeError, KeyError, IndexError) as e:
                print(f"DEBUG CLICK: exception: {e}")
                return None
        if triggered_id == "scatter-disk":
            click_data = ctx.inputs["scatter-disk.clickData"]
            idx = _clicked(click_data)
            if idx is None:
                return dash.no_update
            if idx in current_sel:
                new_sel = [i for i in current_sel if i != idx]
            else:
                new_sel = current_sel + [idx]
                max_points = 1
                if mode == "compare": max_points = 5
                elif mode == "interpolate": max_points = 2
                elif mode == "neighbors": max_points = 1
                if len(new_sel) > max_points:
                    new_sel = new_sel[-max_points:]
            return new_sel
        elif isinstance(triggered_id, dict) and triggered_id.get("type") == "close-button":
            if not ctx.triggered[0]['value']:
                return dash.no_update
            idx_to_remove = triggered_id.get("index")
            if idx_to_remove in current_sel:
                new_sel = [i for i in current_sel if i != idx_to_remove]
                return new_sel
        return dash.no_update

    @app.callback(
        Output("run-interpolate-btn", "disabled"),
        Input("sel", "data"),
        Input("mode", "data"),
    )
    def _update_button_state(sel, mode):
        if mode == "interpolate":
            return not (sel and len(sel) == 2)
        return True

    @app.callback(
        Output("interpolated-point", "data"),
        Input("run-interpolate-btn", "n_clicks"),
        State("sel", "data"),
        State("interpolation-slider", "value"),
        State("emb", "data"),
        State("proj", "value"),
        prevent_initial_call=True,
    )
    def _interpolate(n_clicks, sel, t, edata, proj):
        if not (n_clicks and sel and len(sel) == 2 and edata):
            return None
        emb = np.asarray(edata, dtype=np.float32)
        i, j = sel[:2]
        p1, p2 = emb[i], emb[j]
        interpolated = _interpolate_hyperbolic(p1, p2, t)
        return interpolated.tolist()

    @app.callback(
        [Output("tree-parent", "children"), Output("tree-current", "children"), Output("tree-child", "children")],
        Input("sel", "data"),
        Input("mode", "data"),
        State("meta-store", "data"),
        State("points-store", "data"),
    )
    def _update_tree_view(sel, mode, meta, points):
        if mode != "tree" or not sel or len(sel) != 1 or meta is None or points is None:
            return html.Span(), html.Span(), html.Span()
        idx = sel[0]
        try:
            pt = points[idx]
        except (IndexError, TypeError):
            return html.Span(), html.Span(), html.Span()
        synset_id = pt.get("synset_id", "?")
        pt_kind = pt.get("kind", "?")
        meta_row = meta.get(synset_id, {}) if isinstance(meta, dict) else {}
        parent_div = html.Div([
            html.H4(meta_row.get("name", synset_id), style={"margin": "0.25rem 0", "color": "#007bff"}),
        ], style={"padding": "0.5rem", "backgroundColor": "#eef", "borderRadius": "4px", "marginBottom": "0.5rem"})
        current_div = html.Div([
            html.P(meta_row.get("description", "(no description)"), style={"margin": 0}),
        ], style={"padding": "0.5rem", "borderLeft": "3px solid #99c", "marginLeft": "1rem", "marginBottom": "0.5rem"})
        if pt_kind == "image" and pt.get("image_path"):
            img_rel = pt["image_path"]
        else:
            img_rel = meta_row.get("first_image_path")
        img_src = _encode_image(img_rel) if img_rel else ""
        child_div = html.Div([
            html.Img(src=img_src, style={"maxWidth": "220px", "border": "1px solid #ccc"}),
            html.P(f"Image source: {img_rel}" if img_rel else "", style={"fontSize": "0.75rem", "color": "#666"}),
        ], style={"padding": "0.5rem", "borderLeft": "3px solid #c9c", "marginLeft": "2rem"})
        return parent_div, current_div, child_div

    @app.callback(
        Output("mode", "data"),
        Output("compare-btn", "style"),
        Output("interpolate-mode-btn", "style"),
        Output("tree-mode-btn", "style"),
        Output("neighbors-mode-btn", "style"),
        Output("interpolate-controls", "style"),
        Output("neighbors-controls", "style"),
        Output("tree-traversal-section", "style"),
        Output("mode-instructions", "children"),
        Output("sel", "data", allow_duplicate=True),
        Output("interpolated-point", "data", allow_duplicate=True),
        Input("compare-btn", "n_clicks"),
        Input("interpolate-mode-btn", "n_clicks"),
        Input("tree-mode-btn", "n_clicks"),
        Input("neighbors-mode-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _update_mode(compare_clicks, interpolate_clicks, tree_clicks, neighbors_clicks):
        ctx = callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        base_style = {
            "color": "white",
            "border": "none",
            "padding": "0.5rem 1rem",
            "borderRadius": "6px",
            "cursor": "pointer",
            "width": "100%",
            "minWidth": "0",
            "flex": "1 1 0",
            "boxSizing": "border-box",
            "transition": "background-color 0.2s",
        }
        compare_style = {**base_style, "backgroundColor": "#007bff"}
        interpolate_style = {**base_style, "backgroundColor": "#007bff"}
        tree_style = {**base_style, "backgroundColor": "#007bff"}
        neighbors_style = {**base_style, "backgroundColor": "#007bff"}
        selected_style = {**base_style, "backgroundColor": "green"}
        mode = "compare"
        interpolate_controls_style = {"display": "none"}
        neighbors_controls_style = {"display": "none"}
        tree_traversal_style = {"display": "none"}
        instructions = "Select up to 5 points to compare."
        if triggered_id == "interpolate-mode-btn":
            mode = "interpolate"
            interpolate_style = selected_style
            interpolate_controls_style = {"display": "block"}
            instructions = "Select 2 points to interpolate."
        elif triggered_id == "tree-mode-btn":
            mode = "tree"
            tree_style = selected_style
            tree_traversal_style = {"display": "block"}
            instructions = "Select 1 point to view its lineage."
        elif triggered_id == "neighbors-mode-btn":
            mode = "neighbors"
            neighbors_style = selected_style
            neighbors_controls_style = {"display": "block"}
            instructions = "Select 1 point to view its neighbors."
        else:
            compare_style = selected_style
        return (
            mode, compare_style, interpolate_style, tree_style, neighbors_style,
            interpolate_controls_style, neighbors_controls_style, tree_traversal_style, instructions, [], None
        )

    @app.callback(
        Output("cmp", "children"),
        Output("cmp-instructions", "children"),
        Input("sel", "data"),
        Input("interpolated-point", "data"),
        Input("interpolation-slider", "value"),
        Input("mode", "data"),
        State("labels-store", "data"),
        State("target-names-store", "data"),
        State("images-store", "data"),
        State("emb", "data"),
        Input("neighbors-slider", "value"),
    )
    def _compare(sel, interpolated_point, t_value, mode, labels_data, target_names, images, emb_data, k_neighbors):
        if labels_data is None:
            return html.Div(), html.P("Select a dataset to begin.")
        labels = np.asarray(labels_data)
        sel = sel or []
        instructions = None
        components = []
        if mode == "tree":
            instructions = html.P("Select 1 point to view its lineage.")
            if sel:
                pass
            return html.Div(components), instructions
        if mode == "compare":
            instructions = html.P("Select up to 5 points to compare.")
            for idx in sel:
                label = target_names[labels[idx]] if target_names is not None else labels[idx]
                components.append(
                    html.Div([
                        html.Button(
                            "×",
                            id={"type": "close-button", "index": idx},
                            style={
                                "position": "absolute",
                                "top": "0.25rem",
                                "right": "0.25rem",
                                "width": "1.5rem",
                                "height": "1.5rem",
                                "borderRadius": "50%",
                                "border": "none",
                                "backgroundColor": "#ff4444",
                                "color": "white",
                                "fontSize": "1rem",
                                "lineHeight": "1",
                                "cursor": "pointer",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "padding": "0",
                                "transition": "background-color 0.2s",
                                "hover": {"backgroundColor": "#cc0000"}
                            }
                        ),
                        _create_img_tag(idx, images),
                        html.P(f"Point {idx} label: {label}")
                    ], style={"display": "flex", "alignItems": "center", "padding": "0.5rem", "borderRadius": "4px", "position": "relative", "backgroundColor": "#f8f9fa", "marginBottom": "0.5rem"})
                )
            return html.Div(components), instructions
        if mode == "interpolate":
            instructions = html.P("Select two distinct points to interpolate.")
            if sel:
                i, j = (sel[0], sel[1]) if len(sel) > 1 else (sel[0], None)
                li = target_names[labels[i]] if target_names is not None else labels[i]
                components.append(
                    html.Div([
                        html.Button(
                            "×",
                            id={"type": "close-button", "index": i},
                            style={
                                "position": "absolute",
                                "top": "0.25rem",
                                "right": "0.25rem",
                                "width": "1.5rem",
                                "height": "1.5rem",
                                "borderRadius": "50%",
                                "border": "none",
                                "backgroundColor": "#ff4444",
                                "color": "white",
                                "fontSize": "1rem",
                                "lineHeight": "1",
                                "cursor": "pointer",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "padding": "0",
                                "transition": "background-color 0.2s",
                                "hover": {"backgroundColor": "#cc0000"}
                            }
                        ),
                        _create_img_tag(i, images),
                        html.P(f"Point {i} label: {li}")
                    ], style={"display": "flex", "alignItems": "center", "padding": "0.5rem", "borderRadius": "4px", "position": "relative", "backgroundColor": "#f8f9fa", "marginBottom": "0.5rem"})
                )
                if j is not None:
                    lj = target_names[labels[j]] if target_names is not None else labels[j]
                    components.append(
                        html.Div([
                            html.Button(
                                "×",
                                id={"type": "close-button", "index": j},
                                style={
                                    "position": "absolute",
                                    "top": "0.25rem",
                                    "right": "0.25rem",
                                    "width": "1.5rem",
                                    "height": "1.5rem",
                                    "borderRadius": "50%",
                                    "border": "none",
                                    "backgroundColor": "#ff4444",
                                    "color": "white",
                                    "fontSize": "1rem",
                                    "lineHeight": "1",
                                    "cursor": "pointer",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "padding": "0",
                                    "transition": "background-color 0.2s",
                                    "hover": {"backgroundColor": "#cc0000"}
                                }
                            ),
                            _create_img_tag(j, images),
                            html.P(f"Point {j} label: {lj}")
                        ], style={"display": "flex", "alignItems": "center", "padding": "0.5rem", "borderRadius": "4px", "position": "relative", "backgroundColor": "#f8f9fa", "marginBottom": "0.5rem"})
                    )
            if interpolated_point is not None and len(sel) == 2:
                interpolated_img = _create_interpolated_img_tag(sel[0], sel[1], t_value, images)
                components.append(
                    html.Div([
                        interpolated_img,
                        html.P(f"Interpolated point (t={t_value:.1f})"),
                    ], style={"display": "flex", "alignItems": "center", "marginTop": "0.5rem"})
                )
            return html.Div(components), instructions
        if mode == "neighbors":
            instructions = html.P("Select one point to view its neighbors.")
            if not sel or len(sel) != 1:
                return html.Div(), instructions
            if emb_data is None:
                components.append(html.P("Embedding not available."))
                return html.Div(components), instructions
            emb = np.asarray(emb_data, dtype=np.float32)
            dists = np.linalg.norm(emb - emb[sel[0]], axis=1)
            neighbors = np.argsort(dists)
            neighbors = neighbors[neighbors != sel[0]][:k_neighbors]
            label = target_names[labels[sel[0]]] if target_names is not None else labels[sel[0]]
            components.append(
                html.Div([
                    html.Button(
                        "×",
                        id={"type": "close-button", "index": sel[0]},
                        style={
                            "position": "absolute",
                            "top": "0.25rem",
                            "right": "0.25rem",
                            "width": "1.5rem",
                            "height": "1.5rem",
                            "borderRadius": "50%",
                            "border": "none",
                            "backgroundColor": "#ff4444",
                            "color": "white",
                            "fontSize": "1rem",
                            "lineHeight": "1",
                            "cursor": "pointer",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "padding": "0",
                            "transition": "background-color 0.2s",
                            "hover": {"backgroundColor": "#cc0000"}
                        }
                    ),
                    _create_img_tag(sel[0], images),
                    html.P(f"Selected point {sel[0]} label: {label}")
                ], style={"display": "flex", "alignItems": "center", "padding": "0.5rem", "borderRadius": "4px", "position": "relative", "backgroundColor": "#e0f7fa", "marginBottom": "0.5rem"})
            )
            if len(neighbors) > 0:
                components.append(html.H6("Neighbors:", style={"margin": "1rem 0 0.5rem 0", "color": "#666"}))
                for nidx in neighbors:
                    nlabel = target_names[labels[nidx]] if target_names is not None else labels[nidx]
                    components.append(
                        html.Div([
                            _create_img_tag(nidx, images),
                            html.P(f"Neighbor {nidx} label: {nlabel}")
                        ], style={"display": "flex", "alignItems": "center", "padding": "0.5rem", "borderRadius": "4px", "backgroundColor": "#f8f9fa", "marginBottom": "0.5rem"})
                    )
            else:
                components.append(html.P("No neighbors found.", style={"color": "#666", "fontStyle": "italic"}))
            return html.Div(components), instructions
        return html.Div(), instructions

    @app.callback(
        Output("cmp-header", "children"),
        Input("mode", "data"),
    )
    def _cmp_header(mode):
        if mode == "compare":
            return html.H4("Point comparison")
        elif mode == "interpolate":
            return html.H4("Interpolation")
        elif mode == "tree":
            return html.H4("Tree Traversal")
        elif mode == "neighbors":
            return html.H4("Neighbors")
        else:
            return html.H4("Point comparison")

    # ... (other callbacks: _compute, _select, _update_button_state, _interpolate, _update_tree_view, _update_mode, _compare, _cmp_header) ...