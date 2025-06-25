import dash
from dash import Input, Output, State, callback_context, html, dcc
import numpy as np
import plotly.graph_objs as go
from .projection import _interpolate_hyperbolic

from .image_utils import _encode_image, _create_img_tag, _create_interpolated_img_tag, _create_content_element
from .layout import _tree_node
import json

# All callback functions and register_callbacks go here
# ... (move all @app.callback functions and register_callbacks from main.py) ... 

def _create_simple_scatter(x, y, labels, target_names, emb_labels, title):
    """Create a simple scatter plot without interactions for comparison views"""
    # Define colors matching plotting_utils.py
    colors = {
        'child_image': '#1f77b4',    # tab:blue
        'parent_image': '#ff7f0e',   # tab:orange
        'child_text': '#2ca02c',     # tab:green  
        'parent_text': '#d62728'     # tab:red
    }
    
    traces = []
    
    # If we have emb_labels, create separate traces for each label type
    if emb_labels and len(emb_labels) == len(x):
        unique_label_types = sorted(set(emb_labels))
        
        for label_type in unique_label_types:
            # Find indices for this label type
            indices = [i for i, lbl in enumerate(emb_labels) if lbl == label_type]
            
            if indices:
                x_coords = [x[i] for i in indices]
                y_coords = [y[i] for i in indices]
                hover_text = [
                    f"{i}"
                    for i in indices
                ]
                
                trace = go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    text=hover_text,
                    hoverinfo="text",
                    marker=dict(
                        size=6, 
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
        trace = go.Scatter(
            x=x,
            y=y,
            mode="markers",
            text=[
                f"{i}"
                for i in range(len(x))
            ],
            hoverinfo="text",
            marker=dict(size=6, opacity=0.7, color=labels, colorscale="Viridis"),
            name="Data points",
            showlegend=False,
        )
        traces = [trace]
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, b=60, t=40),
        uirevision="embedding",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        dragmode='pan',
    )
    return fig

def _create_interactive_scatter(x, y, labels, target_names, emb_labels, title, sel):
    """Create an interactive scatter plot with selection highlighting for comparison views"""
    # Define colors matching plotting_utils.py
    colors = {
        'child_image': '#1f77b4',    # tab:blue
        'parent_image': '#ff7f0e',   # tab:orange
        'child_text': '#2ca02c',     # tab:green  
        'parent_text': '#d62728'     # tab:red
    }
    
    traces = []
    
    # If we have emb_labels, create separate traces for each label type
    if emb_labels and len(emb_labels) == len(x):
        unique_label_types = sorted(set(emb_labels))
        
        for label_type in unique_label_types:
            # Find indices for this label type
            indices = [i for i, lbl in enumerate(emb_labels) if lbl == label_type]
            
            if indices:
                x_coords = [x[i] for i in indices]
                y_coords = [y[i] for i in indices]
                hover_text = [
                    f"{i}"
                    for i in indices
                ]
                
                trace = go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    text=hover_text,
                    hoverinfo="text",
                    customdata=indices,  # Store original indices for clicking
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
        trace = go.Scatter(
            x=x,
            y=y,
            mode="markers",
            text=[
                f"{i}"
                for i in range(len(x))
            ],
            hoverinfo="text",
            customdata=list(range(len(x))),  # Store original indices for clicking
            marker=dict(size=8, opacity=0.7, color=labels, colorscale="Viridis"),
            name="Data points",
            showlegend=False,
        )
        traces = [trace]
    
    # Add selected points as a separate trace
    if sel:
        selected_x = [x[i] for i in sel if i < len(x)]
        selected_y = [y[i] for i in sel if i < len(x)]
        
        if selected_x:
            selected_trace = go.Scatter(
                x=selected_x,
                y=selected_y,
                mode="markers",
                marker=dict(size=12, color="red", symbol="circle-open", line=dict(width=3)),
                name="Selected points",
                showlegend=False,
                hoverinfo="skip",
            )
            traces.append(selected_trace)
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, b=60, t=40),
        uirevision="embedding",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        dragmode='pan',
    )
    return fig

def _create_full_interactive_scatter(x, y, labels, target_names, emb_labels, title, sel, neighbor_indices, tree_connections, interp_point, mode):
    """Create a full interactive scatter plot with all mode features for comparison views"""
    # Define colors matching plotting_utils.py
    colors = {
        'child_image': '#1f77b4',    # tab:blue
        'parent_image': '#ff7f0e',   # tab:orange
        'child_text': '#2ca02c',     # tab:green  
        'parent_text': '#d62728'     # tab:red
    }
    
    traces = []
    neighbor_set = set(neighbor_indices) if neighbor_indices is not None else set()
    
    # Add interpolated path FIRST (like single mode) so colored points appear on top
    if interp_point is not None:
        if len(interp_point.shape) == 2 and interp_point.shape[0] > 1:
            # Multiple points forming a path - match single mode styling exactly
            # Plot orange diamond markers for all traversal points
            traces.append(
                go.Scatter(
                    x=interp_point[:, 0],
                    y=interp_point[:, 1],
                    mode="markers",
                    marker=dict(size=12, color="orange", symbol="diamond"),
                    name="Traversal Path Points",
                    text=[f"Traversal {i}" for i in range(len(interp_point))],
                    hoverinfo="text",
                    showlegend=False,
                )
            )
            # Plot a dashed orange line between every two adjacent points
            for i in range(len(interp_point) - 1):
                traces.append(
                    go.Scatter(
                        x=[interp_point[i, 0], interp_point[i+1, 0]],
                        y=[interp_point[i, 1], interp_point[i+1, 1]],
                        mode="lines",
                        line=dict(color="orange", width=2, dash="dash"),
                        name="Traversal Segment" if i == 0 else None,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
        else:
            # Single interpolated point
            traces.append(
                go.Scatter(
                    x=[interp_point[0]],
                    y=[interp_point[1]],
                    mode="markers",
                    marker=dict(size=12, color="orange", symbol="diamond"),
                    name="Interpolated point",
                    text=["Interpolated point"],
                    hoverinfo="text",
                    showlegend=False,
                )
            )
    
    # If we have emb_labels, create separate traces for each label type
    if emb_labels and len(emb_labels) == len(x):
        unique_label_types = sorted(set(emb_labels))
        
        for label_type in unique_label_types:
            # Find indices for this label type
            indices = [i for i, lbl in enumerate(emb_labels) if lbl == label_type]
            
            if indices:
                # Separate regular points and neighbor points
                regular_indices = [i for i in indices if i not in neighbor_set]
                neighbor_indices_for_type = [i for i in indices if i in neighbor_set]
                
                # Regular points trace
                if regular_indices:
                    x_coords = [x[i] for i in regular_indices]
                    y_coords = [y[i] for i in regular_indices]
                    hover_text = [
                        f"{i}"
                        for i in regular_indices
                    ]
                    
                    trace = go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode="markers",
                        text=hover_text,
                        hoverinfo="text",
                        customdata=regular_indices,  # Store original indices for clicking
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
                
                # Neighbor points trace (larger and brighter)
                if neighbor_indices_for_type:
                    x_coords_neighbors = [x[i] for i in neighbor_indices_for_type]
                    y_coords_neighbors = [y[i] for i in neighbor_indices_for_type]
                    hover_text_neighbors = [
                        f"{i} (neighbor)"
                        for i in neighbor_indices_for_type
                    ]
                    
                    neighbor_trace = go.Scatter(
                        x=x_coords_neighbors,
                        y=y_coords_neighbors,
                        mode="markers",
                        text=hover_text_neighbors,
                        hoverinfo="text",
                        customdata=neighbor_indices_for_type,
                        marker=dict(
                            size=12,  # Larger size for neighbors
                            opacity=1.0,  # Full opacity for neighbors
                            color=colors.get(label_type, 'gray'),
                            line=dict(width=2, color='purple')  # Purple border to make them stand out
                        ),
                        name=f"{label_type.replace('_', ' ').title()} (Neighbors)",
                        showlegend=False,  # Don't show in legend to avoid clutter
                    )
                    traces.append(neighbor_trace)
    else:
        # Fallback to single trace with colorscale
        regular_indices = [i for i in range(len(x)) if i not in neighbor_set]
        neighbor_indices_list = [i for i in range(len(x)) if i in neighbor_set]
        
        # Regular points trace
        if regular_indices:
            base = go.Scatter(
                x=[x[i] for i in regular_indices],
                y=[y[i] for i in regular_indices],
                mode="markers",
                text=[
                    f"{i}"
                    for i in regular_indices
                ],
                hoverinfo="text",
                customdata=regular_indices,
                marker=dict(size=8, opacity=0.7, color=[labels[i] for i in regular_indices], colorscale="Viridis"),
                name="Data points",
                showlegend=False,
            )
            traces = [base]
        else:
            traces = []
        
        # Neighbor points trace (larger and brighter)
        if neighbor_indices_list:
            neighbor_trace = go.Scatter(
                x=[x[i] for i in neighbor_indices_list],
                y=[y[i] for i in neighbor_indices_list],
                mode="markers",
                text=[
                    f"{i} (neighbor)"
                    for i in neighbor_indices_list
                ],
                hoverinfo="text",
                customdata=neighbor_indices_list,
                marker=dict(
                    size=12,  # Larger size for neighbors
                    opacity=1.0,  # Full opacity for neighbors
                    color=[labels[i] for i in neighbor_indices_list], 
                    colorscale="Viridis",
                    line=dict(width=2, color='purple')  # Purple border to make them stand out
                ),
                name="Neighbors",
                showlegend=False,
            )
            traces.append(neighbor_trace)

    # Add tree connections
    if tree_connections:
        for conn in tree_connections:
            idx1, idx2 = conn
            if idx1 < len(x) and idx2 < len(x):
                # Create line trace
                x1, y1 = x[idx1], y[idx1]
                x2, y2 = x[idx2], y[idx2]
                
                line_trace = go.Scatter(
                    x=[x1, x2],
                    y=[y1, y2],
                    mode="lines",
                    line=dict(color="gold", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Tree connections"
                )
                traces.append(line_trace)

    # Add selected points as a separate trace
    if sel:
        selected_x = [x[i] for i in sel if i < len(x)]
        selected_y = [y[i] for i in sel if i < len(x)]
        
        if selected_x:
            selected_trace = go.Scatter(
                x=selected_x,
                y=selected_y,
                mode="markers",
                marker=dict(size=12, color="red", symbol="circle-open", line=dict(width=3)),
                name="Selected points",
                showlegend=False,
                hoverinfo="skip",
            )
            traces.append(selected_trace)
    
    # Calculate the maximum distance from origin to any point for boundary circle
    max_distance = 0
    for trace in traces:
        if hasattr(trace, 'x') and hasattr(trace, 'y') and len(trace.x) > 0 and len(trace.y) > 0:
            distances = np.sqrt(np.array(trace.x)**2 + np.array(trace.y)**2)
            max_distance = max(max_distance, np.max(distances))
    
    # Add boundary circle if we have data points
    if max_distance > 0:
        circle_radius = 1.1 * max_distance
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = circle_radius * np.cos(theta)
        circle_y = circle_radius * np.sin(theta)
        
        # Add circle trace (make sure it's first so it renders behind other traces)
        circle_trace = go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hoverinfo='skip'
        )
        traces.insert(0, circle_trace)

    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, b=60, t=40),
        uirevision="embedding",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        dragmode='pan',
    )
    return fig

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
        Input("proj", "data"),
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
            
            # Create points list from embeddings, grouping them properly by tree
            points = []
            synset_ids = []
            images_list = []
            
            # Get the embedding labels to understand what each embedding represents
            embedding_labels = emb_data.get("labels", [])
            
            # Create a mapping from tree components to actual trees
            # Distribute embeddings to ensure each tree has different types
            trees_list = list(meta_data_trees["trees"].items())
            
            # Group embeddings by type first
            embeddings_by_type = {}
            for i, label in enumerate(embedding_labels):
                if label not in embeddings_by_type:
                    embeddings_by_type[label] = []
                embeddings_by_type[label].append(i)
            
            # Distribute each type across trees to create mixed trees
            # Use the number of parent_text embeddings to determine how many trees we need
            num_trees = len(embeddings_by_type.get('parent_text', []))
            
            for i, (embedding_label) in enumerate(embedding_labels):
                # For each embedding type, distribute across trees
                type_embeddings = embeddings_by_type[embedding_label]
                position_in_type = type_embeddings.index(i)
                tree_idx = position_in_type % num_trees
                tree_id, tree_data = trees_list[tree_idx]
                
                synset_id = tree_data["synset_id"]
                synset_ids.append(synset_id)
                
                # Get appropriate image path based on embedding type
                image_path = None
                if embedding_label == "child_image" and "child_images" in tree_data and tree_data["child_images"]:
                    original_path = tree_data["child_images"][0]["path"]
                    image_path = original_path.replace("/data/", "/trees/")
                elif embedding_label in ["parent_text", "child_text"]:
                    # For text embeddings, still use child image for display
                    if "child_images" in tree_data and tree_data["child_images"]:
                        original_path = tree_data["child_images"][0]["path"]
                        image_path = original_path.replace("/data/", "/trees/")
                
                points.append({
                    "synset_id": synset_id,
                    "tree_id": tree_id,
                    "image_path": image_path,
                    "kind": "tree",
                    "embedding_type": embedding_label
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
                embeddings.tolist(),  # data-store - needed for neighbor calculations
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
            
            # Create points list from embeddings, grouping them properly by tree
            points = []
            synset_ids = []
            images_list = []
            
            # Get the embedding labels to understand what each embedding represents
            embedding_labels = emb_data.get("labels", [])
            
            # Create a mapping from tree components to actual trees
            # Distribute embeddings to ensure each tree has different types
            trees_list = list(meta_data_trees["trees"].items())
            
            # Group embeddings by type first
            embeddings_by_type = {}
            for i, label in enumerate(embedding_labels):
                if label not in embeddings_by_type:
                    embeddings_by_type[label] = []
                embeddings_by_type[label].append(i)
            
            # Distribute each type across trees to create mixed trees
            # Use the number of parent_text embeddings to determine how many trees we need
            num_trees = len(embeddings_by_type.get('parent_text', []))
            
            for i, (embedding_label) in enumerate(embedding_labels):
                # For each embedding type, distribute across trees
                type_embeddings = embeddings_by_type[embedding_label]
                position_in_type = type_embeddings.index(i)
                tree_idx = position_in_type % num_trees
                tree_id, tree_data = trees_list[tree_idx]
                
                # GRIT uses sample_key instead of synset_id
                sample_key = tree_data.get("sample_key", tree_id)
                synset_ids.append(sample_key)
                
                # Get appropriate image path based on embedding type
                image_path = None
                if embedding_label == "child_image" and "child_images" in tree_data and tree_data["child_images"]:
                    original_path = tree_data["child_images"][0]["path"]
                    image_path = original_path.replace("/data/", "/trees/")
                elif embedding_label == "parent_image" and "parent_images" in tree_data and tree_data["parent_images"]:
                    original_path = tree_data["parent_images"][0]["path"]
                    image_path = original_path.replace("/data/", "/trees/")
                elif embedding_label in ["parent_text", "child_text"]:
                    # For text embeddings, use child image for display if available
                    if "child_images" in tree_data and tree_data["child_images"]:
                        original_path = tree_data["child_images"][0]["path"]
                        image_path = original_path.replace("/data/", "/trees/")
                    elif "parent_images" in tree_data and tree_data["parent_images"]:
                        original_path = tree_data["parent_images"][0]["path"]
                        image_path = original_path.replace("/data/", "/trees/")
                
                points.append({
                    "synset_id": sample_key,
                    "tree_id": tree_id,
                    "image_path": image_path,
                    "kind": "tree",
                    "embedding_type": embedding_label
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
                embeddings.tolist(),  # data-store - needed for neighbor calculations
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
        Input("proj", "data"),
        Input("interpolated-point", "data"),
        State("labels-store", "data"),
        State("target-names-store", "data"),
        Input("mode", "data"),
        Input("neighbors-slider", "value"),
        State("data-store", "data"),
        Input("dataset-dropdown", "value"),
        State("points-store", "data"),
    )
    def _scatter(edata, sel, proj, traversal_path, labels_data, target_names, mode, k_neighbors, data_store, dataset_name, points):
        if edata is None or labels_data is None:
            print("Warning: No embedding or label data available for plotting")
            return {}
        emb = np.asarray(edata, dtype=np.float32)
        labels = np.asarray(labels_data, dtype=int)
        sel = sel or []
        highlight = sel
        neighbor_indices = []
        selected_idx = sel
        tree_connections = []
        interpolation_lines = []
        interpolation_highlight = []
        traversal_points = None


        if mode == "interpolate" and traversal_path is not None and len(traversal_path) > 0:
            # traversal_path is a list of indices; convert to actual points
            traversal_points = np.asarray([emb[idx] for idx in traversal_path if idx < len(emb)])

        if mode == "neighbors" and sel and len(sel) == 1:
            selected_idx = sel[:1]
            if data_store is not None:
                data_np = np.asarray(data_store, dtype=np.float32)
                dists = np.linalg.norm(data_np - data_np[sel[0]], axis=1)
                neighbor_indices = np.argsort(dists)
                neighbor_indices = neighbor_indices[neighbor_indices != sel[0]][:k_neighbors]
            else:
                neighbor_indices = []
        elif mode == "tree" and sel and len(sel) == 1:
            selected_idx = sel[:1]
            tree_connections = []
            try:
                selected_pt = points[sel[0]]
                selected_tree_id = selected_pt.get("tree_id", "?")
                tree_point_indices = []
                tree_points_by_type = {}
                for i, pt in enumerate(points):
                    if pt.get("tree_id") == selected_tree_id:
                        tree_point_indices.append(i)
                        emb_type = pt.get("embedding_type", "unknown")
                        if emb_type not in tree_points_by_type:
                            tree_points_by_type[emb_type] = []
                        tree_points_by_type[emb_type].append(i)
                if dataset_name == "imagenet":
                    level_order = ['parent_text', 'child_text', 'child_image']
                else:
                    level_order = ['parent_text', 'child_text', 'parent_image', 'child_image']
                for i in range(len(level_order) - 1):
                    current_level = level_order[i]
                    next_level = level_order[i + 1]
                    if current_level in tree_points_by_type and next_level in tree_points_by_type:
                        for curr_pt in tree_points_by_type[current_level]:
                            for next_pt in tree_points_by_type[next_level]:
                                tree_connections.append((curr_pt, next_pt))
                neighbor_indices = tree_point_indices
            except Exception as e:
                print(f"Error finding tree points: {e}")
                neighbor_indices = []
                tree_connections = []
        else:
            neighbor_indices = []

        def _fig_disk(x, y, sel, labels, target_names, traversal_points=None, neighbor_indices=None, emb_labels=None, tree_connections=None, points=None):
            colors = {
                'child_image': '#1f77b4',    # tab:blue
                'parent_image': '#ff7f0e',   # tab:orange
                'child_text': '#2ca02c',     # tab:green  
                'parent_text': '#d62728'     # tab:red
            }
            traces = []
            neighbor_set = set(neighbor_indices) if neighbor_indices is not None else set()
            tree_arrow_annotations = []

            # Highlight traversal_points in interpolation mode
            if traversal_points is not None and len(traversal_points) > 0:
                # Project traversal_points to disk coordinates
                traversal_points = np.asarray(traversal_points)
                if traversal_points.shape[1] > 2:
                    traversal_x = traversal_points[:, 0] / (1.0 + traversal_points[:, 2])
                    traversal_y = traversal_points[:, 1] / (1.0 + traversal_points[:, 2])
                else:
                    traversal_x = traversal_points[:, 0]
                    traversal_y = traversal_points[:, 1]
                # Plot orange diamond markers for all traversal points
                traces.append(
                    go.Scatter(
                        x=traversal_x,
                        y=traversal_y,
                        mode="markers",
                        marker=dict(size=12, color="orange", symbol="diamond"),
                        name="Traversal Path Points",
                        text=[f"Traversal {i}" for i in range(len(traversal_points))],
                        hoverinfo="text",
                        showlegend=False,
                    )
                )
                # Plot a dashed orange line between every two adjacent points
                for i in range(len(traversal_x) - 1):
                    traces.append(
                        go.Scatter(
                            x=[traversal_x[i], traversal_x[i+1]],
                            y=[traversal_y[i], traversal_y[i+1]],
                            mode="lines",
                            line=dict(color="orange", width=2, dash="dash"),
                            name="Traversal Segment" if i == 0 else None,
                            showlegend=False,
                        )
                    )

            # If we have emb_labels, create separate traces for each label type
            if emb_labels and len(emb_labels) == len(x):
                unique_label_types = sorted(set(emb_labels))
                
                for label_type in unique_label_types:
                    # Find indices for this label type
                    indices = [i for i, lbl in enumerate(emb_labels) if lbl == label_type]
                    
                    if indices:
                        # Separate regular points and neighbor points (neighbors can be tree points in tree mode)
                        regular_indices = [i for i in indices if i not in neighbor_set]
                        neighbor_indices_for_type = [i for i in indices if i in neighbor_set]
                        
                        # Regular points trace
                        if regular_indices:
                            x_coords = [x[i] for i in regular_indices]
                            y_coords = [y[i] for i in regular_indices]
                            hover_text = [
                                f"{i}"
                                for i in regular_indices
                            ]
                            
                            trace = go.Scatter(
                                x=x_coords,
                                y=y_coords,
                                mode="markers",
                                text=hover_text,
                                hoverinfo="text",
                                customdata=regular_indices,  # Store original indices directly
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
                        

                        
                        # Neighbor points trace (larger and brighter)
                        if neighbor_indices_for_type:
                            x_coords_neighbors = [x[i] for i in neighbor_indices_for_type]
                            y_coords_neighbors = [y[i] for i in neighbor_indices_for_type]
                            hover_text_neighbors = [
                                f"{i} (neighbor)"
                                for i in neighbor_indices_for_type
                            ]
                            
                            neighbor_trace = go.Scatter(
                                x=x_coords_neighbors,
                                y=y_coords_neighbors,
                                mode="markers",
                                text=hover_text_neighbors,
                                hoverinfo="text",
                                customdata=neighbor_indices_for_type,
                                marker=dict(
                                    size=12,  # Larger size for neighbors
                                    opacity=1.0,  # Full opacity for neighbors
                                    color=colors.get(label_type, 'gray'),
                                    line=dict(width=2, color='purple')  # Purple border to make them stand out
                                ),
                                name=f"{label_type.replace('_', ' ').title()} (Neighbors)",
                                showlegend=False,  # Don't show in legend to avoid clutter
                            )
                            traces.append(neighbor_trace)
            else:
                # Fallback to single trace with colorscale
                
                # Separate regular points and neighbor points
                regular_indices = [i for i in range(len(x)) if i not in neighbor_set]
                neighbor_indices_list = [i for i in range(len(x)) if i in neighbor_set]
                
                # Regular points trace
                if regular_indices:
                    base = go.Scatter(
                        x=[x[i] for i in regular_indices],
                        y=[y[i] for i in regular_indices],
                        mode="markers",
                        text=[
                            f"{i}"
                            for i in regular_indices
                        ],
                        hoverinfo="text",
                        customdata=regular_indices,
                        marker=dict(size=8, opacity=0.7, color=[labels[i] for i in regular_indices], colorscale="Viridis"),
                        name="Data points",
                        showlegend=False,
                    )
                    traces = [base]
                else:
                    traces = []
                

                
                # Neighbor points trace (larger and brighter)
                if neighbor_indices_list:
                    neighbor_trace = go.Scatter(
                        x=[x[i] for i in neighbor_indices_list],
                        y=[y[i] for i in neighbor_indices_list],
                        mode="markers",
                        text=[
                            f"{i} (neighbor)"
                            for i in neighbor_indices_list
                        ],
                        hoverinfo="text",
                        customdata=neighbor_indices_list,
                        marker=dict(
                            size=12,  # Larger size for neighbors
                            opacity=1.0,  # Full opacity for neighbors
                            color=[labels[i] for i in neighbor_indices_list], 
                            colorscale="Viridis",
                            line=dict(width=2, color='purple')  # Purple border to make them stand out
                        ),
                        name="Neighbors",
                        showlegend=False,
                    )
                    traces.append(neighbor_trace)

            # Store tree connections for later arrow annotation
            tree_arrow_annotations = []
            if tree_connections and points:
                for conn in tree_connections:
                    idx1, idx2 = conn
                    if idx1 < len(x) and idx2 < len(x):
                        # Create line trace
                        x1, y1 = x[idx1], y[idx1]
                        x2, y2 = x[idx2], y[idx2]
                        
                        line_trace = go.Scatter(
                            x=[x1, x2],
                            y=[y1, y2],
                            mode="lines",
                            line=dict(color="gold", width=2),
                            hoverinfo="skip",
                            showlegend=False,
                            name="Tree connections"
                        )
                        traces.append(line_trace)
                        
                        # Store arrow annotation info
                        tree_arrow_annotations.append({
                            'x': x2, 'y': y2,
                            'ax': x1, 'ay': y1,
                            'xref': 'x', 'yref': 'y',
                            'axref': 'x', 'ayref': 'y',
                            'arrowhead': 2,
                            'arrowsize': 1.5,
                            'arrowwidth': 2,
                            'arrowcolor': 'gold',
                            'showarrow': True,
                            'text': '',
                        })

            # if traversal_points is not None:
            #     traces.append(
            #         go.Scatter(
            #             x=traversal_points[:, 0],
            #             y=traversal_points[:, 1],
            #             mode="markers",
            #             marker=dict(size=12, color="orange", symbol="diamond"),
            #             name="Interpolated point",
            #             text=["Interpolated point"],
            #             hoverinfo="text",
            #         )
            #     )

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
            
            # Calculate the maximum distance from origin to any point for boundary circle
            max_distance = 0
            for trace in traces:
                if hasattr(trace, 'x') and hasattr(trace, 'y') and len(trace.x) > 0 and len(trace.y) > 0:
                    distances = np.sqrt(np.array(trace.x)**2 + np.array(trace.y)**2)
                    max_distance = max(max_distance, np.max(distances))
            
            # Add boundary circle if we have data points
            if max_distance > 0:
                circle_radius = 1.1 * max_distance
                theta = np.linspace(0, 2*np.pi, 100)
                circle_x = circle_radius * np.cos(theta)
                circle_y = circle_radius * np.sin(theta)
                
                # Add circle trace (make sure it's first so it renders behind other traces)
                circle_trace = go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode='lines',
                    line=dict(color='lightgray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                )
                traces.insert(0, circle_trace)

            fig = go.Figure(data=traces)
            
            # Add arrow annotations for tree connections
            annotations = tree_arrow_annotations
            
            fig.update_layout(
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(scaleanchor="x", scaleratio=1),
                margin=dict(l=0, r=0, b=60, t=30),
                uirevision="embedding",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                ),
                annotations=annotations,
                dragmode='pan',
            )
            return fig

        # Handle 2D embeddings for disk projection
        xh, yh = emb[:, 0], emb[:, 1]
        if emb.shape[1] > 2:
            zh = emb[:, 2]
        else:
            zh = np.zeros(emb.shape[0])  # For 2D embeddings, use z=0
        dx, dy = xh / (1.0 + zh), yh / (1.0 + zh)
        emb_labels = None
        if dataset_name and proj:
            try:
                import pickle
                dataset_dir = {"imagenet": "ImageNet", "grit": "GRIT"}.get(dataset_name, dataset_name)
                emb_file = f"hierchical_datasets/{dataset_dir}/{proj}_embeddings.pkl"
                with open(emb_file, "rb") as f:
                    emb_data_loaded = pickle.load(f)
                emb_labels = emb_data_loaded.get("labels", [])
            except Exception as e:
                emb_labels = []
        fig_disk = _fig_disk(dx, dy, selected_idx if mode == "neighbors" else highlight, labels, target_names, traversal_points=traversal_points, neighbor_indices=neighbor_indices, emb_labels=emb_labels, tree_connections=tree_connections, points=points)
        return fig_disk

    @app.callback(
        Output("sel", "data"),
        [
            Input("scatter-disk", "clickData"),
            Input("scatter-disk-1", "clickData"),
            Input("scatter-disk-2", "clickData"),
            Input({"type": "close-button", "index": dash.ALL}, "n_clicks")
        ],
        State("sel", "data"),
        State("mode", "data"),
        prevent_initial_call=True,
    )
    def _select(click_disk, click_disk_1, click_disk_2, close_clicks, sel, mode):
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
                
                # Check if we have customdata with original indices
                if "customdata" in pt and pt["customdata"] is not None:
                    return int(pt["customdata"])
                
                # Fallback to original logic for single trace
                if curve_number != 0:
                    return None
                return point_index
            except (TypeError, KeyError, IndexError):
                return None
        
        if triggered_id in ["scatter-disk", "scatter-disk-1", "scatter-disk-2"]:
            # Handle clicks from any of the scatter plots
            if triggered_id == "scatter-disk":
                click_data = ctx.inputs["scatter-disk.clickData"]
            elif triggered_id == "scatter-disk-1":
                click_data = ctx.inputs["scatter-disk-1.clickData"]
            else:  # scatter-disk-2
                click_data = ctx.inputs["scatter-disk-2.clickData"]
            
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
        Input("interpolation-slider", "value"),
        State("sel", "data"),
        State("proj", "data"),
        State("dataset-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _interpolate(n_clicks, t, sel, proj, dataset_name):
        if not (n_clicks and sel and len(sel) == 2 and proj and dataset_name):
            return None
        
        # Load the embeddings for the selected projection method
        try:
            import pickle
            dataset_dir = {"imagenet": "ImageNet", "grit": "GRIT"}.get(dataset_name, dataset_name)
            emb_file = f"hierchical_datasets/{dataset_dir}/{proj}_embeddings.pkl"
            
            with open(emb_file, "rb") as f_emb:
                emb_data = pickle.load(f_emb)
            
            emb = np.array(emb_data["embeddings"], dtype=np.float32)
        except Exception as e:
            print(f"Error loading embeddings for interpolation: {e}")
            return None
        
        i, j = sel[:2]
        p1, p2 = emb[i], emb[j]
        traversal_path = _interpolate_hyperbolic(p1, p2, emb, model='tmp', steps=t)
        return traversal_path
        

    @app.callback(
        [Output("tree-levels-above", "children"), Output("tree-selected-level", "children"), Output("tree-levels-below", "children")],
        Input("sel", "data"),
        Input("mode", "data"),
        State("meta-store", "data"),
        State("points-store", "data"),
        Input("dataset-dropdown", "value"),
        Input("proj", "data"),
    )
    def _update_tree_view(sel, mode, meta, points, dataset_name, proj):
        if mode != "tree" or not sel or len(sel) != 1 or meta is None or points is None:
            return html.Span(), html.Span(), html.Span()
        
        idx = sel[0]
        
        # Load embedding labels to determine the level of the selected point
        emb_labels = None
        if dataset_name and proj:
            try:
                import pickle
                dataset_dir = {"imagenet": "ImageNet", "grit": "GRIT"}.get(dataset_name, dataset_name)
                emb_file = f"hierchical_datasets/{dataset_dir}/{proj}_embeddings.pkl"
                with open(emb_file, "rb") as f:
                    emb_data_loaded = pickle.load(f)
                emb_labels = emb_data_loaded.get("labels", [])
            except Exception as e:
                emb_labels = []
        
        if not emb_labels or idx >= len(emb_labels):
            return html.Span(), html.Span(), html.Span()
        
        # Load the tree data to get proper image paths
        tree_data = None
        try:
            import json
            dataset_dir = {"imagenet": "ImageNet", "grit": "GRIT"}.get(dataset_name, dataset_name)
            meta_file = f"hierchical_datasets/{dataset_dir}/meta_data_trees.json"
            with open(meta_file, "r") as f:
                meta_data_trees = json.load(f)
            
            # Find the tree for this point
            pt = points[idx]
            synset_id = pt.get("synset_id", "?")
            
            # Find the tree that matches this synset_id
            for tree_id, tree_info in meta_data_trees["trees"].items():
                if (dataset_name == "imagenet" and tree_info.get("synset_id") == synset_id) or \
                   (dataset_name == "grit" and tree_info.get("sample_key") == synset_id):
                    tree_data = tree_info
                    break
                    
        except Exception as e:
            print(f"Error loading tree data: {e}")
        
        # Define the hierarchy levels based on dataset
        if dataset_name == "imagenet":
            # ImageNet only has 3 levels (no parent_image)
            level_mapping = {
                'parent_text': 1,
                'child_text': 2, 
                'child_image': 3,
            }
            level_names = {
                1: "Level 1",
                2: "Level 2", 
                3: "Level 3",
            }
            max_level = 3
        else:  # GRIT
            level_mapping = {
                'parent_text': 1,
                'child_text': 2, 
                'parent_image': 3,
                'child_image': 4
            }
            level_names = {
                1: "Level 1",
                2: "Level 2", 
                3: "Level 3",
                4: "Level 4"
            }
            max_level = 4
        
        # Get the level of the selected point
        selected_label = emb_labels[idx]
        selected_level = level_mapping.get(selected_label, 0)
        
        if selected_level == 0:
            return html.Span(), html.Span(), html.Span()
        
        try:
            pt = points[idx]
            synset_id = pt.get("synset_id", "?")
            meta_row = meta.get(synset_id, {}) if isinstance(meta, dict) else {}
        except (IndexError, TypeError):
            return html.Span(), html.Span(), html.Span()
        
        # Find all points in the same tree that are actually plotted
        tree_point_indices = []
        tree_points_by_level = {}
        
        for i, point in enumerate(points):
            if i < len(emb_labels):
                # For ImageNet: points in same tree share same synset_id
                # For GRIT: points in same tree might have different sample_key but same tree structure
                point_synset = point.get("synset_id", "?")
                
                # Check if this point belongs to the same tree
                is_same_tree = False
                if dataset_name == "imagenet":
                    is_same_tree = (point_synset == synset_id)
                else:  # GRIT
                    is_same_tree = (point_synset == synset_id)
                
                if is_same_tree:
                    # This point is in the same tree and is plotted
                    tree_point_indices.append(i)
                    
                    # Group by level
                    point_level_label = emb_labels[i]
                    point_level = level_mapping.get(point_level_label, 0)
                    if point_level > 0:
                        if point_level not in tree_points_by_level:
                            tree_points_by_level[point_level] = []
                        tree_points_by_level[point_level].append(i)
        
        # Create level components
        def create_level_component(level, content, is_selected=False):
            border_color = "#28a745" if is_selected else "#6c757d"
            bg_color = "#d4edda" if is_selected else "#f8f9fa"
            
            return html.Div([
                html.H5(level_names[level], style={
                    "margin": "0 0 0.5rem 0", 
                    "color": "#28a745" if is_selected else "#6c757d",
                    "fontWeight": "bold" if is_selected else "normal"
                }),
                content
            ], style={
                "padding": "0.75rem", 
                "backgroundColor": bg_color,
                "border": f"2px solid {border_color}",
                "borderRadius": "6px", 
                "marginBottom": "0.5rem",
                "marginLeft": f"{(level-1) * 1}rem"
            })
        
        # Build the hierarchy display
        levels_above = []
        current_level = None
        levels_below = []
        
        # Level 1: Parent Text
        if selected_level == 1:
            current_level = create_level_component(1, html.P(meta_row.get("name", synset_id), style={"margin": 0, "fontWeight": "bold"}), True)
        elif selected_level > 1:
            levels_above.append(create_level_component(1, html.P(meta_row.get("name", synset_id), style={"margin": 0})))
        
        # Level 2: Child Text  
        if selected_level == 2:
            current_level = create_level_component(2, html.P(meta_row.get("description", "(no description)"), style={"margin": 0, "fontWeight": "bold"}), True)
        elif selected_level > 2:
            levels_above.append(create_level_component(2, html.P(meta_row.get("description", "(no description)"), style={"margin": 0})))
        elif selected_level < 2:
            levels_below.append(create_level_component(2, html.P(meta_row.get("description", "(no description)"), style={"margin": 0, "color": "#6c757d"})))
        
        # Level 3: Parent Image (GRIT) or Child Image (ImageNet)
        if dataset_name == "imagenet":
            # For ImageNet, level 3 is child image - show only child images that are actually plotted
            child_img_components = []
            
            # Get the plotted points at level 3 (child_image)
            level_3_points = tree_points_by_level.get(3, [])
            
            # Instead of complex path matching, just use the point indices directly
            # Show the first N images from tree data where N = number of plotted points
            if tree_data and tree_data.get("child_images") and len(level_3_points) > 0:
                # Take the first len(level_3_points) images from the tree data
                num_images_to_show = min(len(level_3_points), len(tree_data["child_images"]))
                
                for i in range(num_images_to_show):
                    child_img = tree_data["child_images"][i]
                    child_img_path = child_img["path"]
                    child_img_rel = child_img_path.replace("/data/", "/trees/")
                    child_img_src = _encode_image(child_img_rel)
                    
                    if child_img_src:
                        child_img_components.append(
                            html.Div([
                                html.Img(src=child_img_src, style={"maxWidth": "160px", "maxHeight": "160px", "objectFit": "contain", "border": "1px solid #ccc"}),
                                html.P(f"Image {i+1}/{num_images_to_show}", style={"fontSize": "0.7rem", "color": "#666", "margin": "0.2rem 0 0 0", "textAlign": "center"})
                            ], style={"margin": "0.5rem 0"})
                        )
            
            if not child_img_components:
                child_img_components = [html.P("No images available", style={"color": "#6c757d", "fontStyle": "italic"})]
            
            child_img_container = html.Div(child_img_components, style={"margin": 0, "display": "flex", "flexDirection": "column", "alignItems": "center"})
            
            if selected_level == 3:
                current_level = create_level_component(3, child_img_container, True)
            elif selected_level < 3:
                levels_below.append(create_level_component(3, child_img_container))
        else:
            # For GRIT, level 3 is parent image - show only parent images that are actually plotted
            parent_img_components = []
            
            # Get the plotted points at level 3 (parent_image)
            level_3_points = tree_points_by_level.get(3, [])
            
            # Show the first N images from tree data where N = number of plotted points
            if tree_data and tree_data.get("parent_images") and len(level_3_points) > 0:
                # Take the first len(level_3_points) images from the tree data
                num_images_to_show = min(len(level_3_points), len(tree_data["parent_images"]))
                
                for i in range(num_images_to_show):
                    parent_img = tree_data["parent_images"][i]
                    parent_img_path = parent_img["path"]
                    parent_img_rel = parent_img_path.replace("/data/", "/trees/")
                    parent_img_src = _encode_image(parent_img_rel)
                    
                    if parent_img_src:
                        parent_img_components.append(
                            html.Div([
                                html.Img(src=parent_img_src, style={"maxWidth": "160px", "maxHeight": "160px", "objectFit": "contain", "border": "1px solid #ccc"}),
                                html.P(f"Image {i+1}/{num_images_to_show}", style={"fontSize": "0.7rem", "color": "#666", "margin": "0.2rem 0 0 0", "textAlign": "center"})
                            ], style={"margin": "0.5rem 0"})
                        )
            
            if not parent_img_components:
                parent_img_components = [html.P("No images available", style={"color": "#6c757d", "fontStyle": "italic"})]
            
            parent_img_container = html.Div(parent_img_components, style={"margin": 0, "display": "flex", "flexDirection": "column", "alignItems": "center"})
            
            if selected_level == 3:
                current_level = create_level_component(3, parent_img_container, True)
            elif selected_level > 3:
                levels_above.append(create_level_component(3, parent_img_container))
            elif selected_level < 3:
                levels_below.append(create_level_component(3, parent_img_container))
        
        # Level 4: Child Image (GRIT only) - show only child images that are actually plotted
        if dataset_name == "grit" and max_level >= 4:
            child_img_components = []
            
            # Get the plotted points at level 4 (child_image)
            level_4_points = tree_points_by_level.get(4, [])
            
            # Show the first N images from tree data where N = number of plotted points
            if tree_data and tree_data.get("child_images") and len(level_4_points) > 0:
                # Take the first len(level_4_points) images from the tree data
                num_images_to_show = min(len(level_4_points), len(tree_data["child_images"]))
                
                for i in range(num_images_to_show):
                    child_img = tree_data["child_images"][i]
                    child_img_path = child_img["path"]
                    child_img_rel = child_img_path.replace("/data/", "/trees/")
                    child_img_src = _encode_image(child_img_rel)
                    
                    if child_img_src:
                        child_img_components.append(
                            html.Div([
                                html.Img(src=child_img_src, style={"maxWidth": "160px", "maxHeight": "160px", "objectFit": "contain", "border": "1px solid #ccc"}),
                                html.P(f"Image {i+1}/{num_images_to_show}", style={"fontSize": "0.7rem", "color": "#666", "margin": "0.2rem 0 0 0", "textAlign": "center"})
                            ], style={"margin": "0.5rem 0"})
                        )
            
            if not child_img_components:
                child_img_components = [html.P("No images available", style={"color": "#6c757d", "fontStyle": "italic"})]
            
            child_img_container = html.Div(child_img_components, style={"margin": 0, "display": "flex", "flexDirection": "column", "alignItems": "center"})
            
            if selected_level == 4:
                current_level = create_level_component(4, child_img_container, True)
            elif selected_level < 4:
                levels_below.append(create_level_component(4, child_img_container))
        
        # Combine all levels
        all_levels = levels_above + ([current_level] if current_level else []) + levels_below
        
        # Split into three sections for the layout
        if len(all_levels) <= 3:
            # Pad with empty divs if needed
            while len(all_levels) < 3:
                all_levels.append(html.Div())
            return all_levels[0], all_levels[1], all_levels[2]
        else:
            # If more than 3 levels, combine some
            return (
                html.Div(all_levels[:2]) if len(all_levels) > 3 else all_levels[0],
                all_levels[2] if len(all_levels) > 3 else all_levels[1], 
                html.Div(all_levels[3:]) if len(all_levels) > 3 else all_levels[2]
            )

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
        Input("comparison-mode", "data"),
        State("labels-store", "data"),
        State("target-names-store", "data"),
        State("images-store", "data"),
        State("emb", "data"),
        Input("neighbors-slider", "value"),
        State("points-store", "data"),
        State("meta-store", "data"),
    )
    def _compare(sel, traversal_path, t_value, mode, comparison_mode, labels_data, target_names, images, emb_data, k_neighbors, points, meta):
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
                # Get the content to display based on embedding type
                content_element = _create_content_element(idx, images, points, meta)
                
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
                        content_element
                    ], style={"display": "flex", "alignItems": "center", "padding": "0.5rem", "borderRadius": "4px", "position": "relative", "backgroundColor": "#f8f9fa", "marginBottom": "0.5rem"})
                )
            return html.Div(components), instructions
        if mode == "interpolate":
            instructions = html.P("Select two distinct points to interpolate.")
            if traversal_path is not None and len(traversal_path) > 0:
                for idx in traversal_path:
                    # Get the content to display based on embedding type
                    content_element = _create_content_element(idx, images, points, meta)
                    components.append(
                        html.Div([
                            content_element
                        ], style={"display": "flex", "alignItems": "center", "padding": "0.5rem", "borderRadius": "4px", "position": "relative", "backgroundColor": "#f8f9fa", "marginBottom": "0.5rem"})
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
            
            # Get the content to display based on embedding type
            content_element = _create_content_element(sel[0], images, points, meta)
            
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
                    content_element
                ], style={"display": "flex", "alignItems": "center", "padding": "0.5rem", "borderRadius": "4px", "position": "relative", "backgroundColor": "#e0f7fa", "marginBottom": "0.5rem"})
            )
            if len(neighbors) > 0:
                components.append(html.H6("Neighbors:", style={"margin": "1rem 0 0.5rem 0", "color": "#666"}))
                for nidx in neighbors:
                    # Get the content to display based on embedding type
                    neighbor_content_element = _create_content_element(nidx, images, points, meta)
                    
                    components.append(
                        html.Div([
                            neighbor_content_element
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

    @app.callback(
        Output("comparison-mode", "data"),
        Output("compare-projections-btn", "style"),
        Output("single-plot-container", "style"),
        Output("comparison-plot-container", "style"),
        Input("compare-projections-btn", "n_clicks"),
        State("comparison-mode", "data"),
        prevent_initial_call=True,
    )
    def _toggle_comparison_mode(n_clicks, current_mode):
        if not n_clicks:
            return dash.no_update
        
        new_mode = not current_mode
        
        # Button styles
        btn_style_inactive = {
            "backgroundColor": "#6c757d",
            "color": "white",
            "border": "none",
            "padding": "0.5rem 1rem",
            "borderRadius": "6px",
            "cursor": "pointer",
            "width": "100%",
            "marginBottom": "0.5rem",
            "transition": "background-color 0.2s",
        }
        btn_style_active = {
            **btn_style_inactive,
            "backgroundColor": "#28a745"
        }
        
        if new_mode:
            # Comparison mode ON
            return (
                True,
                btn_style_active,
                {"display": "none"},   # Hide single plot
                {"display": "flex"}    # Show comparison plots
            )
        else:
            # Comparison mode OFF
            return (
                False,
                btn_style_inactive,
                {                      # Show single plot
                    "display": "flex",
                    "width": "min(85vh, 50vw)",
                    "height": "min(85vh, 50vw)",
                    "aspectRatio": "1 / 1",
                    "margin": "auto",
                    "maxWidth": "100%",
                    "maxHeight": "100%",
                    "flexShrink": 0,
                    "flexGrow": 0,
                },
                {"display": "none"}    # Hide comparison plots
            )

    @app.callback(
        Output("scatter-disk-2", "figure"),
        Input("dataset-dropdown", "value"),
        Input("sel", "data"),
        Input("mode", "data"),
        Input("neighbors-slider", "value"),
        Input("interpolated-point", "data"),
        State("labels-store", "data"),
        State("target-names-store", "data"),
        State("data-store", "data"),
        State("points-store", "data"),
        Input("comparison-mode", "data"),
        State("proj", "data"),
    )
    def _scatter_plot_2(dataset_name, sel, mode, k_neighbors, traversal_path, labels_data, target_names, data_store, points, comparison_mode, selected_proj):
        if not comparison_mode or labels_data is None or not dataset_name:
            return {}
        
        # Always load CO-SNE for right plot
        try:
            import pickle
            dataset_dir = {"imagenet": "ImageNet", "grit": "GRIT"}.get(dataset_name, dataset_name)
            emb_file = f"hierchical_datasets/{dataset_dir}/cosne_embeddings.pkl"
            
            with open(emb_file, "rb") as f_emb:
                emb_data = pickle.load(f_emb)
            
            embeddings = np.array(emb_data["embeddings"], dtype=np.float32)
            emb = embeddings
        except Exception as e:
            print(f"Error loading CO-SNE embeddings: {e}")
            return {}
        
        labels = np.asarray(labels_data, dtype=int)
        sel = sel or []
        
        # Handle interpolated points - they are calculated in the space of selected_proj
        # traversal_path contains indices into the dataset, not coordinates
        interp_point = None
        if traversal_path is not None and isinstance(traversal_path, list):
            if selected_proj == "cosne":
                # Path was calculated in CO-SNE space, get the coordinates for these indices
                interp_coords = []
                for idx in traversal_path:
                    if idx < len(emb):
                        interp_coords.append(emb[idx])
                if interp_coords:
                    interp_point = np.array(interp_coords)
            else:
                # Path was calculated in HoroPCA space, but we're showing CO-SNE
                # Show the same indices but in CO-SNE coordinates
                interp_coords = []
                for idx in traversal_path:
                    if idx < len(emb):
                        interp_coords.append(emb[idx])
                if interp_coords:
                    interp_point = np.array(interp_coords)
        
        # Handle 2D embeddings for disk projection (same as main scatter)
        xh, yh = emb[:, 0], emb[:, 1]
        if emb.shape[1] > 2:
            zh = emb[:, 2]
        else:
            zh = np.zeros(emb.shape[0])
        dx, dy = xh / (1.0 + zh), yh / (1.0 + zh)
        
        # Transform interpolated path if exists
        interp_transformed = None
        if interp_point is not None:
            if len(interp_point.shape) == 2 and interp_point.shape[0] > 1:
                # Multiple points forming a path - transform each point
                interp_coords = []
                for pt in interp_point:
                    if len(pt) > 2:
                        interp_dx = pt[0] / (1.0 + pt[2])
                        interp_dy = pt[1] / (1.0 + pt[2])
                    else:
                        interp_dx, interp_dy = pt[0], pt[1]
                    interp_coords.append([interp_dx, interp_dy])
                interp_transformed = np.array(interp_coords)
            else:
                # Single point
                if len(interp_point) > 2:
                    interp_dx = interp_point[0] / (1.0 + interp_point[2])
                    interp_dy = interp_point[1] / (1.0 + interp_point[2])
                else:
                    interp_dx, interp_dy = interp_point[0], interp_point[1]
                interp_transformed = np.array([interp_dx, interp_dy])
        
        # Calculate neighbors and tree connections based on mode
        neighbor_indices = []
        tree_connections = []
        
        if mode == "neighbors" and sel and len(sel) == 1:
            if data_store is not None:
                data_np = np.asarray(data_store, dtype=np.float32)
                dists = np.linalg.norm(data_np - data_np[sel[0]], axis=1)
                neighbor_indices = np.argsort(dists)
                neighbor_indices = neighbor_indices[neighbor_indices != sel[0]][:k_neighbors]
            else:
                neighbor_indices = []
        elif mode == "tree" and sel and len(sel) == 1:
            # Find all points in the same tree and create connections between adjacent levels
            tree_connections = []
            try:
                # Get the selected point's tree
                selected_pt = points[sel[0]]
                selected_tree_id = selected_pt.get("tree_id", "?")
                
                # Find all points that belong to the same tree
                tree_point_indices = []
                tree_points_by_type = {}
                
                for i, pt in enumerate(points):
                    if pt.get("tree_id") == selected_tree_id:
                        # Include all tree points in highlighting (including selected)
                        tree_point_indices.append(i)
                        
                        # Group by embedding type for creating connections
                        emb_type = pt.get("embedding_type", "unknown")
                        if emb_type not in tree_points_by_type:
                            tree_points_by_type[emb_type] = []
                        tree_points_by_type[emb_type].append(i)
                
                # Create connections between adjacent hierarchical levels
                if dataset_name == "imagenet":
                    level_order = ['parent_text', 'child_text', 'child_image']
                else:  # GRIT
                    level_order = ['parent_text', 'child_text', 'parent_image', 'child_image']
                
                # Connect consecutive levels
                for i in range(len(level_order) - 1):
                    current_level = level_order[i]
                    next_level = level_order[i + 1]
                    
                    if current_level in tree_points_by_type and next_level in tree_points_by_type:
                        for curr_pt in tree_points_by_type[current_level]:
                            for next_pt in tree_points_by_type[next_level]:
                                tree_connections.append((curr_pt, next_pt))
                
                # Use tree points as neighbor_indices for highlighting
                neighbor_indices = tree_point_indices
                
            except Exception as e:
                print(f"Error finding tree points: {e}")
                neighbor_indices = []
                tree_connections = []
        
        # Load the original label types for color coding (always CO-SNE for right plot)
        emb_labels = emb_data.get("labels", [])
        
        # Create the figure with all mode features
        fig = _create_full_interactive_scatter(dx, dy, labels, target_names, emb_labels, "", sel, neighbor_indices, tree_connections, interp_transformed, mode)
        return fig

    @app.callback(
        Output("scatter-disk-1", "figure"),
        Input("dataset-dropdown", "value"),
        Input("sel", "data"),
        Input("mode", "data"),
        Input("neighbors-slider", "value"),
        Input("interpolated-point", "data"),
        State("labels-store", "data"),
        State("target-names-store", "data"),
        State("data-store", "data"),
        State("points-store", "data"),
        Input("comparison-mode", "data"),
        State("proj", "data"),
    )
    def _scatter_plot_1(dataset_name, sel, mode, k_neighbors, traversal_path, labels_data, target_names, data_store, points, comparison_mode, selected_proj):
        if not comparison_mode or labels_data is None or not dataset_name:
            return {}
        
        # Always load HoroPCA for left plot
        try:
            import pickle
            dataset_dir = {"imagenet": "ImageNet", "grit": "GRIT"}.get(dataset_name, dataset_name)
            emb_file = f"hierchical_datasets/{dataset_dir}/horopca_embeddings.pkl"
            
            with open(emb_file, "rb") as f_emb:
                emb_data = pickle.load(f_emb)
            
            embeddings = np.array(emb_data["embeddings"], dtype=np.float32)
            emb = embeddings
        except Exception as e:
            print(f"Error loading HoroPCA embeddings: {e}")
            return {}
        
        labels = np.asarray(labels_data, dtype=int)
        sel = sel or []
        
        # Handle interpolated points - they are calculated in the space of selected_proj
        # traversal_path contains indices into the dataset, not coordinates
        interp_point = None
        if traversal_path is not None and isinstance(traversal_path, list):
            if selected_proj == "horopca":
                # Path was calculated in HoroPCA space, get the coordinates for these indices
                interp_coords = []
                for idx in traversal_path:
                    if idx < len(emb):
                        interp_coords.append(emb[idx])
                if interp_coords:
                    interp_point = np.array(interp_coords)
            else:
                # Path was calculated in CO-SNE space, but we're showing HoroPCA
                # Show the same indices but in HoroPCA coordinates
                interp_coords = []
                for idx in traversal_path:
                    if idx < len(emb):
                        interp_coords.append(emb[idx])
                if interp_coords:
                    interp_point = np.array(interp_coords)
        
        # Handle 2D embeddings for disk projection (same as main scatter)
        xh, yh = emb[:, 0], emb[:, 1]
        if emb.shape[1] > 2:
            zh = emb[:, 2]
        else:
            zh = np.zeros(emb.shape[0])
        dx, dy = xh / (1.0 + zh), yh / (1.0 + zh)
        
        # Transform interpolated path if exists
        interp_transformed = None
        if interp_point is not None:
            if len(interp_point.shape) == 2 and interp_point.shape[0] > 1:
                # Multiple points forming a path - transform each point
                interp_coords = []
                for pt in interp_point:
                    if len(pt) > 2:
                        interp_dx = pt[0] / (1.0 + pt[2])
                        interp_dy = pt[1] / (1.0 + pt[2])
                    else:
                        interp_dx, interp_dy = pt[0], pt[1]
                    interp_coords.append([interp_dx, interp_dy])
                interp_transformed = np.array(interp_coords)
            else:
                # Single point
                if len(interp_point) > 2:
                    interp_dx = interp_point[0] / (1.0 + interp_point[2])
                    interp_dy = interp_point[1] / (1.0 + interp_point[2])
                else:
                    interp_dx, interp_dy = interp_point[0], interp_point[1]
                interp_transformed = np.array([interp_dx, interp_dy])
        
        # Calculate neighbors and tree connections based on mode
        neighbor_indices = []
        tree_connections = []
        
        if mode == "neighbors" and sel and len(sel) == 1:
            if data_store is not None:
                data_np = np.asarray(data_store, dtype=np.float32)
                dists = np.linalg.norm(data_np - data_np[sel[0]], axis=1)
                neighbor_indices = np.argsort(dists)
                neighbor_indices = neighbor_indices[neighbor_indices != sel[0]][:k_neighbors]
            else:
                neighbor_indices = []
        elif mode == "tree" and sel and len(sel) == 1:
            # Find all points in the same tree and create connections between adjacent levels
            tree_connections = []
            try:
                # Get the selected point's tree
                selected_pt = points[sel[0]]
                selected_tree_id = selected_pt.get("tree_id", "?")
                
                # Find all points that belong to the same tree
                tree_point_indices = []
                tree_points_by_type = {}
                
                for i, pt in enumerate(points):
                    if pt.get("tree_id") == selected_tree_id:
                        # Include all tree points in highlighting (including selected)
                        tree_point_indices.append(i)
                        
                        # Group by embedding type for creating connections
                        emb_type = pt.get("embedding_type", "unknown")
                        if emb_type not in tree_points_by_type:
                            tree_points_by_type[emb_type] = []
                        tree_points_by_type[emb_type].append(i)
                
                # Create connections between adjacent hierarchical levels
                if dataset_name == "imagenet":
                    level_order = ['parent_text', 'child_text', 'child_image']
                else:  # GRIT
                    level_order = ['parent_text', 'child_text', 'parent_image', 'child_image']
                
                # Connect consecutive levels
                for i in range(len(level_order) - 1):
                    current_level = level_order[i]
                    next_level = level_order[i + 1]
                    
                    if current_level in tree_points_by_type and next_level in tree_points_by_type:
                        for curr_pt in tree_points_by_type[current_level]:
                            for next_pt in tree_points_by_type[next_level]:
                                tree_connections.append((curr_pt, next_pt))
                
                # Use tree points as neighbor_indices for highlighting
                neighbor_indices = tree_point_indices
                
            except Exception as e:
                print(f"Error finding tree points: {e}")
                neighbor_indices = []
                tree_connections = []
        
        # Load the original label types for color coding (always HoroPCA for left plot)
        emb_labels = emb_data.get("labels", [])
        
        # Create the figure with all mode features
        fig = _create_full_interactive_scatter(dx, dy, labels, target_names, emb_labels, "", sel, neighbor_indices, tree_connections, interp_transformed, mode)
        return fig



    @app.callback(
        Output("hyperparams-table", "children"),
        Input("proj", "data"),
    )
    def _update_hyperparams_display(projection_method):
        """Update hyperparameters display based on selected projection method."""
        if not projection_method:
            return html.Div()
        
        if projection_method == "horopca":
            # HoroPCA hyperparameters from create_projections.py
            params = [
                {"param": "Components", "value": "2", "description": "Output dimensions"},
                {"param": "Learning Rate", "value": "0.05", "description": "Optimization step size"},
                {"param": "Max Steps", "value": "500", "description": "Maximum iterations"},
            ]
        elif projection_method == "cosne":
            # CO-SNE hyperparameters from create_projections.py
            params = [
                {"param": "Learning Rate", "value": "0.5", "description": "Main learning rate"},
                {"param": "Hyperbolic LR", "value": "0.01", "description": "Hyperbolic learning rate"},
                {"param": "Perplexity", "value": "30", "description": "Local neighborhood size"},
                {"param": "Exaggeration", "value": "12.0", "description": "Early exaggeration factor"},
                {"param": "Gamma", "value": "0.1", "description": "Student-t distribution parameter"},
            ]
        else:
            return html.Div("Unknown projection method")
        
        # Create table rows
        table_rows = []
        for param in params:
            table_rows.append(
                html.Tr([
                    html.Td(param["param"], style={
                        "fontWeight": "600", 
                        "color": "#495057",
                        "fontSize": "0.8rem",
                        "padding": "0.25rem 0.5rem 0.25rem 0",
                        "borderBottom": "1px solid #e9ecef",
                        "width": "35%"
                    }),
                    html.Td(param["value"], style={
                        "color": "#007bff", 
                        "fontFamily": "monospace",
                        "fontSize": "0.8rem",
                        "padding": "0.25rem 0.5rem",
                        "borderBottom": "1px solid #e9ecef",
                        "width": "25%",
                        "textAlign": "center"
                    }),
                    html.Td(param["description"], style={
                        "color": "#6c757d", 
                        "fontSize": "0.75rem",
                        "padding": "0.25rem 0 0.25rem 0.5rem",
                        "borderBottom": "1px solid #e9ecef",
                        "width": "40%"
                    }),
                ])
            )
        
        return html.Table(
            [html.Tbody(table_rows)],
            style={
                "width": "100%",
                "borderCollapse": "collapse",
                "fontSize": "0.8rem"
            }
        )

    @app.callback(
        Output("interpolated-point", "data", allow_duplicate=True),
        Input("clear-path-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _clear_interpolated_point(n_clicks):
        if n_clicks:
            return None
        return dash.no_update


    # Projection button selection callbacks
    @app.callback(
        Output("proj", "data"),
        Output("proj-horopca-btn", "style"),
        Output("proj-cosne-btn", "style"),
        Input("proj-horopca-btn", "n_clicks"),
        Input("proj-cosne-btn", "n_clicks"),
        State("proj", "data"),
        prevent_initial_call=True,
    )
    def _update_projection_selection(horopca_clicks, cosne_clicks, current_proj):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update
        
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Button styles
        active_style = {
            "backgroundColor": "#28a745",
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
        inactive_style = {
            **active_style,
            "backgroundColor": "#6c757d"
        }
        
        if triggered_id == "proj-horopca-btn":
            return "horopca", active_style, inactive_style
        elif triggered_id == "proj-cosne-btn":
            return "cosne", inactive_style, active_style
        
        return dash.no_update, dash.no_update, dash.no_update

    # Interpolation number input callbacks
    @app.callback(
        Output("interpolation-slider", "value"),
        Input("interpolation-increase-btn", "n_clicks"),
        Input("interpolation-decrease-btn", "n_clicks"),
        State("interpolation-slider", "value"),
        prevent_initial_call=True,
    )
    def _update_interpolation_value(increase_clicks, decrease_clicks, current_value):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
        
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if triggered_id == "interpolation-increase-btn":
            return current_value + 1
        elif triggered_id == "interpolation-decrease-btn":
            return max(1, current_value - 1)  # Don't go below 1
        
        return dash.no_update

    # End of callbacks