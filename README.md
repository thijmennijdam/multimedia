# HIVE: Hyperbolic Interactive Visualization Explorer

An interactive dashboard for exploring hierarchical data in hyperbolic space using the Poincaré disk model.

## Setup Instructions

### Prerequisites
First, install uv if you haven't already:
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Running the Dashboard

```bash
# 1. Create a virtual environment
uv venv

# 2. Activate the virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 3. Install dependencies
uv sync

# 4. Run the dashboard
uv run src/main.py
```

The dashboard will be available at `http://localhost:8080`

## Overview

HIVE provides an interactive interface for visualizing and exploring hierarchical relationships in data using hyperbolic geometry. The tool is particularly useful for understanding complex parent-child relationships and taxonomic structures.

### Key Features

- **Interactive Poincaré Disk Visualization**: Explore data in 2D hyperbolic space
- **Multiple Projection Methods**: Compare HoroPCA and CO-SNE embeddings
- **Hierarchical Data Exploration**: Navigate parent-child relationships through tree structures
- **Hyperbolic Space Traversal**: Create and follow geodesic paths between points
- **Real-time Analysis**: Select points to view associated images and text content
- **Dual View Mode**: Side-by-side comparison of different projection methods

### Datasets

- **ImageNet**: Hierarchical image classification with semantic relationships
- **GRIT**: Grounded image-text dataset with parent-child associations

### Use Cases

- Understanding hierarchical embeddings quality
- Exploring taxonomic relationships in data
- Analyzing knowledge graph structures
- Validating hyperbolic space representations
- Educational visualization of hyperbolic geometry

The tool is meant for researchers and practitioners working with hierarchical data and hyperbolic embeddings.
