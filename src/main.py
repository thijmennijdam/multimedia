"""
Main entry point for the Hyperbolic Learning Dashboard.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.app.dashboard import EmbeddingDashboard

def main():
    """
    Main entry point for the dashboard application.
    """
    print("Starting Hyperbolic Learning Dashboard...")
    
    # Initialize and run the dashboard
    dashboard = EmbeddingDashboard.from_backend(dataset_name="iris", reduction="PCA")
    dashboard.run(debug=True, port=8050)

if __name__ == "__main__":
    main() 