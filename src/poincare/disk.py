"""
Core Poincaré disk functionality for hyperbolic geometry operations.
"""
import numpy as np

class PoincareDisk:
    def __init__(self):
        """Initialize the Poincaré disk."""
        self.points = []
        self.clusters = []
        self.hierarchy = None

    def add_point(self, x, y, label=None):
        """Add a point to the disk."""
        if not self._is_in_disk(x, y):
            raise ValueError("Point must be within the unit disk")
        self.points.append({'x': x, 'y': y, 'label': label})

    def _is_in_disk(self, x, y):
        """Check if a point is within the unit disk."""
        return x**2 + y**2 < 1

    def get_points(self):
        """Get all points in the disk."""
        return self.points

    def get_boundary_points(self, num_points=100):
        """Get points on the disk boundary."""
        theta = np.linspace(0, 2*np.pi, num_points)
        return np.cos(theta), np.sin(theta)

    def get_grid_points(self, num_rings=4, points_per_ring=100):
        """Get points for the grid lines."""
        grid_points = []
        for r in np.linspace(0.2, 0.8, num_rings):
            theta = np.linspace(0, 2*np.pi, points_per_ring)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            grid_points.append((x, y))
        return grid_points

    def hyperbolic_distance(self, p1, p2):
        """
        Calculate the hyperbolic distance between two points in the Poincaré disk.
        
        Args:
            p1: First point (x, y)
            p2: Second point (x, y)
            
        Returns:
            float: Hyperbolic distance between the points
        """
        x1, y1 = p1
        x2, y2 = p2
        
        # Convert to complex numbers
        z1 = x1 + 1j * y1
        z2 = x2 + 1j * y2
        
        # Calculate the distance using the Poincaré disk metric
        numerator = abs(z1 - z2)
        denominator = np.sqrt((1 - abs(z1)**2) * (1 - abs(z2)**2))
        
        return 2 * np.arcsinh(numerator / denominator)

    def find_clusters(self, max_distance=0.5):
        """Find clusters of points based on hyperbolic distance."""
        # TODO: Implement clustering algorithm
        pass

    def build_hierarchy(self):
        """Build a hierarchical structure from the points."""
        # TODO: Implement hierarchy building
        pass 