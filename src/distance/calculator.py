"""
Distance calculator for hyperbolic geometry.
"""
import numpy as np

class DistanceCalculator:
    def __init__(self, disk):
        """
        Initialize the distance calculator.
        
        Args:
            disk: PoincareDisk instance to calculate distances for
        """
        self.disk = disk

    def calculate_distance_map(self, resolution=50):
        """
        Calculate a distance map from the center of the disk.
        
        Args:
            resolution: Resolution of the distance map
            
        Returns:
            tuple: (X, Y, Z) meshgrid coordinates and distances
        """
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distances from center
        Z = np.sqrt(X**2 + Y**2)
        
        # Mask points outside the disk
        Z[Z >= 1] = np.nan
        
        return X, Y, Z

    def calculate_point_distances(self, point):
        """
        Calculate distances from a point to all other points.
        
        Args:
            point: (x, y) coordinates of the reference point
            
        Returns:
            list: List of distances to all points
        """
        distances = []
        for p in self.disk.get_points():
            dist = self.disk.hyperbolic_distance(point, (p['x'], p['y']))
            distances.append(dist)
        return distances

    def find_nearest_neighbors(self, point, k=5):
        """
        Find k nearest neighbors to a point.
        
        Args:
            point: (x, y) coordinates of the reference point
            k: Number of nearest neighbors to find
            
        Returns:
            list: List of k nearest points with their distances
        """
        distances = []
        for p in self.disk.get_points():
            dist = self.disk.hyperbolic_distance(point, (p['x'], p['y']))
            distances.append((p, dist))
        
        # Sort by distance and return k nearest
        distances.sort(key=lambda x: x[1])
        return distances[:k] 