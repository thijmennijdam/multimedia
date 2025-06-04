"""
Hierarchy builder for hyperbolic hierarchical structures.
"""
import numpy as np
from collections import defaultdict

class HierarchyBuilder:
    def __init__(self, disk):
        """
        Initialize the hierarchy builder.
        
        Args:
            disk: PoincareDisk instance to build hierarchy for
        """
        self.disk = disk
        self.hierarchy = None

    def build_hierarchy(self, max_distance=0.5):
        """
        Build a hierarchical structure from the points.
        
        Args:
            max_distance: Maximum distance for clustering
            
        Returns:
            dict: Hierarchical structure
        """
        points = self.disk.get_points()
        if not points:
            return None

        # Initialize hierarchy with root node
        hierarchy = {
            'id': 'root',
            'children': [],
            'points': points
        }

        # Build hierarchy recursively
        self._build_level(hierarchy, max_distance)
        
        self.hierarchy = hierarchy
        return hierarchy

    def _build_level(self, node, max_distance):
        """Build a level of the hierarchy."""
        points = node['points']
        if len(points) <= 1:
            return

        # Find clusters at this level
        clusters = self._find_clusters(points, max_distance)
        
        # Create child nodes for each cluster
        for i, cluster in enumerate(clusters):
            child = {
                'id': f"{node['id']}_{i}",
                'children': [],
                'points': cluster
            }
            node['children'].append(child)
            
            # Recursively build next level
            self._build_level(child, max_distance * 0.8)  # Reduce distance for next level

    def _find_clusters(self, points, max_distance):
        """Find clusters of points based on hyperbolic distance."""
        if not points:
            return []

        clusters = []
        remaining = points.copy()

        while remaining:
            # Start a new cluster with the first remaining point
            current = remaining.pop(0)
            cluster = [current]
            
            # Find all points within max_distance
            i = 0
            while i < len(remaining):
                point = remaining[i]
                if self.disk.hyperbolic_distance(
                    (current['x'], current['y']),
                    (point['x'], point['y'])
                ) <= max_distance:
                    cluster.append(remaining.pop(i))
                else:
                    i += 1
            
            clusters.append(cluster)

        return clusters

    def get_hierarchy(self):
        """Get the current hierarchy."""
        return self.hierarchy

    def get_levels(self):
        """Get the number of levels in the hierarchy."""
        if not self.hierarchy:
            return 0
        
        def count_levels(node):
            if not node['children']:
                return 1
            return 1 + max(count_levels(child) for child in node['children'])
        
        return count_levels(self.hierarchy) 