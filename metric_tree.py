import numpy as np
from scipy.spatial.distance import cdist
from typing import Tuple


class MetricTree:
    def __init__(self, points: np.ndarray = None) -> None:
        self.root = None
        self.mu = None
        self.left = None
        self.right = None
        self.size = None
        self.distances = None
        if points is not None:
            self.build(points)

    def median_mu(self, points: np.ndarray, index: int) -> Tuple:
        pivot = points[[index]]
        other_points = np.delete(points, index, 0)
        distance_to_pivot = cdist(pivot, other_points)[0]
        sorted_indices = np.argsort(distance_to_pivot)
        return (
            np.median(distance_to_pivot),
            distance_to_pivot[sorted_indices],
            other_points[sorted_indices],
        )

    def left_right_split(self, mu, distances, points):
        idx = np.searchsorted(distances, mu, side="right")
        distances_left = distances[:idx]
        points_left = points[:idx]
        distances_right = distances[idx:]
        points_right = points[idx:]
        return points_left, points_right, distances_left, distances_right

    def pivot_choice(self, points: np.ndarray) -> int:
        ## For the moment only random choice
        choice_idx = np.random.randint(len(points))
        return choice_idx

    def __repr__(self) -> str:
        return f"MetricTree with root {self.root} and size {self.size}"

    def build(self, points: np.ndarray, distances: np.ndarray = None, tau: float = 0):
        self.size = len(points)
        if distances is not None:
            self.distances = distances
        if len(points) == 0:
            return None
        pivot_idx = self.pivot_choice(points)
        self.root = points[pivot_idx]
        mu, distances, other_points = self.median_mu(points, pivot_idx)
        self.mu = mu
        (
            points_left,
            points_right,
            distances_left,
            distances_right,
        ) = self.left_right_split(mu, distances, other_points)
        self.left = MetricTree().build(points_left, distances_left)
        self.right = MetricTree().build(points_right, distances_right)
        return self
