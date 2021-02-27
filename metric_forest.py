import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist
from typing import Tuple
from mpl_toolkits.mplot3d import Axes3D

colors = [name for name, color in mcolors.TABLEAU_COLORS.items()]


class MetricForestTree:
    def __init__(self, m_proportion, points: np.ndarray = None) -> None:
        # Proportion of middle points to discard during building of the forest
        self.m_proportion = m_proportion
        self.root = None
        self.left = None
        self.min_left = None
        self.max_left = None
        self.right = None
        self.min_right = None
        self.max_right = None
        self.size = None
        self.dim = None
        if points is not None:
            self.build(points)

    def left_middle_right_split(
        self, points: np.ndarray, distances: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
        """
        Splits the points into left, middle and right depending on their distance to root node.
        The proportion of middle points is given by the instance attribut m_proportion
        Points and distances are ordered by ascending distance to root node.

        Parameters
        ----------
        points : np.ndarray
            Array of points to be splitted
        distances : np.ndarray
            Distances of the points to the root node

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, float, float, float, float, np.ndarray]
            Corresponds to the left and right subtree points with
            their min and max distances to the root node
            The final array is the middle (or discarded) points
        """
        if len(points) == 1:
            dist_to_pivot = distances[0]
            return (
                points,
                None,
                dist_to_pivot,
                dist_to_pivot,
                None,
                None,
                np.empty(shape=(0, self.dim)),
            )
        N = len(points)
        idx_left = int(N * self.m_proportion)
        len_middle = int((N - idx_left) / 2)
        points_left = points[:idx_left]
        points_middle = points[idx_left : idx_left + len_middle]
        points_right = points[idx_left + len_middle :]
        return (
            points_left,
            points_right,
            distances[0],
            distances[idx_left - 1],
            distances[idx_left + len_middle],
            distances[-1],
            points_middle,
        )

    def pivot_choice(self, points: np.ndarray) -> int:
        ## For the moment only random choice
        choice_idx = np.random.randint(len(points))
        return choice_idx

    def __repr__(self) -> str:
        return f"MetricForest: root={self.root}"

    def __len__(self) -> int:
        return self.size

    def build(self, points: np.ndarray, subset_size: int = 10000):
        self.size = len(points)
        self.dim = points.shape[1]
        if len(points) == 0:
            return None, np.empty(shape=(0, self.dim))
        if len(points) == 1:
            self.root = points[0]
            return self, np.empty(shape=(0, self.dim))
        ## Selection of the pivot
        pivot_idx = self.pivot_choice(points)
        remaining_points = np.delete(points, pivot_idx, 0)
        self.root = points[pivot_idx]

        ## Compute distances
        distances = cdist(self.root.reshape(1, *self.root.shape), remaining_points)[0]

        ## Sort the points by distance to the root
        sorted_indices = np.argsort(distances)
        distances = distances[sorted_indices]
        remaining_points = remaining_points[sorted_indices]

        ## Split the points into left and right trees
        (
            points_left,
            points_right,
            self.min_left,
            self.max_left,
            self.min_right,
            self.max_right,
            points_middle,
        ) = self.left_middle_right_split(remaining_points, distances)

        ## Build the left and right trees if needed
        discarded_left = np.empty(shape=(0, self.dim))
        discarded_right = np.empty(shape=(0, self.dim))
        if points_left is not None:
            self.left, discarded_left = MetricForestTree(self.m_proportion).build(
                points_left
            )
        if points_right is not None:
            self.right, discarded_right = MetricForestTree(self.m_proportion).build(
                points_right
            )
        discarded_all = np.concatenate((points_middle, discarded_left, discarded_right))
        return self, discarded_all

    def plot(self, color=None, fig=None, ax=None):
        if color is None:
            color = np.random.choice(colors)
        if fig is None:
            # print(self.root.shape)
            if self.root.shape[0] == 2:
                fig, ax = plt.subplots()
            elif self.root.shape[0] == 3:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
        if self.root.shape[0] == 2:
            ax.scatter(self.root[0], self.root[1], color=color)
            if self.left is not None:
                ax.plot(
                    [self.root[0], self.left.root[0]],
                    [self.root[1], self.left.root[1]],
                    color=color,
                )
                self.left.plot(color, fig, ax)
            if self.right is not None:
                ax.plot(
                    [self.root[0], self.right.root[0]],
                    [self.root[1], self.right.root[1]],
                    color=color,
                )
                self.right.plot(color, fig, ax)
        elif self.root.shape[0] == 3:
            ax.scatter(self.root[0], self.root[1], self.root[2], color=color)
            if self.left is not None:
                ax.plot(
                    [self.root[0], self.left.root[0]],
                    [self.root[1], self.left.root[1]],
                    [self.root[2], self.left.root[2]],
                    color=color,
                )
                self.left.plot(color, fig, ax)
            if self.right is not None:
                ax.plot(
                    [self.root[0], self.right.root[0]],
                    [self.root[1], self.right.root[1]],
                    [self.root[2], self.right.root[2]],
                    color=color,
                )
                self.right.plot(color, fig, ax)           


class MetricForest:
    def __init__(self, m_proportion: float, points: np.ndarray = None) -> None:
        self.m_proportion = m_proportion
        self.size = len(points)
        self.forest = []
        if points is not None:
            self.build_forest(points)

    def build_forest(self, points):
        if len(points) == 0:
            return None
        tree, points = MetricForestTree(self.m_proportion).build(points)
        self.forest.append(tree)
        self.build_forest(points)

    def plot(self, points: np.ndarray):
        if points.shape[1] == 2:
            fig, ax = plt.subplots()
        elif points.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        for tree in self.forest:
            tree.plot(fig=fig, ax=ax)
        plt.title('Plotting the Metric Forest')

# if __name__=='__main__':
#     n_points, dim, n = 1000, 3, 10
#     points = n*np.random.random((n_points, dim))
#     forest = MetricForest(0.5, points)
#     forest.plot(points)
#     plt.show()
