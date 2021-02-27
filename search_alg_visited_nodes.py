import numpy as np
import random
from scipy.spatial.distance import euclidean
import math
import matplotlib.pyplot as plt
from metric_tree import MetricTree

def defeatist_search_vn(
    metric_tree: MetricTree,
    query_point: np.ndarray,
    min_dist: float = np.inf,
    result: np.ndarray = None,
    visited_nodes: int = 0):

    if metric_tree.size == 1:
        visited_nodes += 1
        current_dist = euclidean(query_point, metric_tree.root)
        if current_dist < min_dist:
            return metric_tree.root, current_dist, visited_nodes
        else:
            return result, min_dist, visited_nodes

    else:

        current_dist = euclidean(query_point, metric_tree.root)
        if (metric_tree.max_left is not None) and (metric_tree.min_right is not None):
            mu = (metric_tree.max_left + metric_tree.min_right)/2
        else:
            mu = metric_tree.max_left

        if current_dist < min_dist:
            min_dist = current_dist
            result = metric_tree.root

        if (current_dist < mu) and (metric_tree.left is not None):
            visited_nodes += 1
            result, min_dist, visited_nodes = defeatist_search_vn(metric_tree.left, query_point, min_dist, result, visited_nodes)

        elif (metric_tree.right is not None):
            visited_nodes += 1
            result, min_dist, visited_nodes = defeatist_search_vn(metric_tree.right, query_point, min_dist, result, visited_nodes)

        return result, min_dist, visited_nodes


def isin_dilated_interval(value, min, max, dilation):
    lower_bound = value > (min - dilation)
    upper_bound = value < (max + dilation)
    return lower_bound and upper_bound

def search_pruning_vn(
    metric_tree: MetricTree,
    query_point: np.ndarray,
    min_dist: float = np.inf,
    result: np.ndarray = None,
    visited_nodes: int = 0,
):
    if metric_tree.size == 1:
        visited_nodes += 1
        current_dist = euclidean(query_point, metric_tree.root)
        if current_dist < min_dist:
            return metric_tree.root, current_dist, visited_nodes
        else:
            return result, min_dist, visited_nodes
    else:
        left = False
        current_dist = euclidean(query_point, metric_tree.root)
        if current_dist < min_dist:
            result = metric_tree.root
            min_dist = current_dist

        ## Verifying that the left (resp. right) tree is not empty
        ## so min and max are defined
        if (metric_tree.left is not None) and isin_dilated_interval(
            current_dist, metric_tree.min_left, metric_tree.max_left, min_dist
        ):
            left = True
            visited_nodes += 1
            result, min_dist, visited_nodes = search_pruning_vn(
                metric_tree.left, query_point, min_dist, result, visited_nodes
            )

        if (metric_tree.right is not None) and isin_dilated_interval(
            current_dist, metric_tree.min_right, metric_tree.max_right, min_dist
        ):
            if not left:
                visited_nodes += 1
            result, min_dist, visited_nodes = search_pruning_vn(
                metric_tree.right, query_point, min_dist, result, visited_nodes
            )

        return result, min_dist, visited_nodes

def search_pruning_in_forest_vn(metric_forest, q):
    nn = []
    distances = []
    visited_nodes = 0
    for i, tree in enumerate(metric_forest.forest):
        result, min_dist, vn = search_pruning_vn(tree, q)
        nn.append(result)
        distances.append(min_dist)
        visited_nodes += vn
    sorted_indices = np.argsort(distances)
    distances = np.array(distances)[sorted_indices]
    nn = np.array(nn)[sorted_indices]
    return nn[0], distances[0], visited_nodes

def search_defeatist_in_forest_vn(metric_forest, q):
    nn = []
    distances = []
    visited_nodes = 0
    for i, tree in enumerate(metric_forest.forest):
        result, min_dist, vn = defeatist_search_vn(tree, q)
        nn.append(result)
        distances.append(min_dist)
        visited_nodes += vn
    sorted_indices = np.argsort(distances)
    distances = np.array(distances)[sorted_indices]
    nn = np.array(nn)[sorted_indices]
    return nn[0], distances[0], visited_nodes

