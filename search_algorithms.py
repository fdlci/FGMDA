import numpy as np
import random
from scipy.spatial.distance import euclidean
import math
import matplotlib.pyplot as plt
from metric_tree import MetricTree


def exact_nn_search(all_points, q):
    nnq = None
    d = math.inf
    for point in all_points:
        dist = euclidean(point, q)
        if dist < d:
            d = dist
            nnq = point
    return nnq, len(all_points)


def isin_dilated_interval(value, min, max, dilation):
    lower_bound = value > (min - dilation)
    upper_bound = value < (max + dilation)
    return lower_bound and upper_bound


def search_pruning(
    metric_tree: MetricTree,
    query_point: np.ndarray,
    min_dist: float = np.inf,
    result: np.ndarray = None,
):
    if metric_tree.size == 1:
        current_dist = euclidean(query_point, metric_tree.root)
        if current_dist < min_dist:
            return metric_tree.root, current_dist
        else:
            return result, min_dist
    else:
        current_dist = euclidean(query_point, metric_tree.root)
        if current_dist < min_dist:
            result = metric_tree.root
            min_dist = current_dist

        ## Verifying that the left (resp. right) tree is not empty
        ## so min and max are defined
        if (metric_tree.left is not None) and isin_dilated_interval(
            current_dist, metric_tree.min_left, metric_tree.max_left, min_dist
        ):
            result, min_dist = search_pruning(
                metric_tree.left, query_point, min_dist, result
            )

        if (metric_tree.right is not None) and isin_dilated_interval(
            current_dist, metric_tree.min_right, metric_tree.max_right, min_dist
        ):
            result, min_dist = search_pruning(
                metric_tree.right, query_point, min_dist, result
            )

        return result, min_dist


def defeatist_search(
    metric_tree: MetricTree,
    query_point: np.ndarray,
    min_dist: float = np.inf,
    result: np.ndarray = None):

    if metric_tree.size == 1:
        current_dist = euclidean(query_point, metric_tree.root)
        if current_dist < min_dist:
            return metric_tree.root, current_dist
        else:
            return result, min_dist

    else:

        current_dist = euclidean(query_point, metric_tree.root)
        mu = metric_tree.mu

        if current_dist < min_dist:
            min_dist = current_dist
            result = metric_tree.root

        if (current_dist < mu) and (metric_tree.left is not None):
            result, min_dist = defeatist_search(metric_tree.left, query_point, min_dist, result)

        elif (metric_tree.right is not None):
            result, min_dist = defeatist_search(metric_tree.right, query_point, min_dist, result)
        
        return result, min_dist


# def compare_search_algorithms(all_points, metric_tree, q):

#     nn_exact, visit_exact = exact_nn_search(all_points, q)
#     nn_pruning, visit_pruning = search_pruning(metric_tree, q)
#     nn_defeat, visit_defeat = defeatist_search(metric_tree, q)

#     return """Exact search: NN {} visited_nodes: {}
# Exact search with pruning: NN {} visited_nodes: {}
# Defeatist search : NN {} visited_nodes: {}""".format(
#         nn_exact, visit_exact, nn_pruning, visit_pruning, nn_defeat, visit_defeat
#     )
