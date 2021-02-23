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


def search_pruning_clement(
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
            result, min_dist = search_pruning_clement(
                metric_tree.left, query_point, min_dist, result
            )

        if (metric_tree.right is not None) and isin_dilated_interval(
            current_dist, metric_tree.min_right, metric_tree.max_right, min_dist
        ):
            result, min_dist = search_pruning_clement(
                metric_tree.right, query_point, min_dist, result
            )

        return result, min_dist


def search_pruning_metric_tree(metric_tree, q, visited_nodes=0, tau=math.inf, nnq=None):

    if metric_tree.size == 1:
        visited_nodes += 1
        return exact_nn_search([metric_tree.root, nnq], q)[0], visited_nodes

    else:
        nnq1, nnq2 = None, None
        search = True
        pivot = metric_tree.root
        visited_nodes += 1

        l = euclidean(pivot, q)
        # print('l:' + str(l))

        if l < tau:
            tau = l
            nnq = pivot

        if metric_tree.left.size == 1:
            return search_pruning_metric_tree(
                metric_tree.left, q, visited_nodes, tau, nnq
            )
        else:
            min_left, max_left = (
                np.min(metric_tree.left.distances),
                np.max(metric_tree.left.distances),
            )

        min_right, max_right = (
            np.min(metric_tree.right.distances),
            np.max(metric_tree.right.distances),
        )

        d_left = [min_left - tau, max_left + tau]
        d_right = [min_right - tau, max_right + tau]
        # print('d_left: ' + str(d_left))
        # print('d_right: ' + str(d_right))

        if l >= d_left[0] and l <= d_left[1]:
            print("left")
            search = False
            nnq1, v1 = search_pruning_metric_tree(
                metric_tree.left, q, visited_nodes, tau, nnq
            )
            visited_nodes = v1
        if l >= d_right[0] and l <= d_right[1]:
            print("right")
            search = False
            nnq2, v2 = search_pruning_metric_tree(
                metric_tree.right, q, visited_nodes, tau, nnq
            )
            visited_nodes = v2
        if search:
            print("In none of the intervals")
            nnq1, v1 = search_pruning_metric_tree(
                metric_tree.left, q, visited_nodes, tau, nnq
            )
            nnq2, v2 = search_pruning_metric_tree(
                metric_tree.right, q, visited_nodes, tau, nnq
            )
            visited_nodes = v1 + v2 - 1

        if nnq1 is None or nnq2 is None:
            if nnq1 is None:
                return nnq2, visited_nodes
            else:
                return nnq1, visited_nodes
        else:
            return exact_nn_search([nnq1, nnq2], q)[0], visited_nodes


def defeatist_search(metric_tree, q, visited_nodes=0, nnq=None, d=math.inf):

    if metric_tree == None:
        return nnq, visited_nodes

    else:

        pivot = metric_tree.root
        visited_nodes += 1

        tau = euclidean(pivot, q)
        mu = metric_tree.mu

        if tau < d:
            d = tau
            nnq = pivot

        if tau < mu:
            print("left")
            return defeatist_search(metric_tree.left, q, visited_nodes, nnq, d)
        else:
            print("right")
            return defeatist_search(metric_tree.right, q, visited_nodes, nnq, d)


def compare_search_algorithms(all_points, metric_tree, q):

    nn_exact, visit_exact = exact_nn_search(all_points, q)
    nn_pruning, visit_pruning = search_pruning_metric_tree(metric_tree, q)
    nn_defeat, visit_defeat = defeatist_search(metric_tree, q)

    return """Exact search: NN {} visited_nodes: {}
Exact search with pruning: NN {} visited_nodes: {}
Defeatist search : NN {} visited_nodes: {}""".format(
        nn_exact, visit_exact, nn_pruning, visit_pruning, nn_defeat, visit_defeat
    )
