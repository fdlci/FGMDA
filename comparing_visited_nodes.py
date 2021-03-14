import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from metric_tree import MetricTree
from search_alg_visited_nodes import search_pruning_vn
from metric_tree_pivot import MetricTreepivot
from experiments import *

"""In this file you can find the functions used for solving Question 5
in the notebook Project"""

def comparing_visited_nodes(data_size, N, gaussian=False):
    """Compares the average number of visited nodes for both pivot choice
    methods while varying the size of the dataset. If isgaussian is set to False,
    then the points are drawn randomly, otherwise the points are drawn from gaussians"""
    mean_vns = []
    mean_vns_bis = []
    for i in tqdm(data_size):
        if gaussian:
            points = data_drawn_from_gaussians(i, N, 10, int(i/10))
        else:
            points = np.random.rand(i, N)
        tree = MetricTree(points)
        tree_bis = MetricTreepivot(points)
        n_try = 1000
        vns = []
        vns_bis = []
        for t in range(n_try):
            q = np.random.rand(N)
            vn = search_pruning_vn(tree, q)[2]
            vn_bis = search_pruning_vn(tree_bis, q)[2]
            vns.append(vn)
            vns_bis.append(vn_bis)
        mean_vns.append(np.mean(vns))
        mean_vns_bis.append(np.mean(vns_bis))
    return mean_vns, mean_vns_bis, N

def increasing_dimension(dims, gaussian=False, size=10000):
    """Compares the average number of visited nodes for both pivot choice
    methods while varying the dimension of the Euclidean space. If isgaussian 
    is set to False, then the points are drawn randomly, otherwise the points 
    are drawn from gaussians"""
    mean_vns = []
    mean_vns_bis = []
    for i in tqdm(dims):
        if gaussian:
            points = data_drawn_from_gaussians(size, i, 10, int(size/10))
        else:
            points = np.random.rand(size, i)
        tree = MetricTree(points)
        tree_bis = MetricTreepivot(points)
        n_try = 1000
        vns = []
        vns_bis = []
        for t in range(n_try):
            q = np.random.rand(i)
            vn = search_pruning_vn(tree, q)[2]
            vn_bis = search_pruning_vn(tree_bis, q)[2]
            vns.append(vn)
            vns_bis.append(vn_bis)
        mean_vns.append(np.mean(vns))
        mean_vns_bis.append(np.mean(vns_bis))
    return mean_vns, mean_vns_bis

def plotting_vn_size(mean_vns, mean_vns_bis, N, data_size):
    """Plots the average number of visited nodes for both pivot choice methods
    while varying the size of the data set"""
    plt.plot(data_size, mean_vns, label=f'Random pivot for dim={N}')
    plt.plot(data_size, mean_vns_bis, label=f'Optimized pivot for dim={N}')
    plt.xlabel('Size of the dataset')
    plt.ylabel('Average number of visited nodes')
    plt.title('Comparing the number of visited nodes with different pivot approaches for different dimensions')
    plt.legend()
    plt.plot()

def plotting_vn_dim(mean_vns, mean_vns_bis, dims): 
    """Plots the average number of visited nodes for both pivot choice methods
    while varying the dimension of the Euclidean space"""
    plt.plot(dims, mean_vns, label=f'Random pivot')
    plt.plot(dims, mean_vns_bis, label=f'Optimized')
    plt.xlabel('Dimension')
    plt.ylabel('Average number of visited nodes')
    plt.title('Comparing the number of visited nodes according to the dimension of the Eucledian space')
    plt.legend()
    plt.plot()