import numpy as np
import random
from scipy.spatial.distance import euclidean
import math
import matplotlib.pyplot as plt

def random_choice_coordinates(num_points, x_min, x_max, y_min, y_max):
    """Computes num_points random points in the space delimited by
    x_min, x_max, y_min, y_max"""
    points_in_tree = []
    while len(points_in_tree)<num_points:
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        if (x, y) not in points_in_tree:
            points_in_tree.append((x,y))
    return points_in_tree

def visualize_metric_tree_points(all_points):
    """Plots the random points chosen to build the metric tree"""
    x_coords = [all_points[i][0] for i in range(len(all_points))]
    y_coords = [all_points[i][1] for i in range(len(all_points))]
    plt.scatter(x_coords, y_coords)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Random points used to build a metric tree')
    plt.show()

def visualize_metric_nn(all_points, nn, q):
    x_coords = [all_points[i][0] for i in range(len(all_points))]
    y_coords = [all_points[i][1] for i in range(len(all_points))]
    plt.scatter(x_coords, y_coords, c='green', label='"Search points"')
    plt.scatter(nn[0], nn[1], c='blue', label='Nearest Neighbor')
    plt.scatter(q[0], q[1], c='red', label='Query point')
    plt.title('Visualizing the NN given a specific query')
    plt.legend()
    plt.show()