import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm.notebook import tqdm
from metric_forest import MetricForest
from search_alg_visited_nodes import search_defeatist_in_forest_vn, search_pruning_in_forest_vn

"""In this file you can find the functions used for solving Question 3
in the notebook Project"""

def select_random_points(n_points, dim, n):
    """Draws random points of a given dimension and plots them"""
    points = n*np.random.random((n_points, dim))
    if dim == 2:
        plt.scatter(points[:,0], points[:,1])
        plt.title('Points drawn randomly')
        plt.show()
    elif dim == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(points[:,0], points[:,1], points[:,2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.title('Points drawn randomly')
        plt.show()
    return points

def data_drawn_from_gaussians(n_points, dim, n, num_points_gaussian):
    """Draws points from gaussian distributions and plots them"""
    assert n_points%num_points_gaussian == 0, 'Please choose a number of points multiple of num_points_gaussian'
    all_points = np.zeros((n_points, dim))
    for i in range(1, int(n_points/num_points_gaussian)+1):
        sigma, mu = 3*np.random.rand(1), n*np.random.rand(1)
        points = sigma * np.random.randn(num_points_gaussian,dim) + mu
        all_points[(i-1)*num_points_gaussian : i*num_points_gaussian] = points    
    return all_points

def plotting_gaussians(all_points, dim, n, num_points_gaussian):
    """Plots data points drawn from gaussian distributions in 2D and 3D"""

    n_points = len(all_points)
    assert dim == 2 or dim == 3, 'Please make sure the dimension is 2 or 3 to plot the data'
    assert n_points%num_points_gaussian == 0, 'Please choose a number of points multiple of num_points_gaussian'

    if dim == 3:
        fig = plt.figure()
        ax = Axes3D(fig)

    for i in range(1, int(n_points/num_points_gaussian)+1):
        points = all_points[(i-1)*num_points_gaussian : i*num_points_gaussian]
        if dim == 2:
            plt.scatter(points[:,0], points[:,1])
            plt.title('Points drawn from a mixture of gaussians')
        elif dim == 3:
            ax.scatter(points[:,0], points[:,1], points[:,2])
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.title('Points drawn from a mixture of gaussians')

def average_nodes_visited_dim(n_points, avg, max_dim, n, m, isgaussian=False):
    """Computes the average number of visited nodes for both search methods varying
    the dimension of the Euclidean space. If isgaussian is set to False, then the 
    points are drawn randomly, otherwise the points are drawn from gaussians"""
    vn_pruning = []
    vn_defeatist = []
    for d in tqdm(range(1, max_dim + 1)):
        vnp = 0
        vnd = 0
        for i in range(avg):
            query = n*np.random.rand(d)
            if isgaussian:
                points = data_drawn_from_gaussians(n_points, d, n, int(n_points/10))
            else:
                points = n*np.random.random((n_points, d))
            forest = MetricForest(m, points)
            vnp += search_pruning_in_forest_vn(forest, query)[2]
            vnd += search_defeatist_in_forest_vn(forest, query)[2]
        vn_pruning.append(vnp/avg)
        vn_defeatist.append(vnd/avg)
    x = [i for i in range(1, max_dim+1)]
    plt.plot(x, vn_pruning, label='Pruning Search')
    plt.plot(x, vn_defeatist, label='Defeatist Search')
    plt.title('Variations of the average number of visited nodes according to the dimension of the ambient space')
    plt.xlabel('Dimension')
    plt.ylabel('Visited Nodes')
    plt.legend()
    plt.show()

def average_nodes_visited_size(avg, max_num_points, n, m, d, isgaussian=False):
    """Computes the average number of visited nodes for both search methods varying
    the number of trees of the forest (i.e the number of points in the dataset). 
    If isgaussian is set to False, then the  points are drawn randomly, otherwise 
    the points are drawn from gaussians"""
    vn_pruning = []
    vn_defeatist = []
    num_trees = []
    for num_points in tqdm(range(100, max_num_points + 1, 200)):
        vnp = 0
        vnd = 0
        for i in range(avg):
            query = n*np.random.rand(d)
            if isgaussian:
                points = data_drawn_from_gaussians(num_points, d, n, int(num_points/10))
            else:
                points = n*np.random.random((num_points, d))
            forest = MetricForest(m, points)
            vnp += search_pruning_in_forest_vn(forest, query)[2]
            vnd += search_defeatist_in_forest_vn(forest, query)[2]
        vn_pruning.append(vnp/avg)
        vn_defeatist.append(vnd/avg)
        num_trees.append(len(forest.forest))
    # x = [i for i in range(100, max_num_points + 1, 200)]
    plt.plot(num_trees, vn_pruning, label='Pruning Search')
    plt.plot(num_trees, vn_defeatist, label='Defeatist Search')
    plt.title('Variations of the average number of visited nodes according to the database size in dimension {}'.format(d))
    plt.xlabel('Number of trees')
    plt.xscale('log')
    plt.ylabel('Visited Nodes')
    plt.legend()
    plt.show()

def transforming_the_data_from_d_to_D(points, d, D):
    """Transforms a vector of dimension d to a vector of dimension D (D >= d)
    by adding the adequate number of zeros"""
    assert d <= D, 'The dimension of the ambient space must be higher or equal to the instrinsic dimension'
    adding_zeros = np.zeros((points.shape[0], D-d))
    return np.concatenate((points, adding_zeros), axis=1)

def average_nodes_visited_intrinsic_dim(n_points, dim, num_D, n, avg, isgaussian=False):
    """Computes the average number of visited nodes for both search methods varying
    the dimension of the ambient space while keeping the intrinsic dimension constant.
    If isgaussian is set to False, then the  points are drawn randomly, otherwise 
    the points are drawn from gaussians"""
    vn_pruning = []
    vn_defeatist = []
    if isgaussian:
        points = data_drawn_from_gaussians(n_points, dim, n, int(n_points/10))
    else:
        points = n*np.random.random((n_points, dim))
    for d in tqdm(range(dim, dim + num_D)):
        vnp = 0
        vnd = 0
        for i in range(avg):
            query = n*np.random.rand(d)
            new_points = transforming_the_data_from_d_to_D(points, dim, d)
            forest = MetricForest(0.5, new_points)
            vnp += search_pruning_in_forest_vn(forest, query)[2]
            vnd += search_defeatist_in_forest_vn(forest, query)[2]
        vn_pruning.append(vnp/avg)
        vn_defeatist.append(vnd/avg)
    x = [i for i in range(dim, dim + num_D)]
    plt.plot(x, vn_pruning, label='Pruning Search')
    plt.plot(x, vn_defeatist, label='Defeatist Search')
    plt.title('Variations of the average number of visited nodes according to the dimension of the ambient space for a given database')
    plt.xlabel('Dimension')
    plt.ylabel('Visited Nodes')
    plt.legend()
    plt.show()

def applying_both_searches(forest, query):
    """Computes the NN search using both search methods"""
    nn_pruning, min_dist_pruning, vn_pruning = search_pruning_in_forest_vn(forest, query)
    nn_defeatist, min_dist_defeatist, vn_defeatist = search_defeatist_in_forest_vn(forest, query)
    return """Exact search with pruning: NN {}, min_dist {}, visited_nodes: {}
Defeatist search : NN {}, min_dist {}, visited_nodes: {}""".format(
nn_pruning, min_dist_pruning, vn_pruning, nn_defeatist, min_dist_defeatist, vn_defeatist
    )

def plotting3D(points):
    """Plots 3D points"""
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.title('Plotting the 3D points')