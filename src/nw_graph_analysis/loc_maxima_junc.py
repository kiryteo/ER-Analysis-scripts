import torch
import torchvision.transforms.functional as TF
from skimage import io
from skimage import exposure
from skimage import morphology, segmentation
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, feature, color
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.filters import difference_of_gaussians, window

import graph_connector_modules as gcm
import networkx as nx

import copy
from plantcv import plantcv as pcv
import sknw

# img = io.imread('/localhome/asa420/MIAL/data/confocal-data/Climp/preproc/C1/C1_decon_t000_ch00_proc.png')


def get_local_max_junc():
    ref_skel = io.imread('/localhome/asa420/MIAL/data/confocal-data/ATL/er_mean_proc/atl2_proc_skel.png')

    ref_graph = sknw.build_sknw(ref_skel, multi=True)

    ref_nodes, ref_degree_list = ref_graph.nodes(), ref_graph.degree

    # get only the nodes with degree > 2
    ref_node_coords = np.array([ref_nodes[node]['o'] for node in ref_nodes])

    ref_node_set = np.array([ref_node_coords[i] for i, val in enumerate(ref_degree_list) if val[1] > 2])

    for i in range(90, 100):

        er = io.imread(f'/localhome/asa420/MIAL/data/confocal-data/ATL/preproc/A2/A2_decon_t0{i:02d}_ch00_proc.png')

        skel = io.imread(f'/localhome/asa420/MIAL/data/other_data/confocal_movies/ATL/new_op_jul/skel/A2/A2_decon_t0{i:02d}_ch00_skel.png')    

        graph = sknw.build_sknw(skel, multi=True)

        nodes, degree_list = graph.nodes(), graph.degree

        # get only the nodes with degree > 2
        node_coords = np.array([nodes[node]['o'] for node in nodes])

        node_set = np.array([node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2])

        local_max_coords = peak_local_max(er, min_distance=5, threshold_abs=0, indices=True)

        plt.imshow(er, cmap='gray')
        plt.plot(ref_node_set[:, 1], ref_node_set[:, 0], 'o', markerfacecolor='None', markeredgecolor='green', mew=2)
        plt.plot(node_set[:, 1], node_set[:, 0], 'r.')
        plt.plot(local_max_coords[:, 1], local_max_coords[:, 0], 'b.')

        plt.show()


        # updated_dict, g_nodes_array = gcm.get_updated_neighbor_dict(graph)


        # # Create a copy of the graph for node connection
        # temp_graph = copy.deepcopy(graph)

        # er_input = io.imread(er)
        # cost_arr = np.ones((128, 128))

        # for node in dict(graph.degree()):
        #     # Access the first element of graph.neighbors
        #     neighbor = next(iter(graph.neighbors(node)))

        # #     # Connect the node to its neighbor
        #     gcm.connect_nodes(er_input, temp_graph, node, neighbor, updated_dict, cost_arr, g_nodes_array)

        # # # Create a copy of the graph
        # temp_graph_2 = copy.deepcopy(temp_graph)

        # # # Go through each node and adjust the degree
        # for node in temp_graph.nodes():
        #     gcm.process_node(temp_graph_2, node)

        # temp_graph_3 = copy.deepcopy(temp_graph_2)
        # for node in temp_graph_2.nodes():
        #     gcm.process_node(temp_graph_3, node)

        # node_set, degree_list = temp_graph_3.nodes, temp_graph_3.degree

        # node_coords = np.array([node_set[node]['o'] for node in node_set])

        # # return [node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2], temp_graph_3

        # return [node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2], graph


# get_local_max_junc()
# exit()


def add_new_nodes():
    # get reference junctions from reference skel

    ref_skel = io.imread('/localhome/asa420/MIAL/data/confocal-data/ATL/er_mean_proc/atl2_proc_skel.png')

    ref_graph = sknw.build_sknw(ref_skel, multi=True)

    ref_nodes, ref_degree_list = ref_graph.nodes(), ref_graph.degree

    # get only the nodes with degree > 2
    ref_node_coords = np.array([ref_nodes[node]['o'] for node in ref_nodes])

    ref_node_set = np.array([ref_node_coords[i] for i, val in enumerate(ref_degree_list) if val[1] > 2])

    # get current frame junctions from current frame skel

    skel = io.imread('/localhome/asa420/MIAL/data/other_data/confocal_movies/ATL/new_op_jul/skel/A2/A2_decon_t070_ch00_skel.png')

    graph = sknw.build_sknw(skel, multi=True)

    nodes, degree_list = graph.nodes(), graph.degree

    # get only the nodes with degree > 2
    node_coords = np.array([nodes[node]['o'] for node in nodes])

    # node_set = np.array([node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2])

    # node_set = [node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2]

    node_set = [node_coords[i] for i, val in enumerate(degree_list)]# if val[1] > 2]

    # get local maxima from current frame

    er = io.imread('/localhome/asa420/MIAL/data/confocal-data/ATL/preproc/A2/A2_decon_t070_ch00_proc.png')

    # local_max_coords: ndarray
    local_max_coords = peak_local_max(er, min_distance=5, threshold_abs=0, indices=True)


    # Find the local max which are within the neighborhood of the reference junctions and there is no junction from node set within the neighborhood
    # For such local max, find the closest coordinate in the graph and add a node there
    # connect the nearest local max nodes.


    result = []

    for lmc in local_max_coords:
        is_within_nbr = False

        for node in node_set:
            if abs(lmc[0] - node[0]) <= 5//2 and abs(lmc[1] - node[1]) <= 5//2:
                is_within_nbr = True
                break

        if any(abs(ref_node[0] - lmc[0]) <= 5//2 and abs(ref_node[1] - lmc[1]) <= 5//2 for ref_node in ref_node_set):
            result.append(lmc)

    # for res in result:
    #     node_set.append([res[0], res[1]])

    node_set = np.array(node_set)

    # current_nodes_len = len(nodes)

    # for i, res in enumerate(result):
    #     graph.add_node(current_nodes_len + i, pts=res)


    plt.imshow(er, cmap='gray')

    for (s, e) in graph.edges():
        ps = graph[s][e][0]['pts']
        plt.plot(ps[:, 1], ps[:, 0], 'green')

    # for node in nodes:
    #     print(nodes[node]['pts'])
        # plt.plot(nodes[node]['pts'][1], nodes[node]['pts'][0], 'r.')

    plt.plot(ref_node_set[:, 1], ref_node_set[:, 0], 'o', markerfacecolor='None', markeredgecolor='yellow', mew=2)

    # plt.plot(node_set[:, 1], node_set[:, 0], 'b.')

    # # plt.plot(ps[:,1], ps[:,0], 'r.')

    plt.plot(node_set[:, 1], node_set[:, 0], 'r.')

    plt.plot(local_max_coords[:, 1], local_max_coords[:, 0], 'b.')

    plt.show()


    exit()

    # # print(local_max_coords)
    # # Define the coordinates list
    # coordinates = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]

    # # Define the location (x, y)
    # x, y = 4, 5  # Replace with your desired location

    # # Define the size of the neighborhood (5x5 square)
    # neighborhood_size = 5

    # # Find coordinates within the 5x5 neighborhood
    # result = [(a, b) for a, b in coordinates if abs(a - x) <= neighborhood_size // 2 and abs(b - y) <= neighborhood_size // 2]

    # print("Coordinates within a {}x{} neighborhood centered at ({}, {}):".format(neighborhood_size, neighborhood_size, x, y))
    # print(result)




add_new_nodes()


exit()


img = io.imread('/localhome/asa420/MIAL/data/confocal-data/Climp/er_mean_proc/climp1_er_mean_proc.png')

skel = io.imread('/localhome/asa420/MIAL/data/confocal-data/Climp/er_mean_proc/climp1_proc_skel.png')


graph = sknw.build_sknw(skel, multi=True)

nodes, degree_list = graph.nodes(), graph.degree

# get only the nodes with degree > 2
node_coords = np.array([nodes[node]['o'] for node in nodes])

node_set = [node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2]

# filt = difference_of_gaussians(img, 1, 2)

# local_max_coords = peak_local_max(filt, min_distance=5, threshold_abs=0, indices=True)

local_max_coords = peak_local_max(img, min_distance=5, threshold_abs=0, indices=True)

plt.imshow(img, cmap='gray')
plt.plot(node_coords[:, 1], node_coords[:, 0], 'r.')
plt.plot(local_max_coords[:, 1], local_max_coords[:, 0], 'b.')

plt.show()

# # show where filt > 0
# op = np.where(filt > 0, 1, 0)

# plt.imshow(op, cmap='gray')
# plt.show()

exit()


# Load the image
image = img
# Convert to grayscale if it's a color image
# if image.shape[-1] == 3:
#     image = color.rgb2gray(image)

# Define the range of scales to explore
min_sigma = 1
max_sigma = 16
num_scales = 8
sigma_vals = np.linspace(min_sigma, max_sigma, num_scales)

# Create an empty list to store local maxima coordinates
local_maxima_coords = []

# Loop over the scales
for sigma in sigma_vals:
    # Apply Gaussian smoothing at the current scale
    smoothed_image = gaussian(image, sigma=sigma, mode='reflect')

    # Find local maxima in the smoothed image
    local_max_coords = peak_local_max(smoothed_image, min_distance=1, threshold_abs=0, indices=True)

    # Store local maxima coordinates at this scale
    local_maxima_coords.extend([(x, y) for y, x in local_max_coords])

# Convert the list of coordinates to a NumPy array
local_maxima_coords = np.array(local_maxima_coords)

# Plot the original image and overlay local maxima
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray')
plt.scatter(local_maxima_coords[:, 1], local_maxima_coords[:, 0], c='r', marker='x', s=20)
plt.title('Local Maxima')
plt.axis('off')
plt.show()

# You can now use local_maxima_coords for further analysis or processing
print(local_maxima_coords)





















exit()



coords = peak_local_max(img, min_distance=3)#, threshold_abs=0.1, exclude_border=False)

# junctions = io.imread('/localhome/asa420/MIAL/data/other_data/confocal_movies/Climp/new_op_jul/junctions/C1/C1_decon_t000_ch00_junc.png')

plt.imshow(img, cmap='gray')
plt.plot(coords[:, 1], coords[:, 0], 'r.')

# junc = np.where(junctions == 255)

# plt.plot(junc[1], junc[0], 'b.')

plt.show()

exit()

a, b = img.shape

c = copy.deepcopy(img)


def generate_neighborhood(matrix, i, j, n):
    neighborhood = []
    half_n = n // 2  # Calculate half of the neighborhood size

    for row_offset in range(-half_n, half_n + 1):
        for col_offset in range(-half_n, half_n + 1):
            row = i + row_offset
            col = j + col_offset

            # Check if the row and col indices are within the matrix bounds
            if 0 <= row < len(matrix) and 0 <= col < len(matrix[0]):
                neighborhood.append(matrix[row][col])

    return neighborhood


def get_nbrhood_mean(img, i, j):
    # nbrs = [img[i-1,j], img[i+1,j], img[i,j-1], img[i,j+1], img[i-1,j-1], img[i-1,j+1], img[i+1,j-1], img[i+1,j+1]]
    nbrs = generate_neighborhood(img, i, j, 3)
    non_zero_nbrs = [x for x in nbrs if x != 0]
    avg = np.mean(non_zero_nbrs)
    return avg




for i in range(1, a-1):
    for j in range(1, b-1):
        if img[i,j] != 0:
            val = get_nbrhood_mean(img, i, j)
            if val > img[i, j]:
                c[i,j] = val
            else:
                c[i,j] = img[i,j]


# plt.imshow(c, cmap='gray')
# plt.show()

# seg = segmentation.morphological_geodesic_active_contour(img, 10, 'circle')#, 0.1, balloon=-1)
# seg = segmentation.random_walker(c, beta=10, mode='bf')

seg = segmentation.watershed(img)

plt.imshow(seg, cmap='gray')
plt.show()


# p2, p98 = np.percentile(img, (2, 98))
# img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

# # img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.01)

# plt.imshow(img_rescale, cmap='gray')
# plt.show()

# exit()

# img = TF.to_tensor(img)

# img = img.unsqueeze(0)

# # print(img.shape)
# from soft_frangi.soft_frangi_filter2d import SoftFrangiFilter2D

# soft_frangi_filter = SoftFrangiFilter2D(1, 7, [2,4,8], 0.5, 0.5, 'cpu')

# soft_frangi_response = soft_frangi_filter(img)

# print(soft_frangi_response.shape)


# plt.imshow(soft_frangi_response[0,0,:,:], cmap='gray')
# plt.show()