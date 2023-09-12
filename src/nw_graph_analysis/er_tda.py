import imageio
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sknw
import gudhi as gd
import grakel
from grakel.kernels import RandomWalk
from grakel.utils import graph_from_networkx
import seaborn as sns
import pandas as pd


group_data = {'ATL': 26, 'Climp': 31, 'Control': 31, 'RTN': 29}

group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}

# skel1 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t000_ch00_skel.png')

# skel2 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t001_ch00_skel.png')

# skel3 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t002_ch00_skel.png')


# g1 = sknw.build_sknw(skel1, multi=True, iso=False)
# g2 = sknw.build_sknw(skel2, multi=True, iso=False)
# g3 = sknw.build_sknw(skel3, multi=True, iso=False)



# def kernel_similarity(graphs):
#     # Create a random walk kernel instance
#     # random_walk_kernel = RandomWalk()

#     # graphlet_sampling = grakel.kernels.GraphletSampling(4)
#     subgraph_matching = grakel.kernels.SubgraphMatching()

#     # Fit and transform your evolving graphs
#     # kernel_matrix = random_walk_kernel.fit_transform(graphs)
#     # kernel_matrix = graphlet_sampling.fit_transform(graphs)
#     kernel_matrix = subgraph_matching.fit_transform(graphs)

#     # Print the kernel matrix
#     print(kernel_matrix)

#     return kernel_matrix[0][1]

# evolving_graphs = [g1, g2, g3]
# grakel_graphs = graph_from_networkx(evolving_graphs)

# k = kernel_similarity(grakel_graphs)
# print(k)

# exit()


def jaccard_similarity(graph1, graph2):
    nodes1 = set(graph1.nodes)
    nodes2 = set(graph2.nodes)
    intersection = len(nodes1.intersection(nodes2))
    union = len(nodes1.union(nodes2))
    return intersection / union

def get_jaccard_similarity(graphs):
    similarities = []
    for i in range(len(graphs)-1):
        similarities.append(jaccard_similarity(graphs[i], graphs[i+1]))

    return similarities


def get_jaccard_edge_similarity(evolving_graphs):
    jaccard_edge_similarity = []
    for i in range(len(evolving_graphs) - 1):
        edge_set1 = set(evolving_graphs[i].edges)
        edge_set2 = set(evolving_graphs[i + 1].edges)
        jaccard_index = len(edge_set1.intersection(edge_set2)) / len(edge_set1.union(edge_set2))
        jaccard_edge_similarity.append(jaccard_index)

    return jaccard_edge_similarity

# def plot_jaccard_similarity(group):
#     var_data = []
#     for num in range(1, group_data[group]+1):
#         graphs = []
#         for frame in range(100):
#             skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/{group_pref[group]}{num}/{group_pref[group]}{num}_decon_t0{frame:02d}_ch00_skel.png')
#             graph = sknw.build_sknw(skel, multi=True, iso=False)
#             graphs.append(graph)
#         sim = get_jaccard_similarity(graphs)
#         var_data.append(sim)
#     return var_data

import pickle as pkl

def get_jaccard_similarity_data(group):
    var_data = []
    graph_data = pkl.load(open(f'{group.lower()}_graph_data.pkl', 'rb'))
    for series in graph_data:
        # sim = get_jaccard_similarity(series)
        sim = get_jaccard_edge_similarity(series)
        var_data.extend(sim)
    return var_data


def plot_jaccard_similarity():
    
    atl = plot_jaccard_similarity('ATL')
    climp = plot_jaccard_similarity('Climp')
    control = plot_jaccard_similarity('Control')
    rtn = plot_jaccard_similarity('RTN')

    # atl = np.array(atl).T
    # climp = np.array(climp).T
    # control = np.array(control).T
    # rtn = np.array(rtn).T

    df = pd.DataFrame()
    df['Jaccard Similarity'] = pd.Series(atl + climp + control + rtn)

    df['Group'] = pd.Series(['ATL']*len(atl) + ['Climp']*len(climp) + ['Control']*len(control) + ['RTN']*len(rtn))

    ax = sns.boxplot(x='Group', y='Jaccard Similarity', data=df, showfliers=False)

    plt.show()


def get_degree_centrality():
    var_data = []
    graph_data = pkl.load(open(f'atl_graph_data.pkl', 'rb'))
    for series in graph_data:
        



# from sklearn.metrics.pairwise import cosine_similarity

# Convert graphs to binary node feature vectors
# def graph_to_vector(graph):
#     num_nodes = len(graph.nodes)
#     vector = [1 if i in graph.nodes else 0 for i in range(num_nodes)]
#     return vector

# Calculate cosine similarity - assumes similar sized vectors
# NOT really a good measure for evolving graphs where the node number changes over time
# One suggested way to handle is padding with zeros for the vectors but that doesn't seem to make sense

# cosine_similarities = []
# for i in range(len(evolving_graphs) - 1):
#     vector1 = graph_to_vector(evolving_graphs[i])
#     vector2 = graph_to_vector(evolving_graphs[i + 1])
#     similarity = cosine_similarity([vector1], [vector2])[0][0]
#     cosine_similarities.append(similarity)

# # Print the cosine similarities
# for i, similarity in enumerate(cosine_similarities):
#     print(f"Cosine Similarity between time steps {i} and {i + 1}: {similarity}")



import networkx as nx
import numpy as np

def graph_diffusion():
    # Number of time steps or diffusion iterations
    num_steps = 3

    # Initialize node influence scores for each evolving graph
    influence_scores_list = []

    # Start a random walk from a seed node for each evolving graph
    seed_node = 1

    for graph in evolving_graphs:
        influence_scores = {node: 0.0 for node in graph.nodes}
        for _ in range(num_steps):
            current_node = seed_node
            while True:
                neighbors = list(graph.neighbors(current_node))
                if not neighbors:
                    break
                next_node = np.random.choice(neighbors)
                influence_scores[next_node] += 1
                current_node = next_node
        influence_scores_list.append(influence_scores)

    # Normalize influence scores for each evolving graph (optional)
    normalized_influence_scores_list = []

    for influence_scores in influence_scores_list:
        total_influence = sum(influence_scores.values())
        normalized_influence_scores = {node: score / total_influence for node, score in influence_scores.items()}
        normalized_influence_scores_list.append(normalized_influence_scores)

    # Print influence scores for each evolving graph
    for i, influence_scores in enumerate(normalized_influence_scores_list):
        print(f"Evolving Graph {i + 1} - Node Influence Scores:")
        for node, score in influence_scores.items():
            print(f"Node {node}: {score}")


# import networkx as nx
# import numpy as np
# from scipy.optimize import linear_sum_assignment



# # Initialize an alignment matrix for each pair of evolving graphs
# alignment_matrices = []

# # Compute alignment between each pair of evolving graphs
# for i in range(len(evolving_graphs)):
#     alignment_matrix = np.zeros((len(evolving_graphs[i]), len(evolving_graphs[i])))
#     for j in range(len(evolving_graphs)):
#         if i == j:
#             continue  # Skip self-alignment
#         # Calculate a similarity/distance matrix (e.g., Jaccard similarity, graph edit distance)
#         # You can choose a suitable similarity measure based on your graph data
#         similarity_matrix = np.zeros((len(evolving_graphs[i]), len(evolving_graphs[j])))
#         # Compute the optimal node mapping using the Hungarian algorithm
#         row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
#         for r, c in zip(row_ind, col_ind):
#             alignment_matrix[r, c] = 1  # Mark nodes as aligned
#     alignment_matrices.append(alignment_matrix)

# # Print the alignment matrices
# for i, matrix in enumerate(alignment_matrices):
#     print(f"Alignment Matrix for Evolving Graph {i + 1}:")
#     print(matrix)



#########################

# Structural Intergraph Features:

#     Structural features capture the overall topology or structure of the evolving graphs and their changes over time.
#     Examples of structural features:
#         Graph Similarity: Measure the similarity or dissimilarity between consecutive snapshots. Techniques like graph edit distance, Jaccard similarity, or cosine similarity can be used.
#         Centrality Measures: Compute centrality measures (e.g., degree centrality, betweenness centrality) for nodes and track how they change over time.
#         Clustering Coefficients: Calculate the clustering coefficient for nodes and observe how it evolves.

# Temporal Intergraph Features:

#     Temporal features focus on the dynamics and temporal patterns of the evolving graphs.
#     Examples of temporal features:
#         Edge Creation/Deletion: Count the number of edges added or removed between consecutive snapshots.
#         Graph Density: Measure the density of each graph snapshot, indicating how connected the nodes are at each time step.
#         Network Motifs: Identify and track the occurrence of specific network motifs or subgraphs.

# Dynamic Intergraph Features:

#     Dynamic features consider how individual nodes or edges change their properties over time.
#     Examples of dynamic features:
#         Node Degree Evolution: Track how the degree (number of connections) of specific nodes changes over time.
#         Edge Weight Evolution: If your graphs have weighted edges, analyze how edge weights change.
#         Community Evolution: Detect communities in each snapshot and track how nodes move between communities.

# Feature Extraction:

#     Write code to extract the selected intergraph features from your evolving graph data.
#     You may need to iterate through the graph snapshots, compute the relevant measures, and store them in a structured format, such as a DataFrame.

# Analysis and Visualization:

#     Analyze and visualize the intergraph features to gain insights into the evolving graphs.
#     Use statistical analysis, data visualization libraries (e.g., Matplotlib or Seaborn), or dimensionality reduction techniques (e.g., PCA) to explore the data.

# Machine Learning: Optionally, you can use the extracted intergraph features as input features for machine learning models to make predictions or classifications related to the evolving graphs.













################################

# Graph Alignment:

#     Graph alignment techniques aim to find correspondences between nodes or subgraphs in different snapshots of evolving graphs.
#     Alignment-based measures quantify the similarity by considering how well nodes or subgraphs align across time steps.
#     Methods like GraRep and GEALIGN are examples of graph alignment approaches.

# Graphlet-Based Measures:

#     Graphlet-based similarity measures capture evolving graph patterns or motifs.
#     These methods consider the frequencies and distributions of specific subgraphs (graphlets) across different time steps.
#     By comparing the graphlet profiles of evolving graphs, you can assess their similarity.

# Dynamic Graph Embeddings:

#     Dynamic graph embeddings aim to project evolving graphs into lower-dimensional spaces while preserving structural and temporal information.
#     Techniques like DynamicTriad and dynGEM provide embeddings that can be used for similarity computation.
#     Once you have embeddings, you can apply traditional similarity measures in the embedding space.

# Dynamic Graph Edit Distance:

#     Similar to traditional graph edit distance, dynamic graph edit distance measures quantify the dissimilarity between evolving graphs by considering the operations required to transform one graph into another.
#     These measures account for changes in node and edge sets across time steps.

################################