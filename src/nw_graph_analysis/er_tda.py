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
import pickle as pkl
import pandas as pd
import community
import math

import tnetwork as tn


group_data = {'ATL': 26, 'Climp': 31, 'Control': 31, 'RTN': 29}

group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}

# skel1 = imageio.imread('/localhome/asa420/MIAL/data/other_data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t000_ch00_skel.png')

# skel2 = imageio.imread('/localhome/asa420/MIAL/data/other_data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t001_ch00_skel.png')

# skel3 = imageio.imread('/localhome/asa420/MIAL/data/other_data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t002_ch00_skel.png')


# g1 = sknw.build_sknw(skel1, multi=True, iso=False)
# g2 = sknw.build_sknw(skel2, multi=True, iso=False)
# g3 = sknw.build_sknw(skel3, multi=True, iso=False)


# graphs = [g1, g2, g3]






def graph_entropy_evolution(graphs):
    data = []
    for graph in graphs:
        # Calculate the degree distribution
        degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)

        # Calculate the probabilities of each degree
        degree_counts = {}
        total_degrees = len(degree_sequence)
        for degree in degree_sequence:
            if degree in degree_counts:
                degree_counts[degree] += 1
            else:
                degree_counts[degree] = 1

        degree_probabilities = [count / total_degrees for count in degree_counts.values()]

        # Calculate the Shannon entropy
        entropy = -sum(p * math.log2(p) for p in degree_probabilities)

        data.append(entropy)
    
    return data


def assortativity_evolution(graphs):
    return [nx.degree_assortativity_coefficient(graph) for graph in graphs]


def clustering_coefficient_evolution(graphs):
    return [nx.average_clustering(graph) for graph in graphs]


def pearson_correlation_evolution(graphs):
    return [nx.degree_pearson_correlation_coefficient(graph) for graph in graphs]


def local_efficiency_evolution(graphs):
    return [nx.local_efficiency(graph) for graph in graphs]


def global_efficiency_evolution(graphs):
    return [nx.global_efficiency(graph) for graph in graphs]


def avg_degree_connectivity_evolution(graphs):
    return [nx.average_degree_connectivity(graph) for graph in graphs]


def graph_density_evolution(graphs):
    return [nx.density(graph) for graph in graphs]


def avg_degree_evolution(graphs):
    return [sum(dict(graph.degree()).values()) / len(graph) for graph in graphs]


def transitivity_evolution(graphs):
    return [nx.transitivity(graph) for graph in graphs]


def degree_centrality_evolution(graphs):
    return [nx.degree_centrality(graph) for graph in graphs]

def betweenness_centrality_evolution(graphs):
    return [nx.betweenness_centrality(graph) for graph in graphs]

def closeness_centrality_evolution(graphs):
    return [nx.closeness_centrality(graph) for graph in graphs]

def eigenvector_centrality_evolution(graphs):
    return [nx.eigenvector_centrality(graph) for graph in graphs]

def katz_centrality_evolution(graphs):
    return [nx.katz_centrality(graph) for graph in graphs]


def calculate_graph_similarity(graph1, graph2):
    # You can use various graph similarity metrics here
    # For example, graph edit distance, structural similarity index, Jaccard index, etc.
    # Here, we use the number of common edges as a simple example.
    common_edges = len(set(graph1.edges()).intersection(set(graph2.edges())))
    total_edges = len(set(graph1.edges()).union(set(graph2.edges())))
    return common_edges / total_edges

# # Calculate the graph evolution rate for each pair of consecutive time steps
# for i in range(1, len(graphs)):
#     similarity = calculate_graph_similarity(graphs[i - 1], graphs[i])
#     evolution_rate = 1 - similarity  # You can also normalize this value if needed
#     graph_evolution_rates.append(evolution_rate)

def jaccard_similarity(graph1, graph2):
    nodes1 = set(graph1.nodes)
    nodes2 = set(graph2.nodes)
    intersection = len(nodes1.intersection(nodes2))
    union = len(nodes1.union(nodes2))
    return intersection / union

def get_jaccard_similarity(graphs):
    return [
        jaccard_similarity(graphs[i], graphs[i + 1])
        for i in range(len(graphs) - 1)
    ]


def get_jaccard_edge_similarity(evolving_graphs):
    jaccard_edge_similarity = []
    for i in range(len(evolving_graphs) - 1):
        edge_set1 = set(evolving_graphs[i].edges)
        edge_set2 = set(evolving_graphs[i + 1].edges)
        jaccard_index = len(edge_set1.intersection(edge_set2)) / len(edge_set1.union(edge_set2))
        jaccard_edge_similarity.append(jaccard_index)

    return jaccard_edge_similarity


def get_jaccard_similarity_data(group):
    var_data = []
    graph_data = pkl.load(open(f'{group.lower()}_graph_data.pkl', 'rb'))
    for series in graph_data:
        # sim = get_jaccard_similarity(series)
        sim = get_jaccard_edge_similarity(series)
        var_data.extend(sim)
    return var_data


def plot_jaccard_similarity():
    
    atl = get_jaccard_similarity_data('ATL')
    climp = get_jaccard_similarity_data('Climp')
    control = get_jaccard_similarity_data('Control')
    rtn = get_jaccard_similarity_data('RTN')

    # atl = np.array(atl).T
    # climp = np.array(climp).T
    # control = np.array(control).T
    # rtn = np.array(rtn).T

    df = pd.DataFrame()
    df['Jaccard Similarity'] = pd.Series(atl + climp + control + rtn)

    df['Group'] = pd.Series(['ATL']*len(atl) + ['Climp']*len(climp) + ['Control']*len(control) + ['RTN']*len(rtn))

    ax = sns.boxplot(x='Group', y='Jaccard Similarity', data=df, showfliers=False)

    plt.show()




# atl_graph_data shape: (26, 100)
atl_graph_data = pkl.load(open('atl_graph_data.pkl', 'rb'))

for series in atl_graph_data:
    entropy_evolution = [graph_entropy_evolution(series)]

# plot the entropy evolution
plt.plot(entropy_evolution)
plt.show()




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






#########################

# Structural Intergraph Features:

#     Structural features capture the overall topology or structure of the evolving graphs and their changes over time.
#     Examples of structural features:
#         Graph Similarity: Measure the similarity or dissimilarity between consecutive snapshots. Techniques like graph edit distance, Jaccard similarity, or cosine similarity can be used.
#           Jaccard similarity done, cosine not possible due to different vector lengths
#         Centrality Measures: Compute centrality measures (e.g., degree centrality, betweenness centrality) for nodes and track how they change over time.
#            degree, betnness centrality done
#         Clustering Coefficients: Calculate the clustering coefficient for nodes and observe how it evolves.
#            clustering coeff done

# Temporal Intergraph Features:

#     Temporal features focus on the dynamics and temporal patterns of the evolving graphs.
#     Examples of temporal features:
#         Edge Creation/Deletion: Count the number of edges added or removed between consecutive snapshots.
            # NOT Sure
#         Graph Density: Measure the density of each graph snapshot, indicating how connected the nodes are at each time step.
            # DONE
#         Network Motifs: Identify and track the occurrence of specific network motifs or subgraphs.
            # TODO

# Dynamic Intergraph Features:

#     Dynamic features consider how individual nodes or edges change their properties over time.
#     Examples of dynamic features:
#         Node Degree Evolution: Track how the degree (number of connections) of specific nodes changes over time. -> DONE
#         Edge Weight Evolution: If your graphs have weighted edges, analyze how edge weights change. -> NOT Sure
#         Community Evolution: Detect communities in each snapshot and track how nodes move between communities. -> TODO

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
            # Graph alignment for iso and fuzzy regions -> TODO

# Graphlet-Based Measures:

#     Graphlet-based similarity measures capture evolving graph patterns or motifs. -> TODO
#     These methods consider the frequencies and distributions of specific subgraphs (graphlets) across different time steps.
#     By comparing the graphlet profiles of evolving graphs, you can assess their similarity.
            # NOT sure

# Dynamic Graph Embeddings:

#     Dynamic graph embeddings aim to project evolving graphs into lower-dimensional spaces while preserving structural and temporal information.
#     Techniques like DynamicTriad and dynGEM provide embeddings that can be used for similarity computation.
#     Once you have embeddings, you can apply traditional similarity measures in the embedding space.
            # TODO

# Dynamic Graph Edit Distance:

#     Similar to traditional graph edit distance, dynamic graph edit distance measures quantify the dissimilarity between evolving graphs by considering the operations required to transform one graph into another.
#     These measures account for changes in node and edge sets across time steps.
            # Not feasible

    # Graph Diffusion:
    #     Graph diffusion methods capture how information or influence propagates through a graph over time.
    #     You can model evolving graphs as time-dependent diffusion processes, where nodes exchange information based on the evolving connectivity.
    #     Similarity between evolving graphs can be assessed by comparing the dynamics of information diffusion, such as the spread of influence or labels.
        # TODO




# Temporal Metrics:

#     Graph Density: The ratio of actual edges to possible edges in a graph at each time step. -> DONE

#     Edge Turnover Rate: The rate at which edges are added or removed between consecutive time steps. -> jaccaard similarity

#     Node Membership Stability: Measures the persistence of nodes across time frames. -> TODO

#     Graph Evolution Rate: Measures the overall change in the graph structure over time.

#     Graph Growth: The increase in the number of nodes and edges over time.


# Centrality Metrics:

#     Degree Centrality: The number of edges connected to a node. -> DONE

#     Betweenness Centrality: Measures the extent to which a node lies on paths between other nodes. -> DONE

#     Closeness Centrality: Measures how close a node is to all other nodes in terms of shortest paths. -> DONE

#     Eigenvector Centrality: Takes into account a node's connections to high-degree nodes. -> DONE

#     Katz Centrality: Incorporates the number of paths of different lengths to assess node importance. -> DONE



# Link Prediction Metrics:

#     Common Neighbors: Counts the number of common neighbors between two nodes. -> TODO

#     Jaccard Similarity: Measures the similarity of sets of neighbors between two nodes. -> TODO

#     Adamic-Adar Index: Assigns higher importance to common neighbors with lower degrees. -> TODO

#     Resource Allocation Index: Similar to Adamic-Adar but accounts for neighbor degree. -> TODO

#     Preferential Attachment: Measures the likelihood of forming a new edge based on node degrees. -> NOT SURE

# Clustering Metrics:

#     Clustering Coefficient: Measures the extent to which nodes in a neighborhood form cliques. -> DONE

#     Transitivity: A global measure of clustering in the graph. -> DONE

#     Local Clustering Coefficient: Measures clustering at the node level. -> DONE




# Network Robustness Metrics:

#     Connectivity: Measures the degree to which a graph remains connected as edges are removed.

#     Diameter: The maximum shortest path length in the graph.

#     Average Path Length: The average length of the shortest paths in the graph.

#     Network Resilience: Measures the ability of the graph to withstand random failures or targeted attacks. -> NOT SURE




# Information Diffusion Metrics:

#     Influence Spread: Measures how information or influence propagates through the graph.

#     Cascade Size: The number of nodes affected by a contagion or information cascade.

#     Time to Cascade: Measures how quickly a cascade spreads through the network.




# Graph Similarity Metrics:

#     Graph Alignment: Aligning nodes and edges between two graphs to assess their similarity.




# Other Metrics:

#     Assortative Mixing: Measures the correlation between the degrees of connected nodes.

#     Core-Periphery Structure: Identifies nodes that form a densely connected core and a sparsely connected periphery.

#     Motif Analysis: Identifies recurring subgraph patterns within the evolving graph.

#     Temporal Motif Analysis: Extends motif analysis to account for temporal patterns.

#     Graph Entropy: Measures the randomness or predictability of the graph structure over time. - DONE

# Community Detection Metrics:

#     Modularity: Measures the strength of the division of a network into communities.

#     Community Size: The number of nodes within each community over time.

#     Community Evolution: How communities change or merge over time.

#     Community Persistence: Measures the lifespan of communities in the evolving graph.

#     Overlap Coefficient: Measures the degree to which nodes belong to multiple communities.








# communities_generator = nx.community.girvan_newman(graphs[0])

# top_level_communities = next(communities_generator)
# print(top_level_communities)
# next_level_communities = next(communities_generator)

# print(sorted(map(sorted, next_level_communities)))

# all_communities = []

# for graph in graphs:
#     partition = community.best_partition(graph)
#     all_communities.append(partition)

# jaccard_threshold = 0.5

# stable_communities = []

# for i in range(1, len(all_communities)):
#     for j in range(i):
#         jaccard_index = len(set(all_communities[i].values()).intersection(set(all_communities[j].values()))) / len(set(all_communities[i].values()).union(set(all_communities[j].values())))
#         if jaccard_index >= jaccard_threshold:
#             stable_communities.append((i, j, set(all_communities[i].values())))

# # Print the communities and stable communities
# for i, communities in enumerate(all_communities):
#     print(f"Communities at Time {i}: {communities}")

# for i, j, stable_community in stable_communities:
#     print(f"Stable Community between Time {j} and Time {i}: {stable_community}")



# dg_sn = tn.DynGraphSN()
# dg_sn.add_interactions_from(graphs[0])
# dg_sn.add_interactions_from(graphs[1])
# dg_sn.add_interactions_from(graphs[2])

# print(dg_sn.interactions())

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


# Initialize a list to store node membership stability for each node
# node_membership_stability = []

# # Create NetworkX graphs from adjacency matrices
# graphs = [nx.Graph(adj_matrix) for adj_matrix in adj_matrices]

# # Define the 3x3 neighborhood proximity
# neighborhood_size = 1  # Adjust as needed based on your data

# # Function to check if two nodes are within the neighborhood proximity
# def is_within_neighborhood(node1, node2):
#     pos1 = graphs[i].nodes[node1]['pos']
#     pos2 = graphs[i - 1].nodes[node2]['pos']
#     return abs(pos1[0] - pos2[0]) <= neighborhood_size and abs(pos1[1] - pos2[1]) <= neighborhood_size

# # Calculate node membership stability for each node
# for node in graphs[0].nodes():
#     membership = set([node])
#     persistence = 1.0  # Initialize persistence to 1 for nodes present in the first time step

#     for i in range(1, len(graphs)):
#         for neighbor in graphs[i].nodes():
#             if is_within_neighborhood(node, neighbor):
#                 membership.add(neighbor)
#             else:
#                 persistence = persistence * (1 - 1 / (i + 1))

#     node_membership_stability.append((node, membership, persistence))