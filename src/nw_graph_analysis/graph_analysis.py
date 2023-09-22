import sknw
import imageio
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def create_graph_data_pickles(group):
    graph_data = []
    group_len = {'ATL': 26, 'Climp': 31, 'RTN': 29, 'Control':31}
    group_prefix = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}
    for ser in range(1, group_len[group]+1):
        ser_data = []
        for frame in range(100):
            skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/{group_prefix[group]}{ser}/{group_prefix[group]}{ser}_decon_t0{frame:02d}_ch00_skel.png')
            graph = sknw.build_sknw(skel, multi=False)

            ser_data.append(graph)
        graph_data.append(ser_data)

    with open(f'{group.lower()}_graph_data.pkl', 'wb') as f:
        pickle.dump(graph_data, f)



# def compute_average_neighbor_degree(graphs):
#     # provides per node avg degree -> now we can either track the evolution for each node avg degree 
#     # NOT REALLY TRACTABLE
#     return [nx.average_neighbor_degree(graph) for graph in graphs]


# def compute_average_node_connectivity(graphs):
#     # may not be tractable
#     return [nx.average_node_connectivity(graph) for graph in graphs]


# def compute_number_of_cliques(graphs):
#     return [sum(1 for _ in nx.find_cliques(graph)) for graph in graphs]


# def compute_clique_number(graphs):
#     return [max(len(c) for c in nx.find_cliques(graph)) for graph in graphs]


# def compute_num_connected_components(graphs):
#     return [nx.number_connected_components(graph) for graph in graphs]


# def compute_diameter(graphs):
#     return [nx.diameter(graph) for graph in graphs]


# def compute_radius(graphs):
#     return [nx.radius(graph) for graph in graphs]


# def compute_s_metric(graphs):
#     return [nx.s_metric(graph) for graph in graphs]

import math

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


def assortativity_coefficient_evolution(graphs):
    # float
    return [nx.degree_assortativity_coefficient(graph) for graph in graphs]


def clustering_coefficient_evolution(graphs):
    # float
    return [nx.average_clustering(graph) for graph in graphs]


def pearson_correlation_coefficient_evolution(graphs):
    # float
    return [nx.degree_pearson_correlation_coefficient(graph) for graph in graphs]


def local_efficiency_evolution(graphs):
    # float
    return [nx.local_efficiency(graph) for graph in graphs]


def global_efficiency_evolution(graphs):
    # float
    return [nx.global_efficiency(graph) for graph in graphs]


# def avg_degree_connectivity_evolution(graphs):
#     return [nx.average_degree_connectivity(graph) for graph in graphs]


def graph_density_evolution(graphs):
    # float
    return [nx.density(graph) for graph in graphs]


def avg_degree_evolution(graphs):
    # float
    return [sum(dict(graph.degree()).values()) / len(graph) for graph in graphs]


def transitivity_evolution(graphs):
    # float
    return [nx.transitivity(graph) for graph in graphs]


# def degree_centrality_evolution(graphs):
#     return [nx.degree_centrality(graph) for graph in graphs]

# def betweenness_centrality_evolution(graphs):
#     return [nx.betweenness_centrality(graph) for graph in graphs]

# def closeness_centrality_evolution(graphs):
#     return [nx.closeness_centrality(graph) for graph in graphs]

# def eigenvector_centrality_evolution(graphs):
#     return [nx.eigenvector_centrality(graph) for graph in graphs]

# def katz_centrality_evolution(graphs):
#     return [nx.katz_centrality(graph) for graph in graphs]


def calculate_graph_similarity(graph1, graph2):
    # You can use various graph similarity metrics here
    # For example, graph edit distance, structural similarity index, Jaccard index, etc.
    # Here, we use the number of common edges as a simple example.
    common_edges = len(set(graph1.edges()).intersection(set(graph2.edges())))
    total_edges = len(set(graph1.edges()).union(set(graph2.edges())))
    return common_edges / total_edges


def graph_similarity_evolution(graphs):
    # Calculate the graph similarity for each pair of consecutive time steps
    return [
        calculate_graph_similarity(graphs[i], graphs[i + 1])
        for i in range(len(graphs) - 1)
    ]


def jaccard_edge_similarity_evolution(evolving_graphs):
    jaccard_edge_similarity = []
    for i in range(len(evolving_graphs) - 1):
        edge_set1 = set(evolving_graphs[i].edges)
        edge_set2 = set(evolving_graphs[i + 1].edges)
        jaccard_index = len(edge_set1.intersection(edge_set2)) / len(edge_set1.union(edge_set2))
        jaccard_edge_similarity.append(jaccard_index)

    return jaccard_edge_similarity


def jaccard_similarity(graph1, graph2):
    nodes1 = set(graph1.nodes)
    nodes2 = set(graph2.nodes)
    intersection = len(nodes1.intersection(nodes2))
    union = len(nodes1.union(nodes2))
    return intersection / union

def jaccard_similarity_evolution(graphs):
    return [
        jaccard_similarity(graphs[i], graphs[i + 1])
        for i in range(len(graphs) - 1)
    ]





def temporal_graph_analyzer(group, feature_func):
    # get temporal graphs per series per group

    with open(f'{group.lower()}_graph_data.pkl', 'rb') as f:
        graph_data = pickle.load(f)

    # # get temporal graph properties per series per group
    feature_data = []

    # graph_data = np.array(graph_data)
    # print(graph_data.shape)
    # for ser in range(graph_data.shape[0]):
    #     ser_data = []
        # for frame in range(100):
        #     dac = feature_func(graph_data[ser, frame])
        #     ser_data.append(dac)
        # feature_data.append(ser_data)

    for series in graph_data:
        feature_data.append(feature_func(series))

    # mean_data = np.mean(feature_data, axis=0)
    # std_data = np.std(feature_data, axis=0)
    # print(mean_data.shape)
    # print(mean_data)

    # return np.array(feature_data).flatten()
    return np.array(feature_data)
    # return mean_data, std_data


def qcd(xs):
    q1, q3 = np.quantile(xs, [0.25, 0.75])
    return (q3-q1)/(q3+q1)

def get_evolution_plots(measure):
    atl = temporal_graph_analyzer('ATL', measure)
    climp = temporal_graph_analyzer('Climp', measure)
    control = temporal_graph_analyzer('Control', measure)
    rtn = temporal_graph_analyzer('RTN', measure)


    # atl_data = [abs(np.fft.fftshift(np.fft.fft(series))) for series in atl]

    # climp_data = [abs(np.fft.fftshift(np.fft.fft(series))) for series in climp]

    # control_data = [abs(np.fft.fftshift(np.fft.fft(series))) for series in control]

    # rtn_data = [abs(np.fft.fftshift(np.fft.fft(series))) for series in rtn]

    # plt.plot(atl_data, label='ATL')
    # plt.plot(climp_data, label='Climp')
    # plt.plot(control_data, label='Control')
    # plt.plot(rtn_data, label='RTN')

    # df = pd.DataFrame()

    # # Tframe = []
    # # vals = []
    # # group_name = []

    # # for _ in range(1, 27):
    # #     Tframe.extend(np.arange(100))

    # # for _ in range(1, 32):
    # #     Tframe.extend(np.arange(100))

    # # for _ in range(1, 32):
    # #     Tframe.extend(np.arange(100))

    # # for _ in range(1, 30):
    # #     Tframe.extend(np.arange(100))


    # # group_name.extend(['ATL']*2600)
    # # group_name.extend(['Climp']*3100)
    # # group_name.extend(['Control']*3100)
    # # group_name.extend(['RTN']*2900)

    # # vals.extend(np.fft.fft(atl.flatten()))
    # # vals.extend(np.fft.fft(climp.flatten()))
    # # vals.extend(np.fft.fft(control.flatten()))
    # # vals.extend(np.fft.fft(rtn.flatten()))

    # plt.plot(np.fft.fft(atl.flatten()), label='ATL')
    # plt.plot(np.fft.fft(climp.flatten()), label='Climp')
    # plt.plot(np.fft.fft(control.flatten()), label='Control')
    # plt.plot(np.fft.fft(rtn.flatten()), label='RTN')

    # df['Tframe'] = pd.Series(Tframe)
    # df['vals'] = pd.Series(vals)
    # df['group_name'] = pd.Series(group_name)

    # plt.figure(figsize=(12, 6))

    # sns.lineplot(x='Tframe', y='vals', hue='group_name', data=df)

    # measure_name = str(measure).split('_evolution')[0][10:]

    # # plt.xlabel('Time (frames)', fontsize=16)
    # plt.ylabel(measure_name, fontsize=16)
    # plt.title(f'{measure_name} evolution across groups', fontsize=18)

    # plt.savefig(f'graph_evolution_plots/{measure_name}_evolution.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    # plt.close()
    # plt.legend()

    # plt.show()

    # Calculate the mean and standard deviation for each dataset
    # mean_atl = atl.mean(axis=0)
    # # qcd_atl = qcd(atl.flatten())
    mean_atl = abs(np.fft.fftshift(np.fft.fft(atl.mean(axis=0))))
    # # std_atl = atl.std(axis=1)

    # mean_climp = climp.mean(axis=0)
    # # qcd_climp = qcd(climp.flatten())
    mean_climp = abs(np.fft.fftshift(np.fft.fft(climp.mean(axis=0))))
    # # std_climp = climp.std(axis=1)

    # mean_control = control.mean(axis=0)
    # # qcd_control = qcd(control.flatten())
    mean_control = abs(np.fft.fftshift(np.fft.fft(control.mean(axis=0))))
    # # std_control = control.std(axis=1)

    # mean_rtn = rtn.mean(axis=0)
    # # qcd_rtn = qcd(rtn.flatten())
    mean_rtn = abs(np.fft.fftshift(np.fft.fft(rtn.mean(axis=0))))
    # std_rtn = rtn.std(axis=1)

    # Create a figure and axis for the plot
    # plt.figure(figsize=(12, 6))

    # Plot the mean data as a line plot
    # sns.lineplot(x=range(len(mean_atl)), y=mean_atl, label='ATL')
    # sns.lineplot(x=range(len(mean_climp)), y=mean_climp, label='Climp')
    # sns.lineplot(x=range(len(mean_control)), y=mean_control, label='Control')
    # sns.lineplot(x=range(len(mean_rtn)), y=mean_rtn, label='RTN')


    # plt.plot(qcd_atl, label='ATL')
    # plt.plot(qcd_climp, label='Climp')
    # plt.plot(qcd_control, label='Control')
    # plt.plot(qcd_rtn, label='RTN')    

    plt.plot(mean_atl, label='ATL')
    plt.plot(mean_climp, label='Climp')
    plt.plot(mean_control, label='Control')
    plt.plot(mean_rtn, label='RTN')


    # # Fill the standard deviation as a shaded region around the mean
    # plt.fill_between(range(len(mean_atl)), mean_atl - std_atl, mean_atl + std_atl, alpha=0.2)
    # plt.fill_between(range(len(mean_climp)), mean_climp - std_climp, mean_climp + std_climp, alpha=0.2)
    # plt.fill_between(range(len(mean_control)), mean_control - std_control, mean_control + std_control, alpha=0.2)
    # plt.fill_between(range(len(mean_rtn)), mean_rtn - std_rtn, mean_rtn + std_rtn, alpha=0.2)


    # Add labels and a legend
    # plt.xlabel('Time (frames)', fontsize=16)
    # plt.ylabel('Assortativity Coefficient', fontsize=16)
    # plt.title('Assortativity Coefficient evolution across groups', fontsize=18)

    plt.legend()

    # Show the plot
    plt.show()

get_evolution_plots(assortativity_coefficient_evolution)
# get_evolution_plots(clustering_coefficient_evolution)
# get_evolution_plots(pearson_correlation_coefficient_evolution)
# get_evolution_plots(local_efficiency_evolution)
# get_evolution_plots(global_efficiency_evolution)
# get_evolution_plots(graph_density_evolution)
# get_evolution_plots(avg_degree_evolution)
# get_evolution_plots(transitivity_evolution)
# get_evolution_plots(graph_similarity_evolution)
# get_evolution_plots(graph_entropy_evolution)

# get_evolution_plots(jaccard_similarity_evolution)
# get_evolution_plots(jaccard_edge_similarity_evolution)


# def get_graph_edits(group):
#     with open(f'{group.lower()}_graph_data.pkl', 'rb') as f:
#         graph_data = pickle.load(f)

#     graph_data = np.array(graph_data)
#     graph_edits = []
#     for ser in range(graph_data.shape[0]):
#         ser_data = []
#         for frame in range(1, 100):
#             graph_edit = compute_graph_edit_distance(graph_data[ser, frame-1], graph_data[ser, frame])
#             ser_data.append(graph_edit)
#         graph_edits.append(ser_data)

#     return np.array(graph_edits).flatten()








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