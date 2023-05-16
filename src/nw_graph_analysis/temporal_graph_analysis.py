import sknw
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import data
from junction_analysis_modules import JunctionAnalysis as JA
import imageio
import pickle

junc_analysis = JA('/localhome/asa420/MIAL/data/confocal_movies/')


groups = {'ATL': 26, 'Climp': 31, 'RTN': 29, 'Control':31}
group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}

def get_temporal_graph(group, num):
    er_prefix = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/std_egfp/'
    skel_prefix = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/'
    graph_data = []
    for i in range(100):
        er_path = f'{er_prefix}{group_pref[group]}{num}_decon_t0{i:02d}_ch00_std.png'
        skel_path = f'{skel_prefix}{group_pref[group]}{num}/{group_pref[group]}{num}_decon_t0{i:02d}_ch00_skel.png'
        _, graph = junc_analysis.get_junctions(er_path, skel_path)
        graph_data.append(graph)
    
    return graph_data


def get_group_temporal_graphs(group):
    num_series = groups[group]
    return [get_temporal_graph(group, i) for i in range(1, num_series+1)]


def create_group_graph_pickles(group):
    group_graphs = get_group_temporal_graphs(group)
    with open(f'{group}_graphs.pkl', 'wb') as f:
        pickle.dump(group_graphs, f)


# create_group_graph_pickles('ATL')
# create_group_graph_pickles('Climp')
# create_group_graph_pickles('RTN')
# create_group_graph_pickles('Control')

# atl_graphs = pickle.load(open('ATL_graphs.pkl', 'rb'))


# a1 = atl_graphs[0]
# for i in range(100):
#     print(len(a1[i].nodes()))
# exit()

# for i in range(26):
#     print(len(atl_graphs[i][0].nodes()))
# exit()


def get_mean_skel_graphs(group, num_series):

    data = []
    for num in range(1, num_series+1):
        skel_mean_path = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/er_mean_proc/{group.lower()}{num}_proc_skel.png'

        graph = sknw.build_sknw(imageio.imread(skel_mean_path), multi=True, iso=False)
        data.append(graph)

    with open(f'{group}_mean_graphs.pkl', 'wb') as f:
        pickle.dump(data, f)


# get_mean_skel_graphs('ATL', 26)
# get_mean_skel_graphs('Climp', 31)
# get_mean_skel_graphs('RTN', 29)
# get_mean_skel_graphs('Control', 31)

import networkx as nx

# Step 1: Define subgraph size
num_nodes = 5
num_edges = 8

# Step 2: Snapshot-based approach
# Assume 'snapshots' is a list of NetworkX graphs representing the temporal graph snapshots

# Initialize variables to track the most consistent subgraph
most_consistent_subgraph = None
highest_consistency_score = 0


snapshots = pickle.load(open('ATL_graphs.pkl', 'rb'))[0]


# print(snapshots)
subgraphs = [snapshots[0].subgraph(c) for c in nx.connected_components(snapshots[0])]
# print(subgraphs)
print(subgraphs[0].nodes())
print(subgraphs[0].edges())
exit()

import networkx as nx

# Step 1: Define subgraph size
num_nodes = 5
num_edges = 8

# Step 2: Snapshot-based approach
# Assume 'snapshots' is a list of NetworkX graphs representing the temporal graph snapshots

# Initialize variables to track the most consistent subgraph
most_consistent_subgraph = None
highest_consistency_score = 0

for i in range(len(snapshots)):
    snapshot = snapshots[i]

    # Step 2a: Extract subgraphs
    subgraphs = [snapshot.subgraph(c) for c in nx.connected_components(snapshot)]
    
    for subgraph in subgraphs:
        # Step 2b: Compare subgraph consistency
        consistency_score = 0
        for j in range(i+1, len(snapshots)):
            next_snapshot = snapshots[j]
            next_subgraphs = [next_snapshot.subgraph(c) for c in nx.connected_components(next_snapshot)]

            for next_subgraph in next_subgraphs:
                # Compare the number of nodes and edges between subgraphs
                if len(subgraph.nodes) == len(next_subgraph.nodes) == num_nodes and len(subgraph.edges) == len(next_subgraph.edges) == num_edges:
                    consistency_score += 1

        # Step 2c: Select the most consistent subgraph
        if consistency_score > highest_consistency_score:
            highest_consistency_score = consistency_score
            most_consistent_subgraph = subgraph

# Step 3: Event-based approach
# Assume 'events' is a list of events describing changes in the graph structure

# Initialize variables to track the most consistent subgraph
# most_consistent_subgraph = None
# highest_consistency_score = 0

# for i in range(len(events)):
#     event = events[i]
#     # Update the graph based on the event (add/remove nodes/edges)

#     # Step 3b: Identify persistent subgraphs
#     subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
    
#     for subgraph in subgraphs:
#         # Step 3c: Evaluate subgraph consistency
#         consistency_score = 0
#         for j in range(i+1, len(events)):
#             next_event = events[j]
#             # Update the graph based on the next event

#             next_subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
#             for next_subgraph in next_subgraphs:
#                 # Compare the number of nodes and edges between subgraphs
#                 if len(subgraph.nodes) == len(next_subgraph.nodes) == num_nodes and len(subgraph.edges) == len(next_subgraph.edges) == num_edges:
#                     consistency_score += 1

#         # Step 3d: Select the most consistent subgraph
#         if consistency_score > highest_consistency_score:
#             highest_consistency_score = consistency_score
#             most_consistent_subgraph = subgraph

# The 'most_consistent_subgraph' variable will contain the most consistent subgraph found based on the number of nodes and edges.


print(most_consistent_subgraph.nodes())
print(most_consistent_subgraph.edges())

