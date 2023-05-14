import sknw
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import data
from junction_analysis_modules import JunctionAnalysis as JA
import imageio

junc_analysis = JA('/localhome/asa420/MIAL/data/confocal_movies/')

er_prefix = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_egfp/'
skel_prefix = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/'


groups = {'ATL': 26, 'Climp': 31, 'RTN': 29, 'Control':31}
group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}

def get_temporal_graph(group, num):
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


create_group_graph_pickles('ATL')
create_group_graph_pickles('Climp')
create_group_graph_pickles('RTN')
create_group_graph_pickles('Control')