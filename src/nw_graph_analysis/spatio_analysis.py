import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sknw
import imageio
import itertools
import seaborn as sns
import pandas as pd
import statannot
from junction_analysis_modules import JunctionAnalysis as JA

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'

junc_analysis = JA(confocal_data_path)

seq_per_group = {'ATL': 26, 'RTN': 29, 'Climp':30, 'Control':30}
prefix = {'ATL': 'A', 'RTN': 'R', 'Climp': 'C', 'Control': 'Ct'}

def get_connected_temporal_graphs(group):

    avg_degree_conn = []

    for series in range(1, seq_per_group[group]+1):
        graphs = junc_analysis.get_all_junc(group, series)

        # get degree per graph
        avg_degree_conn.extend(
            nx.degree_assortativity_coefficient(graph) for graph in graphs
        )

    return avg_degree_conn


def get_temporal_graphs(group):

    avg_degree = []

    for series, frame in itertools.product(range(1, seq_per_group[group]+1), range(100)):
        skeleton_path = f'{confocal_data_path}{group}/new_op_jul/skel/{prefix[group]}{series}/{prefix[group]}{series}_decon_t0{frame:02d}_ch00_skel.png'

        graph = junc_analysis.skel_to_graph(skeleton_path)

        avg_degree.append(nx.degree_assortativity_coefficient(graph))

    return avg_degree



atl_degree_data = get_temporal_graphs('ATL')
climp_degree_data = get_temporal_graphs('Climp')
rtn_degree_data = get_temporal_graphs('RTN')
control_degree_data = get_temporal_graphs('Control')

sns.distplot(control_degree_data, rug=True, hist=False, label='Control')
sns.distplot(rtn_degree_data, rug=True, hist=False, label='RTN')
sns.distplot(climp_degree_data, rug=True, hist=False, label='Climp')
sns.distplot(atl_degree_data, rug=True, hist=False, label='ATL')

plt.legend()

plt.show()
    


