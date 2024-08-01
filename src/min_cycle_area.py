import imageio
import sknw
import networkx as nx
import numpy as np
import math
from shapely.geometry import Polygon
import pickle
import pandas as pd

import os
import imageio

import matplotlib.pyplot as plt
import seaborn as sns

home = os.path.expanduser('~')

def calculate_angle(point, reference_point):
    x, y = point[0] - reference_point[0], point[1] - reference_point[1]
    return math.atan2(y, x)

def sort_points_clockwise(points):
    # Find the centroid as the reference point
    reference_point = [sum(x[0] for x in points) / len(points),
                       sum(x[1] for x in points) / len(points)]

    # Sort the points based on the angle with respect to the reference point
    sorted_points = sorted(points, key=lambda p: calculate_angle(p, reference_point))

    return sorted_points


def get_cycle_area(group, seq_num):
    area_data = []
    for frame in range(1, seq_num + 1):
        skel = imageio.imread(f'{home}/MIAL/data/confocal-data/vess_enh_unet/{group}/skel/{group}{frame}_proc_skel.png')

        # build graph from skeleton
        graph = sknw.build_sknw(skel, multi=False, iso=False)

        # get the minimum cycle basis of the graph.
        min_cycles = nx.minimum_cycle_basis(graph.to_undirected())

        node_set = graph.nodes

        node_coords = np.array([node_set[node]['o'] for node in node_set])

        for cycle in min_cycles:
            # shoelace formula needs the points to be in clockwise order

            sorted_points = sort_points_clockwise(node_coords[cycle])

            xl = [point[0] for point in sorted_points]
            yl = [point[1] for point in sorted_points]
            
            if len(xl) > 3:
                polygon = Polygon(zip(xl, yl))
                area_data.append(polygon.area)

    return area_data


def runner():
    # Plot boxplots
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(x='Group', y='Cycles_area', data=df, showfliers=False)

    yt = ax.get_yticks()
    ax.set_yticklabels([f'{y}' for y in yt])

    ax.set_xlabel('Group', fontsize=18)
    ax.set_ylabel('Cycle Area', fontsize=18)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.gcf().set_size_inches(3.5, 10)
    plt.savefig('cycle_area_boxplot.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

if __name__ == "__main__":
    # Load data
    home = os.path.expanduser('~')
    data_files = ['atl_cycles.pkl', 'climp_cycles.pkl', 'control_cycles.pkl', 'rtn_cycles.pkl']
    groups = ['ATL', 'Climp', 'Control', 'RTN']

    data = {}
    for group, file in zip(groups, data_files):
        with open(file, 'rb') as f:
            data[group] = pickle.load(f)

    # Prepare DataFrame
    df = pd.DataFrame({
        'Cycles_area': np.concatenate([data[group] for group in groups]),
        'Group': np.concatenate([[group] * len(data[group]) for group in groups])
    })

    runner()