import imageio
import sknw
import networkx as nx
import numpy as np
import math
from shapely.geometry import Polygon


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
        skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal-data/vess_enh_unet/{group}/skel/{group}{frame}_proc_skel.png')

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


atl_data = get_cycle_area('atl', 26)
climp_data = get_cycle_area('climp', 31)
control_data = get_cycle_area('control', 31)
rtn_data = get_cycle_area('rtn', 29)

# plot boxplots for each group

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

data = [atl_data, climp_data, control_data, rtn_data]

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title('Cycle Area Distribution')
ax.set_xlabel('Group')
ax.set_ylabel('Cycle Area')
ax.set_xticklabels(['ATL', 'CLIMP', 'Control', 'RTN'])
ax.boxplot(data)
plt.show()
