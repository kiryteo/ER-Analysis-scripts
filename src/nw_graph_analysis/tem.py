import imageio
import matplotlib.pyplot as plt
import numpy as np
from junction_analysis_modules import JunctionAnalysis as JA

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'
junc_analysis = JA(confocal_data_path)


path_skel = '/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/er_mean_proc/control5_er_mean_proc_enhance_skel.png'

# path_skel = '/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/er_mean_proc/control5_proc_skel.png'

path_er = '/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/er_mean/control5_er_mean.png'


_, graph = junc_analysis.get_junctions(path_er, path_skel)

# print(graph)

plt.imshow(imageio.imread(path_er), cmap='gray')

# print(graph.edges(data=True))

# graph = junc_analysis.skel_to_graph(path_skel)

# draw edges by pts
for (s,e) in graph.edges():
    ps = graph[s][e][0]['pts']
    plt.plot(ps[:,1], ps[:,0], 'green')
    try:
        ps = graph[s][e][1]['pts']
        plt.plot(ps[:,1], ps[:,0], 'green')
    except:
        pass
    
# draw node by o
nodes = graph.nodes()
ps = np.array([nodes[i]['o'] for i in nodes])
plt.plot(ps[:,1], ps[:,0], 'r.')

plt.show()