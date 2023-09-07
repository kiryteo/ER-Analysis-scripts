import imageio
import matplotlib.pyplot as plt
import numpy as np
from junction_analysis_modules import JunctionAnalysis as JA
import scipy
from scipy import ndimage

data = np.zeros((128, 128))

for i in range(99):
    skel1 = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/skel/R6/R6_decon_t0{i:02d}_ch00_skel.png')
    skel2 = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/skel/R6/R6_decon_t0{i+1:02d}_ch00_skel.png')


    # get distance transform of skeleton
    # invert skeleton
    skel1 = np.invert(skel1)
    skel2 = np.invert(skel2)

    plt.imshow(skel1)
    plt.show()

    exit()

    # get distance transform
    edt1 = ndimage.distance_transform_edt(skel1)
    edt2 = ndimage.distance_transform_edt(skel2)

    # get the difference
    # diff = abs(edt2 - edt1)
    diff = edt2 - edt1

    # # diff = diff + np.abs(np.min(diff))
    # # plt.hist(diff.flatten())

    # plt.imshow(diff)
    # plt.show()

    data = data + diff

    # mean_skel = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/er_mean_proc/rtn6_proc_skel.png')

    # # skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/skel/R6/R6_decon_t0{i:02d}_ch00_skel.png')

    # # skel = np.invert(skel)
    # mean_skel = np.invert(mean_skel)

    # edt = ndimage.distance_transform_edt(mean_skel)

    # # data = data + edt
    # plt.imshow(edt)
    # plt.show()

# plt.imshow(data)
# plt.show()

exit()


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