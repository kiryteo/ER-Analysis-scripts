import glob
import numpy as np
import imageio
from scipy.spatial import cKDTree as ckdtree

def get_junctions(ref_junctions):
    """
    get junctions as a list of tuples
    required by kdtree call

    """

    X, Y = np.where(ref_junctions == 255)[0], np.where(ref_junctions == 255)[1]
    points = []
    for x, y in zip(X, Y):
        points.append((x, y))
    return points

def get_tree():
    """
    get neighbours for each pixel
    with ckdtree

    """
    img = []
    for i in range(128):
        for j in range(128):
            img.append((i, j))

    tree = ckdtree(np.array(img))
    return tree

def get_neighbours(file, points, tree):
    """
    get dict with junction and neighbours
    """
    file = imageio.imread(file)
    nbrs_dict = {}
    for eachpt in points:
        idx = tree.query_ball_point([eachpt[0], eachpt[1]], 2)
        nbrs_dict[eachpt] = []
        for each in tree.data[idx]:
            nbrs_dict[eachpt].append((each[0], each[1]))
    return file, nbrs_dict

def get_intensities(imgfile, nbrs_dict):
    """
    get intensities in a neighborhood
    """
    intensity_vals = []
    for values in nbrs_dict.values():
        for val in values:
            intensity_vals.append(imgfile[int(val[0])][int(val[1])])

    cnt = 0
    for intensity in intensity_vals:
        if intensity == 0:
            cnt += 1

    print(sum(intensity_vals)/len(intensity_vals)-cnt)
    print(np.mean(intensity_vals))
    print()


## TODO: Remove hardcoded paths

# ctrl_brpts frame
ctrl_ref_junctions = imageio.imread('/home/ashwin/MIAL/aggregation-with-median/Control/avg/brpts/series2-avg_brpts.png')
# ctrl_files_dir
ctrl_files = glob.glob('/home/ashwin/MIAL/live-cell-movies/COSKDEL/COSKDEL/Decon/Series002_decon_converted/std/*')

# climp_brpts_frame
climp_ref_junctions = imageio.imread('/home/ashwin/MIAL/aggregation-with-median/Climp/avg/brpts/ClimpSeries1-avg_brpts.png')
# climp_files_dir
climp_files = glob.glob('/home/ashwin/MIAL/live-cell-movies/COSKDELCLIMP/COSKDELCLIMP/Decon/Series001_decon_converted/std/*')

# rtn_brpts_frame
rtn_ref_junctions = imageio.imread('/home/ashwin/MIAL/aggregation-with-median/RTN/avg/brpts/RTNSeries4-avg_brpts.png')
# rtn_files_dir
rtn_files = glob.glob('/home/ashwin/MIAL/live-cell-movies/COSKDELRTN/COSKDELRTN/Decon/Series004_decon_converted/std/*')



def runner(files):
    for file in files:
        points = get_junctions(ref_junctions)
        tree = get_tree()
        imgfile, nbrs_dict = get_neighbours(file, points, tree)
        get_intensities(imgfile, nbrs_dict)

# runner(files)
