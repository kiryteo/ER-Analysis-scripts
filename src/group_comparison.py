import os.path
import glob
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as ckdtree

home = os.path.expanduser('~')

def get_junctions(ref_junctions):
    """
    get junctions as a list of tuples
    required by kdtree call

    """

    X, Y = np.where(ref_junctions == 255)[0], np.where(ref_junctions == 255)[1]
    return list(zip(X, Y))

def get_tree():
    """
    get neighbours for each pixel
    with ckdtree

    """
    img = []
    for i in range(128):
        img.extend((i, j) for j in range(128))
    return ckdtree(np.array(img))

def get_neighbours(file, points, tree, radius):
    """
    get dict with junction and neighbours
    """
    file = imageio.imread(file)
    nbrs_dict = {}
    for eachpt in points:
        idx = tree.query_ball_point([eachpt[0], eachpt[1]], radius)
        nbrs_dict[eachpt] = [(each[0], each[1]) for each in tree.data[idx]]
    return file, nbrs_dict

def get_intensities(imgfile, nbrs_dict):
    """
    get intensities in a neighborhood
    """
    intensity_vals = []
    for values in nbrs_dict.values():
        intensity_vals.extend(imgfile[int(val[0])][int(val[1])] for val in values)
    # cnt = 0
    # for intensity in intensity_vals:
    #     if intensity == 0:
    #         cnt += 1

    # print(sum(intensity_vals)/len(intensity_vals)-cnt)
    # print(np.mean(intensity_vals))
    return np.mean(intensity_vals)




# ctrl_brpts frame
ctrl_ref_junctions = imageio.imread(f'{home}/MIAL/aggregation-with-median/Control/avg/brpts/series6-avg_brpts.png')

# ctrl_files_dir
ctrl_files = glob.glob(f'{home}/MIAL/live-cell-movies/COSKDEL/COSKDEL/Decon/Series006_decon_converted/std/*')


# climp_brpts_frame
climp_ref_junctions = imageio.imread(f'{home}/MIAL/aggregation-with-median/Climp/avg/brpts/ClimpSeries4-avg_brpts.png')

# climp_files_dir
climp_files = glob.glob(f'{home}/MIAL/live-cell-movies/COSKDELCLIMP/COSKDELCLIMP/Decon/Series004_decon_converted/std/*')


# rtn_brpts_frame
rtn_ref_junctions = imageio.imread(f'{home}/MIAL/aggregation-with-median/RTN/avg/brpts/RTNSeries14-avg_brpts.png')

# rtn_files_dir
rtn_files = glob.glob(f'{home}/MIAL/live-cell-movies/COSKDELRTN/COSKDELRTN/Decon/Series014_decon_converted/std/*')



def runner(files, ref_junctions, radius=2):
    mean_intensities = []
    for file in files:
        points = get_junctions(ref_junctions)
        tree = get_tree()
        imgfile, nbrs_dict = get_neighbours(file, points, tree, radius)
        mean_intensities.append(get_intensities(imgfile, nbrs_dict))
    return mean_intensities


def combined_analysis(numbins=20):
    ctrl = runner(ctrl_files, ctrl_ref_junctions, radius=2)
    climp = runner(climp_files, climp_ref_junctions, radius=2)
    rtn = runner(rtn_files, rtn_ref_junctions, radius=2)
    min_limit = min(min(ctrl), min(climp), min(rtn))
    max_limit = max(max(ctrl), max(climp), max(rtn))
    bins = np.linspace(min_limit, max_limit, numbins+1)
    plt.hist(ctrl, bins, alpha=0.33, color='Blue', label='Control')
    plt.hist(climp, bins, alpha=0.33, color='Red', label='Climp')
    plt.hist(rtn, bins, alpha=0.34, color='Green', label='RTN')
    plt.legend(loc='upper right')
    plt.suptitle('Junction Area analysis', size=15)
    plt.title('Mean intensity within specified radius around the reference frame junctions (radius/ euc. dist = 2)', size=12)
    plt.show()
    plt.close()

combined_analysis()

# from matplotlib.ticker import PercentFormatter

# def plot_overlapped_histogram( x1, label1, x2, label2, numbins=10, cumulative=False):
#     min_limit = min( min(x1), min(x2) )
#     max_limit = max( max(x1), max(x2) )
#     bins = np.linspace(min_limit, max_limit, numbins+1)
#     plt.hist(x1, bins, alpha=0.5, label=label1, color='Red', weights=np.ones(len(x1)) / len(x1), cumulative=cumulative)
#     plt.hist(x2, bins, alpha=0.5, label=label2, color='Blue', weights=np.ones(len(x2)) / len(x2), cumulative=cumulative)
#     plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
#     plt.legend(loc='upper right')
#     #plt.savefig("dn_hist_test.png", dpi=1050)
#     plt.show()

# plot_overlapped_histogram(fwhm_vals, 'gt',fwhm_vals_pr, 'pred')