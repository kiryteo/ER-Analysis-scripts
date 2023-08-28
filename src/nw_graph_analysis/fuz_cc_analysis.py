# Fuz CC analysis

import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import label, regionprops
from skimage.morphology import dilation, closing
from junction_analysis_modules import JunctionAnalysis as JA

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'
junc_analysis = JA(confocal_data_path)


def get_cc_ids(labelled_img, region):
    """
    Get CC ids for the specified region
    @param labelled_img: labelled image
    @param region: isolated or fuzzy
    @return: list of CC ids
    """

    # Create a dictionary to store per component data
    cc_data = {cc_id: [] for cc_id in np.unique(labelled_img)}

    # Populate the dictionary with locations for the specified region
    for loc in region:
        loc_x, loc_y = loc[0], loc[1]
        cc_id = labelled_img[loc_x, loc_y]
        cc_data[cc_id].append(loc)

    # Extract CC ids for the specified region
    cc_ids = [cc_id for cc_id, data in cc_data.items() if cc_id > 0 and len(data) > 0]

    return cc_ids

def get_fuz_cc_outline(group, series_num):
    ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, series_num)


    # dict with ids as key and (x, y) as value
    label_ids, unassigned_cc_dict = junc_analysis.separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)

    # iso, fuz, unk: list of lists with x, y
    iso, fuz, unk = junc_analysis.get_junction_areas(label_ids, unassigned_cc_dict)



    # iso_cc = get_cc_ids(labelled_img, iso)
    fuz_cc = get_cc_ids(labelled_img, fuz)
    # # unk_cc = get_cc_ids(labelled_img, unk)

    # iso_cc_coords = {each: np.where(labelled_img==each) for each in iso_cc}
    fuz_cc_coords = {each: np.where(labelled_img==each) for each in fuz_cc}
    # unk_cc_coords = {each: np.where(labelled_img==each) for each in unk_cc}

    spread_img = np.zeros((128, 128))
    for k, v in fuz_cc_coords.items():
        spread_img[v[0], v[1]] = 255.

    # spread_img = dilation(spread_img)
    spread_img = closing(spread_img)
    # spread_img = dilation(spread_img)

    lab_img = label(spread_img, connectivity=2)

    return lab_img

def get_intersection(a, b):
    # a = (np.array([19, 20, 20, 124]), np.array([124, 122, 123, 43]))
    # b = (np.array([19, 21, 23, 25, 26]), np.array([124, 125, 127, 110, 46]))

    # Combine values from the first and second arrays in each tuple
    combined_a = np.stack(a, axis=1)
    combined_b = np.stack(b, axis=1)

    # Convert the combined arrays to sets of tuples
    set_a = {tuple(row) for row in combined_a}
    set_b = {tuple(row) for row in combined_b}

    # Find the intersection of sets
    intersection_set = set_a.intersection(set_b)

    # Convert the intersection set back to a list of arrays
    intersection = [np.array(row) for row in intersection_set]

    return intersection


lab_img = get_fuz_cc_outline('ATL', 1)
fuzzy_coords = np.where(lab_img)



def get_fuz_cc_ids(a, b):

    a = set(a)
    b = set(b)

    intersection = a.intersection(b)

    return list(intersection)


atl_data = []

for i in range(100):
    skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t0{i:02d}_ch00_skel.png')

    graph = sknw.build_sknw(skel, multi=True, iso=False)

    nodes = graph.nodes

    node_coords = np.array([nodes[node]['o'] for node in nodes])

    node_coords_list = list(zip(node_coords[:,0], node_coords[:,1]))

    cc_data = []

    for cc_id in range(1, lab_img.max()+1):
        cc_id_coords = np.where(lab_img==cc_id)

        cc_id_coords_list = list(zip(cc_id_coords[0], cc_id_coords[1]))

        fuz_cc_nodes = get_fuz_cc_ids(node_coords_list, cc_id_coords_list)

        degree_data = []
        for node in fuz_cc_nodes:
            if node in node_coords_list:
                idx = node_coords_list.index(node)
                degree_data.append(graph.degree[idx])

        if degree_data:
            cc_data.append(np.sum(degree_data)/len(degree_data))

    atl_data.append(cc_data)


def get_fuz_cc_area(lab_img):
    cc_area_list = []
    for cc_id in range(1, lab_img.max()+1):
        cc_id_coords = np.where(lab_img==cc_id)
        cc_area_list.append(len(cc_id_coords[0]))
    return cc_area_list


def get_skel_per_fuz_cc(lab_img):
    data = []
    for cc_id in range(1, lab_img.max()+1):
        cc_id_coords = np.where(lab_img==cc_id)
        cc_data = []
        for frame in range(100):

            er = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_egfp/A1_decon_t0{frame:02d}_ch00_std.png')

            skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t0{frame:02d}_ch00_skel.png')
            skel_coords = np.where(skel)

            fuz_skel_pixels = get_intersection(cc_id_coords, skel_coords)

            # print(fuz_skel_pixels)
            # exit()
            values_at_coordinates = [er[coord[0], coord[1]] for coord in fuz_skel_pixels]

            # Calculate the mean of the extracted values
            mean_value = np.mean(values_at_coordinates)

            # cc_data.append(fuz_skel_pixels)
            cc_data.append(mean_value)
        data.append(cc_data)

    return data