import imageio
import matplotlib.pyplot as plt

from skimage import exposure
import numpy as np

from skimage.measure import regionprops
from PIL import Image




def frame_creator():
    for i in range(100):
        img = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/Climp/std/C1_decon_t0{i:02d}_ch00_std.png')
        nerdy = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/Climp/preproc/C1/C1_decon_t0{i:02d}_ch00_proc_enhance.png')
        nerdynet = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/Climp/nerdynet_seg/C1_decon_t0{i:02d}_ch00_std_seg.png')
        
        plt.subplot(131)
        plt.imshow(exposure.equalize_adapthist(img), cmap='gray')
        plt.title('Input')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(nerdy, cmap='gray')
        plt.title('nERdy')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(nerdynet, cmap='gray')
        plt.title('nERdyNet')
        plt.axis('off')
        
        plt.subplots_adjust(wspace=0.02)
        
    #     plt.show()
        plt.savefig(f'/localhome/asa420/MIAL/data/sted-data/Climp/C1_frames/Climp1_frame_{i:02d}.png', bbox_inches='tight', pad_inches=0.02)
        plt.close()



def crops_creator(modality, group, seq):
    # path_pref = confocal_path if modality == 'confocal' else sted_path
    group_pref = {'Control': 'ct', 'Climp': 'C', 'RTN': 'R', 'ATL': 'A'}
    for i in range(100):
        # img = imageio.imread(f'/localhome/asa420/MIAL/data/{modality}-data/{group}/std/{group_pref[group]}{seq}_decon_t0{i:02d}_ch00_std.png')
        img = imageio.imread(f'/localhome/asa420/MIAL/data/{modality}-data/{group}/ct6_single/{group_pref[group]}{seq}_junc_repr_t{i:02d}.png')

        # crop = img[57:127, 57:127] # conf atl16
        # crop = img[65:127, 0:65] # conf climp19
        # crop = img[66:110, 0:44] # sted rtn4
        # crop = img[50:82, 0:32] # conf ctrl27

        crop = img[1059: 2103 , 1540:2584] # sted ctrl6 highres

        imageio.imsave(f'/localhome/asa420/MIAL/data/{modality}-data/{group}/ct{seq}_single_crops/crop_{i:02d}.png', crop)


# crops_creator('sted', 'Control', 6)

# exit()

def confocal_frame_creator(group, seq):
    # group_pref = {'Control': 'Ct', 'Climp': 'C', 'RTN': 'R', 'ATL': 'A'}
    input = imageio.imread('/localhome/asa420/MIAL/data/sted-data/Control/control6_er_mean_annot.png')

    # input = imageio.imread(f'/localhome/asa420/MIAL/data/confocal-data/{group}/er_mean/{group.lower()}{seq}_er_mean_annot.png')

    for i in range(100):
        # img = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/Control/ct6_crops/crop_{i:02d}.png')
        # img = imageio.imread(f'/localhome/asa420/MIAL/data/confocal-data/{group}/{group.lower()}{seq}_crops/crop_{i:02d}.png')
        img = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/Control/ct6_single_crops/crop_{i:02d}.png')
        plt.subplot(122)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.subplot(121)
        plt.imshow(input, cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(wspace=0.01)
        
        
        # plt.show()
        plt.savefig(f'/localhome/asa420/MIAL/data/sted-data/Control/ct6_single_frames/{group.lower()}{seq}_frame_{i:02d}.png', bbox_inches='tight', pad_inches=0.01, dpi=500)
        plt.close()

# confocal_frame_creator('Control', 6)

# exit()


def runner(writer, group, seq):
    group_pref = {'Control': 'Ct', 'Climp': 'C', 'RTN': 'R', 'ATL': 'A'}
    for frame in range(100):
        # atl2_frame_00.png
        # file = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/Control/ct6_repr_frames/control6_frame_{frame:02d}.png')
        file = Image.open(f'/localhome/asa420/MIAL/data/sted-data/Control/ct6_single_frames/control6_frame_{frame:02d}.png')
        file = file.resize((2496, 1248), Image.LANCZOS)
        file = np.array(file)
        # file = imageio.imread(f'/localhome/asa420/MIAL/data/confocal-data/{group}/{group.lower()}{seq}_frames/{group.lower()}{seq}_frame_{frame:02d}.png')
        # file = imageio.imread(f'/localhome/asa420/MIAL/data/confocal-data/{group}/std/{group[0]}{seq}_decon_t0{frame:02d}_ch00_std.png')
        writer.append_data(file)

def create_sequence(group, seq):
    with imageio.get_writer(f'STED_{group}{seq}_junc_repr_new.mp4', mode='I', fps=1) as writer:
        runner(writer, group, seq)

# create_sequence('Control', 6)



# img = imageio.imread('/localhome/asa420/MIAL/data/sted-data/Control/control6_er_mean.png')
# crop = img[53:103, 77:127]
# plt.imshow(crop, cmap='gray')
# plt.axis('off')
# plt.show()
# def crop_creator():

# for i in range(100):
#     img = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/Control/std/Ct6_decon_t0{i:02d}_ch00_std.png')
#     crop = img[53:103, 77:127]
#     imageio.imsave(f'/localhome/asa420/MIAL/data/sted-data/Control/ct6_crops/crop_{i:02d}.png', crop)





import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.morphology import label, closing
import sknw
import itertools
from skimage.morphology import skeletonize

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

def get_skeleton(img_path):
    img = imageio.imread(img_path)
    return skeletonize(img/255.)

def skel_to_graph(skel_img_path):
    skel = get_skeleton(skel_img_path)
    return sknw.build_sknw(skel, multi=True, iso=False)


def get_junctions(path_skel):
    """
    @param path_er: path to er image
    @param path_skel: path to skeleton image
    @return: junctions (list) - provides all junctions with degree > 2 from the mean projection proc skeleton
    """

    graph = skel_to_graph(path_skel)

    node_set, degree_list = graph.nodes, graph.degree

    node_coords = np.array([node_set[node]['o'] for node in node_set])

    return [node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2]

def get_all_junc():

    mean_skel = '/localhome/asa420/MIAL/data/sted-data/Control/control6_er_mean_pred_skel.png'

    ref_junctions = get_junctions(mean_skel)
    ref_junctions = [[each[0], each[1]] for each in ref_junctions]

    per_frame_junctions = []
    pf_junc = []
    
    for frame in range(100):

        # NerdyNet pipeline
        skeleton_path = f'/localhome/asa420/MIAL/data/sted-data/Control/nerdynet_skel_no_erosion/Ct6_decon_t0{frame:02d}_ch00_std_skel.png'

        junctions = get_junctions(skeleton_path)

        junc_array = [[junc[0], junc[1]] for junc in junctions]
        per_frame_junctions.extend(junc_array)
        # pf_junc.append(junc_array)
        pf_junc.append(np.array(junctions))

    return ref_junctions, per_frame_junctions, pf_junc

def label_junctions(group, series_num):

    ref_junctions, per_frame_junctions, pf_junc = get_all_junc()

    ref_junctions = np.array(ref_junctions)
    per_frame_junctions = np.array(per_frame_junctions)

    spread_img = np.zeros((128, 128))
    for each in per_frame_junctions:
        spread_img[each[0], each[1]] = 255.

    labelled_img = label(spread_img, connectivity=2)

    return ref_junctions, per_frame_junctions, labelled_img, pf_junc

def get_ref_junc_per_CC_id(reference_junctions, connected_components):
    # get CC_id and corresponding junctions, get CC_ids with at
    """
    Return the reference junctions per connected component and the list of connected components with at least 1 reference
    junction.

    :param reference_junctions: (ndarray) the reference junctions
    :param connected_components: (ndarray) the connected components for the junctions
    :return: label_values (dict) provides corresponding reference junctions per cc_id,
                assigned_components (list) provides cc with at least 1 reference junction
    """
    label_ids = {}
    assigned_components = []

    for junction in reference_junctions:
        if connected_components[junction[0], junction[1]] != 0:
            cc_id = connected_components[junction[0], junction[1]]
            if cc_id not in label_ids:
                label_ids[cc_id] = []
            label_ids[cc_id].append([junction[0], junction[1]])
            assigned_components.append(cc_id)

    return label_ids, assigned_components

def separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img):
    """
    Return ref junctions per CC and CCs without reference junction dicts.

    """
    regions = regionprops(labelled_img)

    num_components = np.unique(labelled_img)

    label_ids, assigned_components = get_ref_junc_per_CC_id(ref_junctions, labelled_img)
    # print(label_vals)

    # unassigned_cc_dict = get_uncertain_junctions(labelled_img, per_frame_junctions, num_components, assigned_components)

    # return label_vals, cc_area_dict, unassigned_cc_dict
    return label_ids

def get_junction_areas(label_ids):
    """
    Returns junctions arrays for iso, fuz and unknown classes
    """

    isolated_junctions = []
    fuzzy_junctions = []

    for cc_id, junctions in label_ids.items():
        if cc_id != 0:
            if len(junctions) == 1:
                isolated_junctions.append(junctions[0])
                #isolated_junc_area.append(cc_area_dict[k])
            else:
                fuzzy_junctions.append(junctions)


    isolated_junctions = np.array(isolated_junctions)

    fuzzy_junctions = list(itertools.chain.from_iterable(fuzzy_junctions))
    fuzzy_junctions = np.array(fuzzy_junctions)

    return isolated_junctions, fuzzy_junctions


# create STED frame viz
# outline from proj frame, per frame junc in blue, ref junc from proj frame
def get_sted_viz():

    ref_junctions, per_frame_junctions, labelled_img, pf_junc = label_junctions('Control', 6)

    # print(per_frame_junctions)
    # print(pf_junc)
    # print(pf_junc[0][:, 1])

    # exit()

    # dict with ids as key and (x, y) as value
    label_ids = separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)

    # iso, fuz, unk: list of lists with x, y
    iso, fuz = get_junction_areas(label_ids)


    iso_cc = get_cc_ids(labelled_img, iso)
    fuz_cc = get_cc_ids(labelled_img, fuz)

    iso_cc_coords = {each: np.where(labelled_img==each) for each in iso_cc}
    fuz_cc_coords = {each: np.where(labelled_img==each) for each in fuz_cc}

    im = np.zeros((128, 128))

    for v in iso_cc_coords.values():
        im[v[0], v[1]] = 255

    for v in fuz_cc_coords.values():
        im[v[0], v[1]] = 255

    # imageio.imsave('sted-ctrl-6.png', im)

    # exit()

    lab = label(im, connectivity=2)

    # plt.imshow(lab, cmap='gray', interpolation=None)
    # plt.show()

    # perform area closing operation on lab
    lab = closing(lab)

    # imageio.imsave('sted-ctrl-6-close.png', lab)

    # exit()
    cntrs = measure.find_contours(lab, 0.8)
    largest_contour_idx = max(range(len(cntrs)), key=lambda i: len(cntrs[i]))

    for i in range(100):
        er = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/Control/std/Ct6_decon_t0{i:02d}_ch00_std.png')

        plt.axis('off')

        plt.imshow(er, cmap='gray', interpolation=None)
        # cntrs = measure.find_contours(lab, 0.8)#, fully_connected='high')
        # for cntr in cntrs:
        #     y, x = cntr.T
        #     plt.plot(x, y, color='cyan', linewidth=0.4)
        cntr = cntrs[largest_contour_idx]
        y, x = cntr.T
        plt.plot(x, y, color='cyan', linewidth=0.4)

        plt.plot(ref_junctions[:, 1], ref_junctions[:, 0], color='red', marker='o', linestyle='None', markersize=1)

        plt.plot(pf_junc[i][:, 1], pf_junc[i][:, 0], color='blue', marker='o', linestyle='None', markersize=1)

        # plt.show()
        plt.savefig(f'/localhome/asa420/MIAL/data/sted-data/Control/ct6_single/ct6_junc_repr_t{i:02d}.png', bbox_inches='tight', pad_inches=0, dpi=700)

        plt.close()

# get_sted_viz()