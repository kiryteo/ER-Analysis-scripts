import numpy as np
from skimage.measure import label, regionprops
import itertools

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'

def get_junctions(graph):
    """

    @param graph: Input graph to obtain the junctions
    @return: nodes (junctions) with degree > 2
    """
    # get all the nodes from the graph
    nodes_list = graph.nodes()
    node_coords = np.array([nodes_list[node]['o'] for node in nodes_list])

    # degree_list provides list of tuples with node id followed by its degree
    degree_list = graph.degree

    return [node_coords[node_num] for node_num, degree_val in enumerate(degree_list) if degree_val[1] > 2]


def get_all_junc(group, num_series):
    """

    @param group: group to be analyzed
    @param num_series: sequence number
    @return: nps (list) - provides all junctions with degree > 2 from the mean projection proc skeleton, skdata (list) - provides all junctions per skel frame
    """
    # mean_img = 'confocal_data_pathATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'

    group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}

    mean_img = f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'



    # Get junction coordinates from projection frame
    newps = get_junctions(mean_img)

    nps = [[each[0], each[1]] for each in newps]
    skdata = []
    for frame in range(100):

        sk_img = f'{confocal_data_path}{group}/new_op_jul/skel/{group_pref[group]}{num_series}/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'



        sk_newps = get_junctions(sk_img)

        sk_nps = [[each[0], each[1]] for each in sk_newps]
        # sk_nps = np.array(sk_nps)
        skdata.extend(sk_nps)

    return nps, skdata



def label_junctions(group, series_num):

    # fig, ax = plt.subplots()
    nps, skdata = get_all_junc(group, series_num)

    nps = np.array(nps)
    skdata = np.array(skdata)

    spread_img = np.zeros((128, 128))
    for each in skdata:
        spread_img[each[0], each[1]] = 255.


    # fig.add_subplot(1,2,1)
    # plt.imshow(spread_img)
    #
    labelled_img = label(spread_img, connectivity=2)
    # imageio.imsave('Climp12_junc_labelled.png', labelled_img)
    # fig.add_subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(labelled_img, cmap='gray')
    # plt.savefig('Climp12_junc_labelled.png', bbox_inches='tight', pad_inches=0, dpi=700)
    # plt.close()
    # plt.show()
    # exit()

    return nps, skdata, labelled_img

def get_junction_types(nps, lab):
    """

    @param nps: (ndarray) reference junctions
    @param lab: (ndarray) connected components for junctions
    @return: label_vals (dict) provides corresponding reference junctions per cc_id, assigned_components (list) provides cc with at least 1 reference junction
    """
    label_vals = {}

    assigned_components = []

    for each in nps:
        if lab[each[0], each[1]] != 0:
            if lab[each[0], each[1]] not in label_vals.keys():
                label_vals[(lab[each[0], each[1]])] = []
            label_vals[(lab[each[0], each[1]])].append([each[0], each[1]])
            assigned_components.append(lab[each[0], each[1]])
    return label_vals, assigned_components


def get_uncertain_junctions(lab, skdata, num_components, assigned_components):
    unassigned_components = [x for x in num_components if x not in assigned_components]

    unassigned_cc_dict = {}

    for each in skdata:
        cc_label = lab[each[0], each[1]]
        if cc_label != 0 and cc_label in unassigned_components:
            if cc_label not in unassigned_cc_dict.keys():
                unassigned_cc_dict[(lab[each[0], each[1]])] = []
            unassigned_cc_dict[(lab[each[0], each[1]])].append([each[0], each[1]])
    return unassigned_cc_dict


def separate_junc_cc(nps, skdata, labelled_img):
    regions = regionprops(labelled_img)

    # cc_list = []
    # for idx in range(1, labelled_img.max()):
    #     lab_i = props[idx].label

    # cc_area_dict = {}
    # for idx, props in enumerate(regions):
    #     cc_area_dict[idx] = props.area
    # cc_area_dict[idx] = [props.area, props.axis_major_length]

    num_components = np.unique(labelled_img)
    # print(num_components)

    label_vals, assigned_components = get_junction_types(nps, labelled_img)
    # print(label_vals)

    unassigned_cc_dict = get_uncertain_junctions(labelled_img, skdata, num_components, assigned_components)

    # return label_vals, cc_area_dict, unassigned_cc_dict
    return label_vals, unassigned_cc_dict


def get_junction_areas(label_vals, unassigned_cc_dict):
    isolated_junc = []
    isolated_junc_area = []
    fuzzy_junc = []
    fuzzy_junc_area = []
    for k, v in label_vals.items():
        if k != 0:
            if len(v) == 1:
                isolated_junc.append(v[0])
                # try:
                #     isolated_junc_area.append(cc_area_dict[k])
                # except:
                #     pass
            else:
                fuzzy_junc.append(v)
                # try:
                #     fuzzy_junc_area.append(cc_area_dict[k])
                # except:
                #     pass

    unknown_junc = [v for k, v in unassigned_cc_dict.items()]
    iso = np.array(isolated_junc)

    fuz = list(itertools.chain.from_iterable(fuzzy_junc))
    fuz = np.array(fuz)

    unk = list(itertools.chain.from_iterable(unknown_junc))
    unk = np.array(unk)

    # iso_area = list(itertools.chain.from_iterable(isolated_junc_area))
    # iso_area = np.array(isolated_junc_area)

    # fuz_area = list(itertools.chain.from_iterable(fuzzy_junc_area))
    # fuz_area = np.array(fuzzy_junc_area)


    return iso, fuz, unk#, iso_area, fuz_area