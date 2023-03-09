import numpy as np
from skimage.measure import label, regionprops
import itertools
import sknw
import imageio

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'

class JunctionAnalysis:

    def __init__(self, confocal_data_path):
        self.confocal_data_path = confocal_data_path

    def skel_to_graph(self, skel_img_path):
        """
        @param skel_img_path:
        @return:
        """
        return sknw.build_sknw(imageio.imread(skel_img_path), multi=True, iso=False)

    def get_junctions(self, graph):
        """

        @param graph: Input graph to obtain the junctions
        @return: nodes (junctions) with degree > 2
        """
        # get all the nodes from the graph
        node_coords = np.array([graph.nodes[node]['o'] for node in graph.nodes()])

        return [node_coords[node_num] for node_num, degree_val in enumerate(graph.degree) if degree_val[1] > 2]

    def get_all_junc(self, group, num_series):
        """

        @param group: group to be analyzed
        @param num_series: sequence number
        @return: nps (list) - provides all junctions with degree > 2 from the mean projection proc skeleton, per_frame_junctions (list) - provides all junctions per skel frame
        """

        group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}
        mean_img = f'{self.confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_er_mean_proc_enhance_skel.png'

        # Get junction coordinates from projection frame
        graph = self.skel_to_graph(mean_img)
        newps = self.get_junctions(graph)

        ref_junctions = [[each[0], each[1]] for each in newps]

        per_frame_junctions = []
        for frame in range(100):
            skeleton_path = f'{confocal_data_path}{group}/new_op_jul/skel/{group_pref[group]}{num_series}/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'
            graph = self.skel_to_graph(skeleton_path)
            junctions = self.get_junctions(graph)

            junc_array = [[junc[0], junc[1]] for junc in junctions]
            # sk_nps = np.array(sk_nps)
            per_frame_junctions.extend(junc_array)

        return ref_junctions, per_frame_junctions

    def label_junctions(self, group, series_num):

        # fig, ax = plt.subplots()
        ref_junctions, per_frame_junctions = self.get_all_junc(group, series_num)

        ref_junctions = np.array(ref_junctions)
        per_frame_junctions = np.array(per_frame_junctions)

        spread_img = np.zeros((128, 128))
        for each in per_frame_junctions:
            spread_img[each[0], each[1]] = 255.

        labelled_img = label(spread_img, connectivity=2)
        # imageio.imsave('Climp12_junc_labelled.png', labelled_img)
        # fig.add_subplot(1,2,2)
        # plt.axis('off')
        # plt.imshow(labelled_img, cmap='gray')
        # plt.savefig('Climp12_junc_labelled.png', bbox_inches='tight', pad_inches=0, dpi=700)
        # plt.close()
        # plt.show()
        # exit()

        return ref_junctions, per_frame_junctions, labelled_img

    def get_junction_types(self, reference_junctions, connected_components):
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

    def get_uncertain_junctions(self, labelled_img, per_frame_junctions, num_components, assigned_components):
        unassigned_components = [x for x in num_components if x not in assigned_components]

        unassigned_cc_dict = {}

        for each in per_frame_junctions:
            cc_label = labelled_img[each[0], each[1]]
            if cc_label != 0 and cc_label in unassigned_components:
                if cc_label not in unassigned_cc_dict.keys():
                    unassigned_cc_dict[(labelled_img[each[0], each[1]])] = []
                unassigned_cc_dict[(labelled_img[each[0], each[1]])].append([each[0], each[1]])
        return unassigned_cc_dict

    def separate_junc_cc(self, ref_junctions, per_frame_junctions, labelled_img):
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

        label_ids, assigned_components = self.get_junction_types(ref_junctions, labelled_img)
        # print(label_vals)

        unassigned_cc_dict = self.get_uncertain_junctions(labelled_img, per_frame_junctions, num_components, assigned_components)

        # return label_vals, cc_area_dict, unassigned_cc_dict
        return label_ids, unassigned_cc_dict

    def get_junction_areas(self, label_ids, unassigned_cc_dict):
        isolated_junctions = []
        fuzzy_junctions = []
        unknown_junctions = []

        for cc_id, junctions in label_ids.items():
            if cc_id != 0:
                if len(junctions) == 1:
                    isolated_junctions.append(junctions[0])
                else:
                    fuzzy_junctions.append(junctions)

        for junctions in unassigned_cc_dict.values():
            unknown_junctions.extend(junctions)

        isolated_junctions = np.array(isolated_junctions)

        fuzzy_junctions = list(itertools.chain.from_iterable(fuzzy_junctions))
        fuz = np.array(fuzzy_junctions)

        unknown_junctions = list(itertools.chain.from_iterable(unknown_junctions))
        unknown_junctions = np.array(unknown_junctions)

        return isolated_junctions, fuzzy_junctions, unknown_junctions
