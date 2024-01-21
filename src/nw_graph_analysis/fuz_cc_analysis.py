# Fuz CC analysis

import imageio
import numpy as np
import seaborn as sns
import pandas as pd
import sknw
import pickle as pkl
import cv2
import statannot
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import label, regionprops
from skimage.morphology import dilation, closing
from statannotations.Annotator import Annotator
from junction_analysis_modules import JunctionAnalysisModules as JA

confocal_data_path = '/localhome/asa420/MIAL/data/confocal-data/'
sted_data_path = '/localhome/asa420/MIAL/data/sted-data/'
# junc_analysis = JA(confocal_data_path)
# junc_analysis = JA(sted_data_path)

junc_analysis = JA('sted')
# junc_analysis = JA('confocal')


group_dict = {'ATL': 26, 'Climp': 31, 'Control': 31, 'RTN': 29}

group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}


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

#ref_junctions, per_frame_junctions, labelled_img, _ = junc_analysis.label_junctions('ATL', 2)
#print(ref_junctions)

#exit()

def get_fuz_cc_outline(group, series_num, region):
    ref_junctions, per_frame_junctions, labelled_img, _ = junc_analysis.label_junctions(group, series_num)

    if len(ref_junctions) == 0:
        return None

    # dict with ids as key and (x, y) as value
    label_ids = junc_analysis.separate_junc_cc(ref_junctions, labelled_img)

    # iso, fuz, unk: list of lists with x, y
    iso, fuz = junc_analysis.get_junction_areas(label_ids)

    if region == 'iso':
        cc = get_cc_ids(labelled_img, iso)
    else:
        cc = get_cc_ids(labelled_img, fuz)
    # # unk_cc = get_cc_ids(labelled_img, unk)

    cc_coords = {each: np.where(labelled_img==each) for each in cc}

    # ratio_data = []

    xdata = []
    ydata = []

    for id in cc:
        num = len(label_ids[id])
        den = len(cc_coords[id][0])
        # ratio_data.append(num/ den)
        # ratio_data.append(den/ num)

        xdata.append(num)
        ydata.append(den)

    # iso_cc_coords = {each: np.where(labelled_img==each) for each in iso_cc}
    # fuz_cc_coords = {each: np.where(labelled_img==each) for each in fuz_cc}
    # unk_cc_coords = {each: np.where(labelled_img==each) for each in unk_cc}

    # spread_img = np.zeros((128, 128))
    # for v in cc_coords.values():
    #     spread_img[v[0], v[1]] = 255.

    # # spread_img = dilation(spread_img)
    # spread_img = closing(spread_img)
    # # spread_img = dilation(spread_img)

    # lab_img = label(spread_img, connectivity=2)

#    return lab_img
    # return ratio_data
    return xdata, ydata


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

def get_fuz_cc_variance_data(group):
    # data = []
    X = []
    Y = []
    for series in range(1, group_dict[group]+1):
#        ser_data = []
#        lab_img = get_fuz_cc_outline(group, series, 'fuz')
#        num_fuz_patches = np.unique(lab_img).shape[0] - 1
        # ratio_data = get_fuz_cc_outline(group, series, 'fuz')

        xd, yd = get_fuz_cc_outline(group, series, 'fuz')

        if xd and yd:
            X.extend(xd)
            Y.extend(yd)
        else:
            continue


        # if ratio_data:
        #     data.extend(ratio_data)
        # else:
        #     continue


#        try:
            #img = imageio.imread(f'/localhome/asa420/MIAL/data/confocal-data/{group}/er_mean/{group.lower()}{series}_er_mean.png')
#            img = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/{group.lower()}/{group.lower()}{series}_er_mean.png')

#            img = (img - np.min(img)) / (np.max(img) - np.min(img))

#            for patch in range(1, num_fuz_patches+1):
                # get variance per patch in img
#                patch_coords = np.where(lab_img==patch)
#                size = len(patch_coords[0])

#                patch_pixels = img[patch_coords]

                #patch_var = np.var(patch_pixels)
#                patch_mean = np.mean(patch_pixels)

                #data.append(patch_var/size)
#                data.append(patch_mean)
#                ser_data.append(size)
#        except:
#            pass
#        data.append(np.mean(ser_data))


    # return data
    return X, Y

from plantcv import plantcv as pcv


def get_skeleton(self, img_path):
    """
    @param img_path: path to image
    @return: skeleton (ndarray) - skeleton of the image
    """
    img = imageio.imread(img_path)
    # # for blur sted
    # img = skimage.transform.resize(img, (32, 32), anti_aliasing=True)
    return pcv.morphology.skeletonize(mask=img)

def skel_to_graph(skel_img_path):
    """
    @param skel_img_path:
    @return:
    """
    # fname_suffix = skel_img_path.split('/')[-1].split('.')[0].split('_')[-1]
    # if fname_suffix == 'skel':
    #     return sknw.build_sknw(imageio.imread(skel_img_path), multi=True, iso=False)
    # elif fname_suffix == 'filt':
    skel = get_skeleton(skel_img_path)
    return sknw.build_sknw(skel, multi=False, iso=False)

# def get_all_junc(group, num_series):
#     # get reference junctions based on mean projection frame and per frame junctions for each series, all groups
#     """

#     @param group: group to be analyzed
#     @param num_series: sequence number
#     @return: nps (list) - provides all junctions with degree > 2 from the mean projection proc skeleton, per_frame_junctions (list) - provides all junctions per skel frame
#     """

#     # pr
#     group_pref = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}

#     # mean_er = f'{self.confocal_data_path}{group}/new_op_jul/er_mean/{group.lower()}{num_series}_er_mean.png'

#     # mean_skel = f'{self.confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{num_series}_proc_skel.png'
#     # mean_skel = f'{self.data_path}{group}/er_mean_proc/{group.lower()}{num_series}_proc_skel.png'

#     mean_skel_path = f'{sted_data_path}vess_enh_unet/{group.lower()}/gt_skel/sted_{group.lower()}{num_series}_proc_skel.png'
#     # else:
#     #     mean_skel_path = f'{self.data_path}vess_enh_unet/{group.lower()}/gt_skel/{group.lower()}{num_series}_proc_skel.png'

#     if os.path.exists(mean_skel_path):
#         ref_graph = skel_to_graph(mean_skel_path)

#         # ref_junctions = self.get_ref_junctions(self.skel_to_graph(mean_skel))
#         ref_junctions = get_junctions(mean_skel_path)
#         ref_junctions = [[each[0], each[1]] for each in ref_junctions]

#         per_frame_junctions = []
        
#         # for frame in range(fr_start, fr_end):
#         for frame in range(100):

#             # er_path = f'{self.data_path}{group}/std/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_std.png'

#             # skeleton_path = f'{self.data_path}{group}/preproc/{group_pref[group]}{num_series}/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_proc_enhance.png'
            
#             # pipeline
#             # skeleton_path = f'{self.data_path}{group}/skel/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'

#             # UNet pipeline
#             skeleton_path = f'{self.data_path}/vess_enh_unet/{group.lower()}/skel/{group_pref[group]}{num_series}_decon_t0{frame:02d}_ch00_skel.png'

#             junctions = self.get_junctions(skeleton_path)

#             junc_array = [[junc[0], junc[1]] for junc in junctions]
#             per_frame_junctions.extend(junc_array)
    
#     else:
#         ref_junctions = []
#         per_frame_junctions = []
#         ref_graph = []

#     return ref_junctions, per_frame_junctions, ref_graph


# def label_junctions(group, series_num):
#         """
#         Get connected component output for junction classification
#         """


#         ref_junctions, per_frame_junctions, ref_graph = self.get_all_junc(group, series_num)

#         if len(ref_junctions) == 0:
#             return [], [], [], []

#         ref_junctions = np.array(ref_junctions)
#         per_frame_junctions = np.array(per_frame_junctions)

#         junc_spread_img = np.zeros((128, 128))

#         for junction in per_frame_junctions:
#             junc_spread_img[junction[0], junction[1]] = 255.

#         labelled_img = label(junc_spread_img, connectivity=2)
#         # imageio.imsave('Climp12_junc_labelled.png', labelled_img)
#         # fig.add_subplot(1,2,2)
#         # plt.axis('off')
#         # plt.imshow(labelled_img, cmap='gray')
#         # plt.savefig('Climp12_junc_labelled.png', bbox_inches='tight', pad_inches=0, dpi=700)
#         # plt.close()
#         # plt.show()
#         # exit()

#         return ref_junctions, per_frame_junctions, labelled_img, ref_graph

def get_junc_sum_per_frame(group, series_num, region):

    ref_junctions, per_frame_junctions, labelled_img, _ = junc_analysis.label_junctions(group, series_num)

    # dict with ids as key and (x, y) as value
    label_ids = junc_analysis.separate_junc_cc(ref_junctions, labelled_img)

    # iso, fuz, unk: list of lists with x, y
    iso, fuz = junc_analysis.get_junction_areas(label_ids)

    if region == 'iso':
        cc = get_cc_ids(labelled_img, iso)
    else:
        cc = get_cc_ids(labelled_img, fuz)
    # # unk_cc = get_cc_ids(labelled_img, unk)

    cc_coords = {each: np.where(labelled_img==each) for each in cc}

    new_dt = {}

    for k, v in cc_coords.items():
        pairs = list(zip(v[0], v[1]))
        new_dt[k] = pairs

    mean_frame_dt = {}
    pf_dt = {}

    mean_proj_frame = imageio.imread(f'{sted_data_path}/vess_enh_unet/{group.lower()}/nerdynet_v2/sted_{group.lower()}{series_num}_er_mean_pred.png')

    mean_proj_frame = (mean_proj_frame - np.min(mean_proj_frame)) / (np.max(mean_proj_frame) - np.min(mean_proj_frame))

    mean_skel = pcv.morphology.skeletonize(mask=mean_proj_frame)

    mean_graph = sknw.build_sknw(mean_skel, multi=False, iso=False)

    mean_node_set, mean_degree_list = mean_graph.nodes, mean_graph.degree

    # get node coordinates
    mean_node_coords = np.array([mean_node_set[node]['o'] for node in mean_node_set])

    # return coordinates of junctions with degree > 2
    mean_junctions = [mean_node_coords[i] for i, val in enumerate(mean_degree_list) if val[1] > 2]

    for frame in range(100):

        # UNet pipeline
        skeleton_path = f'{sted_data_path}/vess_enh_unet/{group.lower()}/skel/{group_pref[group]}{series_num}_decon_t0{frame:02d}_ch00_skel.png'

        # img = imageio.imread(skeleton_path)
        # skel = pcv.morphology.skeletonize(mask=img)
        skel = imageio.imread(skeleton_path)
        graph = sknw.build_sknw(skel, multi=False, iso=False)

        node_set, degree_list = graph.nodes, graph.degree

        # get node coordinates
        node_coords = np.array([node_set[node]['o'] for node in node_set])

        # return coordinates of junctions with degree > 2    
        junctions = [node_coords[i] for i, val in enumerate(degree_list) if val[1] > 2]

        # print(junctions)
        for junction in junctions:
            for key, value in new_dt.items():
                if (junction[0], junction[1]) in value:
                    if key not in pf_dt:
                        pf_dt[key] = 1
                    pf_dt[key] += 1

    for junction in mean_junctions:
        for key, value in new_dt.items():
            if (junction[0], junction[1]) in value:
                if key not in mean_frame_dt:
                    mean_frame_dt[key] = 1
                mean_frame_dt[key] += 1

    # print(pf_dt)
    # print(mean_frame_dt)
    # exit()
    return pf_dt, mean_frame_dt

def get_junc_ratio_data(group):
    ratio_data = []
    for ser in range(1, 17):
        try:
            pf_dt, mean_frame_dt = get_junc_sum_per_frame(group, ser, 'fuz')
            keys = list(pf_dt.keys())
            for k in keys:
                ratio_data.append(pf_dt[k]/mean_frame_dt[k])
        except:
            continue
    return ratio_data

# print(ratio_data)
# plt.plot(ratio_data)
# plt.show()

ctrl = get_junc_ratio_data('Control')
rtn = get_junc_ratio_data('RTN')
climp = get_junc_ratio_data('Climp')

df = pd.DataFrame()
df['Ratio'] = pd.Series(np.concatenate((ctrl, rtn, climp)))
df['Group'] = pd.Series(np.concatenate((['Control']*len(ctrl), ['Reticulon']*len(rtn), ['Climp']*len(climp))))

ax = sns.boxplot(data=df, x='Group', y='Ratio', showfliers=False, width=0.6)

box_pairs = [('Climp', 'Control'), ('Climp', 'Reticulon'), ('Control', 'Reticulon')]
annotator = Annotator(ax, box_pairs, data=df, x='Group', y='Ratio')
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=9)
annotator.apply_and_annotate()

yt = ax.get_yticks()
yt = [f'{y:.2f}' for y in yt]
ax.set_xticklabels(ax.get_xticklabels(), fontsize=9, rotation=45)
ax.set_yticklabels(yt, fontsize=9)

plt.xlabel('Group', fontsize=12)
plt.ylabel('Ratio', fontsize=12)
plt.gcf().set_size_inches(2, 6)

plt.savefig('sted_fuz_cc_junc_sum_ratio_v5.png', dpi=300, bbox_inches='tight')

# plt.show()
plt.close()

exit()



# with open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/ATL_fuz_cc_patches.pkl', 'rb') as f:
#     atl = pkl.load(f)

# with open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/Climp_fuz_cc_patches.pkl', 'rb') as f:
#     climp = pkl.load(f)

# with open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/Control_fuz_cc_patches.pkl', 'rb') as f:
#     control = pkl.load(f)

# with open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/RTN_fuz_cc_patches.pkl', 'rb') as f:
#     rtn = pkl.load(f)

# print(atl)

# exit()

# atl = get_fuz_cc_variance_data('ATL')
# climp = get_fuz_cc_variance_data('Climp')
# control = get_fuz_cc_variance_data('Control')
# rtn = get_fuz_cc_variance_data('RTN')


climpX, climpY = get_fuz_cc_variance_data('Climp')
rtnX, rtnY = get_fuz_cc_variance_data('RTN')
controlX, controlY = get_fuz_cc_variance_data('Control')
atlX, atlY = get_fuz_cc_variance_data('ATL')
# if element in climpX is greater than 15, remove it along with corresponding element in climpY

climpX = [x for x in climpX if x < 15]
climpY = [y for y in climpY if y < 500]

rtnX = [x for x in rtnX if x < 30]
rtnY = [y for y in rtnY if y < 1000]

# controlX = [x for x in controlX if x < 30]
# controlY = [y for y in controlY if y < 1000]

# atlX = [x for x in atlX if x < 30]
# atlY = [y for y in atlY if y < 1000]


fig = plt.figure()
ax = fig.add_subplot(111)


ax.scatter(climpX, climpY, label='Climp', alpha=0.5)
ax.scatter(rtnX, rtnY, label='RTN', alpha=0.5)
ax.scatter(controlX, controlY, label='Control', alpha=0.5)
ax.scatter(atlX, atlY, label='ATL', alpha=0.5)
# plt.scatter(controlX, controlY)
# plt.scatter(rtnX, rtnY)
# plt.scatter(atlX, atlY)

plt.ylim(0, 800)
plt.xlim(0, 30)

plt.legend()

plt.title('#junctions vs #reference junctions in Overlapping CC')

plt.xlabel('#reference junctions')
plt.ylabel('#junctions')

plt.savefig('Tot_junc_ref_junc_scatter.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.close()

# plt.show()

exit()


# with open('atl_tot_junc_ref_junc_ratio.pkl', 'wb') as f:
#     pkl.dump(atl, f)

# with open('sted_climp_tot_junc_ref_junc_ratio.pkl', 'wb') as f:
#     pkl.dump(climp, f)

# with open('sted_control_tot_junc_ref_junc_ratio.pkl', 'wb') as f:
#     pkl.dump(control, f)

# with open('sted_rtn_tot_junc_ref_junc_ratio.pkl', 'wb') as f:
#     pkl.dump(rtn, f)

# with open('atl_tot_junc_ref_junc_ratio.pkl', 'rb') as f:
#     atl = pkl.load(f)

with open('sted_climp_tot_junc_ref_junc_ratio.pkl', 'rb') as f:
    climp = pkl.load(f)

with open('sted_control_tot_junc_ref_junc_ratio.pkl', 'rb') as f:
    control = pkl.load(f)

with open('sted_rtn_tot_junc_ref_junc_ratio.pkl', 'rb') as f:
    rtn = pkl.load(f)



#atl = (atl - np.min(atl)) / (np.max(atl) - np.min(atl))
#climp = (climp - np.min(climp)) / (np.max(climp) - np.min(climp))
#control = (control - np.min(control)) / (np.max(control) - np.min(control))
#rtn = (rtn - np.min(rtn)) / (np.max(rtn) - np.min(rtn))

#with open('sted_climp_fuz_var_data.pkl', 'wb') as f:
#    pkl.dump(climp, f)

#with open('sted_control_fuz_var_data.pkl', 'wb') as f:
#    pkl.dump(control, f)

#with open('sted_rtn_fuz_var_data.pkl', 'wb') as f:
#    pkl.dump(rtn, f)


#with open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/sted_Climp_fuz_cc_variance_data.pkl', 'rb') as f:
#    climp = pkl.load(f)

#with open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/sted_Control_fuz_cc_variance_data.pkl', 'rb') as f:
#    control = pkl.load(f)

#with open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/sted_RTN_fuz_cc_variance_data.pkl', 'rb') as f:  
#    rtn = pkl.load(f)


# with open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/conf_ATL_fuz_cc_variance_data.pkl', 'rb') as f:
#     atl = pkl.load(f)

# with open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/conf_Climp_fuz_cc_variance_data.pkl', 'rb') as f:
#     climp = pkl.load(f)

# with open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/conf_Control_fuz_cc_variance_data.pkl', 'rb') as f:
#     control = pkl.load(f)

# with open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/conf_RTN_fuz_cc_variance_data.pkl', 'rb') as f:
#     rtn = pkl.load(f)


df = pd.DataFrame()
# df['Variance'] = pd.Series(np.concatenate((control, rtn, climp, atl)))
# df['Group'] = pd.Series(np.concatenate((['Control']*len(control), ['Reticulon']*len(rtn), ['Climp']*len(climp), ['Atlastin']*len(atl))))

df['Variance'] = pd.Series(np.concatenate((control, rtn, climp)))
df['Group'] = pd.Series(np.concatenate((['Control']*len(control), ['Reticulon']*len(rtn), ['Climp']*len(climp))))

colors = sns.color_palette(n_colors=4)

# pal = {'Control': colors[0], 'Reticulon': colors[1], 'Climp': colors[2], 'Atlastin': colors[3]}

pal = {'Control': colors[0], 'Reticulon': colors[1], 'Climp': colors[2]}

ax = sns.boxplot(data=df, x='Group', y='Variance', showfliers=False, width=0.6, palette=pal)

# box_pairs = [('Atlastin', 'Climp'), ('Atlastin', 'Control'), ('Atlastin', 'Reticulon'), ('Climp', 'Control'), ('Climp', 'Reticulon'), ('Control', 'Reticulon')]



box_pairs = [('Climp', 'Control'), ('Climp', 'Reticulon'), ('Control', 'Reticulon')]

annotator = Annotator(ax, box_pairs, data=df, x='Group', y='Variance')
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=9)

annotator.apply_and_annotate()



# statannot.add_stat_annotation(ax, x='Region', y='Variance', data=df, box_pairs=box_pairs,
#                                     test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=15)

# plt.ylim(0, 45)

yt = ax.get_yticks()
yt = [f'{y:.1f}' for y in yt]
ax.set_xticklabels(ax.get_xticklabels(), fontsize=9, rotation=45)
ax.set_yticklabels(yt, fontsize=9)

# plt.legend(['ATL', 'Climp', 'Control', 'RTN'], fontsize=15)

plt.xlabel('Group', fontsize=12)
#plt.ylabel('Mean CC area', fontsize=13)
#plt.ylabel('CC area mean', fontsize=13)
plt.ylabel('Ratio', fontsize=12)
plt.gcf().set_size_inches(2, 6)
#plt.title(f'Mean of Overlapping CC in \n mean projection over CC area (STED) ', fontsize=12)
#plt.title('Isolated CC area mean \n per time-series (Confocal)', fontsize=12)
#plt.title('Overlapping CC area mean \n per time-series (STED)', fontsize=12)
#plt.title('Mean of Overlapping CC intensity in \n mean projection per time series (STED)', fontsize=12)

# plt.title('Ratio of Total junctions and reference \n junctions in Overlapping CC (STED)', fontsize=12)

plt.savefig('sted_fuz_cc_area_ref_junc_ratio_v5.png', dpi=300, bbox_inches='tight')
# plt.savefig('conf_fuz_cc_variance.png', dpi=300, bbox_inches='tight')
plt.close()

exit()

def get_ref_junc_per_fuz_CC(group):
    data = []
    lab_img_data = []
    ref_junc_data = []
    for series in range(1, group_dict[group]+1):
        lab_img, ref_junctions = get_fuz_cc_outline(group, series, 'fuz')

        # ref_junctions: list of lists [x y]
        lab_img_data.append(lab_img)
        ref_junc_data.append(ref_junctions)

        # get ref_junctions per CC
        for cc_id in range(1, lab_img.max()+1):
            cc_id_coords = np.where(lab_img==cc_id)

            cc_id_coords = np.stack(cc_id_coords, axis=1)

            # get the intersection of ref_junctions and cc_id_coords

            set_ref_junc = {tuple(row) for row in ref_junctions}
            set_cc_id_coords = {tuple(row) for row in cc_id_coords}

            intersection_set = set_ref_junc.intersection(set_cc_id_coords)

            intersection = [np.array(row) for row in intersection_set]

            data.append(len(intersection))

    with open(f'{group}_fuz_cc_patches.pkl', 'wb') as f:
            pkl.dump(lab_img_data, f)
            
    with open(f'{group}_ref_junc.pkl', 'wb') as f:
            pkl.dump(ref_junc_data, f)

    return data

data = get_ref_junc_per_fuz_CC('RTN')
plt.hist(data)

plt.show()

data = get_ref_junc_per_fuz_CC('ATL')

exit()


# lab_img = get_fuz_cc_outline('ATL', 1)
# fuzzy_coords = np.where(lab_img)

# def check_fuz_cc_edt():
#     lab_img = get_fuz_cc_outline('RTN', 5, 'fuz')
#     fuzzy_coords = np.where(lab_img)

#     lab_img_iso = get_fuz_cc_outline('RTN', 5, 'iso')
#     iso_coords = np.where(lab_img_iso)

#     data = np.zeros((128, 128))

#     for i in range(100):
#         skel1 = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/skel/R5/R5_decon_t0{i:02d}_ch00_skel.png')

#         skel1 = np.invert(skel1)

#         edt1 = ndimage.distance_transform_edt(skel1)

#         data = data + edt1

#     data = data / 100

#     # print(data[fuzzy_coords])
#     plt.hist(data[fuzzy_coords])
#     plt.show()

#     plt.hist(data[iso_coords])
#     plt.show()

#     # for cc_id in range(1, lab_img.max()+1):
#     #     cc_id_coords = np.where(lab_img==cc_id)

#         # cc_id_coords = np.stack(cc_id_coords, axis=1)


# check_fuz_cc_edt()
# exit()


def get_fuz_cc_ids(a, b):

    a = set(a)
    b = set(b)

    intersection = a.intersection(b)

    return list(intersection)


# lab_img_iso = get_fuz_cc_outline('ATL', 1, 'iso')
# l = []
# for ser in range(1, 30):
#     lab_img_fuz = get_fuz_cc_outline('RTN', ser, 'fuz')
#     l.extend(
#         len(np.where(lab_img_fuz == cc_id)[0])
#         for cc_id in range(1, lab_img_fuz.max() + 1)
#     )

# plt.hist(l)
# plt.show()

# exit()


# lab_img = get_fuz_cc_outline('ATL', 1, 'iso')
# group = 'ATL'
# series_num = 1
# ratio = []
# for cc_id in range(1, lab_img.max()+1):
#     cc_id_coords = np.where(lab_img==cc_id)


#     er = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/std_egfp/{group_pref[group]}{series_num}_decon_t000_ch00_std.png')

#     skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/{group_pref[group]}{series_num}/{group_pref[group]}{series_num}_decon_t000_ch00_skel.png')

#     skel_coords = np.where(skel)

#     skel_pixels = get_intersection(cc_id_coords, skel_coords)

#     ratio.append(len(skel_pixels)/len(cc_id_coords[0]))

# print(ratio)

# exit()


def get_single_frame_cc_intensity(group, series_num, lab_img):
    data_skel = []
    data_cc = []
    for cc_id in range(1, lab_img.max()+1):
        cc_id_coords = np.where(lab_img==cc_id)


        er = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/std_egfp/{group_pref[group]}{series_num}_decon_t000_ch00_std.png')

        skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/{group_pref[group]}{series_num}/{group_pref[group]}{series_num}_decon_t000_ch00_skel.png')

        skel_coords = np.where(skel)

        skel_pixels = get_intersection(cc_id_coords, skel_coords)

        values_at_skel_coordinates = [er[coord[0], coord[1]] for coord in skel_pixels]

        values_at_cc_coordinates = er[cc_id_coords]

        data_skel.extend(values_at_skel_coordinates)
        data_cc.extend(values_at_cc_coordinates)

    if sum(data_cc) != 0:
        return sum(data_skel) / sum(data_cc)


def get_skel_intensity_under_CC(group):

    group_name = {'ATL': 'Atlastin', 'Climp': 'Climp', 'Control': 'Control', 'RTN': 'Reticulon'}

    data_iso = []
    data_fuz = []
    for series in range(1, group_dict[group]+1):
        lab_img_iso = get_fuz_cc_outline(group, series, 'iso')
        iso_val = get_single_frame_cc_intensity(group, series, lab_img_iso)

        lab_img_fuz = get_fuz_cc_outline(group, series, 'fuz')
        fuz_val = get_single_frame_cc_intensity(group, series, lab_img_fuz)

        data_iso.append(iso_val)
        data_fuz.append(fuz_val)

    df = pd.DataFrame()
    df['Intensity'] = pd.Series(data_iso + data_fuz)
    df['Region'] = pd.Series([f'{group}_iso']*len(data_iso) + [f'{group}_fuz']*len(data_fuz))

    ax = sns.boxplot(data=df, x='Region', y='Intensity', showfliers=False)

    box_pairs = [(f'{group}_iso', f'{group}_fuz')]
    statannot.add_stat_annotation(ax, x='Region', y='Intensity', data=df, box_pairs=box_pairs,
                                    test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=15)

    yt = ax.get_yticks()
    yt = [f'{y:.2f}' for y in yt]
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(yt, fontsize=15)

    plt.xlabel('Region', fontsize=18)
    plt.ylabel('Intensity', fontsize=18)
    plt.gcf().set_size_inches(2.5, 6)
    # plt.title(f'{group_name[group]}: Intensity under skeleton over CC area for fuzzy and isolated regions per sequence', fontsize=16)

    plt.show()


# get_skel_intensity_under_CC('ATL')
# get_skel_intensity_under_CC('Climp')
# get_skel_intensity_under_CC('Control')
# get_skel_intensity_under_CC('RTN')

# exit()


replicate_data_iso = []
replicate_data_fuz = []
for series in range(1, 11):
    atl_iso_img = get_fuz_cc_outline('ATL', series, 'iso')
    atl1_iso = get_single_frame_cc_intensity('ATL', series, atl_iso_img)
    replicate_data_iso.extend(atl1_iso)

    atl_fuz_img = get_fuz_cc_outline('ATL', series, 'fuz')
    atl1_fuz = get_single_frame_cc_intensity('ATL', series, atl_fuz_img)
    replicate_data_fuz.extend(atl1_fuz)


print(len(replicate_data_iso))
print(len(replicate_data_fuz))

df = pd.DataFrame()
df['Intensity'] = pd.Series(replicate_data_iso + replicate_data_fuz)
df['Region'] = pd.Series(['ATL_iso']*len(replicate_data_iso) + ['ATL_fuz']*len(replicate_data_fuz))

# df['Intensity'] = pd.Series(atl1_iso + atl1_fuz)
# df['Region'] = pd.Series(['ATL_iso']*len(atl1_iso) + ['ATL_fuz']*len(atl1_fuz))

ax = sns.boxplot(data=df, x='Region', y='Intensity', showfliers=False)

box_pairs = [('ATL_iso', 'ATL_fuz')]

statannot.add_stat_annotation(ax, x='Region', y='Intensity', data=df, box_pairs=box_pairs,
                              test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=15)


plt.xlabel('Region', fontsize=16)
plt.ylabel('Intensity', fontsize=16)
plt.gcf().set_size_inches(4, 8)

plt.show()

# sns.boxplot(data=[atl1_iso, atl1_fuz], orient='v')
# plt.legend(['Isolated', 'Fuzzy'])
# plt.xlabel('CC Type', fontsize=16)
# plt.ylabel('Intensity', fontsize=16)
# plt.show()

exit()


def get_cc_intensity(group):
    group_data_iso = []
    group_data_fuz = []
    for series_num in range(1, group_dict[group]+1):
        lab_img_iso = get_fuz_cc_outline(group, series_num, 'iso')
        lab_img_fuz = get_fuz_cc_outline(group, series_num, 'fuz')

        data_iso = get_single_frame_cc_intensity(group, series_num, lab_img_iso)
        data_fuz = get_single_frame_cc_intensity(group, series_num, lab_img_fuz)

        group_data_iso.append(np.mean(data_iso))
        group_data_fuz.append(np.mean(data_fuz))

    return group_data_iso, group_data_fuz


atl_data_iso, atl_data_fuz = get_cc_intensity('ATL')
sns.boxplot(data=[atl_data_iso, atl_data_fuz], orient='v')
plt.legend(['Isolated', 'Fuzzy'])
plt.xlabel('CC Type', fontsize=16)
plt.ylabel('Intensity', fontsize=16)
plt.show()

exit()

def get_skel_intensity_over_cc_intensity(lab_img):
    """
    Get the mean intensity of the skeleton over the mean intensity per CC based on region type
    """
    data = []
    for cc_id in range(1, lab_img.max()+1):
        cc_id_coords = np.where(lab_img==cc_id)

        cc_data = []

        for i in range(100):
            er = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_egfp/A1_decon_t0{i:02d}_ch00_std.png')

            skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t0{i:02d}_ch00_skel.png')

            skel_coords = np.where(skel)

            skel_pixels = get_intersection(cc_id_coords, skel_coords)

            values_at_skel_coordinates = [er[coord[0], coord[1]] for coord in skel_pixels]

            values_at_cc_coordinates = er[cc_id_coords]

            val = sum(values_at_skel_coordinates) / sum(values_at_cc_coordinates)

            cc_data.append(val)

        data.append(cc_data)
    return data


def get_iso_fuz_cc_skel_data(group):
    group_data_iso = []
    group_data_fuz = []
    for series_num in range(1, group_dict[group]+1):
        lab_img_iso = get_fuz_cc_outline(group, series_num, 'iso')
        lab_img_fuz = get_fuz_cc_outline(group, series_num, 'fuz')

        data_iso = get_skel_intensity_over_cc_intensity(lab_img_iso)
        data_fuz = get_skel_intensity_over_cc_intensity(lab_img_fuz)        

        group_data_iso.extend(data_iso)
        group_data_fuz.extend(data_fuz)

    return group_data_iso, group_data_fuz


def plot_iso_fuz_skel_intensity_variation():
    atl_data_iso, atl_data_fuz = get_iso_fuz_cc_skel_data('ATL')
    # climp_data_iso, climp_data_fuz = get_iso_fuz_cc_skel_data('Climp')
    # control_data_iso, control_data_fuz = get_iso_fuz_cc_skel_data('Control')
    # rtn_data_iso, rtn_data_fuz = get_iso_fuz_cc_skel_data('RTN')

    atl_data_iso = [item for sublist in atl_data_iso for item in sublist]
    atl_data_fuz = [item for sublist in atl_data_fuz for item in sublist]


    df = pd.DataFrame()
    df['Intensity'] = pd.Series(np.concatenate((atl_data_iso, atl_data_fuz)))
    df['Region'] = pd.Series(['ATL_iso']*len(atl_data_iso) + ['ATL_fuz']*len(atl_data_fuz))

    sns.boxplot(data=df, x='Region', y='Intensity', showfliers=False)

    plt.xlabel('Region', fontsize=16)
    plt.ylabel('Intensity', fontsize=16)

    plt.show()
    
# plot_iso_fuz_skel_intensity_variation()
# exit()

# data_iso, data_fuz = get_iso_fuz_cc_skel_data('ATL', 1)

# data_iso = [item for sublist in data_iso for item in sublist]
# data_fuz = [item for sublist in data_fuz for item in sublist]

# sns.boxplot(data=[data_iso, data_fuz], orient='v')
# plt.legend(['Isolated', 'Fuzzy'])
# plt.xlabel('CC Type', fontsize=16)
# plt.ylabel('Intensity', fontsize=16)
# plt.show()


exit()

def replace_zeros(data):
    for i, l in enumerate(data):
        if 0 in l:
            avg = np.mean(l)
            idx = l.index(0)
            data[i][idx] = avg
    return data


def get_df_series(group, series_num, region):

    # group_dict = {'Atlastin': atl_data, 'Climp': climp_data, 'Control': control_data, 'Reticulon': rtn_data}

    data_iso, data_fuz = get_iso_fuz_cc_skel_data(group, series_num)
    data_iso = replace_zeros(data_iso)
    data_fuz = replace_zeros(data_fuz)

    region_dict = {'iso': data_iso, 'fuz': data_fuz}

    Tframe = []
    vals = []
    group_name = []

    # for i, l in enumerate(group_dict[group]):
        # Tframe.extend([i]*len(l))

    for i in range(100):
        Tframe.extend([i]*(len(region_dict[region])-1))

    for i in range(len(region_dict[region])):
        group_name.extend([group + '_' + region]*len(region_dict[region][i]))

    transposed_list = [list(row) for row in zip(*region_dict[region])]

    for transposed in transposed_list:
        vals.extend(transposed)

    return Tframe, vals, group_name

    
def create_dataframe():
    # tf_atl, vals_atl, group_atl = get_df_series('Atlastin')
    # tf_climp, vals_climp, group_climp = get_df_series('Climp')
    # tf_control, vals_control, group_control = get_df_series('Control')
    # tf_rtn, vals_rtn, group_rtn = get_df_series('Reticulon')
    atl_tf_iso, atl_vals_iso, atl_group_iso = get_df_series('ATL', 1, 'iso')
    atl_tf_fuz, atl_vals_fuz, atl_group_fuz = get_df_series('ATL', 1, 'fuz')

    climp_tf_iso, climp_vals_iso, climp_group_iso = get_df_series('Climp', 1, 'iso')
    climp_tf_fuz, climp_vals_fuz, climp_group_fuz = get_df_series('Climp', 1, 'fuz')

    control_tf_iso, control_vals_iso, control_group_iso = get_df_series('Control', 1, 'iso')
    control_tf_fuz, control_vals_fuz, control_group_fuz = get_df_series('Control', 1, 'fuz')

    rtn_tf_iso, rtn_vals_iso, rtn_group_iso = get_df_series('RTN', 1, 'iso')
    rtn_tf_fuz, rtn_vals_fuz, rtn_group_fuz = get_df_series('RTN', 1, 'fuz')
    

    df = pd.DataFrame()
    df['Time'] = pd.Series(np.concatenate([atl_tf_iso, atl_tf_fuz, climp_tf_iso, climp_tf_fuz, control_tf_iso, control_tf_fuz, rtn_tf_iso, rtn_tf_fuz]))
    df['Values'] = pd.Series(np.concatenate([atl_vals_iso, atl_vals_fuz, climp_vals_iso, climp_vals_fuz, control_vals_iso, control_vals_fuz, rtn_vals_iso, rtn_vals_fuz]))
    df['Group'] = pd.Series(np.concatenate([atl_group_iso, atl_group_fuz, climp_group_iso, climp_group_fuz, control_group_iso, control_group_fuz, rtn_group_iso, rtn_group_fuz]))

    return df



df = create_dataframe()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Time', y='Values', hue='Group')
plt.xlabel('Time', fontsize=16)
plt.ylabel('Values', fontsize=16)

plt.show()

exit()

# data_iso = replace_zeros(data_iso)
# data_fuz = replace_zeros(data_fuz)

# Tframe = []
# vals = []
# group_name = []

# for i in range(100):
#     Tframe.extend([i]*(len(data_iso)-1))

# for i in range(len(data_iso)):
#     group_name.extend(['ATL_iso']*len(data_iso[i]))

# transposed_list = [list(row) for row in zip(*data_iso)]

# for transposed in transposed_list:
#     vals.extend(transposed)

# df = pd.DataFrame()
# df['Time'] = pd.Series(Tframe)
# df['Values'] = pd.Series(vals)
# df['Group'] = pd.Series(group_name)

# sns.lineplot(data=df, x='Time', y='Values', hue='Group')
# plt.xlabel('Time', fontsize=16)
# plt.ylabel('Values', fontsize=16)
# # plt.title('Fuzzy CC skeleton mean intensity over CC area per frame', fontsize=18)

# plt.show()

# exit()




def fuz_cc_degree_variation(group):
    group_data = []
    for series in range(1, group_dict[group]+1):
        lab_img = get_fuz_cc_outline(group, series)

        data = []

        for cc_id in range(1, lab_img.max()+1):
            cc_id_coords = np.where(lab_img==cc_id)
            cc_id_coords_list = list(zip(cc_id_coords[0], cc_id_coords[1]))
            cc_data = []
            for i in range(100):
                skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A1/A1_decon_t0{i:02d}_ch00_skel.png')

                graph = sknw.build_sknw(skel, multi=True, iso=False)

                nodes = graph.nodes

                node_coords = np.array([nodes[node]['o'] for node in nodes])

                node_coords_list = list(zip(node_coords[:,0], node_coords[:,1]))                

                fuz_cc_nodes = get_fuz_cc_ids(node_coords_list, cc_id_coords_list)

                degree_data = []
                for node in fuz_cc_nodes:
                    if node in node_coords_list:
                        idx = node_coords_list.index(node)
                        degree_data.append(graph.degree[idx])

                if degree_data:
                    cc_data.append(np.sum(degree_data)/len(degree_data))

            data.append(cc_data)
        group_data.extend(data)

    return group_data


atl_degree_variation = fuz_cc_degree_variation('ATL')
climp_degree_variation = fuz_cc_degree_variation('Climp')
control_degree_variation = fuz_cc_degree_variation('Control')
rtn_degree_variation = fuz_cc_degree_variation('RTN')

with open('atl_fuz_cc_degree_variation.pkl', 'wb') as f:
    pickle.dump(atl_degree_variation, f)

with open('climp_fuz_cc_degree_variation.pkl', 'wb') as f:
    pickle.dump(climp_degree_variation, f)

with open('control_fuz_cc_degree_variation.pkl', 'wb') as f:
    pickle.dump(control_degree_variation, f)

with open('rtn_fuz_cc_degree_variation.pkl', 'wb') as f:
    pickle.dump(rtn_degree_variation, f)

exit()

# TO CHECK the following code

def fuz_cc_degree_variation(group):
    group_data = []
    for ser_num in range(1, group_dict[group]+1):
        lab_img = get_fuz_cc_outline(group, ser_num)

        data = []
        for cc_id in range(1, lab_img.max()+1):
            cc_id_coords = np.where(lab_img==cc_id)
            cc_data = []
            for frame in range(100):

                skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/{group_pref[group]}{ser_num}/{group_pref[group]}{ser_num}_decon_t0{frame:02d}_ch00_skel.png')

                graph = sknw.build_sknw(skel, multi=True, iso=False)

                nodes = graph.nodes

                node_coords = np.array([nodes[node]['o'] for node in nodes])

                node_coords_list = list(zip(node_coords[:,0], node_coords[:,1]))

                fuz_cc_nodes = get_fuz_cc_ids(node_coords_list, cc_id_coords)

                degree_data = []
                for node in fuz_cc_nodes:
                    if node in node_coords_list:
                        idx = node_coords_list.index(node)
                        degree_data.append(graph.degree[idx])

                if degree_data:
                    cc_data.append(np.sum(degree_data)/len(degree_data))

            data.append(cc_data)

        # skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/{group_pref[group]}{ser_num}/{group_pref[group]}{ser_num}_decon_t000_ch00_skel.png')

        # graph = sknw.build_sknw(skel, multi=True, iso=False)

        # nodes = graph.nodes

        # node_coords = np.array([nodes[node]['o'] for node in nodes])

        # node_coords_list = list(zip(node_coords[:,0], node_coords[:,1]))

        

        # cc_data = []

        # for cc_id in range(1, lab_img.max()+1):
        #     cc_id_coords = np.where(lab_img==cc_id)

        #     cc_id_coords_list = list(zip(cc_id_coords[0], cc_id_coords[1]))

        #     fuz_cc_nodes = get_fuz_cc_ids(node_coords_list, cc_id_coords_list)

        #     degree_data = []
        #     for node in fuz_cc_nodes:
        #         if node in node_coords_list:
        #             idx = node_coords_list.index(node)
        #             degree_data.append(graph.degree[idx])

        #     if degree_data:
        #         cc_data.append(np.sum(degree_data)/len(degree_data))

        # data.append(cc_data)
    return data


def get_fuz_cc_area(lab_img):
    cc_area_list = []
    for cc_id in range(1, lab_img.max()+1):
        cc_id_coords = np.where(lab_img==cc_id)
        cc_area_list.append(len(cc_id_coords[0]))
    return cc_area_list


def get_skel_per_fuz_cc(group):

    group_data = []
    for ser_num in range(1, group_dict[group]+1):
        lab_img = get_fuz_cc_outline(group, ser_num)
        data = []
        for cc_id in range(1, lab_img.max()+1):
            cc_id_coords = np.where(lab_img==cc_id)
            cc_data = []
            for frame in range(100):

                er = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/std_egfp/{group_pref[group]}{ser_num}_decon_t0{frame:02d}_ch00_std.png')

                skel = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/skel/{group_pref[group]}{ser_num}/{group_pref[group]}{ser_num}_decon_t0{frame:02d}_ch00_skel.png')
                skel_coords = np.where(skel)

                fuz_skel_pixels = get_intersection(cc_id_coords, skel_coords)

                values_at_coordinates = [er[coord[0], coord[1]] for coord in fuz_skel_pixels]
                
                if values_at_coordinates:
                    # Calculate the mean of the extracted values
                    # mean_value = np.mean(values_at_coordinates)
                    # cc_data.append(mean_value/255.)

                    if cc_id_coords:
                        mean_val_over_area = (np.mean(values_at_coordinates)/255.) / len(cc_id_coords[0])
                    else:
                        mean_val_over_area = 0
                    
                    cc_data.append(mean_val_over_area)
                else:
                    cc_data.append(0)
                    # cc_data.append(mean_val_over_area)
            data.append(cc_data)
        group_data.extend(data)
    return group_data


def plot_fuz_skel_intensity_variation(group):
    data = get_skel_per_fuz_cc(group)
    group_names = {'ATL': 'Atlastin', 'Climp': 'Climp63', 'Control': 'Control', 'RTN': 'Reticulon'}
    data = np.array(data)

    df = pd.DataFrame(data, columns=[f't={i}' for i in range(data.shape[1])])

    sns.boxplot(data=df, orient='v')  # 'orient' specifies vertical orientation
    plt.xlabel('Frames', fontsize=16)
    plt.ylabel('Mean Intensity', fontsize=16)
    # plt.title(f'Fuzzy CC skeleton mean intensity per frame over CC area - {group_names[group]}', fontsize=18)
    plt.title(f'Fuzzy CC skeleton mean intensity per frame - {group_names[group]}', fontsize=18)
    plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
    plt.show()


import pickle

# atl_data = get_skel_per_fuz_cc('ATL')
# with open('atl_fuz_cc_intensity_over_area.pkl', 'wb') as f:
#     pickle.dump(atl_data, f)

# climp_data = get_skel_per_fuz_cc('Climp')
# with open('climp_fuz_cc_intensity_over_area.pkl', 'wb') as f:
#     pickle.dump(climp_data, f)

# control_data = get_skel_per_fuz_cc('Control')
# with open('control_fuz_cc_intensity_over_area.pkl', 'wb') as f:
#     pickle.dump(control_data, f)

# rtn_data = get_skel_per_fuz_cc('RTN')
# with open('rtn_fuz_cc_intensity_over_area.pkl', 'wb') as f:
#     pickle.dump(rtn_data, f)

# exit()

atl_data = pickle.load(open('atl_fuz_cc_intensity.pkl', 'rb'))
climp_data = pickle.load(open('climp_fuz_cc_intensity.pkl', 'rb'))
control_data = pickle.load(open('control_fuz_cc_intensity.pkl', 'rb'))
rtn_data = pickle.load(open('rtn_fuz_cc_intensity.pkl', 'rb'))


def replace_zeros(data):
    for i, l in enumerate(data):
        if 0 in l:
            avg = np.mean(l)
            idx = l.index(0)
            data[i][idx] = avg
    return data

atl_data = replace_zeros(atl_data)
climp_data = replace_zeros(climp_data)
control_data = replace_zeros(control_data)
rtn_data = replace_zeros(rtn_data)


def get_df_series(group):

    group_dict = {'Atlastin': atl_data, 'Climp': climp_data, 'Control': control_data, 'Reticulon': rtn_data}

    Tframe = []
    vals = []
    group_name = []

    # for i, l in enumerate(group_dict[group]):
        # Tframe.extend([i]*len(l))

    for i in range(100):
        Tframe.extend([i]*(len(group_dict[group])-1))

    for i in range(len(group_dict[group])):
        group_name.extend([group]*len(group_dict[group][i]))

    transposed_list = [list(row) for row in zip(*group_dict[group])]

    for transposed in transposed_list:
        vals.extend(transposed)

    return Tframe, vals, group_name

    
def create_dataframe():
    tf_atl, vals_atl, group_atl = get_df_series('Atlastin')
    tf_climp, vals_climp, group_climp = get_df_series('Climp')
    tf_control, vals_control, group_control = get_df_series('Control')
    tf_rtn, vals_rtn, group_rtn = get_df_series('Reticulon')


    df = pd.DataFrame()
    df['Time'] = pd.Series(np.concatenate([tf_atl, tf_climp, tf_control, tf_rtn]))
    df['Values'] = pd.Series(np.concatenate([vals_atl, vals_climp, vals_control, vals_rtn]))
    df['Group'] = pd.Series(np.concatenate([group_atl, group_climp, group_control, group_rtn]))

    return df



import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# # Concatenate and reshape data for Seaborn's lineplot
# df_combined = pd.concat([df.melt(var_name='Time', value_name='Values') for df_name, df in dfs.items()])
# df_combined['Dataset'] = np.concatenate([[df_name] * len(df) for df_name, df in dfs.items()])

df = create_dataframe()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Time', y='Values', hue='Group')
plt.xlabel('Time', fontsize=16)
plt.ylabel('Values', fontsize=16)
# plt.title('Lineplot for Different Groups', fontsize=18)
# plt.title('Fuzzy CC skeleton mean intensity over CC area per frame', fontsize=18)
plt.title('Fuzzy CC skeleton mean intensity per frame', fontsize=18)
plt.show()


exit()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# # Generate example data (replace this with your actual data)
# num_time_points = 100
# num_values = 50
# num_classes = 4

# # Simulate data for different classes
# data = np.random.rand(num_time_points, num_values, num_classes)

# # Create a DataFrame
# time_points = np.arange(num_time_points)
# df = pd.DataFrame(data.reshape(-1, num_classes), columns=[f'Class_{i+1}' for i in range(num_classes)])
# df['Time'] = np.repeat(time_points, num_values)

# # Melt the DataFrame for Seaborn's lmplot
# df_melted = df.melt(id_vars=['Time'], var_name='Class', value_name='Values')

# plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# # Using lmplot to overlay regression lines for each class
# sns.lmplot(data=df_melted, x='Time', y='Values', hue='Class', scatter_kws={'s': 10})
# plt.xlabel('Time')
# plt.ylabel('Values')
# plt.title('Regression Lines for Different Classes')
# plt.show()



import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate example data (replace this with your actual data)
num_time_points = 100
num_values = 50
num_classes = 4

# Simulate data for different classes
data = np.random.rand(num_time_points, num_values, num_classes)

# Create a DataFrame
time_points = np.arange(num_time_points)
df = pd.DataFrame(data.reshape(-1, num_classes), columns=[f'Class_{i+1}' for i in range(num_classes)])

# print(df)

df['Time'] = np.repeat(time_points, num_values)

# Melt the DataFrame for Seaborn's lineplot
df_melted = df.melt(id_vars='Time', var_name='Class', value_name='Values')

print(df_melted)
exit()

plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# Using lineplot
sns.lineplot(data=df_melted, x='Time', y='Values', hue='Class')

# Using regplot (comment out the lineplot above if using regplot)
# sns.regplot(data=df_melted, x='Time', y='Values', hue='Class', scatter=False)

plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Lineplot of Values per Time Point for Different Classes')
plt.legend(title='Classes')
plt.show()



# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # Generate example data (replace this with your actual data)
# num_time_points = 100
# num_values = 50
# num_classes = 4

# # Simulate data for different classes
# data = np.random.rand(num_time_points, num_values, num_classes)

# # Create a DataFrame
# time_points = np.arange(num_time_points)
# df = pd.DataFrame(data.reshape(-1, num_classes), columns=[f'Class_{i+1}' for i in range(num_classes)])
# df['Time'] = np.repeat(time_points, num_values)

# # Loop through each class and create a regression plot
# for class_col in df.columns[:-1]:  # Exclude the 'Time' column
#     plt.figure(figsize=(8, 5))  # Adjust the figure size as needed
#     sns.regplot(data=df, x='Time', y=class_col, scatter_kws={'s': 10})
#     plt.xlabel('Time')
#     plt.ylabel('Values')
#     plt.title(f'Regression Plot for {class_col}')
#     plt.show()


