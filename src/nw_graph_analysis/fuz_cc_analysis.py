# Fuz CC analysis

import imageio
import numpy as np
import seaborn as sns
import pandas as pd
import sknw
import pickle
import cv2
import statannot
import matplotlib.pyplot as plt
from skimage import measure
from skimage.measure import label, regionprops
from skimage.morphology import dilation, closing
from junction_analysis_modules import JunctionAnalysis as JA

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'
junc_analysis = JA(confocal_data_path)


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

def get_fuz_cc_outline(group, series_num, region):
    ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, series_num)

    # dict with ids as key and (x, y) as value
    label_ids, unassigned_cc_dict = junc_analysis.separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)

    # iso, fuz, unk: list of lists with x, y
    iso, fuz, unk = junc_analysis.get_junction_areas(label_ids, unassigned_cc_dict)


    if region == 'iso':
        cc = get_cc_ids(labelled_img, iso)
    else:
        cc = get_cc_ids(labelled_img, fuz)
    # # unk_cc = get_cc_ids(labelled_img, unk)

    cc_coords = {each: np.where(labelled_img==each) for each in cc}

    # iso_cc_coords = {each: np.where(labelled_img==each) for each in iso_cc}
    # fuz_cc_coords = {each: np.where(labelled_img==each) for each in fuz_cc}
    # unk_cc_coords = {each: np.where(labelled_img==each) for each in unk_cc}

    spread_img = np.zeros((128, 128))
    for v in cc_coords.values():
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


# lab_img = get_fuz_cc_outline('ATL', 1)
# fuzzy_coords = np.where(lab_img)


def get_fuz_cc_ids(a, b):

    a = set(a)
    b = set(b)

    intersection = a.intersection(b)

    return list(intersection)



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
get_skel_intensity_under_CC('Climp')
get_skel_intensity_under_CC('Control')
get_skel_intensity_under_CC('RTN')

exit()


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


