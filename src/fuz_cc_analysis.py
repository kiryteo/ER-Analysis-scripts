import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import imageio
import sknw
import pickle
import cv2
from skimage import measure
from skimage.morphology import closing
from junction_analysis_modules import JunctionAnalysis as JA

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'
junc_analysis = JA(confocal_data_path)

GROUP_DICT = {'ATL': 26, 'Climp': 31, 'Control': 31, 'RTN': 29}
GROUP_PREF = {'ATL':'A', 'Climp':'C', 'Control':'Ct', 'RTN':'R'}
GROUP_NAMES = {'ATL': 'Atlastin', 'Climp': 'Climp63', 'Control': 'Control', 'RTN': 'Reticulon'}

def get_cc_ids(labelled_img, region):
    cc_data = {cc_id: [] for cc_id in np.unique(labelled_img)}
    for loc in region:
        cc_id = labelled_img[loc[0], loc[1]]
        cc_data[cc_id].append(loc)
    return [cc_id for cc_id, data in cc_data.items() if cc_id > 0 and data]

def get_fuz_cc_outline(group, series_num, region):
    ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, series_num)
    label_ids, unassigned_cc_dict = junc_analysis.separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)
    iso, fuz, _ = junc_analysis.get_junction_areas(label_ids, unassigned_cc_dict)
    
    cc = get_cc_ids(labelled_img, iso if region == 'iso' else fuz)
    cc_coords = {each: np.where(labelled_img==each) for each in cc}
    
    spread_img = np.zeros((128, 128))
    for v in cc_coords.values():
        spread_img[v[0], v[1]] = 255.
    
    spread_img = closing(spread_img)
    return measure.label(spread_img, connectivity=2), ref_junctions

def get_intersection(a, b):
    set_a = set(map(tuple, np.stack(a, axis=1)))
    set_b = set(map(tuple, np.stack(b, axis=1)))
    return [np.array(x) for x in set_a.intersection(set_b)]

def get_single_frame_cc_intensity(group, series_num, lab_img):
    er = imageio.imread(f'{confocal_data_path}{group}/new_op_jul/std_egfp/{GROUP_PREF[group]}{series_num}_decon_t000_ch00_std.png')
    skel = imageio.imread(f'{confocal_data_path}{group}/new_op_jul/skel/{GROUP_PREF[group]}{series_num}/{GROUP_PREF[group]}{series_num}_decon_t000_ch00_skel.png')
    
    skel_coords = np.where(skel)
    data_skel, data_cc = [], []
    
    for cc_id in range(1, lab_img.max() + 1):
        cc_id_coords = np.where(lab_img == cc_id)
        skel_pixels = get_intersection(cc_id_coords, skel_coords)
        
        values_at_skel_coordinates = [er[coord[0], coord[1]] for coord in skel_pixels]
        values_at_cc_coordinates = er[cc_id_coords]
        
        data_skel.extend(values_at_skel_coordinates)
        data_cc.extend(values_at_cc_coordinates)
    
    return sum(data_skel) / sum(data_cc) if sum(data_cc) != 0 else None

def get_skel_intensity_under_CC(group):
    data_iso, data_fuz = [], []
    for series in range(1, GROUP_DICT[group] + 1):
        lab_img_iso, _ = get_fuz_cc_outline(group, series, 'iso')
        lab_img_fuz, _ = get_fuz_cc_outline(group, series, 'fuz')
        
        iso_val = get_single_frame_cc_intensity(group, series, lab_img_iso)
        fuz_val = get_single_frame_cc_intensity(group, series, lab_img_fuz)
        
        data_iso.append(iso_val)
        data_fuz.append(fuz_val)
    
    df = pd.DataFrame({
        'Intensity': data_iso + data_fuz,
        'Region': [f'{group}_iso'] * len(data_iso) + [f'{group}_fuz'] * len(data_fuz)
    })
    
    plot_boxplot(df, 'Region', 'Intensity', f'{GROUP_NAMES[group]}: Intensity under skeleton over CC area')

def plot_boxplot(df, x, y, title):
    plt.figure(figsize=(6, 8))
    ax = sns.boxplot(data=df, x=x, y=y, showfliers=False)
    plt.xlabel(x, fontsize=16)
    plt.ylabel(y, fontsize=16)
    plt.title(title, fontsize=18)
    plt.show()

def get_skel_per_fuz_cc(group):
    group_data = []
    for ser_num in range(1, GROUP_DICT[group] + 1):
        lab_img, _ = get_fuz_cc_outline(group, ser_num, 'fuz')
        
        for cc_id in range(1, lab_img.max() + 1):
            cc_id_coords = np.where(lab_img == cc_id)
            cc_data = []
            
            for frame in range(100):
                er = imageio.imread(f'{confocal_data_path}{group}/new_op_jul/std_egfp/{GROUP_PREF[group]}{ser_num}_decon_t0{frame:02d}_ch00_std.png')
                skel = imageio.imread(f'{confocal_data_path}{group}/new_op_jul/skel/{GROUP_PREF[group]}{ser_num}/{GROUP_PREF[group]}{ser_num}_decon_t0{frame:02d}_ch00_skel.png')
                
                skel_coords = np.where(skel)
                fuz_skel_pixels = get_intersection(cc_id_coords, skel_coords)
                
                if fuz_skel_pixels:
                    mean_val = np.mean([er[coord[0], coord[1]] for coord in fuz_skel_pixels]) / 255.
                    mean_val_over_area = mean_val / len(cc_id_coords[0]) if cc_id_coords[0].size else 0
                    cc_data.append(mean_val_over_area)
                else:
                    cc_data.append(0)
            
            group_data.append(cc_data)
    
    return group_data

def plot_fuz_skel_intensity_variation(group):
    data = get_skel_per_fuz_cc(group)
    df = pd.DataFrame(data, columns=[f't={i}' for i in range(len(data[0]))])
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, orient='v')
    plt.xlabel('Frames', fontsize=16)
    plt.ylabel('Mean Intensity', fontsize=16)
    plt.title(f'Fuzzy CC skeleton mean intensity per frame - {GROUP_NAMES[group]}', fontsize=18)
    plt.xticks(rotation=90)
    plt.show()

def create_combined_dataframe():
    group_data = {}
    for group in GROUP_DICT.keys():
        with open(f'{group.lower()}_fuz_cc_intensity.pkl', 'rb') as f:
            group_data[group] = pickle.load(f)
    
    df_list = []
    for group, data in group_data.items():
        df = pd.DataFrame(data).melt(var_name='Time', value_name='Values')
        df['Group'] = GROUP_NAMES[group]
        df_list.append(df)
    
    return pd.concat(df_list, ignore_index=True)

def plot_combined_lineplot():
    df = create_combined_dataframe()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Time', y='Values', hue='Group')
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Values', fontsize=16)
    plt.title('Fuzzy CC skeleton mean intensity per frame', fontsize=18)
    plt.show()

def replace_zeros(data):
    return [[np.mean(l) if val == 0 else val for val in l] for l in data]

def fuz_cc_degree_variation(group):
    group_data = []
    for ser_num in range(1, GROUP_DICT[group]+1):
        lab_img, _ = get_fuz_cc_outline(group, ser_num, 'fuz')
        
        for cc_id in range(1, lab_img.max()+1):
            cc_id_coords = np.where(lab_img==cc_id)
            cc_id_coords_set = set(zip(cc_id_coords[0], cc_id_coords[1]))
            cc_data = []
            
            for frame in range(100):
                skel = imageio.imread(f'{confocal_data_path}{group}/new_op_jul/skel/{GROUP_PREF[group]}{ser_num}/{GROUP_PREF[group]}{ser_num}_decon_t0{frame:02d}_ch00_skel.png')
                graph = sknw.build_sknw(skel, multi=True, iso=False)
                
                node_coords = np.array([node['o'] for node in graph.nodes.values()])
                node_coords_set = set(map(tuple, node_coords))
                
                fuz_cc_nodes = cc_id_coords_set.intersection(node_coords_set)
                
                if fuz_cc_nodes:
                    degrees = [graph.degree[list(node_coords_set).index(node)] for node in fuz_cc_nodes]
                    cc_data.append(np.mean(degrees))
                else:
                    cc_data.append(0)
            
            group_data.append(cc_data)
    
    return group_data

def degree_variation_runner():
    for group in GROUP_DICT.keys():
        data = fuz_cc_degree_variation(group)
        with open(f'{group.lower()}_fuz_cc_degree_variation.pkl', 'wb') as f:
            pickle.dump(data, f)

def get_fuz_cc_area(lab_img):
    return [np.sum(lab_img == cc_id) for cc_id in range(1, lab_img.max()+1)]

# This function was partially implemented in the optimized version, but here's a standalone version
def get_skel_intensity_over_cc_intensity(lab_img):
    data = []
    for cc_id in range(1, lab_img.max()+1):
        cc_id_coords = np.where(lab_img==cc_id)
        cc_data = []
        
        for i in range(100):
            er = imageio.imread(f'{confocal_data_path}ATL/new_op_jul/std_egfp/A1_decon_t0{i:02d}_ch00_std.png')
            skel = imageio.imread(f'{confocal_data_path}ATL/new_op_jul/skel/A1/A1_decon_t0{i:02d}_ch00_skel.png')
            
            skel_coords = np.where(skel)
            skel_pixels = get_intersection(cc_id_coords, skel_coords)
            
            values_at_skel_coordinates = [er[coord[0], coord[1]] for coord in skel_pixels]
            values_at_cc_coordinates = er[cc_id_coords]
            
            val = sum(values_at_skel_coordinates) / sum(values_at_cc_coordinates) if sum(values_at_cc_coordinates) else 0
            cc_data.append(val)
        
        data.append(cc_data)
    return data

def get_iso_fuz_cc_skel_data(group):
    group_data_iso, group_data_fuz = [], []
    for series_num in range(1, GROUP_DICT[group]+1):
        lab_img_iso, _ = get_fuz_cc_outline(group, series_num, 'iso')
        lab_img_fuz, _ = get_fuz_cc_outline(group, series_num, 'fuz')
        
        data_iso = get_skel_intensity_over_cc_intensity(lab_img_iso)
        data_fuz = get_skel_intensity_over_cc_intensity(lab_img_fuz)        
        
        group_data_iso.extend(data_iso)
        group_data_fuz.extend(data_fuz)
    
    return group_data_iso, group_data_fuz

def plot_iso_fuz_skel_intensity_variation():
    for group in GROUP_DICT.keys():
        data_iso, data_fuz = get_iso_fuz_cc_skel_data(group)
        data_iso = [item for sublist in data_iso for item in sublist]
        data_fuz = [item for sublist in data_fuz for item in sublist]
        
        df = pd.DataFrame({
            'Intensity': np.concatenate((data_iso, data_fuz)),
            'Region': [f'{group}_iso']*len(data_iso) + [f'{group}_fuz']*len(data_fuz)
        })
        
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='Region', y='Intensity', showfliers=False)
        plt.xlabel('Region', fontsize=16)
        plt.ylabel('Intensity', fontsize=16)
        plt.title(f'Skeleton intensity variation - {GROUP_NAMES[group]}', fontsize=18)
        plt.show()

if __name__ == "__main__":
    # Example usage of new functions
    # degree_variation_runner()
    # plot_iso_fuz_skel_intensity_variation()
    pass