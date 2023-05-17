import numpy as np
from skimage.measure import label, regionprops
import imageio
import sknw
import networkx as nx
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from numpy.polynomial.polynomial import polyfit
import pandas as pd
import scipy
import os
from scipy.stats import pearsonr, sem
import statannot
from statsmodels.stats.multicomp import MultiComparison
from scipy.stats import kruskal, mannwhitneyu
import pickle as pkl

from structure_extraction import node_connector, get_updated_degree_nodes
from junction_analysis_modules import JunctionAnalysis as JA
from junction_analysis import cc_area_measure, cc_signal, cc_signal_net_norm, get_per_CC_pixel_data, get_mean_std_per_CC_pixel_data, get_region_areas_per_group

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies'
# pickle_path_prefix = '/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/pickles/CC_junctions/'
pickle_path_prefix = '/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/'

GROUP_PREF = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}
VALID_GROUPS = ['ATL', 'Climp', 'RTN', 'Control']
VALID_CONNECTIONS = ['iso-iso', 'iso-fuz', 'fuz-fuz']
VALID_CHANNELS = ['egfp', 'mch']
VALID_MEASURES = ['tubules', 'tub-mean']

junc_analysis = JA(confocal_data_path)


def get_std_img(path):
    img = imageio.imread(path)
    return (img - img.min()) / (img.max() - img.min())


def filter_data(data):
    return [arr for arr in data if arr is not None and not np.all(arr == None)]


# def per_pixel_correlation(egfp_data, mch_data):
#     egfp = filter_data(egfp_data)
#     mch = filter_data(mch_data)

#     corr_vals = []

#     for s1, s2 in zip(egfp, mch):
#         for t1, t2 in zip(s1, s2):
#             transposed_egfp = list(map(list, zip(*t1)))
#             transposed_mch = list(map(list, zip(*t2)))
#             corr_vals.extend(np.corrcoef(e1, e2)[0, 1] for e1, e2 in zip(transposed_egfp, transposed_mch))

#     return corr_vals




def get_mean_group_data(data, measure):
    func = np.mean if measure == 'mean' else np.std
    group_data = []
    for series in data:
        group_data.extend(func(cc) for cc in series)
    return group_data


def get_CC_mean_variation(channel, region, measure):
    # atl = cc_signal('ATL', channel, region)
    # pkl.dump(atl, open(f'ATL_{channel}_{region}_CC_mean.pkl', 'wb'))
    # climp = cc_signal('Climp', channel, region)
    # pkl.dump(climp, open(f'Climp_{channel}_{region}_CC_mean.pkl', 'wb'))
    # rtn = cc_signal('RTN', channel, region)
    # pkl.dump(rtn, open(f'RTN_{channel}_{region}_CC_mean.pkl', 'wb'))
    # control = cc_signal('Control', channel, region)
    # pkl.dump(control, open(f'Control_{channel}_{region}_CC_mean.pkl', 'wb'))

    atl = pkl.load(open(f'ATL_{channel}_{region}_CC_mean.pkl', 'rb'))
    climp = pkl.load(open(f'Climp_{channel}_{region}_CC_mean.pkl', 'rb'))
    rtn = pkl.load(open(f'RTN_{channel}_{region}_CC_mean.pkl', 'rb'))
    
    atl = get_mean_group_data(atl, measure)
    climp = get_mean_group_data(climp, measure)
    rtn = get_mean_group_data(rtn, measure)

    df = pd.DataFrame()

    if channel == 'egfp':
        control = pkl.load(open(f'Control_{channel}_{region}_CC_mean.pkl', 'rb'))
        control = get_mean_group_data(control, measure)
        df['CC_mean'] = pd.Series(np.concatenate((control, rtn, climp, atl)))
        df['Group'] = pd.Series(np.concatenate((['Control'] * len(control), ['RTN'] * len(rtn), ['Climp'] * len(climp), ['ATL'] * len(atl) )))
    else:
        df['CC_mean'] = pd.Series(np.concatenate((rtn, climp, atl)))
        df['Group'] = pd.Series(np.concatenate((['RTN'] * len(rtn), ['Climp'] * len(climp), ['ATL'] * len(atl))))

    # ax = sns.boxplot(data=df, x='Group', y='CC_mean', showfliers=False, whis=0.5, linewidth=2)
    ax = sns.boxplot(data=df, x='Group', y='CC_mean', showfliers=False, linewidth=2)
    # plt.show()
    # plt.yscale('log')
    ax.set_ylim(0, 0.85) # for egfp
    #ax.set_ylim(0, 0.5) # for mch
    # ax.set_ylim(0, ymax+0.01)
    yt = ax.get_yticks()
    yt = [f'{y:.2f}' for y in yt]
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    ax.set_yticklabels(yt, fontsize=20)

    # box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('ATL', 'Control'), ('Climp', 'RTN'), ('Climp', 'Control'), ('RTN', 'Control')]
    # box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('Climp', 'RTN')]

    # statannot.add_stat_annotation(ax, x='Group', y='CC_mean', data=df, box_pairs=box_pairs,
    #                               test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize=20)

    region_name = 'Isolated' if region == 'iso' else 'Fuzzy'
    ch_name = 'ERmoxGFP' if channel == 'egfp' else 'mCherry'
    measure_name = 'Mean' if measure == 'mean' else 'Standard Deviation'

    # plt.suptitle(f'{region_name} CC, per junction correlation across conditions', fontsize=20)
    plt.title(f'{measure_name} over 100 frames for Junction CC mean \n in {region_name} CC ({ch_name})', fontsize=24)
    plt.grid(True)
    # plt.tight_layout()
    # plt.subplots_adjust(hspace = 1, wspace = 0)
    plt.xlabel('Group', fontsize=24)
    plt.ylabel(f'{measure_name} over sequence per CC mean', fontsize=24)
    plt.gcf().set_size_inches(12, 12)
    plt.savefig(f'Seq_{measure_name}_CC_mean_{region_name}_{channel}', bbox_inches='tight', pad_inches=0.4)
    plt.close()
    # plt.show()



# get_CC_mean_variation('egfp', 'iso', 'mean')
# get_CC_mean_variation('egfp', 'fuz', 'mean')
# get_CC_mean_variation('mch', 'iso', 'mean')
# get_CC_mean_variation('mch', 'fuz', 'mean')
# exit()


# def get_CC_mean_correlation(channel, region, measure):
def get_CC_mean_correlation():
    atl_egfp_iso = pkl.load(open('ATL_egfp_iso_CC_mean.pkl', 'rb'))
    atl_egfp_fuz = pkl.load(open('ATL_egfp_fuz_CC_mean.pkl', 'rb'))
    atl_mch_iso = pkl.load(open('ATL_mch_iso_CC_mean.pkl', 'rb'))
    atl_mch_fuz = pkl.load(open('ATL_mch_fuz_CC_mean.pkl', 'rb'))

    climp_egfp_iso = pkl.load(open('Climp_egfp_iso_CC_mean.pkl', 'rb'))
    climp_egfp_fuz = pkl.load(open('Climp_egfp_fuz_CC_mean.pkl', 'rb'))
    climp_mch_iso = pkl.load(open('Climp_mch_iso_CC_mean.pkl', 'rb'))
    climp_mch_fuz = pkl.load(open('Climp_mch_fuz_CC_mean.pkl', 'rb'))

    rtn_egfp_iso = pkl.load(open('RTN_egfp_iso_CC_mean.pkl', 'rb'))
    rtn_egfp_fuz = pkl.load(open('RTN_egfp_fuz_CC_mean.pkl', 'rb'))
    rtn_mch_iso = pkl.load(open('RTN_mch_iso_CC_mean.pkl', 'rb'))
    rtn_mch_fuz = pkl.load(open('RTN_mch_fuz_CC_mean.pkl', 'rb'))




get_CC_mean_correlation()
exit()



def get_er_area(group, num_series):
    signal_len = []
    for i in range(1, num_series+1):
        data = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/er_mean/{group.lower()}{i}_er_mean.png')
        signal = np.where(data > 0)
        signal_len.append(len(signal[0]))
    with open(f'{group}_er_area.pkl', 'wb') as f:
        pkl.dump(signal_len, f)

# get_er_area('ATL', 26)
# get_er_area('Climp', 31)
# get_er_area('RTN', 29)
# get_er_area('Control', 31)

# exit()


def plot_num_junctions():
    """
    Plot number of junctions per group
    """
    atl = pkl.load(open('ATL_egfp_iso_CC_mean.pkl', 'rb'))
    climp = pkl.load(open('Climp_egfp_iso_CC_mean.pkl', 'rb'))
    rtn = pkl.load(open('RTN_egfp_iso_CC_mean.pkl', 'rb'))
    control = pkl.load(open('Control_egfp_iso_CC_mean.pkl', 'rb'))

    atl_area = pkl.load(open('ATL_er_area.pkl', 'rb'))
    climp_area = pkl.load(open('Climp_er_area.pkl', 'rb'))
    rtn_area = pkl.load(open('RTN_er_area.pkl', 'rb'))
    control_area = pkl.load(open('Control_er_area.pkl', 'rb'))

    atl_num = [len(series) for series in atl]
    climp_num = [len(series) for series in climp]
    rtn_num = [len(series) for series in rtn]
    control_num = [len(series) for series in control]

    atl_num = [num / area for num, area in zip(atl_num, atl_area)]
    climp_num = [num / area for num, area in zip(climp_num, climp_area)]
    rtn_num = [num / area for num, area in zip(rtn_num, rtn_area)]
    control_num = [num / area for num, area in zip(control_num, control_area)]

    df = pd.DataFrame()
    df['Num_junctions'] = pd.Series(np.concatenate((control_num, rtn_num, climp_num, atl_num)))
    df['Group'] = pd.Series(np.concatenate((['Control'] * len(control_num), ['Reticulon'] * len(rtn_num), ['Climp'] * len(climp_num), ['Atlastin'] * len(atl_num))))

    ax = sns.boxplot(data=df, x='Group', y='Num_junctions', showfliers=False, linewidth=2)
    # ax = sns.boxplot(data=df, x='Group', y='Num_junctions', showfliers=False, whis=0.5, linewidth=2)

    # ax = sns.barplot(data=df, x='Group', y='Num_junctions')#, ci='sd', capsize=0.2, linewidth=2, errwidth=2)

    # yt = ax.get_yticks()
    # yt = [f'{y:.2f}' for y in yt]
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    # ax.set_yticklabels(yt, fontsize=18)

    yt = ax.get_yticks()
    yt = [f'{y:.2f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)

    # sns.pointplot(x='Group', y='Num_junctions', data=df.groupby('Group', as_index=False).mean(), ax=ax)

    # ctrl_sem = np.std(control_num) / np.sqrt(len(control_num))
    # rtn_sem = np.std(rtn_num) / np.sqrt(len(rtn_num))
    # climp_sem = np.std(climp_num) / np.sqrt(len(climp_num))
    # atl_sem = np.std(atl_num) / np.sqrt(len(atl_num))

    # plt.errorbar(x=[0, 1, 2, 3], y=[np.mean(control_num), np.mean(rtn_num), np.mean(climp_num), np.mean(atl_num)], yerr=[np.std(control_num), np.std(rtn_num), np.std(climp_num), np.std(atl_num)], fmt='o', color='black', capsize=5, markersize=8)

    # plt.errorbar(x=[0, 1, 2, 3], y=[np.mean(control_num), np.mean(rtn_num), np.mean(climp_num), np.mean(atl_num)], yerr=[ctrl_sem, rtn_sem, climp_sem, atl_sem], fmt='o', color='black', capsize=5, markersize=8)

    # plt.rcParams['figure.figsize'] = (5, 20)
    plt.title('Number of isolated junctions per sequence (normalized by ER area)', fontsize=24)
    plt.grid(True)
    plt.xlabel('Group', fontsize=24)
    plt.ylabel('Number of junctions (normalized)', fontsize=24)
    # plt.show()
    plt.gcf().set_size_inches(14, 8)
    plt.savefig('num_juncs_iso_norm.png', bbox_inches='tight', pad_inches=0.6)
    plt.close()

plot_num_junctions()
exit()

def get_iso_fuz_ratio():
    atl_iso = pkl.load(open('ATL_egfp_iso_CC_mean.pkl', 'rb'))

    # atl_iso_num = [len(series) for series in atl_iso]
    # print(atl_iso_num)
    # exit()

    climp_iso = pkl.load(open('Climp_egfp_iso_CC_mean.pkl', 'rb'))
    rtn_iso = pkl.load(open('RTN_egfp_iso_CC_mean.pkl', 'rb'))
    control_iso = pkl.load(open('Control_egfp_iso_CC_mean.pkl', 'rb'))

    atl_fuz = pkl.load(open('ATL_egfp_fuz_CC_mean.pkl', 'rb'))
    climp_fuz = pkl.load(open('Climp_egfp_fuz_CC_mean.pkl', 'rb'))
    rtn_fuz = pkl.load(open('RTN_egfp_fuz_CC_mean.pkl', 'rb'))
    control_fuz = pkl.load(open('Control_egfp_fuz_CC_mean.pkl', 'rb'))

    atl_iso_num = [len(series) for series in atl_iso]
    climp_iso_num = [len(series) for series in climp_iso]
    rtn_iso_num = [len(series) for series in rtn_iso]
    control_iso_num = [len(series) for series in control_iso]

    atl_fuz_num = [len(series) for series in atl_fuz]
    climp_fuz_num = [len(series) for series in climp_fuz]
    rtn_fuz_num = [len(series) for series in rtn_fuz]
    control_fuz_num = [len(series) for series in control_fuz]

    # atl_ratio = [iso / fuz for iso, fuz in zip(atl_iso_num, atl_fuz_num)]
    # climp_ratio = [iso / fuz for iso, fuz in zip(climp_iso_num, climp_fuz_num)]
    # rtn_ratio = [iso / fuz for iso, fuz in zip(rtn_iso_num, rtn_fuz_num)]
    
    # control_ratio = [iso / fuz for iso, fuz in zip(control_iso_num, control_fuz_num) if fuz != 0]

    atl_ratio = [fuz/ iso for fuz, iso in zip(atl_fuz_num, atl_iso_num)]
    climp_ratio = [fuz / iso for fuz, iso in zip(climp_fuz_num, climp_iso_num)]
    rtn_ratio = [fuz / iso for fuz, iso in zip(rtn_fuz_num, rtn_iso_num)]
    control_ratio = [fuz / iso for fuz, iso in zip(control_fuz_num, control_iso_num) if iso != 0]

    # print(atl_ratio)
    # print(climp_ratio)
    # print(rtn_ratio)
    # print(control_ratio)

    # print(np.median(atl_ratio)) # 6
    # print(np.median(climp_ratio)) # 13
    # print(np.median(rtn_ratio)) # 2
    # print(np.median(control_ratio)) # 13

    # exit()

    df = pd.DataFrame()
    df['Group'] = pd.Series(np.concatenate((['Control'] * len(control_ratio), ['Reticulon'] * len(rtn_ratio), ['Climp'] * len(climp_ratio), ['Atlastin'] * len(atl_ratio))))

    df['Ratio'] = pd.Series(np.concatenate((control_ratio, rtn_ratio, climp_ratio, atl_ratio)))

    ax = sns.boxplot(data=df, x='Group', y='Ratio', showfliers=False, whis=0.5, linewidth=2)
    # ax = sns.swarmplot(data=df, x='Group', y='Ratio', color='black', size=8)

    yt = ax.get_yticks()
    yt = [f'{y:.2f}' for y in yt]
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    ax.set_yticklabels(yt, fontsize=20)

    # plt.rcParams['figure.figsize'] = (5, 20)
    # plt.title(f'Isolated to Overlapping junctions ratio', fontsize=24)
    plt.title(f'Overlapping to Isolated junctions ratio', fontsize=24)
    plt.grid(True)
    plt.xlabel('Group', fontsize=24)
    plt.ylabel('Ratio', fontsize=24)

    plt.gcf().set_size_inches(12, 12)
    # plt.savefig('iso_overlap_juncs_ratio.png', bbox_inches='tight', pad_inches=0.4)
    plt.savefig('overlap_iso_juncs_ratio.png', bbox_inches='tight', pad_inches=0.4)
    plt.close()
    # plt.show()

# get_iso_fuz_ratio()
# exit()

def get_iso_fuz_area_ratio():
    # atl_iso = get_region_areas_per_group('ATL', 26, 'iso')
    # with open('ATL_iso_area.pkl', 'wb') as f:
    #     pkl.dump(atl_iso, f)
    
    # climp_iso = get_region_areas_per_group('Climp', 31, 'iso')
    # with open('Climp_iso_area.pkl', 'wb') as f:
    #     pkl.dump(climp_iso, f)

    # rtn_iso = get_region_areas_per_group('RTN', 29, 'iso')
    # with open('RTN_iso_area.pkl', 'wb') as f:
    #     pkl.dump(rtn_iso, f)

    # control_iso = get_region_areas_per_group('Control', 31, 'iso')
    # with open('Control_iso_area.pkl', 'wb') as f:
    #     pkl.dump(control_iso, f)

    # atl_fuz = get_region_areas_per_group('ATL', 26, 'fuz')
    # with open('ATL_fuz_area.pkl', 'wb') as f:
    #     pkl.dump(atl_fuz, f)

    # climp_fuz = get_region_areas_per_group('Climp', 31, 'fuz')
    # with open('Climp_fuz_area.pkl', 'wb') as f:
    #     pkl.dump(climp_fuz, f)

    # rtn_fuz = get_region_areas_per_group('RTN', 29, 'fuz')
    # with open('RTN_fuz_area.pkl', 'wb') as f:
    #     pkl.dump(rtn_fuz, f)

    # control_fuz = get_region_areas_per_group('Control', 31, 'fuz')
    # with open('Control_fuz_area.pkl', 'wb') as f:
    #     pkl.dump(control_fuz, f)

    atl_iso = pkl.load(open('ATL_iso_area.pkl', 'rb'))
    climp_iso = pkl.load(open('Climp_iso_area.pkl', 'rb'))
    rtn_iso = pkl.load(open('RTN_iso_area.pkl', 'rb'))
    control_iso = pkl.load(open('Control_iso_area.pkl', 'rb'))

    atl_fuz = pkl.load(open('ATL_fuz_area.pkl', 'rb'))
    climp_fuz = pkl.load(open('Climp_fuz_area.pkl', 'rb'))
    rtn_fuz = pkl.load(open('RTN_fuz_area.pkl', 'rb'))
    control_fuz = pkl.load(open('Control_fuz_area.pkl', 'rb'))

    # atl_ratio = [sum(iso) / sum(fuz) for iso, fuz in zip(atl_iso, atl_fuz)]
    # climp_ratio = [sum(iso) / sum(fuz) for iso, fuz in zip(climp_iso, climp_fuz)]
    # rtn_ratio = [sum(iso) / sum(fuz) for iso, fuz in zip(rtn_iso, rtn_fuz)]
    # control_ratio = [sum(iso) / sum(fuz) for iso, fuz in zip(control_iso, control_fuz) if sum(fuz) != 0]

    atl_ratio = [sum(fuz) / sum(iso) for fuz, iso in zip(atl_fuz, atl_iso)]
    climp_ratio = [sum(fuz) / sum(iso) for fuz, iso in zip(climp_fuz, climp_iso)]
    rtn_ratio = [sum(fuz) / sum(iso) for fuz, iso in zip(rtn_fuz, rtn_iso)]
    control_ratio = [sum(fuz) / sum(iso) for fuz, iso in zip(control_fuz, control_iso) if sum(iso) != 0]

    # print(atl_ratio)
    # print(np.median(atl_ratio)) # 2
    # print(climp_ratio)
    # print(np.median(climp_ratio)) # 15
    # print(rtn_ratio)
    # print(np.median(rtn_ratio)) # 9
    # print(control_ratio)
    # print(np.median(control_ratio)) # 7
    # exit()


    df = pd.DataFrame()
    df['Group'] = pd.Series(np.concatenate((['Control'] * len(control_ratio), ['Reticulon'] * len(rtn_ratio), ['Climp'] * len(climp_ratio), ['Atlastin'] * len(atl_ratio))))

    df['Ratio'] = pd.Series(np.concatenate((control_ratio, rtn_ratio, climp_ratio, atl_ratio)))

    ax = sns.boxplot(data=df, x='Group', y='Ratio', showfliers=False, whis=0.5, linewidth=2)

    yt = ax.get_yticks()
    yt = [f'{y:.2f}' for y in yt]
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    ax.set_yticklabels(yt, fontsize=18)

    # plt.rcParams['figure.figsize'] = (5, 20)
    # plt.title(f'Isolated to Overlapping CC area ratio', fontsize=24)
    plt.title('Overlapping to Isolated CC area ratio', fontsize=24)
    plt.grid(True)
    plt.xlabel('Group', fontsize=24)
    plt.ylabel('Ratio', fontsize=24)

    plt.gcf().set_size_inches(12, 12)
    plt.savefig('overlap_iso_CC_area_ratio.png', bbox_inches='tight', pad_inches=0.4)
    plt.close()
    # plt.show()

# get_iso_fuz_area_ratio()
# exit()


def get_variation_from_pickles():
    # atl_egfp = pkl.load(open('ATL_egfp_fuz_CC_mean.pkl', 'rb'))
    # climp_egfp = pkl.load(open('Climp_egfp_fuz_CC_mean.pkl', 'rb'))
    # rtn_egfp = pkl.load(open('RTN_egfp_fuz_CC_mean.pkl', 'rb'))
    # control_egfp = pkl.load(open('Control_egfp_fuz_CC_mean.pkl', 'rb'))

    atl_mch = pkl.load(open('ATL_mch_iso_CC_mean.pkl', 'rb'))
    climp_mch = pkl.load(open('Climp_mch_iso_CC_mean.pkl', 'rb'))
    rtn_mch = pkl.load(open('RTN_mch_iso_CC_mean.pkl', 'rb'))
    # control_mch = pkl.load(open('Control_mch_fuz_CC_mean.pkl', 'rb'))



    atl = get_mean_group_data(atl_mch, 'std')
    climp = get_mean_group_data(climp_mch, 'std')
    rtn = get_mean_group_data(rtn_mch, 'std')
    # control = get_mean_group_data(control_egfp, 'std')

    df = pd.DataFrame()
    df['CC_mean'] = pd.Series(np.concatenate((atl, climp, rtn)))
    df['Group'] = pd.Series(np.concatenate((['ATL'] * len(atl), ['Climp'] * len(climp), ['RTN'] * len(rtn))))#, ['Control'] * len(control))))

    ax = sns.boxenplot(data=df, x='Group', y='CC_mean')

    box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('Climp', 'RTN')]

    statannot.add_stat_annotation(ax, x='Group', y='CC_mean', data=df, box_pairs=box_pairs,
                                    test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize=20)
    
    plt.title('Standard deviation over 100 frames for Junction CC mean in Isolated CC (mCherry)', fontsize=24)
    plt.grid(True)
    plt.xlabel('Group', fontsize=24)
    plt.ylabel('Standard deviation over sequence per CC mean', fontsize=24)
    plt.show()

# get_variation_from_pickles()
# exit()

def plot_num_junctions_per_group():
    # atl = get_num_junctions_per_group('ATL', 26)
    # climp = get_num_junctions_per_group('Climp', 31)
    # rtn = get_num_junctions_per_group('RTN', 29)
    # control = get_num_junctions_per_group('Control', 31)


    df = pd.DataFrame()
    df['Num_junctions'] = pd.Series(np.concatenate((atl, climp, rtn, control)))
    df['Group'] = pd.Series(np.concatenate((['ATL'] * len(atl), ['Climp'] * len(climp), ['RTN'] * len(rtn), ['Control'] * len(control))))

    ax = sns.boxplot(data=df, y='Group', x='Num_junctions')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    xt = ax.get_xticks()
    xt = [f'{x:.2f}' for x in xt]
    ax.set_xticklabels(xt, fontsize=18)

    # box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('ATL', 'Control'), ('Climp', 'RTN'), ('Climp', 'Control'), ('RTN', 'Control')]

    # statannot.add_stat_annotation(ax, x='Group', y='Num_junctions', data=df, box_pairs=box_pairs,
                                  
    #                                 test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize=20)
    
    plt.title('Number of isolated reference junctions', fontsize=24)
    plt.grid(True)
    plt.xlabel('Group', fontsize=24)
    plt.ylabel('Number of junctions', fontsize=24)
    plt.show()
    
# plot_num_junctions_per_group()    
# exit()


def per_CC_pixel_correlation(egfp_data, mch_data):
    corr_vals = []

    for group_num in range(len(egfp_data)):
        for junc_e, junc_m in zip(egfp_data[group_num], mch_data[group_num]):
            corr_vals.append(np.corrcoef(junc_e, junc_m)[0, 1])
        
    return corr_vals


def per_CC_pixel_variation(region):
    atl_egfp = pkl.load(open(f'{pickle_path_prefix}ATL_egfp_{region}_data.pkl', 'rb'))
    climp_egfp = pkl.load(open(f'{pickle_path_prefix}Climp_egfp_{region}_data.pkl', 'rb'))
    rtn_egfp = pkl.load(open(f'{pickle_path_prefix}RTN_egfp_{region}_data.pkl', 'rb'))

    atl_mch = pkl.load(open(f'{pickle_path_prefix}ATL_mch_{region}_data.pkl', 'rb'))
    climp_mch = pkl.load(open(f'{pickle_path_prefix}Climp_mch_{region}_data.pkl', 'rb'))
    rtn_mch = pkl.load(open(f'{pickle_path_prefix}RTN_mch_{region}_data.pkl', 'rb'))

    atl_corr = per_CC_pixel_correlation(atl_egfp, atl_mch)
    climp_corr = per_CC_pixel_correlation(climp_egfp, climp_mch)
    rtn_corr = per_CC_pixel_correlation(rtn_egfp, rtn_mch)

    df = pd.DataFrame()
    df['data_tubule_mean'] = pd.Series(np.concatenate((atl_corr, climp_corr, rtn_corr)))
    df['Group'] = pd.Series(np.concatenate((
        ['ATL'] * len(atl_corr), ['Climp'] * len(climp_corr), ['RTN'] * len(rtn_corr))))
    

    ax = sns.boxenplot(data=df, x='Group', y='data_tubule_mean')
    # ax = sns.boxenplot(data=df, x='Replicate', y='data_tubule_mean', hue='Group', dodge=True)  # , yscale='log')
    # plt.show()
    # plt.yscale('log')
    ax.set_ylim(-0.6, 1)
    yt = ax.get_yticks()
    yt = [f'{y:.2f}' for y in yt]
    # ax.set_ylim([0.0, 0.5])
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    ax.set_yticklabels(yt, fontsize=18)


    box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('Climp', 'RTN')]

    statannot.add_stat_annotation(ax, x='Group', y='data_tubule_mean', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize=20)

    region_name = 'Isolated' if region == 'iso' else 'Fuzzy'
    # ch_name = 'ERmoxGFP' if channel == 'egfp' else 'mCherry'

    # plt.suptitle(f'{region_name} CC, per junction correlation across conditions', fontsize=20)
    plt.title(f'Correlation between channels over 100 frames for each junction within {region_name} CC', fontsize=24)
    plt.grid(True)
    plt.xlabel('Group', fontsize=24)
    plt.ylabel('Cross-correlation value', fontsize=24)
    plt.show()


# per_CC_pixel_variation('iso')
# per_CC_pixel_variation('fuz')
# exit()

# fpath = '/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/pickles/CC_junctions/ATL_egfp_iso_data.pkl'
# atl_data = pkl.load(open(fpath, 'rb'))
# print(len(atl_data))
# print(len(atl_data[0]))
# print(atl_data[0][0])
# print(np.array(atl_data[0][0], dtype='f')/255.)
# exit()



def get_CC_variation(channel, region, measure):
    atl_data = pkl.load(open(f'{pickle_path_prefix}ATL_{channel}_{region}_data.pkl', 'rb'))
    climp_data = pkl.load(open(f'{pickle_path_prefix}Climp_{channel}_{region}_data.pkl', 'rb'))
    rtn_data = pkl.load(open(f'{pickle_path_prefix}RTN_{channel}_{region}_data.pkl', 'rb'))
    if channel == 'egfp':
        control_data = pkl.load(open(f'{pickle_path_prefix}Control_{channel}_{region}_data.pkl', 'rb'))
        control_measure_data = get_mean_std_per_CC_pixel_data(control_data, measure)

    atl_measure_data = get_mean_std_per_CC_pixel_data(atl_data, measure)
    climp_measure_data = get_mean_std_per_CC_pixel_data(climp_data, measure)
    rtn_measure_data = get_mean_std_per_CC_pixel_data(rtn_data, measure)

    df = pd.DataFrame()
    if channel == 'egfp':
        df['data_tubule_mean'] = pd.Series(np.concatenate((atl_measure_data, climp_measure_data, rtn_measure_data, control_measure_data)))
        df['Group'] = pd.Series(np.concatenate((
            ['ATL'] * len(atl_measure_data), ['Climp'] * len(climp_measure_data), ['RTN'] * len(rtn_measure_data), ['Control'] * len(control_measure_data))))
    else:
        df['data_tubule_mean'] = pd.Series(np.concatenate((atl_measure_data, climp_measure_data, rtn_measure_data)))
        df['Group'] = pd.Series(np.concatenate((
            ['ATL'] * len(atl_measure_data), ['Climp'] * len(climp_measure_data), ['RTN'] * len(rtn_measure_data))))
    


    ax = sns.boxenplot(data=df, x='Group', y='data_tubule_mean')
    # ax = sns.boxenplot(data=df, x='Replicate', y='data_tubule_mean', hue='Group', dodge=True)  # , yscale='log')
    yt = ax.get_yticks()
    yt = [f'{y:.2f}' for y in yt]
    # ax.set_ylim([0.0, 0.5])
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    ax.set_yticklabels(yt, fontsize=18)
    # sns.set(font_scale=2)
    # plt.show()
    # plt.yscale('log')


    if channel == 'egfp':
        box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('ATL', 'Control'), ('Climp', 'RTN'), ('Climp', 'Control'), ('RTN', 'Control')]
    else:
        box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('Climp', 'RTN')]

    statannot.add_stat_annotation(ax, x='Group', y='data_tubule_mean', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize=20)

    region_name = 'Isolated' if region == 'iso' else 'Fuzzy'
    measure_name = 'Mean' if measure == 'mean' else 'Standard Deviation'
    ch_name = 'ERmoxGFP' if channel == 'egfp' else 'mCherry'

    # plt.suptitle(f'{region_name} CC, per junction {measure_name} across conditions ({ch_name})', fontsize=20)
    # plt.title(f'{measure_name} over 100 frames for each junction location within a CC', fontsize=18)
    plt.title(f'{measure_name} over 100 frames for each junction within a CC ({region_name} CCs, {ch_name})', fontsize=24)
    plt.grid(True)
    plt.xlabel('Group', fontsize=24)
    plt.ylabel(f'Junction Intensity {measure_name}', fontsize=24)
    plt.show()


# get_CC_variation('egfp', 'iso', 'mean')
# get_CC_variation('egfp', 'fuz', 'mean')
# get_CC_variation('mch', 'iso', 'mean')
# get_CC_variation('mch', 'fuz', 'mean')
# get_CC_variation('egfp', 'iso', 'std')
# get_CC_variation('egfp', 'fuz', 'std')
# get_CC_variation('mch', 'iso', 'std')
# get_CC_variation('mch', 'fuz', 'std')
# exit()



def plot_region_areas(channel):
    atl = get_region_areas_per_group('ATL', 26, 'fuz')
    a1, a2, a3 = atl[:10], atl[10:20], atl[20:]
    climp = get_region_areas_per_group('Climp', 31, 'fuz')
    c1, c2, c3 = climp[:10], climp[10:20], climp[20:]
    rtn = get_region_areas_per_group('RTN', 29, 'fuz')
    r1, r2, r3 = rtn[:10], rtn[10:20], rtn[20:]

    df = pd.DataFrame()

    # df['data_tubule_mean'] = pd.Series(np.concatenate((atl, climp, rtn)))

    if channel == 'egfp':
        ctrl = get_region_areas_per_group('Control', 31, 'fuz')
        ct1, ct2, ct3 = ctrl[:10], ctrl[10:20], ctrl[20:]
        df['data_tubule_mean'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3, ct1, ct2, ct3)))
        df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
                                                    ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
                                                    ['R3'] * len(r3), ['R1'] * len(ct1), ['R2'] * len(ct2),
                                                    ['R3'] * len(ct3))))
        df['Group'] = pd.Series(np.concatenate((
            ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1), ['Climp'] * len(c2),
            ['Climp'] * len(c3), ['RTN'] * len(r1), ['RTN'] * len(r2), ['RTN'] * len(r3), ['Control'] * len(ct1), ['Control'] * len(ct2), ['Control'] * len(ct3))))
    else:
        df['data_tubule_mean'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3)))
        df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
                                                    ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
                                                    ['R3'] * len(r3))))
        df['Group'] = pd.Series(np.concatenate((
            ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1), ['Climp'] * len(c2),
            ['Climp'] * len(c3), ['RTN'] * len(r1), ['RTN'] * len(r2), ['RTN'] * len(r3))))


    # ax = sns.boxenplot(data=df, x='Group', y='data_tubule_mean')
    ax = sns.boxenplot(data=df, x='Replicate', y='data_tubule_mean', hue='Group', dodge=True)  # , yscale='log')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    # egfp
    replicates = ['R1', 'R2', 'R3']

    if channel == 'egfp':
        groups = ['ATL', 'Climp', 'RTN', 'Control']
    else:
        groups = ['ATL', 'Climp', 'RTN']
    box_pairs = [((x, y), (x, z)) for i, x in enumerate(replicates) for j, y in enumerate(groups) for z in groups[j + 1 :]]

    statannot.add_stat_annotation(ax, x='Replicate', y='data_tubule_mean', hue='Group', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    # statannot.add_stat_annotation(ax, x='Group', y='data_tubule_mean', data=df, box_pairs=box_pairs,
    #                               test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    # plt.title('Cross-correlation between ERmoxGFP and mCherry over sequence for tubule intensity mean in all tubules',
    #           fontsize=18)
    plt.title('CC area for Fuzzy CCs across conditions', fontsize=20)
    plt.grid(True)
    plt.xlabel('Replicate', fontsize=20)
    plt.ylabel('Fuzzy CCs area', fontsize=18)

    plt.show()


# atl_e = ref_junc_data_per_group('ATL', 10, 'egfp')
# atl_mean = ref_junc_variation(atl_e, 'mean')
# climp_e = ref_junc_data_per_group('Climp', 10, 'egfp')
# climp_mean = ref_junc_variation(climp_e, 'mean')
# rtn_e = ref_junc_data_per_group('RTN', 10, 'egfp')
# rtn_mean = ref_junc_variation(rtn_e, 'mean')
# ctrl_e = ref_junc_data_per_group('Control', 10, 'egfp')
# ctrl_mean = ref_junc_variation(ctrl_e, 'mean')
#
# # print(atl_mean)
# sns.distplot(atl_mean, hist=False, label='atl')
# sns.distplot(climp_mean, hist=False, label='climp')
# sns.distplot(rtn_mean, hist=False, label='rtn')
# sns.distplot(ctrl_mean, hist=False, label='ctrl')
# plt.legend()
# plt.show()
#
# exit()


# plot_ref_junc_variation()


# exit()





    # dict to store the coords for each cc id
    # iso_cc_coords = {}
    # for each in iso_cc:
    #     iso_cc_coords[each] = np.where(labelled_img==each)


##################################################################




def junction_cc_mean_distplot(channel):
    atl = calc_egfp_deposit('ATL', channel, 26, 'iso')
    climp = calc_egfp_deposit('Climp', channel, 31, 'fuz')
    ctrl = calc_egfp_deposit('Control', channel, 31, 'fuz')
    rtn = calc_egfp_deposit('RTN', channel, 29, 'fuz')

    sns.distplot(atl, label='ATL', hist=False)
    sns.distplot(climp, label='Climp', hist=False)
    sns.distplot(ctrl, label='Control', hist=False)
    sns.distplot(rtn, label='RTN', hist=False)

    plt.title(f'{channel} Mean Intensity per fuzzy area junction CC patch', fontsize=18)
    plt.xlabel(f'{channel} intensity mean values', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    # plt.legend(fontsize=14) # for distplot only

    plt.savefig(f'{channel}_fuzzy_CCs_mean_intensity_per_patch', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    # plt.show()


def junction_cc_mean_boxplot(channel, region):
    region_name = 'isolated' if region == 'iso' else 'fuzzy'
    atl = calc_egfp_deposit('ATL', channel, 26, region)

    climp = calc_egfp_deposit('Climp', channel, 31, region)
    ctrl = calc_egfp_deposit('Control', channel, 31, region)
    rtn = calc_egfp_deposit('RTN', channel, 29, region)

    df = pd.DataFrame()
    df['Values'] = pd.Series(np.concatenate((atl, climp, ctrl, rtn)))
    df['ids'] = pd.Series(np.concatenate((np.arange(1, len(atl) + 1), np.arange(1, len(climp) + 1),
                                          np.arange(1, len(ctrl) + 1), np.arange(1, len(rtn) + 1))))
    df['Group'] = pd.Series(
        np.concatenate((['ATL'] * len(atl), ['Climp'] * len(climp), ['Control'] * len(ctrl), ['RTN'] * len(rtn))))
    #
    ax = sns.violinplot(data=df, y='Group', x='Values')
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=13)
    # sns.scatterplot(data=df, x='ids', y='Values', hue='Group', style='Group')

    # plt.suptitle(f'{channel} deposit in isolated region junction CCs across conditions', fontsize=22)
    # plt.title('Variance of junction CC mean per patch over 100 frames', fontsize=14)
    plt.title(f'{channel} Mean Intensity per {region_name} area junction CC patch', fontsize=18)
    plt.xlabel(f'{channel} intensity mean values', fontsize=14)
    plt.ylabel('ER-Shaping proteins', fontsize=14)
    # plt.legend(fontsize=14) # for distplot only

    plt.savefig(f'{channel}_{region_name}_CCs_mean_intensity_per_patch_violinplot', bbox_inches='tight', pad_inches=0.2)
    plt.close()
    # plt.show()


# junction_cc_mean_boxplot('ERmoxGFP', 'fuz')
# junction_cc_mean_boxplot('mCherry', 'fuz')




def plot_cc_area(a1, a2, a3, c1, c2, c3, r1, r2, r3, ct1, ct2, ct3, region):
    df = pd.DataFrame()
    # df['CC_area'] = pd.Series(np.concatenate((cc_area_atl, cc_area_climp, cc_area_rtn, cc_area_ctrl)))
    # df['Group'] = pd.Series(np.concatenate((['ATL'] * len(cc_area_atl), ['Climp'] * len(cc_area_climp),
    #                                         ['RTN'] * len(cc_area_rtn), ['Control'] * len(cc_area_ctrl))))

    df['CC_area'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3, ct1, ct2, ct3)))
    df['Group'] = pd.Series(np.concatenate((
        ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1),
        ['Climp'] * len(c2), ['Climp'] * len(c3), ['RTN'] * len(r1),
        ['RTN'] * len(r2), ['RTN'] * len(r3), ['Control'] * len(r3),
        ['Control'] * len(r3), ['Control'] * len(r3))))

    df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
                                                ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
                                                ['R3'] * len(r3), ['R1'] * len(ct1), ['R2'] * len(ct2),
                                                ['R3'] * len(ct3))))

    # ax = sns.swarmplot(data=df, x='Group', y='CC_area', hue='Replicate', dodge=True)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax = sns.boxenplot(data=df, x='Replicate', y='CC_area', hue='Group', dodge=True)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    # sns.boxplot(data=df, x='Group', y='CC_area', hue='replicate', color='white', dodge=True)

    plt.yscale('log')

    channel = 'egfp'


    replicates = ['R1', 'R2', 'R3']

    if channel == 'egfp':
        groups = ['ATL', 'Climp', 'RTN', 'Control']
    else:
        groups = ['ATL', 'Climp', 'RTN']
    box_pairs = [((x, y), (x, z)) for i, x in enumerate(replicates) for j, y in enumerate(groups) for z in groups[j + 1 :]]

    statannot.add_stat_annotation(ax, x='Replicate', y='CC_area', hue='Group', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    region_name = 'Isolated' if region == 'iso' else 'Fuzzy'

    plt.suptitle(f'{region_name} CC area across conditions', fontsize=20)
    plt.title('CC area denotes the total movement of each junction', fontsize=18)
    plt.grid(True)
    plt.xlabel('Replicate', fontsize=18)
    plt.ylabel('CC_area (movement of junctions), log scale', fontsize=18)
    plt.show()



def plot_cc_area_all(a1, c1, r1, ct1, region):
    df = pd.DataFrame()
    # df['CC_area'] = pd.Series(np.concatenate((cc_area_atl, cc_area_climp, cc_area_rtn, cc_area_ctrl)))
    # df['Group'] = pd.Series(np.concatenate((['ATL'] * len(cc_area_atl), ['Climp'] * len(cc_area_climp),
    #                                         ['RTN'] * len(cc_area_rtn), ['Control'] * len(cc_area_ctrl))))

    df['CC_area'] = pd.Series(np.concatenate((ct1, r1, c1, a1)))
    df['Group'] = pd.Series(np.concatenate((['Control'] * len(ct1), ['RTN'] * len(r1), ['Climp'] * len(c1),
        ['ATL'] * len(a1))))

    # df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
    #                                             ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
    #                                             ['R3'] * len(r3), ['R1'] * len(ct1), ['R2'] * len(ct2),
    #                                             ['R3'] * len(ct3))))

    # ax = sns.swarmplot(data=df, x='Group', y='CC_area', hue='Replicate', dodge=True)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)


    ax = sns.boxenplot(data=df, x='Group', y='CC_area')
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    yt = ax.get_yticks()
    yt = [f'{y:.2f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=18)
    # sns.boxplot(data=df, x='Group', y='CC_area', hue='replicate', color='white', dodge=True)

    # plt.yscale('log')

    # box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('ATL', 'Control'), ('Climp', 'RTN'), ('Climp', 'Control'),
    #              ('RTN', 'Control')]

    # statannot.add_stat_annotation(ax, x='Group', y='CC_area', data=df, box_pairs=box_pairs,
    #                               test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize=20)

    region_name = 'Isolated' if region == 'iso' else 'Fuzzy'

    # plt.suptitle(f'{region_name} CC area across conditions', fontsize=24)
    # plt.title('CC area denotes the total movement of each junction', fontsize=24)
    plt.title(f'{region_name} CC area across conditions', fontsize=24)
    plt.grid(True)
    plt.xlabel('Group', fontsize=24)
    # plt.ylabel('CC_area (movement of junctions), log scale', fontsize=24)
    plt.ylabel('CC_area (movement of junctions)', fontsize=24)
    plt.show()


# a1 = pd.Series(cc_area_measure('ATL', 'iso', 1, 26))
# c1 = pd.Series(cc_area_measure('Climp', 'iso', 1, 31))
# r1 = pd.Series(cc_area_measure('RTN', 'iso', 1, 29))
# ct1 = pd.Series(cc_area_measure('Control', 'iso', 1, 31))
ct1 = pkl.load(open('cc_area_Control_fuz.pkl', 'rb'))
r1 = pkl.load(open('cc_area_RTN_fuz.pkl', 'rb'))
c1 = pkl.load(open('cc_area_Climp_fuz.pkl', 'rb'))
a1 = pkl.load(open('cc_area_ATL_fuz.pkl', 'rb'))

plot_cc_area_all(a1, c1, r1, ct1, 'fuz')
exit()


def get_data_cc_area(region):
    # at1 = (pd.Series(cc_area_measure('ATL', region, 1, 10)))

    # sns.displot(at3)
    # plt.xlabel('CC Area', fontsize=15)
    # plt.title('ATL-Replicate3 fuzzy CC distribution', fontsize=20)
    # plt.show()

    at1 = (pd.Series(cc_area_measure('ATL', region, 1, 10)))
    cl1 = (pd.Series(cc_area_measure('Climp', region, 1, 10)))
    rt1 = (pd.Series(cc_area_measure('RTN', region, 1, 10)))
    ctrl1 = (pd.Series(cc_area_measure('Control', region, 1, 10)))
    at2 = (pd.Series(cc_area_measure('ATL', region, 11, 20)))
    cl2 = (pd.Series(cc_area_measure('Climp', region, 11, 20)))
    rt2 = (pd.Series(cc_area_measure('RTN', region, 11, 20)))
    ctrl2 = (pd.Series(cc_area_measure('Control', region, 11, 20)))
    at3 = (pd.Series(cc_area_measure('ATL', region, 21, 26)))
    cl3 = (pd.Series(cc_area_measure('Climp', region, 21, 31)))
    rt3 = (pd.Series(cc_area_measure('RTN', region, 21, 29)))
    ctrl3 = (pd.Series(cc_area_measure('Control', region, 21, 31)))

    plot_cc_area(at1, at2, at3, cl1, cl2, cl3, rt1, rt2, rt3, ctrl1, ctrl2, ctrl3, region)


# get_data_cc_area('fuz')
# exit()



def create_tubule_junc_plot(er_path, skeleton_path):
    er_img = imageio.imread(er_path)
    plt.imshow(er_img, cmap='gray')

    graph = node_connector(er_path, skeleton_path)

    exclude_edges = [(node1, node2) for node1, node2 in graph.edges() if
                     graph.degree(node1) == 1 or graph.degree(node2) == 1]

    for (start_node, end_node) in graph.edges():
        if graph[start_node][end_node][0]:
            ps = graph[start_node][end_node][0]['pts']
            if (start_node, end_node) not in exclude_edges:
                plt.plot(ps[:, 1], ps[:, 0], 'red')
            else:
                plt.plot(ps[:, 1], ps[:, 0], 'green')

    deg_one_nodes, deg_two_nodes, high_deg_nodes = get_updated_degree_nodes(graph)

    if len(deg_one_nodes) != 0:
        plt.plot(deg_one_nodes[:, 1], deg_one_nodes[:, 0], 'o', markerfacecolor='yellow', markeredgecolor='yellow',
                 mew=0.5, markersize=3)

    if len(high_deg_nodes) != 0:
        plt.plot(high_deg_nodes[:, 1], high_deg_nodes[:, 0], 'o', markerfacecolor='blue', markeredgecolor='blue',
                 mew=0.5, markersize=3)

    plt.show()


def get_intersection(a, b):
    # get common elements between 2 ndarrays (list of nodes)
    return np.array([x for x in a if np.any(np.all(x == b, axis=1))])



# edges, conn_graph = get_tubule_data('ATL', 1, 'iso-iso')
# edge_pts = [conn_graph[u][v][0]['pts'] for (u, v) in edges]
#
# def get_er_input(num, group, channel, series_num, ch_id):
#     er_input_path = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/std_{channel}/A1_decon_t0{num:02d}_ch0{ch_id}_std.png'
#     er = imageio.imread(er_input_path)
#     # er = (er - er.min()) / (er.max() - er.min())
#     return er / 255
#
# egfp = get_er_input(0, 'ATL', 'egfp', 1, 0)
# mch = get_er_input(0, 'ATL', 'mch', 1, 1)
#
# # egfp = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_egfp/A1_decon_t000_ch00_std.png') / 255
# # mch = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_mch/A1_decon_t000_ch01_std.png') / 255
#
# a = edge_pts[0]
# print(egfp[a[:, 0], a[:, 1]])
# print(mch[a[:, 0], a[:, 1]])
# exit()





def get_group_data(group):
    a1e = get_pickle_data(group, 'iso-iso', 'egfp')
    a2e = get_pickle_data(group, 'iso-fuz', 'egfp')
    a3e = get_pickle_data(group, 'fuz-fuz', 'egfp')
    a1m = get_pickle_data(group, 'iso-iso', 'mch')
    a2m = get_pickle_data(group, 'iso-fuz', 'mch')
    a3m = get_pickle_data(group, 'fuz-fuz', 'mch')

    eg = a1e + a2e + a3e
    mc = a1m + a2m + a3m

    egr1 = eg[:10]
    mcr1 = mc[:10]
    egr2 = eg[10:20]
    mcr2 = mc[10:20]
    egr3 = eg[20:]
    mcr3 = mc[20:]

    corr_data_r1 = []
    corr_data_r2 = []
    corr_data_r3 = []

    for tub_eg, tub_mch in zip(egr1, mcr1):
        corr_data_r1.extend(np.corrcoef(i, j)[0][1] for i, j in zip(tub_eg, tub_mch))

    for tub_eg, tub_mch in zip(egr2, mcr2):
        corr_data_r2.extend(np.corrcoef(i, j)[0][1] for i, j in zip(tub_eg, tub_mch))

    for tub_eg, tub_mch in zip(egr3, mcr3):
        corr_data_r3.extend(np.corrcoef(i, j)[0][1] for i, j in zip(tub_eg, tub_mch))

    return corr_data_r1, corr_data_r2, corr_data_r3


def plot_group_data():

    a1, a2, a3 = get_group_data('ATL')
    c1, c2, c3 = get_group_data('Climp')
    r1, r2, r3 = get_group_data('RTN')
    # control = get_group_data('Control')


    df = pd.DataFrame()

    # df['data_tubule_mean'] = pd.Series(np.concatenate((atl, climp, rtn)))
    df['data_tubule_mean'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3)))

    df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
                                                ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
                                                ['R3'] * len(r3))))

    df['Group'] = pd.Series(np.concatenate((
        ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1), ['Climp'] * len(c2),
        ['Climp'] * len(c3), ['RTN'] * len(r1), ['RTN'] * len(r2), ['RTN'] * len(r3))))

    # ax = sns.boxenplot(data=df, x='Group', y='data_tubule_mean')
    ax = sns.boxenplot(data=df, x='Replicate', y='data_tubule_mean', hue='Group', dodge=True)  # , yscale='log')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    # egfp
    channel = 'egfp'
    replicates = ['R1', 'R2', 'R3']

    if channel == 'egfp':
        groups = ['ATL', 'Climp', 'RTN', 'Control']
    else:
        groups = ['ATL', 'Climp', 'RTN']

    box_pairs = [((x, y), (x, z)) for i, x in enumerate(replicates) for j, y in enumerate(groups) for z in groups[j + 1 :]]

    # box_pairs = [(('R1', 'ATL'), ('R1', 'Climp')), (('R1', 'ATL'), ('R1', 'RTN')), (('R1', 'Climp'), ('R1', 'RTN')),
    #              (('R2', 'ATL'), ('R2', 'Climp')), (('R2', 'ATL'), ('R2', 'RTN')), (('R2', 'Climp'), ('R2', 'RTN')),
    #              (('R3', 'ATL'), ('R3', 'Climp')), (('R3', 'ATL'), ('R3', 'RTN')), (('R3', 'Climp'), ('R3', 'RTN'))]

    statannot.add_stat_annotation(ax, x='Replicate', y='data_tubule_mean', hue='Group', data=df, box_pairs=box_pairs,
                                test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    # statannot.add_stat_annotation(ax, x='Group', y='data_tubule_mean', data=df, box_pairs=box_pairs,
    #                               test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    plt.title('Cross-correlation between ERmoxGFP and mCherry over sequence for tubule intensity mean in all tubules',
            fontsize=18)
    plt.grid(True)
    plt.xlabel('Replicate', fontsize=20)
    plt.ylabel('Cross-correlation value', fontsize=18)

    plt.show()





# for i in range(1, 11):
#     tubule_intensity_analysis('Control', i, 'iso-iso')
# exit()


# atl = []
# climp = []
# rtn = []
# ctrl = []

# for i in range(1, 11):
#     vals = tubule_intensity_analysis('ATL', i, 'iso-iso', 'mean')
#     atl.extend(vals)
#     vals = tubule_intensity_analysis('Climp', i, 'iso-iso', 'mean')
#     climp.extend(vals)
#     vals = tubule_intensity_analysis('RTN', i, 'iso-iso', 'mean')
#     rtn.extend(vals)
#     vals = tubule_intensity_analysis('Control', i, 'iso-iso', 'mean')
#     ctrl.extend(vals)

# df = pd.DataFrame()

# df['Tubule mean '] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3)))

# sns.distplot(atl, hist=False, label='ATL')
# sns.distplot(climp, hist=False, label='Climp')
# sns.distplot(rtn, hist=False, label='RTN')
# sns.distplot(ctrl, hist=False, label='Control')
# plt.show()
# exit()





# atl1 = get_group_len_data('ATL', 'iso-iso')
# atl2 = get_group_len_data('ATL', 'iso-fuz')
# atl3 = get_group_len_data('ATL', 'fuz-fuz')

# climp1 = get_group_len_data('Climp', 'iso-iso')
# climp2 = get_group_len_data('Climp', 'iso-fuz')
# climp3 = get_group_len_data('Climp', 'fuz-fuz')
# # ctrl = get_group_len_data('Control', 'iso-iso')
# rtn1 = get_group_len_data('RTN', 'iso-iso')
# rtn2 = get_group_len_data('RTN', 'iso-fuz')
# rtn3 = get_group_len_data('RTN', 'fuz-fuz')


# sns.distplot(atl1+atl2+atl3, hist=False, label='ATL')
# sns.distplot(climp1+climp2+climp3, hist=False, label='Climp')
# # sns.distplot(ctrl, hist=False, label='Control')
# sns.distplot(rtn1+rtn2+rtn3, hist=False, label='RTN')
# plt.show()
# exit()


def get_above_mean_across_groups():
    atl_list_eg = []
    atl_list_mch = []

    climp_list_eg = []
    climp_list_mch = []

    rtn_list_eg = []
    rtn_list_mch = []

    ctrl_list_eg = []

    # for i in range(1, 27):
    #     ln_egfp, ln_mch = calc_deposit(i, 'ATL', 'iso')
    #
    #     cnt_eg_list = []
    #     cnt_mch_list = []
    #     for each_eg, each_mch in zip(ln_egfp, ln_mch):
    #         eg_mean = np.mean(each_eg)
    #         mch_mean = np.mean(each_mch)
    #         cnt_eg = sum(t > eg_mean for t in each_eg)
    #         cnt_eg_list.append(cnt_eg/100)
    #         cnt_mch = sum(t > mch_mean for t in each_mch)
    #         cnt_mch_list.append(cnt_mch/100)
    #
    #     # print(cnt_list)
    #     atl_list_eg.extend(cnt_eg_list)
    #     atl_list_mch.extend(cnt_mch_list)

    for i in range(1, 27):
        ln_egfp = calc_deposit(i, 'ATL', 'iso')

        cnt_eg_list = []
        for each_eg in ln_egfp:
            eg_mean = np.mean(each_eg)
            cnt_eg = sum(t > eg_mean for t in each_eg)
            cnt_eg_list.append(cnt_eg / 100)

        atl_list_eg.extend(cnt_eg_list)

    for i in range(1, 32):
        ln_egfp = calc_deposit(i, 'Climp', 'iso')

        cnt_eg_list = []
        for each_eg in ln_egfp:
            eg_mean = np.mean(each_eg)
            cnt_eg = sum(t > eg_mean for t in each_eg)
            cnt_eg_list.append(cnt_eg / 100)

        climp_list_eg.extend(cnt_eg_list)

    # for i in range(1, 32):
    #     ln_egfp = calc_deposit(i, 'Control', 'iso')
    #
    #     cnt_eg_list = []
    #     for each_eg in ln_egfp:
    #         eg_mean = np.mean(each_eg)
    #         cnt_eg = sum(t > eg_mean for t in each_eg)
    #         cnt_eg_list.append(cnt_eg/100)
    #
    #     ctrl_list_eg.extend(cnt_eg_list)

    for i in range(1, 30):
        ln_egfp = calc_deposit(i, 'RTN', 'iso')

        # print(ln_egfp)
        cnt_eg_list = []
        for each_eg in ln_egfp:
            eg_mean = np.mean(each_eg)
            # print(eg_mean)
            cnt_eg = sum(t > eg_mean for t in each_eg)
            # print(cnt_eg)
            cnt_eg_list.append(cnt_eg / 100)

        rtn_list_eg.extend(cnt_eg_list)

    # print(above_mean_list)
    # print(len(above_mean_list))
    plt.title('Isolated CCs in mCherry across movies with intensity above mean value for all conditions', fontsize=18)
    plt.xlabel('Fraction of timeframes with intensity above mean', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    sns.distplot(atl_list_eg, label='ATL', hist=False)
    sns.distplot(climp_list_eg, label='Climp', hist=False)
    # sns.distplot(ctrl_list_eg, label='Control', hist=False)
    sns.distplot(rtn_list_eg, label='RTN', hist=False)
    plt.legend(fontsize=13)
    plt.show()


def junc_bar_plots(ser_num, group, junc_num):
    ln_egfp, ln_mch, region_cc_coords = calc_deposit(ser_num, group, 'iso')

    df = pd.DataFrame({'ERmoxGFP': ln_egfp[junc_num], 'mCherry': ln_mch[junc_num]}, index=np.arange(0, 10))

    df.plot.bar(rot=45)

    plt.ylabel('Mean intensity value', fontsize=14)
    plt.xlabel('Timeframe', fontsize=14)
    #    xr = np.arange(0, len(ma_egfp_5))
    #    plt.xticks(xr, rotation=45)
    # plt.title(f'{group} series 1, isolated CC {junc_num+1} mean intensity variation for both channels', fontsize=18)
    #    plt.title(f'{group} Series{ser_num} isolated junc{junc_num} - CC mean intensity variation (mov. avg 3)', fontsize=18)
    plt.title(f'{group} Series{ser_num} isolated junc{junc_num} - CC mean intensity variation', fontsize=18)
    plt.legend()
    #    plt.legend(loc='upper right')
    # plt.savefig(f'ATL1_j{ser_num}_ma5', bbox_inches='tight', pad_inches=1)
    # plt.close()
    plt.show()


def junc_line_charts_init(ser_num, group, junc_num):
    global junc_id
    ln_egfp, ln_mch, region_cc_coords = calc_deposit(ser_num, group, 'iso')

    ids = list(region_cc_coords.keys())

    # print(ids)
    # exit()

    for i, val in enumerate(ids):
        if junc_num == val:
            junc_id = i
            break

    plt.plot(ln_egfp[junc_id], label='EGFP')
    plt.plot(ln_mch[junc_id], label='mCherry')

    plt.ylabel('Mean intensity value', fontsize=14)
    plt.xlabel('Timeframe', fontsize=14)

    plt.title(f'{group} Series{ser_num} isolated junc{junc_num} - CC mean intensity variation', fontsize=18)
    plt.legend()
    plt.show()


def junc_line_charts(ser_num, group, junc_num):
    global junc_id
    ln_egfp, ln_mch, region_cc_coords = calc_deposit(ser_num, group, 'iso')

    print(ln_egfp.shape)
    print(ln_egfp[0])
    exit()

    ln_egfp_mod = []

    for junc_data in ln_egfp:
        # print(junc_data)
        # exit()
        data = []
        for each in junc_data:
            data.extend(each)
        # print(np.amax(junc_data))

        # print(max(data))
        # print(min(data))
        # print(data)

        # print(junc_data - min(data))
        # print(max(data) - min(data))
        #
        # exit()
        std_data = (junc_data - min(data)) / (max(data) - min(data))
        # std_data = np.array(std_data)
        # print(std_data)
        # print(np.mean(std_data.T))
        # print(junc_data)
        l = []
        for e in std_data:
            l.append(np.mean(e))
            ln_egfp_mod.append(l)

    print(np.array(ln_egfp_mod).shape)
    exit()

    # print(ln_egfp.shape)
    # print(ln_egfp[0][:])
    # print(ln_egfp[0].T)
    # exit()

    # df = pd.DataFrame({'Timeframe': np.arange(0, 100), 'Mean Intensity value': ln_egfp[junc_num]})

    # df.plot.bar(x='Timeframe', y='Mean Intensity value', rot=45)

    ##################
    #    df = pd.DataFrame({'ERmoxGFP': ln_egfp[junc_num], 'mCherry': ln_mch[junc_num]}, index = np.arange(0, 10))

    #    df.plot.bar(rot=45)
    ##################

    #    plt.show()

    # cumsum_vec_mch = np.cumsum(np.insert(ln_mch[junc_num], 0, 0))
    # w = mva
    # ma_vec_mch = (cumsum_vec_mch[w:] - cumsum_vec_mch[:-w]) / w

    # cumsum_vec_eg = np.cumsum(np.insert(ln_egfp[junc_num], 0, 0))
    # w = mva
    # ma_vec_eg = (cumsum_vec_eg[w:] - cumsum_vec_eg[:-w]) / w

    #    ma_egfp_3 = np.convolve(ln_egfp[junc_num], np.ones(3), 'valid') / 3

    # ma_mch_3 = np.convolve(ln_mch[junc_num], np.ones(3), 'valid') / 3

    # ma_egfp_5 = np.convolve(ln_egfp[junc_num], np.ones(5), 'valid') / 5

    #    ma_mch_3 = np.convolve(ln_mch[junc_num], np.ones(3), 'valid') / 3

    #    ma_mch_5 = np.convolve(ln_mch[junc_num], np.ones(5), 'valid') / 5

    # ma_egfp_7 = np.convolve(ln_egfp[junc_num], np.ones(7), 'valid') / 7

    # ma_mch_7 = np.convolve(ln_mch[junc_num], np.ones(7), 'valid') / 7

    # ma_egfp_9 = np.convolve(ln_egfp[junc_num], np.ones(9), 'valid') / 9

    # ma_mch_9 = np.convolve(ln_mch[junc_num], np.ones(9), 'valid') / 9

    #    print(ln_egfp[:5])

    #    exit()

    # print(ln_egfp[junc_num])
    # print(ln_mch[junc_num])

    #############################################

    ids = list(region_cc_coords.keys())

    # print(ids)
    # exit()

    for i, val in enumerate(ids):
        if junc_num == val:
            junc_id = i
            break

    #############################################

    # idx = label_vals[junc_num]
    # print(idx)

    # for i, val in enumerate(iso):
    #     if idx[0][0] == val[0] and idx[0][1] == val[1]:
    #         junc_id = i
    #         break

    # print(junc_id)
    # print(iso)
    # print(region_cc_coords)
    # exit()

    #############################################

    # print(ln_mch[junc_id])
    # print(ln_mch[junc_id+1])

    std_egfp = (ln_egfp[junc_id] - ln_egfp[junc_id].min()) / (ln_egfp[junc_id].max() - ln_egfp[junc_id].min())
    std_mch = (ln_mch[junc_id] - ln_mch[junc_id].min()) / (ln_mch[junc_id].max() - ln_mch[junc_id].min())

    plt.plot(std_egfp, label='EGFP')
    plt.plot(std_mch, label='mCherry')

    # plt.plot(ln_egfp[junc_id], label='EGFP')
    # plt.plot(ln_mch[junc_id], label='mCherry')

    # plt.plot(ln_egfp[junc_id+1], label='CC mean intensity (EGFP)')
    # plt.plot(ln_mch[junc_id+1], label='CC mean intensity (mCherry)')

    #    plt.plot(ln_egfp[junc_num], label='EGFP')
    #    plt.bar(ln_egfp[junc_num], label='EGFP')

    #    plt.plot(ma_egfp_3, label='EGFP')
    # plt.plot(ma_egfp_5, label='EGFP')
    # plt.plot(ma_mch_9, label='mCherry')
    # plt.plot(ma_egfp_9, label='EGFP')
    # plt.plot(ma_mch_9, label='mCherry')
    # plt.plot(ma_egfp_5, label='EGFP, mva=5')
    #    plt.plot(ln_mch[junc_num], label='mCherry')
    #    plt.bar(ln_mch[junc_num], label='mCherry')

    #    plt.plot(ma_mch_3, label='mCherry')
    #    plt.plot(ma_mch_5, label='mCherry')
    # plt.plot(ma_egfp_7, label='EGFP, mva=7')
    # plt.plot(ma_mch_7, label='mCherry, mva=7')
    # plt.plot(ma_egfp_9, label='EGFP, mva=9')
    # plt.plot(ma_mch_9, label='mCherry, mva=9')

    plt.ylabel('Mean intensity value', fontsize=14)
    plt.xlabel('Timeframe', fontsize=14)
    #    xr = np.arange(0, len(ma_egfp_5))
    #    plt.xticks(xr, rotation=45)

    # xr = np.arange(30, 60)
    # print(xr)
    # plt.xticks(range(30, 60))
    # exit()
    # plt.xticks(ln_egfp[junc_id], labels=xr)
    # ax.set_xticklabels(xr)

    # plt.title(f'{group} series 1, isolated CC {junc_num+1} mean intensity variation for both channels', fontsize=18)
    #    plt.title(f'{group} Series{ser_num} isolated junc{junc_num} - CC mean intensity variation (mov. avg 3)', fontsize=18)
    plt.title(f'{group} Series{ser_num} isolated junc{junc_num} - CC mean intensity variation', fontsize=18)
    plt.legend()
    #    plt.legend(loc='upper right')
    # plt.savefig(f'ATL1_j{ser_num}_ma5', bbox_inches='tight', pad_inches=1)
    # plt.close()
    plt.show()


# def junc_line_mean_std(group, num_movies, region):
def junc_line_mean_std(group, repl_start, repl_end, region, measure):
    egfp_list = []
    mch_list = []

    for ser_num in range(repl_start, repl_end + 1):
        global junc_id

        ln_egfp, ln_mch, region_cc_coords = cc_signal_net_norm(ser_num, group, region)


        ids = list(region_cc_coords.keys())

        l_eg = []
        l_mc = []

        for idx, val in enumerate(ids):
            mx_egfp = max(ln_egfp[idx])
            mn_egfp = min(ln_egfp[idx])
            std_egfp = (ln_egfp[idx] - mn_egfp) / (mx_egfp - mn_egfp)
            if measure == 'mean':
                l_eg.append(np.mean(std_egfp))
            else:
                l_eg.append(np.std(std_egfp))
            egfp_list.extend(l_eg)
            if group != 'Control':
                mx_mch = max(ln_mch[idx])
                mn_mch = min(ln_mch[idx])
                std_mch = (ln_mch[idx] - mn_mch) / (mx_mch - mn_mch)
                if measure == 'mean':
                    l_mc.append(np.mean(std_mch))
                else:
                    l_mc.append(np.std(std_mch))
                mch_list.extend(l_mc)
            else:
                mch_list = None

    return egfp_list, mch_list
    # return mch_list





def corr_plots(region):
    a1 = correlation_analysis('ATL', 1, 10, region)
    a2 = correlation_analysis('ATL', 11, 20, region)
    a3 = correlation_analysis('ATL', 21, 26, region)
    c1 = correlation_analysis('Climp', 1, 10, region)
    c2 = correlation_analysis('Climp', 11, 20, region)
    c3 = correlation_analysis('Climp', 21, 31, region)
    r1 = correlation_analysis('RTN', 1, 10, region)
    r2 = correlation_analysis('RTN', 11, 20, region)
    r3 = correlation_analysis('RTN', 21, 29, region)

    df = pd.DataFrame()

    df['Channel correlation'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3)))

    df['Group'] = pd.Series(np.concatenate((
        ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1),
        ['Climp'] * len(c2), ['Climp'] * len(c3), ['RTN'] * len(r1),
        ['RTN'] * len(r2), ['RTN'] * len(r3))))
    df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
                                                ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
                                                ['R3'] * len(r3))))

    ax = sns.boxplot(data=df, x='Replicate', y='Channel correlation', hue='Group', dodge=True)  # , yscale='log')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    box_pairs = [(('R1', 'ATL'), ('R1', 'Climp')), (('R1', 'ATL'), ('R1', 'RTN')), (('R1', 'Climp'), ('R1', 'RTN')),
                 (('R2', 'ATL'), ('R2', 'Climp')), (('R2', 'ATL'), ('R2', 'RTN')), (('R2', 'Climp'), ('R2', 'RTN')),
                 (('R3', 'ATL'), ('R3', 'Climp')), (('R3', 'ATL'), ('R3', 'RTN')), (('R3', 'Climp'), ('R3', 'RTN'))]
    statannot.add_stat_annotation(ax, x='Replicate', y='Channel correlation', hue='Group', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    # plt.suptitle('Isolated CC area across conditions', fontsize=20)
    # plt.title('Standard Deviation per sequence for junction CC mean intensity (isolated junctions)', fontsize=18)
    # measure_name = 'Standard deviation' if measure == 'std' else 'Mean'
    region_name = 'isolated' if region == 'iso' else 'fuzzy'
    plt.title(f'Cross correlation between EGFP and mCherry channels in {region_name} CC mean intensity sequences',
              fontsize=18)
    plt.grid(True)
    plt.xlabel('Replicate', fontsize=18)
    plt.ylabel('Pearson correlation coefficient value', fontsize=18)
    plt.show()

    # plt.title(
    #     f'{group} cross correlation between EGFP annd mCherry {reg} CC mean intensity sequences across replicates',
    #     fontsize=18)
    # plt.xlabel('Pearson correlation coefficient value')
    # plt.legend()
    # plt.show()


# corr_plots('fuz')
# exit()


def total_data_variation_plots(group, region, measure):
    # atl_egfp, atl_mch = junc_line_mean_std('ATL', 1, 10, 'fuz')
    # climp_egfp, climp_mch = junc_line_mean_std('Climp', 1, 10, 'fuz')
    # rtn_egfp, rtn_mch = junc_line_mean_std('RTN', 1, 10, 'fuz')
    # # ctrl_egfp, ctrl_mch = junc_line_mean_std('Control', 1, 10, 'iso')
    #
    # # sns.distplot(atl_egfp, hist=False, label='ATL_egfp')
    # sns.distplot(atl_mch, hist=False, label='ATL_mch')
    # # sns.distplot(climp_egfp, hist=False, label='Climp_egfp')
    # sns.distplot(climp_mch, hist=False, label='Climp_mch')
    # # sns.distplot(rtn_egfp, hist=False, label='RTN_egfp')
    # sns.distplot(rtn_mch, hist=False, label='RTN_mch')
    # # sns.distplot(ctrl_egfp, hist=False, label='Control_egfp')
    # plt.title('Standard deviation of Junction CC mean intensity for fuzzy junction CCs (mCherry) - Replicate 1')
    # plt.xlabel('Standard deviation over 100 frames per CC intensity mean value')
    # plt.legend()
    # plt.show()

    if group == 'ATL':
        r3end = 26
    elif group == 'Climp' or group == 'Control':
        r3end = 31
    else:
        r3end = 29

    egfp_r1, mch_r1 = junc_line_mean_std(group, 1, 10, region, measure)
    egfp_r2, mch_r2 = junc_line_mean_std(group, 11, 20, region, measure)
    egfp_r3, mch_r3 = junc_line_mean_std(group, 21, r3end, region, measure)

    reg = 'fuzzy' if region == 'fuz' else 'isolated'

    sns.distplot(egfp_r1, hist=False, label=f'{group}_r1')
    sns.distplot(egfp_r2, hist=False, label=f'{group}_r2')
    sns.distplot(egfp_r3, hist=False, label=f'{group}_r3')
    plt.title(
        f'{group} - Standard deviation of Junction CC mean intensity for {reg} junction CCs (egfp) across replicates',
        fontsize=16)
    plt.xlabel('Standard deviation over 100 frames per CC intensity mean value')
    plt.legend()
    plt.show()
    # climp_egfp, climp_mch = junc_line_mean_std('Climp', 1, 10, 'fuz')
    # rtn_egfp, rtn_mch = junc_line_mean_std('RTN', 1, 10, 'fuz')


# total_data_variation_plots('Control', 'iso', 'std')
# exit()

def full_data_variation_plots(region, measure, channel):
    a_r1_egfp = junc_line_mean_std('ATL', 1, 2, region, measure)
    a_r2_egfp = junc_line_mean_std('ATL', 11, 12, region, measure)
    a_r3_egfp = junc_line_mean_std('ATL', 21, 22, region, measure)
    c_r1_egfp = junc_line_mean_std('Climp', 1, 2, region, measure)
    c_r2_egfp = junc_line_mean_std('Climp', 11, 12, region, measure)
    c_r3_egfp = junc_line_mean_std('Climp', 21, 22, region, measure)
    r_r1_egfp = junc_line_mean_std('RTN', 1, 2, region, measure)
    r_r2_egfp = junc_line_mean_std('RTN', 11, 12, region, measure)
    r_r3_egfp = junc_line_mean_std('RTN', 21, 22, region, measure)
    # ct_r1_egfp = junc_line_mean_std('Control', 1, 10, region, measure)
    # ct_r2_egfp = junc_line_mean_std('Control', 11, 20, region, measure)
    # ct_r3_egfp = junc_line_mean_std('Control', 21, 31, region, measure)

    df = pd.DataFrame()

    df['data_junc_CC_mean'] = pd.Series(np.concatenate((
        a_r1_egfp, a_r2_egfp, a_r3_egfp, c_r1_egfp, c_r2_egfp, c_r3_egfp,
        r_r1_egfp, r_r2_egfp,
        r_r3_egfp)))  # , ct_r1_egfp, ct_r2_egfp, ct_r3_egfp)))
    df['Group'] = pd.Series(np.concatenate((
        ['ATL'] * len(a_r1_egfp), ['ATL'] * len(a_r2_egfp), ['ATL'] * len(a_r3_egfp),
        ['Climp'] * len(c_r1_egfp), ['Climp'] * len(c_r2_egfp),
        ['Climp'] * len(c_r3_egfp), ['RTN'] * len(r_r1_egfp),
        ['RTN'] * len(r_r2_egfp), ['RTN'] * len(
            r_r3_egfp))))  # , ['Control']*len(ct_r1_egfp), ['Control']*len(ct_r2_egfp), ['Control']*len(ct_r3_egfp))))
    df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a_r1_egfp), ['R2'] * len(a_r2_egfp),
                                                ['R3'] * len(a_r3_egfp), ['R1'] * len(c_r1_egfp),
                                                ['R2'] * len(c_r2_egfp), ['R3'] * len(c_r3_egfp),
                                                ['R1'] * len(r_r1_egfp), ['R2'] * len(r_r2_egfp), ['R3'] * len(
        r_r3_egfp))))  # ,['R1']*len(ct_r1_egfp),['R2']*len(ct_r2_egfp),['R3']*len(ct_r3_egfp))))
    ax = sns.boxplot(data=df, x='Replicate', y='data_junc_CC_mean', hue='Group', dodge=True)  # , yscale='log')

    # sns.swarmplot(data=df, x='Replicate', y='data_junc_CC_mean', hue='Group', dodge=True, linewidth=0)

    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    # sns.boxplot(data=df, x='Group', y='CC_area', hue='replicate', color='white', dodge=True)
    # hue_order = ['ATL', 'Climp', 'RTN', 'Control']
    # box_pairs = [(('R1', 'ATL'), ('R1', 'Climp')), (('R1', 'ATL'), ('R1', 'RTN')), (('R1', 'ATL'), ('R1', 'Control')), (('R1', 'Climp'), ('R1', 'RTN')), (('R1', 'Climp'), ('R1', 'Control')), (('R1', 'RTN'), ('R1', 'Control')), (('R2', 'ATL'), ('R2', 'Climp')), (('R2', 'ATL'), ('R2', 'RTN')), (('R2', 'ATL'), ('R2', 'Control')), (('R2', 'Climp'), ('R2', 'RTN')), (('R2', 'Climp'), ('R2', 'Control')), (('R2', 'RTN'), ('R2', 'Control')), (('R3', 'ATL'), ('R3', 'Climp')), (('R3', 'ATL'), ('R3', 'RTN')), (('R3', 'ATL'), ('R3', 'Control')), (('R3', 'Climp'), ('R3', 'RTN')), (('R3', 'Climp'), ('R3', 'Control')), (('R3', 'RTN'), ('R3', 'Control'))]

    replicates = ['R1', 'R2', 'R3']
    groups = ['ATL', 'Climp', 'RTN', 'Control']

    box_pairs = [((x, y), (x, z)) for i, x in enumerate(replicates) for j, y in enumerate(groups) for z in groups[j + 1 :]]


    # box_pairs = [(('R1', 'ATL'), ('R1', 'Climp')), (('R1', 'ATL'), ('R1', 'RTN')), (('R1', 'Climp'), ('R1', 'RTN')),
    #              (('R2', 'ATL'), ('R2', 'Climp')), (('R2', 'ATL'), ('R2', 'RTN')), (('R2', 'Climp'), ('R2', 'RTN')),
    #              (('R3', 'ATL'), ('R3', 'Climp')), (('R3', 'ATL'), ('R3', 'RTN')), (('R3', 'Climp'), ('R3', 'RTN'))]
    statannot.add_stat_annotation(ax, x='Replicate', y='data_junc_CC_mean', hue='Group', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    # plt.suptitle('Isolated CC area across conditions', fontsize=20)
    # plt.title('Standard Deviation per sequence for junction CC mean intensity (isolated junctions)', fontsize=18)
    measure_name = 'Standard deviation' if measure == 'std' else 'Mean'
    region_name = 'isolated' if region == 'iso' else 'fuzzy'
    plt.title(
        f'{measure_name} per sequence for junction CC mean intensity ({region_name} junctions) - {channel} channel',
        fontsize=18)
    plt.grid(True)
    plt.xlabel('Replicate', fontsize=18)
    plt.ylabel(f'{measure_name} value per sequence', fontsize=18)
    plt.show()

def get_group_box_pairs(channel):
    groups = (
        ['ATL', 'Climp', 'RTN']
        if channel == 'mch'
        else ['ATL', 'Climp', 'RTN', 'Control']
    )
    return list(itertools.combinations(groups, 2))


def full_data_variation_plots_all(region, measure):
    atl_egfp, atl_mch = junc_line_mean_std('ATL', 1, 26, region, measure)
    climp_egfp, climp_mch = junc_line_mean_std('Climp', 1, 31, region, measure)
    rtn_egfp, rtn_mch = junc_line_mean_std('RTN', 1, 29, region, measure)
    control_egfp, control_mch = junc_line_mean_std('Control', 1, 31, region, measure)
    # atl_egfp = junc_line_mean_std('ATL', 1, 27, region, measure)
    # climp_egfp = junc_line_mean_std('Climp', 1, 32, region, measure)
    # rtn_egfp = junc_line_mean_std('RTN', 1, 30, region, measure)
    # control_egfp = junc_line_mean_std('Control', 1, 32, region, measure)

    with open(f'CC_mean_atl_egfp_{region}_{measure}.pkl', 'wb') as f:
        pickle.dump(atl_egfp, f)
    with open(f'CC_mean_climp_egfp_{region}_{measure}.pkl', 'wb') as f:
        pickle.dump(climp_egfp, f)
    with open(f'CC_mean_rtn_egfp_{region}_{measure}.pkl', 'wb') as f:
        pickle.dump(rtn_egfp, f)
    with open(f'CC_mean_control_egfp_{region}_{measure}.pkl', 'wb') as f:
        pickle.dump(control_egfp, f)
    
    with open(f'CC_mean_atl_mch_{region}_{measure}.pkl', 'wb') as f:
        pickle.dump(atl_mch, f)
    with open(f'CC_mean_climp_mch_{region}_{measure}.pkl', 'wb') as f:
        pickle.dump(climp_mch, f)
    with open(f'CC_mean_rtn_mch_{region}_{measure}.pkl', 'wb') as f:
        pickle.dump(rtn_mch, f)
    

    df = pd.DataFrame()

    df['data_junc_CC_mean'] = pd.Series(np.concatenate((
        atl_egfp, climp_egfp, rtn_egfp, control_egfp)))
    df['Group'] = pd.Series(np.concatenate((
        ['ATL'] * len(atl_egfp),
        ['Climp'] * len(climp_egfp), ['RTN'] * len(rtn_egfp), ['Control']*len(control_egfp))))
    
    ax = sns.boxplot(data=df, x='Group', y='data_junc_CC_mean', dodge=True)  # , yscale='log')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    # plt.suptitle('Isolated CC area across conditions', fontsize=20)
    # plt.title('Standard Deviation per sequence for junction CC mean intensity (isolated junctions)', fontsize=18)

    box_pairs = get_group_box_pairs('egfp')
    statannot.add_stat_annotation(ax, x='Group', y='data_junc_CC_mean', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize=12)

    measure_name = 'Standard deviation' if measure == 'std' else 'Mean'
    region_name = 'isolated' if region == 'iso' else 'fuzzy'
    plt.title(
        f'{measure_name} per sequence for junction CC mean intensity ({region_name} junctions) - {channel} channel',
        fontsize=18)
    plt.grid(True)
    plt.xlabel('Group', fontsize=18)
    plt.ylabel(f'{measure_name} value per sequence', fontsize=18)
    plt.show()




full_data_variation_plots('iso', 'std', 'mCherry')
# total_data_variation_plots('Control', 'iso', 'std')
exit()


def junc_line_charts_norm(ser_num, group, junc_num):
    egfp_list = []
    mch_list = []
    global junc_id
    # ln_egfp, ln_mch, region_cc_coords = cc_signal_cc_norm(ser_num, group, 'iso')

    ln_egfp, ln_mch, region_cc_coords = cc_signal_net_norm(ser_num, group, 'fuz')

    # print(region_cc_coords)
    # exit()
    # print(ln_egfp)
    # print(ln_mch)
    # exit()

    ids = list(region_cc_coords.keys())
    # print(ids)
    # exit()

    for i, val in enumerate(ids):
        if junc_num == val:
            junc_id = i
            break

    mx_egfp = max(ln_egfp[junc_id])
    mn_egfp = min(ln_egfp[junc_id])

    mx_mch = max(ln_mch[junc_id])
    mn_mch = min(ln_mch[junc_id])

    std_egfp = (ln_egfp[junc_id] - mn_egfp) / (mx_egfp - mn_egfp)
    std_mch = (ln_mch[junc_id] - mn_mch) / (mx_mch - mn_mch)

    print(pearsonr(std_egfp, std_mch))
    exit()

    # plt.plot(ln_egfp[junc_id], label='EGFP')
    # plt.plot(ln_mch[junc_id], label='mCherry')

    # plt.plot(std_egfp, label='EGFP')
    # plt.plot(std_mch, label='mCherry')

    plt.ylabel('Mean intensity value', fontsize=14)
    plt.xlabel('Timeframe', fontsize=14)
    #    xr = np.arange(0, len(ma_egfp_5))
    #    plt.xticks(xr, rotation=45)

    # xr = np.arange(30, 60)
    # print(xr)
    # plt.xticks(range(30, 60))
    # exit()
    # plt.xticks(ln_egfp[junc_id], labels=xr)
    # ax.set_xticklabels(xr)

    # plt.title(f'{group} series 1, isolated CC {junc_num+1} mean intensity variation for both channels', fontsize=18)
    #    plt.title(f'{group} Series{ser_num} isolated junc{junc_num} - CC mean intensity variation (mov. avg 3)', fontsize=18)

    # plt.title(f'{group} Series{ser_num} isolated junc{junc_num} - CC mean intensity variation', fontsize=18)
    # plt.legend()

    plt.title(f'{group} Series{ser_num} fuzzy CC{junc_num} - mean intensity variation', fontsize=18)
    plt.legend()

    #    plt.legend(loc='upper right')
    # plt.savefig(f'ATL1_j{ser_num}_ma5', bbox_inches='tight', pad_inches=1)
    # plt.close()
    plt.show()


# junc_line_charts_init(1, 'ATL', 2)
junc_line_charts_norm(13, 'RTN', 106)
exit()


def get_lincharts(group, junc_num):
    ln_egfp, ln_mch, region_cc_coords = calc_deposit(1, group, 'iso')

    # print(list(region_cc_coords.values())[junc_num])
    # print(region_cc_coords.values())
    # exit()

    # mov_avg = np.convolve(ln_mch[5], np.ones(5), 'valid') / 5
    eg_mean = np.mean(ln_egfp[junc_num])
    mch_mean = np.mean(ln_mch[junc_num])
    scaled_ln_mch = ln_mch[junc_num] * (eg_mean / mch_mean)

    cumsum_vec_mch = np.cumsum(np.insert(scaled_ln_mch, 0, 0))
    w = 5
    ma_vec_mch = (cumsum_vec_mch[w:] - cumsum_vec_mch[:-w]) / w

    cumsum_vec_eg = np.cumsum(np.insert(ln_egfp[junc_num], 0, 0))
    w = 5
    ma_vec_eg = (cumsum_vec_eg[w:] - cumsum_vec_eg[:-w]) / w

    bin_list_eg = []
    bin_list_mch = []
    # for each_eg in ln_egfp:

    bin_ma_mch = []
    bin_ma_egfp = []

    for each in ma_vec_eg:
        if each > eg_mean:
            bin_ma_egfp.append(1)
        else:
            bin_ma_egfp.append(0)

    for each in ma_vec_mch:
        if each > mch_mean:
            bin_ma_mch.append(1)
        else:
            bin_ma_mch.append(0)

    for t in ln_egfp[junc_num]:
        if t > eg_mean:
            bin_list_eg.append(1)
        else:
            bin_list_eg.append(0)

    for t in scaled_ln_mch:
        if t > mch_mean:
            bin_list_mch.append(1)
        else:
            bin_list_mch.append(0)

    egfp_mean_lt = [eg_mean] * 100

    # import scipy.signal
    #
    # corr = scipy.signal.correlate(ma_vec_eg, ma_vec_mch)
    #
    # lags = scipy.signal.correlation_lags(len(ma_vec_mch), len(ma_vec_eg))
    #
    # corr /= np.max(corr)
    #
    # plt.plot(lags, corr)
    # plt.show()
    # corr = (len(ma_vec_eg) - len(ma_vec_mch) + 1) * [0]

    # Go through lag components one-by-one
    # for l in range(len(corr)):
    #     corr[l] = sum([ma_vec_eg[i+l] * ma_vec_mch[i] for i in range(len(ma_vec_mch))])

    # print(corr)

    # # Remove padded correlations
    #     print(ma_vec_eg)
    #     cr_corr = corr[(len(ma_vec_eg)-len(ma_vec_mch)-1):len(corr)-((len(ma_vec_eg)-len(ma_vec_mch)-1))]

    # print(cr_corr)
    # plt.plot(cr_corr)

    # plt.plot(bin_list_eg, label='EGFP_bin')
    # plt.plot(bin_list_mch, label='mCherry_bin')
    # plt.plot(egfp_mean_lt, label='Mean value')

    plt.plot(ln_egfp[junc_num], label='CC mean intensity (EGFP)')
    plt.plot(scaled_ln_mch, label='CC mean intensity (mCherry)')
    # plt.plot(ma_vec_mch, label='mov_avg_mCherry')
    # plt.plot(ma_vec_eg, label='mov_avg_egfp')
    plt.plot(bin_ma_mch, label='binarized mov_avg mCherry')
    plt.plot(bin_ma_egfp, label='binarized mov_avg EGFP')
    plt.plot(egfp_mean_lt, label='Mean value')

    plt.ylabel('Mean intensity value', fontsize=14)
    plt.xlabel('Timeframe', fontsize=14)
    plt.title(f'{group} series 1, isolated CC {junc_num + 1} mean intensity variation for both channels', fontsize=18)
    plt.legend()
    plt.show()
    exit()
    #
    #     # print(eg_mean)
    #     # cnt_eg = sum(t > eg_mean for t in each_eg)
    #     # print(cnt_eg)
    #     # cnt_eg_list.append(cnt_eg / 100)
    #
    #
    plt.plot(ln_egfp[5], label='CC mean intensity (EGFP)')
    plt.plot(scaled_ln_mch, label='CC mean intensity (mCherry)')
    plt.plot(ma_vec_mch, label='mov_avg_mCherry')
    plt.plot(ma_vec_eg, label='mov_avg_egfp')
    plt.xlabel()
    plt.legend()
    plt.show()
    #
    exit()

    mch_mean = np.mean(ln_mch[5])

    egfp_mean = np.mean(ln_egfp[5])

    mch_mean_lt = [mch_mean] * 100
    egfp_mean_lt = [egfp_mean] * 100

    plt.plot(mch_mean_lt, label='mCherry_mean')
    plt.plot(egfp_mean_lt, label='EGFP_mean')
    plt.plot(ln_mch[5], label='mCherry')
    plt.plot(ln_egfp[5], label='EGFP')

    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Mean Intensity value', fontsize=15)
    plt.title('ATL series 1, isolated CC 6 mean intensity over time', fontsize=18)

    plt.legend()
    plt.show()


get_lincharts('RTN', 20)
exit()

# op = np.logical_and(ln_egfp, ln_mch)
# print(op)

# print(ln.shape)


exit()

from scipy.stats import pearsonr


def ref_junc_deposit():
    egfp_img_path = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean/atl1_er_mean.png'

    egfp_img = get_std_img(egfp_img_path)


def per_ref_junc_deposit(group, region, series_num):
    # egfp_img_path = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean/atl1_er_mean.png'

    # egfp_img = get_std_img(egfp_img_path)
    # print(egfp_img.max())

    # mch_stack = np.zeros((128, 128))
    # for i in range(100):
    #     mch_img = f'/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A1_decon_t0{i:02d}_ch01.tif'
    #     mch_img = get_std_img(mch_img)
    #     mch_stack += mch_img
    # mch_op = mch_stack / 100

    # region_name = 'isolated' if region == 'iso' else 'fuzzy'
    atl_mch_iso = calc_egfp_deposit(group, 'mCherry', series_num, region)
    atl_egfp_iso = calc_egfp_deposit(group, 'ERmoxGFP', series_num, region)

    corr = pearsonr(atl_egfp_iso, atl_mch_iso)
    # print(corr)
    # print(len(atl_egfp_iso))

    df = pd.DataFrame()

    df['ERmoxGFP'] = pd.Series(atl_egfp_iso)
    df['mCherry'] = pd.Series(atl_mch_iso)

    fig = plt.gcf()
    fig.set_size_inches(15, 15, forward=True)
    fig.set_dpi(150)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    # plt.figure(figsize=(10,10), dpi=200)
    # sns.scatterplot(data=df, x='mCherry', y='ERmoxGFP')
    b, m = polyfit(df['mCherry'], df['ERmoxGFP'], 1)
    num_pts = len(df['mCherry'])
    plt.plot(df['mCherry'], df['ERmoxGFP'], '.', alpha=(1 / num_pts) * 8000)
    plt.plot(df['mCherry'], b + m * df['mCherry'], '-')
    plt.title('%s - %s junctions, corr=%.2f' % (group, region, corr[0]), fontsize=20)
    plt.xlabel('mCherry', fontsize=18)
    plt.ylabel('ERmoxGFP', fontsize=18)

    plt.show()
    # plt.savefig(f'{group}_{region}_junc_intensity', bbox_inches='tight', pad_inches=0.2)
    # plt.close()

    # exit()

    # climp = calc_egfp_deposit('Climp', channel, 31, region)
    # ctrl = calc_egfp_deposit('Control', channel, 31, region)
    # rtn = calc_egfp_deposit('RTN', channel, 29, region)

    # cv2.imwrite('ATL1_mch_mean.png', mch_op)
    # exit()

    # print(mch_op.max())
    # exit()

    # ref_skel = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc_enhance_skel.png'
    # junctions = get_junctions(ref_skel)
    #
    # # print(junctions)
    # egfp_data = []
    # mch_data = []
    # for each in junctions:
    #     x, y = each[0], each[1]
    #     mean_egfp_val = (egfp_img[x-1, y-1] + egfp_img[x-1, y] + egfp_img[x-1, y+1] + egfp_img[x, y-1] + egfp_img[x, y] + egfp_img[x, y+1] + egfp_img[x+1, y-1] + egfp_img[x+1, y] + egfp_img[x+1, y+1]) / 9
    #     egfp_data.append(mean_egfp_val)
    #     mean_mch_val = (mch_op[x-1, y-1] + mch_op[x-1, y] + mch_op[x-1, y+1] + mch_op[x, y-1] + mch_op[x, y] + mch_op[x, y+1] + mch_op[x+1, y-1] + mch_op[x+1, y] + mch_op[x+1, y+1]) / 9
    #     mch_data.append(mean_mch_val)

    # print(max(egfp_data))
    # print(max(mch_data))
    # exit()

    # plt.plot(egfp_data, label='EGFP')
    # plt.plot(mch_data, label='mCherry')
    # df = pd.DataFrame()
    # df['egfp'] = pd.Series(egfp_data)
    # df['mcherry'] = pd.Series(mch_data)
    # df['intensity'] = pd.Series(np.concatenate((egfp_data, mch_data)))
    # df['junction_num'] = pd.Series(np.concatenate((np.arange( len(egfp_data)+1), np.arange(len(mch_data)+1))))
    # df['channel'] = pd.Series(np.concatenate((['EGFP'] * len(egfp_data), ['mCherry'] * len(mch_data))))

    # df['egfp'] = pd.Series(egfp_data)
    # df['mcherry'] = pd.Series(mch_data)
    #
    # # print(df['channel'])
    # # print(df['junctions'])
    # plt.title('ATL series 1 junction intensity values (mean per 3x3 patch)', fontsize=14)
    #
    # sns.lmplot(data=df, x='egfp', y='mcherry', lowess=True)
    #
    # plt.show()


per_ref_junc_deposit('ATL', 'iso', 26)
# per_ref_junc_deposit('ATL', 'fuz', 26)
# per_ref_junc_deposit('Climp', 'iso', 31)
# per_ref_junc_deposit('Climp', 'fuz', 31)
# per_ref_junc_deposit('RTN', 'iso', 29)
# per_ref_junc_deposit('RTN', 'fuz', 29)
# per_ref_junc_deposit('Climp', 'iso', 31)
# per_ref_junc_deposit('iso')
# per_ref_junc_deposit('iso')
# per_ref_junc_deposit('iso')
# per_ref_junc_deposit('iso')
exit()


def junction_cc_mean_distplot_per_group(group, series_num):
    egfp_data = calc_egfp_deposit(group, 'ERmoxGRP', series_num, 'fuz')
    mch_data = calc_egfp_deposit(group, 'mCherry', series_num, 'fuz')
    return egfp_data if group == 'Control' else (egfp_data, mch_data)
    # climp = calc_egfp_deposit('Climp', channel, 31, 'fuz')
    # ctrl = calc_egfp_deposit('Control', channel, 31, 'fuz')
    # rtn = calc_egfp_deposit('RTN', channel, 29, 'fuz')


atl_egfp, atl_mch = junction_cc_mean_distplot_per_group('ATL', 1)
# climp_egfp, climp_mch = junction_cc_mean_distplot_per_group('Climp', 31)
# rtn_egfp, rtn_mch = junction_cc_mean_distplot_per_group('RTN', 29)
# ctrl_egfp = junction_cc_mean_distplot_per_group('Control', 31)


# plt.scatter(atl_egfp, label='ATL_ERmoxGFP')
# plt.scatter(atl_mch, label='ATL_mCherry')

df = pd.DataFrame()
df['Values'] = pd.Series(np.concatenate((atl_egfp, atl_mch)))
df['channel'] = pd.Series(np.concatenate((np.arange(1, len(atl_egfp) + 1), np.arange(1, len(atl_mch) + 1))))

sns.scatterplot(data=df, x='Values', y='channel', hue='Values')

# plt.legend()

plt.show()

exit()

# sns.distplot(atl_egfp, label='ATL_ERmoxGFP', hist=False)
# sns.distplot(atl_mch, label='ATL_mCherry', hist=False)
# sns.distplot(climp_egfp, label='Climp_ERmoxGFP', hist=False)
# sns.distplot(climp_mch, label='Climp_mCherry', hist=False)
# sns.distplot(rtn_egfp, label='RTN_ERmoxGFP', hist=False)
# sns.distplot(rtn_mch, label='RTN_mCherry', hist=False)
# sns.distplot(ctrl_egfp, label='Control_ERmoxGFP', hist=False)

# sns.distplot(climp, label='Climp', hist=False)
# sns.distplot(ctrl, label='Control', hist=False)
# sns.distplot(rtn, label='RTN', hist=False)

plt.title('Mean Intensity per fuzzy junction area CC patch for all conditions', fontsize=18)
plt.xlabel('Intensity mean values', fontsize=15)
plt.ylabel('Density', fontsize=15)
plt.legend(fontsize=12)  # for distplot only

plt.savefig('All_conditions_fuzzy_CCs_mean_intensity_per_patch_across_channels', bbox_inches='tight', pad_inches=0.2)
plt.close()
# plt.show()

# junction_cc_mean_distplot_per_group('RTN', 29)


def junction_nbrhood_mean_plot():
    atl = calc_egfp_junction_intensity_nbrhood('ATL', 26, 'mCherry')
    climp = calc_egfp_junction_intensity_nbrhood('Climp', 31, 'mCherry')
    # # ctrl = calc_egfp_junction_intensity_nbrhood('Control', 31, 'EGFP')
    rtn = calc_egfp_junction_intensity_nbrhood('RTN', 29, 'mCherry')
    #
    # sns.distplot(atl, label='ATL', hist=False)
    # sns.distplot(climp, label='Climp', hist=False)
    # # sns.distplot(ctrl, label='Control', hist=False)
    # sns.distplot(rtn, label='RTN', hist=False)
    #

    df = pd.DataFrame()
    df['Values'] = pd.Series(np.concatenate((atl, climp, rtn)))
    # df['ids'] = pd.Series(np.concatenate((np.arange(1, len(atl)+1), np.arange(1, len(climp)+1), np.arange(1, len(ctrl)+1), np.arange(1, len(rtn)+1))))
    df['Group'] = pd.Series(np.concatenate((['ATL'] * len(atl), ['Climp'] * len(climp), ['RTN'] * len(rtn))))
    #
    sns.swarmplot(data=df, y='Group', x='Values')

    plt.suptitle('mCherry deposit in isolated reference junction 3x3 neighbourhood across conditions', fontsize=16)
    # # plt.title('Variance of junction CC mean per patch over 100 frames', fontsize=14)
    plt.title('Mean Intensity per junction neighbourhood', fontsize=14)
    # plt.xlabel('mCherry intensity mean values (3x3)', fontsize=12)
    # plt.legend()
    plt.show()

def junction_nbrhood_mean_plot():
    atl = calc_egfp_junction_intensity_nbrhood('ATL', 26, 'mCherry')
    climp = calc_egfp_junction_intensity_nbrhood('Climp', 31, 'mCherry')
    # # ctrl = calc_egfp_junction_intensity_nbrhood('Control', 31, 'EGFP')
    rtn = calc_egfp_junction_intensity_nbrhood('RTN', 29, 'mCherry')
    #
    # sns.distplot(atl, label='ATL', hist=False)
    # sns.distplot(climp, label='Climp', hist=False)
    # # sns.distplot(ctrl, label='Control', hist=False)
    # sns.distplot(rtn, label='RTN', hist=False)
    #

    df = pd.DataFrame()
    df['Values'] = pd.Series(np.concatenate((atl, climp, rtn)))
    # df['ids'] = pd.Series(np.concatenate((np.arange(1, len(atl)+1), np.arange(1, len(climp)+1), np.arange(1, len(ctrl)+1), np.arange(1, len(rtn)+1))))
    df['Group'] = pd.Series(np.concatenate((['ATL'] * len(atl), ['Climp'] * len(climp), ['RTN'] * len(rtn))))
    #
    sns.swarmplot(data=df, y='Group', x='Values')

    plt.suptitle('mCherry deposit in isolated reference junction 3x3 neighbourhood across conditions', fontsize=16)
    # # plt.title('Variance of junction CC mean per patch over 100 frames', fontsize=14)
    plt.title('Mean Intensity per junction neighbourhood', fontsize=14)
    # plt.xlabel('mCherry intensity mean values (3x3)', fontsize=12)
    # plt.legend()
    plt.show()

def junction_cc_mean_plot():
    atl = calc_egfp_deposit('ATL', 'mCherry', 26, 'iso')

    climp = calc_egfp_deposit('Climp', 'mCherry', 31, 'iso')
    # ctrl = calc_egfp_deposit('Control', 'EGFP', 31, 'iso')
    rtn = calc_egfp_deposit('RTN', 'mCherry', 29, 'iso')

    # print(len(atl))
    # print(len(climp))
    # # print(len(ctrl))
    # print(len(rtn))
    #
    # exit()

    sns.distplot(atl, label='ATL', hist=False)
    sns.distplot(climp, label='Climp', hist=False)
    # sns.distplot(ctrl, label='Control', hist=False)
    sns.distplot(rtn, label='RTN', hist=False)

    # sns.boxplot(atl, label='ATL')
    # sns.boxplot(climp, label='Climp')
    # sns.boxplot(ctrl, label='Control')
    # sns.boxplot(rtn, label='RTN')

    # df = pd.DataFrame()
    # df['Values'] = pd.Series(np.concatenate((atl, climp, ctrl, rtn)))
    # df['ids'] = pd.Series(np.concatenate((np.arange(1, len(atl)+1), np.arange(1, len(climp)+1), np.arange(1, len(ctrl)+1), np.arange(1, len(rtn)+1))))
    # df['Group'] = pd.Series(np.concatenate((['ATL'] * len(atl), ['Climp'] * len(climp), ['Control'] * len(ctrl), ['RTN'] * len(rtn))))
    # #
    # # sns.boxplot(data=df, y='Group', x='Values')
    # sns.scatterplot(data=df, x='ids', y='Values', hue='Group', style='Group')

    plt.suptitle('mCherry deposit in isolated region junction CCs across conditions', fontsize=16)
    # plt.title('Variance of junction CC mean per patch over 100 frames', fontsize=14)
    plt.title('Mean Intensity per junction CC patch', fontsize=14)
    # plt.xlabel('EGFP intensity mean values', fontsize=12)
    plt.legend()
    plt.show()

def plot_per_movie_junction_dist():
    ATL_iso, ATL_fuz = per_movie_num_junctions('ATL', 26)
    Climp_iso, Climp_fuz = per_movie_num_junctions('Climp', 31)
    Ctrl_iso, Ctrl_fuz = per_movie_num_junctions('Control', 31)
    RTN_iso, RTN_fuz = per_movie_num_junctions('RTN', 29)

    # sns.distplot(ATL_iso, hist=False, label='atl_iso')
    sns.distplot(ATL_fuz, hist=False, label='atl_fuz')
    # sns.distplot(Climp_iso, hist=False, label='climp_iso')
    sns.distplot(Climp_fuz, hist=False, label='climp_fuz')
    # sns.distplot(Ctrl_iso, hist=False, label='control_iso')
    sns.distplot(Ctrl_fuz, hist=False, label='control_fuz')
    # sns.distplot(RTN_iso, hist=False, label='rtn_iso')
    sns.distplot(RTN_fuz, hist=False, label='rtn_fuz')

    # and fuzzy region
    # (iso: isolated, fuz: fuzzy)
    plt.title('Distribution of fuzzy region junctions across conditions ', fontsize=16)
    plt.xlabel('Number of junctions (per movie)')
    plt.legend()

    plt.show()

def plot_per_movie_junction_area_dist():
    ATL_iso, ATL_fuz = per_movie_junctions_area('ATL', 26)
    Climp_iso, Climp_fuz = per_movie_junctions_area('Climp', 31)
    Ctrl_iso, Ctrl_fuz = per_movie_junctions_area('Control', 31)
    RTN_iso, RTN_fuz = per_movie_junctions_area('RTN', 29)

    sns.distplot(ATL_iso, hist=False, label='atl_iso_area')
    # sns.distplot(ATL_fuz, hist=False, label='atl_fuz_area')
    sns.distplot(Climp_iso, hist=False, label='climp_iso_area')
    # sns.distplot(Climp_fuz, hist=False, label='climp_fuz_area')
    sns.distplot(Ctrl_iso, hist=False, label='control_iso_area')
    # sns.distplot(Ctrl_fuz, hist=False, label='control_fuz_area')
    sns.distplot(RTN_iso, hist=False, label='rtn_iso_area')
    # sns.distplot(RTN_fuz, hist=False, label='rtn_fuz_area')

    # and fuzzy region
    # (iso: isolated, fuz: fuzzy)
    # plt.title('Distribution of fuzzy region junctions across conditions ', fontsize=16)
    plt.title('Distribution of isolated region area across conditions ', fontsize=16)
    plt.xlabel('Region area values (per movie)')
    plt.legend()

    plt.show()

def fuz_isolated_junctions(group, series_num):
    nps, skdata = get_all_junc(group, series_num)

    nps = np.array(nps)
    skdata = np.array(skdata)

    spread_img = np.zeros((128, 128))
    for each in skdata:
        spread_img[each[0], each[1]] = 255.

    # plt.imshow(spread_img)
    # plt.show()
    #
    # exit()

    labelled_img = label(spread_img, connectivity=2)

    # plt.imshow(labelled_img)
    # plt.show()
    #
    # exit()

    regions = regionprops(labelled_img)

    # cc_list = []
    # for idx in range(1, labelled_img.max()):
    #     lab_i = props[idx].label

    cc_area_dict = {idx: props.area for idx, props in enumerate(regions)}
    # print(cc_area_dict)

    # exit()

    num_components = np.unique(labelled_img)

    label_vals, assigned_components = get_ref_junc_per_CC_id(nps, labelled_img)

    unassigned_cc_dict = get_uncertain_junctions(labelled_img, skdata, num_components, assigned_components)

    isolated_junc = []
    isolated_junc_area = []
    fuzzy_junc = []
    fuzzy_junc_area = []
    for k, v in label_vals.items():
        if k != 0:
            if len(v) == 1:
                isolated_junc.append(v[0])
                isolated_junc_area.append(cc_area_dict[k])
            else:
                fuzzy_junc.append(v)
                fuzzy_junc_area.append(cc_area_dict[k])

    print(isolated_junc_area)
    print(fuzzy_junc_area)

    unknown_junc = [v for k, v in unassigned_cc_dict.items()]
    iso = np.array(isolated_junc)

    fuz = list(itertools.chain.from_iterable(fuzzy_junc))
    fuz = np.array(fuz)

    unk = list(itertools.chain.from_iterable(unknown_junc))
    unk = np.array(unk)

    # img = imageio.imread(
    #     '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean_proc/atl1_er_mean_proc.png')

    for i in range(100):
        plt.axis('off')
        if group == 'Control':
            img = imageio.imread(confocal_data_path + f'{group}/files/img_{series_num}_decon_t0{i:02d}.tif')

        else:
            img = imageio.imread(confocal_data_path + f'{group}/files/{group[0]}{series_num}_decon_t0{i:02d}_ch00.tif')

        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img, cmap='gray')
        plt.plot(iso[:, 1], iso[:, 0], 'o', markerfacecolor='None', markeredgecolor='red')
        if len(fuz) > 0:
            plt.plot(fuz[:, 1], fuz[:, 0], 'o', markerfacecolor='None', markeredgecolor='blue')
        plt.plot(unk[:, 1], unk[:, 0], 'o', markerfacecolor='None', markeredgecolor='green')
        plt.plot(skdata[:, 1], skdata[:, 0], 'x', markerfacecolor='None', markeredgecolor='yellow')
        plt.show()
        # if group == 'Control':
        #     plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junc_types_movies/Ct%s_decon_t0%s_ch00.png'%(f'{group}', f'{series_num}', f'{i:02d}'), bbox_inches='tight', pad_inches=0)
        # else:
        #     plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junc_types_movies/%s_decon_t0%s_ch00.png'%(f'{group}', f'{group[0]}{series_num}', f'{i:02d}'), bbox_inches='tight', pad_inches=0)
        # plt.close()


# fuz_isolated_junctions('Control', 13)

def plot_junc_spread(group, n1, n2, num_series):
    fig = plt.gcf()
    ax = fig.gca()
    # gr = cm.Greens(np.linspace(n3arr.min()[0], n3arr.max()[0], num=len(n3)))
    # mcmap = mcolors.LinearSegmentedColormap.from_list('mcmap', gr)
    img = imageio.imread((confocal_data_path + f'{group}/new_op_jul/er_mean/{group.lower()}{num_series}_er_mean.png'))

    plt.imshow(img, cmap='gray', interpolation='none')
    # plt.plot(n2[:, 1], n2[:, 0], 'b.')
    plt.scatter(n2[:, 1], n2[:, 0], c=n3val, cmap='Blues', marker='o')
    # plt.colorbar()
    plt.plot(n1[:, 1], n1[:, 0], 'o', markerfacecolor='None', markeredgecolor='red', mew=1.5)  # , ms=4)
    # plt.plot(n1[:, 1], n1[:, 0], 'r.')
    # c = Circle((n1[0, 1], n1[0, 0]), radius=3, linewidth=2, facecolor='none', edgecolor='green', alpha=0.7)
    # ax.add_patch(c)
    # plt.plot(n2[:, 1], n2[:, 0], 'o', markerfacecolor='blue', markeredgecolor='blue')

    for i, each in enumerate(n1):
        c1 = plt.Circle((n1[i, 1], n1[i, 0]), 3, color='r', fill=False, linestyle='--')
        ax.add_patch(c1)
        # if n5[i] > med:
        s = '(' + '%.2f' % n5[i] + ',' + str(n6[i]) + ')'
        # s = '(' + str(n1[i,1]) + ',' + str(n1[i,0]) + ',' + '%.2f'%n5[i] + ',' + str(n6[i]) + ')'
        ax.text(n1[i, 1], n1[i, 0], s, c='yellow')

    plt.axis('off')
    plt.suptitle(f'Climp series {num_series} junctions movement variance')
    plt.title('Variance of list with distances for matched junctions per frame w.r.t. reference frame junctions')
    # plt.savefig('RTN1_junc_spread.png', bbox_inches='tight', pad_inches=0)
    plt.show()


# plot_junc_spread('Climp', n1, n2, 3)
# exit()

def inter_channel_correlation_runner():
    atl_data = inter_channel_correlation(atl_egfp, atl_mc)
    climp_data = inter_channel_correlation(climp_egfp, climp_mc)
    rtn_data = inter_channel_correlation(rtn_egfp, rtn_mc)

    sns.distplot(atl_data, label='ATL')
    sns.distplot(climp_data, label='Climp')
    sns.distplot(rtn_data, label='RTN')
    plt.legend()
    plt.title('Per patch correlation coefficient between EGFP and mCherry channels', fontsize=16)
    plt.xlabel('Correlation coefficient', fontsize=12)
    plt.show()

def graph_node_connector(group, series):
    global rel
    pref = 'Ct' if group == 'Control' else group[0]

    # projection frame analysis
    # path = '{confocal_data_path}Climp/new_op_jul/er_mean_proc/climp16_er_mean_proc_enhance_skel.png'
    path = f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{series}_er_mean_proc_enhance_skel.png'

    # path_frame = f'{confocal_data_path}{group}/new_op_jul/skel/{group[0]}{series}/{group[0]}{series}_decon_t006_ch00_skel.png'


    path_proc_enh = f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{series}_er_mean_proc_enhance.png'

    path_er = f'{confocal_data_path}{group}/new_op_jul/er_mean/{group.lower()}{series}_er_mean.png'

    path_er_proc = f'{confocal_data_path}{group}/new_op_jul/er_mean_proc/{group.lower()}{series}_er_mean_proc.png'

    # all junction from the graph with degree > 2
    junc_analysis = JA(confocal_data_path)
    graph = junc_analysis.skel_to_graph(path)
    junctions = junc_analysis.get_junctions(graph)

    relevant_nodes = np.array(junctions)

    fin_dict, g_nodes_array = gcm.get_updated_neighbor_dict(graph)

    temp_graph = copy.deepcopy(graph)

    # er_proc = imageio.imread(path_er_proc)
    # er_proc_bg = np.where(er_proc==0)

    er_input = imageio.imread(path_er)
    cost_arr = np.ones((128, 128))
    # cost_arr[er_proc_bg] = 0

    for node in dict(graph.degree()):

        # access the first element of graph.neighbors
        neighbor = next(iter(graph.neighbors(node)))

        gcm.connect_nodes(er_input, temp_graph, node, neighbor, fin_dict, cost_arr, g_nodes_array)

    tgraph = copy.deepcopy(temp_graph)
    for node in temp_graph.nodes():
        gcm.process_node(tgraph, node)

    tgraph2 = copy.deepcopy(tgraph)
    for node in tgraph.nodes():
        gcm.process_node(tgraph2, node)


    # for (st, end) in tgraph2.edges():
    #     print(st, end)
    #
    #
    exclude_edges = []
    for (node1, node2) in tgraph2.edges():
        if tgraph2.degree(node1) == 1 or tgraph2.degree(node2) == 1:
            exclude_edges.append((node1, node2))
    #
    # print(exclude_edges)
    # exit()

    ### Plotting the updated graph
    deg_one_nodes, deg_two_nodes, high_deg_nodes = gcm.get_updated_degree_nodes(tgraph2)

    er_mean = imageio.imread(
        f'{confocal_data_path}{group}/new_op_jul/er_mean/{group.lower()}{series}_er_mean.png')
    plt.imshow(er_mean, cmap='gray')

    for (start_node, end_node) in tgraph2.edges():
        if tgraph2[start_node][end_node][0]:
            ps = tgraph2[start_node][end_node][0]['pts']
            if (start_node, end_node) not in exclude_edges:
                plt.plot(ps[:, 1], ps[:, 0], 'red')
            else:
                plt.plot(ps[:, 1], ps[:, 0], 'green')
        # elif temp_graph[start_node][end_node][1]:
        #     ps = tgraph2[start_node][end_node][1]['pts']
        #     plt.plot(ps[:, 1], ps[:, 0], 'red')

    if len(deg_one_nodes) != 0:
        plt.plot(deg_one_nodes[:, 1], deg_one_nodes[:, 0], 'o', markerfacecolor='yellow', markeredgecolor='yellow',
                 mew=0.5, markersize=3)

    # if len(deg_two_nodes) != 0:
    #     plt.plot(deg_two_nodes[:, 1], deg_two_nodes[:, 0], 'o', markerfacecolor='magenta', markeredgecolor='magenta',
    #              mew=0.5, markersize=3)

    if len(high_deg_nodes) != 0:
        plt.plot(high_deg_nodes[:, 1], high_deg_nodes[:, 0], 'o', markerfacecolor='blue', markeredgecolor='blue',
                 mew=0.5, markersize=3)

    # plt.axis('off')
    # plt.savefig(f'graphs/connected/repair/{group}_{series}_edge_graph_projection_connected_final_v2', bbox_inches='tight', pad_inches=0)
    # plt.savefig(f'graphs/connected/{group}_{series}_v2-2', bbox_inches='tight', pad_inches=0)
    # plt.close()
    # #
    plt.show()