import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import numpy as np
import pdb
import os

# Features
# Index(['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
#        'Orientation', 'ConvexArea', 'Circularity', 'EulerNumber', 'Perimeter',
#        'MaxFeretDiameter', 'MaxFeretAngle', 'MaxFeretCoordinates_1',
#        'MaxFeretCoordinates_2', 'MaxFeretCoordinates_3',
#        'MaxFeretCoordinates_4', 'MinFeretDiameter', 'MinFeretAngle',
#        'MinFeretCoordinates_1', 'MinFeretCoordinates_2',
#        'MinFeretCoordinates_3', 'MinFeretCoordinates_4'],
#       dtype='object')

home = os.path.expanduser('~')

# prefix = home + '/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/'
prefix = home + '/Documents/CROP/'
# control = 'control-features/'
# rtn = 'rtn-features/'
# climp = 'climp-features/'
control = 'CTL-features/'
rtn = 'RTN-features/'
climp = 'CLIMP-features/'
atl = 'ATL-features/'

Beads_features = ['Area', 'MajorAxisLength', 'Eccentricity', 'MinorAxisLength']


prefix_atl = prefix + atl
prefix_control = prefix + control
prefix_climp = prefix + climp
prefix_rtn = prefix + rtn


def load_data(group, modality):
    if group == 'control':
        pref = prefix_control
    elif group == 'climp':
        pref = prefix_climp
    elif group == 'rtn':
        pref = prefix_rtn
    elif group == 'atl':
        pref = prefix_atl
    else:
        print("Only control, climp, rtn groups supported.")
        return
    # return glob.glob(pref + '*-' + modality + '.csv')
    return glob.glob(pref + '*')


def get_features(group, modality, feature):
    group_modality_files = load_data(group, modality)
    group_modality_feature = []
    for group in group_modality_files:
        group_modality_data = pd.read_csv(group)[feature]
        # group_modality_feature += list(group_modality_data)
        group_modality_feature += [x/25 for x in group_modality_data]
    return group_modality_feature


def remove_pixel_beads(data):
    ones_indices = np.where(np.array(data, copy=False)==1)


def remove_ones(feature_list):
    return [val for val in feature_list if val != 1]


def multimodal_plots(group, feature):

    conf_ft = get_features(group, 'conf', feature)
    sted_ft = get_features(group, 'sted', feature)
    synth_ft = get_features(group, 'synth', feature)

    conf_ft = remove_ones(conf_ft)
    sted_ft = remove_ones(sted_ft)
    synth_ft = remove_ones(synth_ft)

    df = pd.DataFrame()
    Feature_list = conf_ft + sted_ft + synth_ft

    # ones_indices = np.where(np.array(Feature_list, copy=False)==1)

    df['Feature'] = pd.Series(Feature_list)

    #For case with dropping the single pixel beads
    # df.drop(df.index[ones_indices])

    conf_len_list = ['Confocal'] * len(conf_ft)
    sted_len_list = ['STED'] * len(sted_ft)
    synth_len_list = ['Synthetic'] * len(synth_ft)

    df['Modality'] = pd.Series(conf_len_list + sted_len_list + synth_len_list)

    sns.set_theme(style="whitegrid")
    # plt.yscale("log") # For Area feature
    # sns.stripplot(y=df['Modality'], x=df['Feature'])
    # sns.swarmplot(y=df['Modality'], x=df['Feature'])
    sns.boxplot(x=df['Feature'], y=df['Modality'])
    # sns.violinplot(y=df['Modality'], x=df['Feature'])
    # plt.legend(loc='upper right')
    plt.suptitle(feature + ' feature values across multiple modalities in ' + group + ' group for all samples')
    plt.show()

def thresh_vals(feature_list):
    return [val for val in feature_list if val < 30]

def multigrp_plots(modality, feature):
    control_ft = get_features('control', modality, feature)
    climp_ft = get_features('climp', modality, feature)
    rtn_ft = get_features('rtn', modality, feature)
    atl_ft = get_features('atl', modality, feature)

    # print(len(control_ft))

    # control_ft = remove_ones(control_ft)
    # climp_ft = remove_ones(climp_ft)
    # rtn_ft = remove_ones(rtn_ft)

    control_ft = thresh_vals(control_ft)
    climp_ft = thresh_vals(climp_ft)
    rtn_ft = thresh_vals(rtn_ft)
    atl_ft = thresh_vals(atl_ft)

    # print(len(control_ft))

    df = pd.DataFrame()
    Feature_list = control_ft + climp_ft + rtn_ft + atl_ft

    df['Feature'] = pd.Series(Feature_list)

    control_len_list = ['Control'] * len(control_ft)
    climp_len_list = ['Climp'] * len(climp_ft)
    rtn_len_list = ['RTN'] * len(rtn_ft)
    atl_len_list = ['ATL'] * len(atl_ft)

    df['Group'] = pd.Series(control_len_list + climp_len_list + rtn_len_list + atl_len_list)

    sns.set_theme(style="whitegrid")
    # plt.yscale("log") # For Area feature
    # sns.stripplot(y=df['Modality'], x=df['Feature'])
    # sns.swarmplot(y=df['Modality'], x=df['Feature'])

    sns.boxplot(x=df['Feature'], y=df['Group'])

    # sns.kdeplot(data=control_ft, label='Control')
    # sns.kdeplot(data=climp_ft, label='Climp')
    # sns.kdeplot(data=rtn_ft, label='RTN')
    # sns.kdeplot(data=atl_ft, label='ATL')

    # sns.violinplot(y=df['Modality'], x=df['Feature'])
    # plt.legend(loc='upper right')
    plt.xlabel('Area of beads', size=14)
    plt.suptitle(feature + ' feature values across multiple groups in ' + modality + ' modality for all samples (PlosBio data)')
    plt.show()

# multimodal_plots('rtn', 'Area')
# multigrp_plots('STED', 'Area')
multigrp_plots('STED', 'MajorAxisLength')

# from scipy.stats import mannwhitneyu
#
# def stat_tests(group, feature):
#     conf_ft = get_features(group, 'conf', feature)
#     sted_ft = get_features(group, 'sted', feature)
#
#     # print(conf_ft[50:100])
#     # print(sted_ft[100:150])
#
#     print(len(conf_ft))
#     print(len(sted_ft))
#
#     # print(conf_ft[4])
#
#     results = mannwhitneyu(conf_ft, sted_ft)
#     print(results)
#     # print(U1)
#     # print(p)
#
# stat_tests('control', 'Area')
