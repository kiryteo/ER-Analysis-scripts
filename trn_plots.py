import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import numpy as np
import pdb

# Features
# Index(['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
#        'Orientation', 'ConvexArea', 'Circularity', 'EulerNumber', 'Perimeter',
#        'MaxFeretDiameter', 'MaxFeretAngle', 'MaxFeretCoordinates_1',
#        'MaxFeretCoordinates_2', 'MaxFeretCoordinates_3',
#        'MaxFeretCoordinates_4', 'MinFeretDiameter', 'MinFeretAngle',
#        'MinFeretCoordinates_1', 'MinFeretCoordinates_2',
#        'MinFeretCoordinates_3', 'MinFeretCoordinates_4'],
#       dtype='object')


prefix = '/localhome/asa420/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/'
control = 'control-features/'
rtn = 'rtn-features/'
climp = 'climp-features/'


def multimodal_plots():
    pass


def multigroup_plots():
    pass


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
    else:
        print("Only control, climp, rtn groups supported.")
        return
    return glob.glob(pref + '*-' + modality + '.csv')


def get_features(group, modality, feature):
    group_modality_files = load_data(group, modality)
    group_modality_feature = []
    for group in group_modality_files:
        group_modality_data = pd.read_csv(group)[feature]
        group_modality_feature += list(group_modality_data)
    return group_modality_feature

Beads_features = ['Area', 'MajorAxisLength', 'Eccentricity', 'MinorAxisLength']

def multimodal_plots(group, feature):
    conf_ft = get_features(group, 'conf', feature)
    sted_ft = get_features(group, 'sted', feature)
    synth_ft = get_features(group, 'synth', feature)

    df = pd.DataFrame()
    Feature_list = conf_ft + sted_ft + synth_ft

    df['Feature'] = pd.Series(Feature_list)
    conf_len_list = ['Confocal'] * len(conf_ft)
    sted_len_list = ['STED'] * len(sted_ft)
    synth_len_list = ['Synthetic'] * len(synth_ft)

    df['Modality'] = pd.Series(conf_len_list + sted_len_list + synth_len_list)

    sns.set_theme(style="whitegrid")
    # plt.yscale("log") # For Area feature
    # sns.stripplot(y=df['Modality'], x=df['Feature'])
    # sns.swarmplot(y=df['Modality'], x=df['Feature'])
    sns.boxplot(y=df['Group'], x=df['Feature'])
    # sns.violinplot(y=df['Modality'], x=df['Feature'])
    # plt.legend(loc='upper right')
    plt.suptitle('Eccentricity feature values across multiple groups in Synthetic STED modality for all samples')
    # plt.suptitle('MajorAxisLength feature values across Synthetic STED modality in RTN group for all samples in replicate 1 Vs. replicate 2')
    plt.show()

def multigrp_plots(modality, feature):
    control_ft = get_features('control', modality, feature)
    climp_ft = get_features('climp', modality, feature)
    rtn_ft = get_features('rtn', modality, feature)

    df = pd.DataFrame()
    Feature_list = pd.Series(control_ft + climp_ft + rtn_ft)

    df['Feature'] = Feature_list

    control_len_list = ['Control'] * len(control_ft)
    climp_len_list = ['Climp'] * len(climp_ft)
    rtn_len_list = ['RTN'] * len(rtn_ft)

    df['Group'] = pd.Series(control_len_list + climp_len_list + rtn_len_list)


def create_plots():
    df = pd.DataFrame()
    # Feature_list = climp_synth_r1_feature + ctrl_synth_r1_feature + rtn_synth_r1_feature

    # Feature_list = conf_climp_r1_feature + sted_climp_r1_feature + synth_climp_r1_feature
    # Feature_list = conf_ctrl_r1_feature + sted_ctrl_r1_feature + synth_ctrl_r1_feature
    Feature_list = conf_rtn_r1_feature + sted_rtn_r1_feature + synth_rtn_r1_feature

    # Feature_list = synth_rtn_r1_feature + synth_rtn_r2_feature
    df['Feature'] = pd.Series(Feature_list)

    # climp_list = ['Climp'] * len(climp_synth_r1_feature)
    # ctrl_list = ['Control'] * len(ctrl_synth_r1_feature)
    # rtn_list = ['RTN'] * len(rtn_synth_r1_feature)

    conf_list = ['Confocal'] * len(conf_rtn_r1_feature)
    sted_list = ['STED'] * len(sted_rtn_r1_feature)
    synth_list = ['Synthetic'] * len(synth_rtn_r1_feature)

    # df['Group'] = pd.Series(climp_list + ctrl_list + rtn_list)
    df['Modality'] = pd.Series(conf_list + sted_list + synth_list)

    df.reset_index()
    sns.set_theme(style="whitegrid")
    # plt.yscale("log") # For Area feature
    # sns.stripplot(y=df['Modality'], x=df['Feature'])
    # sns.swarmplot(y=df['Modality'], x=df['Feature'])
    sns.boxplot(y=df['Group'], x=df['Feature'])
    # sns.violinplot(y=df['Modality'], x=df['Feature'])
    # plt.legend(loc='upper right')
    plt.suptitle('Eccentricity feature values across multiple groups in Synthetic STED modality for all samples')
    # plt.suptitle('MajorAxisLength feature values across Synthetic STED modality in RTN group for all samples in replicate 1 Vs. replicate 2')
    plt.show()




#df = pd.DataFrame()
# Feature_series = pd.concat([climp_sted_r1_feature, ctrl_sted_r1_feature, rtn_sted_r1_feature])
# Feature_list = climp_sted_r1_feature + ctrl_sted_r1_feature + rtn_sted_r1_feature
#Feature_list = climp_synth_r1_feature + ctrl_synth_r1_feature + rtn_synth_r1_feature

# Feature_list = conf_climp_r1_feature + sted_climp_r1_feature + synth_climp_r1_feature
# Feature_list = conf_ctrl_r1_feature + sted_ctrl_r1_feature + synth_ctrl_r1_feature
# Feature_list = conf_rtn_r1_feature + sted_rtn_r1_feature + synth_rtn_r1_feature

# Feature_list = synth_rtn_r1_feature + synth_rtn_r2_feature
#df['Feature'] = pd.Series(Feature_list)

# climp_list = ['Climp'] * len(climp_synth_r1_feature)
# ctrl_list = ['Control'] * len(ctrl_synth_r1_feature)
# rtn_list = ['RTN'] * len(rtn_synth_r1_feature)

# conf_list = ['Confocal'] * len(conf_rtn_r1_feature)
# sted_list = ['STED'] * len(sted_rtn_r1_feature)
# synth_list = ['Synthetic'] * len(synth_rtn_r1_feature)

# r1list = ['replicate 1'] * len(synth_rtn_r1_feature)
# r2list = ['replicate 2'] * len(synth_rtn_r2_feature)

# df['Group'] = pd.Series(climp_list + ctrl_list + rtn_list)
# df['Modality'] = pd.Series(conf_list + sted_list + synth_list)
# df['Replicate'] = pd.Series(r1list + r2list)

# df.reset_index()
#
# sns.set_theme(style="whitegrid")
# plt.yscale("log") # For Area feature
# sns.stripplot(y=df['Modality'], x=df['Feature'])
# sns.swarmplot(y=df['Modality'], x=df['Feature'])
# sns.boxplot(y=df['Group'], x=df['Feature'])
# sns.violinplot(y=df['Modality'], x=df['Feature'])
# plt.legend(loc='upper right')
# plt.suptitle('Eccentricity feature values across multiple groups in Synthetic STED modality for all samples')
# plt.suptitle('MajorAxisLength feature values across Synthetic STED modality in RTN group for all samples in replicate 1 Vs. replicate 2')
# plt.show()

# exit()
# conf_data_8 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x707_y2014_coverage42-conf.csv')
# sted_data_8 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x707_y2014_coverage42-sted.csv')
# synth_data_8 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x707_y2014_coverage42-synth.csv')


# ecc_conf = conf_data_8['Eccentricity']
# ecc_sted = sted_data_8['Eccentricity']
# ecc_synth = synth_data_8['Eccentricity']
#
# # pdb.set_trace()
#
# Eccentricity_series = pd.concat([ecc_conf, ecc_sted, ecc_synth])
#
# df['Eccentricity'] = Eccentricity_series
# df['Modality'] = 'Confocal'
#
# df.reset_index()
#
# df['Modality'].iloc[106:212] = 'STED'
# df['Modality'].iloc[212:] = 'Synthetic STED'
#
# df.reset_index()
# # df['label'] = 'Conf' * len(df['conf']) + 'STED' * len(df['sted']) + 'Synth' * len(df['synth'])
# sns.set_theme(style="whitegrid")
# sns.swarmplot(x=df['Modality'], y=df['Eccentricity'])
# plt.suptitle('#Feature across modalities for sample - ...')
# plt.show()
#
# exit()
#
# # sns.swarmplot(x=df['Eccentricity'],y)
# plt.show()
#
# exit()
#
# conf_data_9 = pd.read_csv(
#     prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x725_y1383_coverage30-conf.csv')
# sted_data_9 = pd.read_csv(
#     prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x725_y1383_coverage30-sted.csv')
# synth_data_9 = pd.read_csv(
#     prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x725_y1383_coverage30-synth.csv')
#
# sns.set_theme(style="whitegrid")
#
# fig, axes = plt.subplots(1, 3)
#
# sns.swarmplot(x=conf_data_8['Eccentricity'], color='red', label='Confocal', ax=axes[0], size=4)
# axes[0].set_title('Series004_x707_y2014', size=10)
# axes[0].set_xlabel('Eccentricity-Confocal', fontsize=10)
# axes[0].legend(loc='upper right')
#
# sns.swarmplot(x=sted_data_8['Eccentricity'], color='blue', label='STED', ax=axes[1], size=4)
# axes[1].set_title('Series004_x707_y2014', size=10)
# axes[1].set_xlabel('Eccentricity-STED', fontsize=10)
# axes[1].legend(loc='upper right')
#
# sns.swarmplot(x=synth_data_8['Eccentricity'], color='green', label='Synth', ax=axes[2], size=4)
# axes[2].set_title('Series004_x707_y2014', size=10)
# axes[2].set_xlabel('Eccentricity-Synth', fontsize=10)
# axes[2].legend(loc='upper right')

# sns.swarmplot(x=conf_data_9['Eccentricity'], color='red', label='Confocal', ax=axes[1], size=4)
# sns.swarmplot(x=sted_data_9['Eccentricity'], color='blue', label='STED', ax=axes[1], size=4)
# sns.swarmplot(x=synth_data_9['Eccentricity'], color='green', label='Synth', ax=axes[1], size=4)
# axes[1].set_title('Series004_x725_y1383', size=10)
# axes[1].set_xlabel('Eccentricity', fontsize=10)
# axes[1].legend(loc='upper right')

# sns.stripplot(x=conf_data_10['Perimeter'], color='red', label='Confocal', ax=axes[0,1], size=1)
# sns.stripplot(x=sted_data_10['Perimeter'], color='blue', label='STED', ax=axes[0,1], size=1)
# axes[0,1].set_title('Series010_decon', size=10)
# axes[0,1].set_xlabel('Perimeter', fontsize=10)
# axes[0,1].legend(loc='upper right')
#
# sns.stripplot(x=conf_data_12['Perimeter'], color='red', label='Confocal', ax=axes[0,2], size=1)
# sns.stripplot(x=sted_data_12['Perimeter'], color='blue', label='STED', ax=axes[0,2], size=1)
# axes[0,2].set_title('Series012_decon', size=10)
# axes[0,2].set_xlabel('Perimeter', fontsize=10)
# axes[0,2].legend(loc='upper right')
#
# sns.stripplot(x=conf_data_13['Perimeter'], color='red', label='Confocal', ax=axes[0,3], size=1)
# sns.stripplot(x=sted_data_13['Perimeter'], color='blue', label='STED', ax=axes[0,3], size=1)
# axes[0,3].set_title('Series013_decon', size=10)
# axes[0,3].set_xlabel('Perimeter', fontsize=10)
# axes[0,3].legend(loc='upper right')
#
# sns.stripplot(x=conf_data_14['Perimeter'], color='red', label='Confocal', ax=axes[1,0], size=1)
# sns.stripplot(x=sted_data_14['Perimeter'], color='blue', label='STED', ax=axes[1,0], size=1)
# axes[1,0].set_title('Series014_decon', size=10)
# axes[1,0].set_xlabel('Perimeter', fontsize=10)
# axes[1,0].legend(loc='upper right')
#
# sns.stripplot(x=conf_data_15['Perimeter'], color='red', label='Confocal', ax=axes[1,1], size=1)
# sns.stripplot(x=sted_data_15['Perimeter'], color='blue', label='STED', ax=axes[1,1], size=1)
# axes[1,1].set_title('Series015_decon', size=10)
# axes[1,1].set_xlabel('Perimeter', fontsize=10)
# axes[1,1].legend(loc='upper right')
#
# sns.stripplot(x=conf_data_16['Perimeter'], color='red', label='Confocal', ax=axes[1,2], size=1)
# sns.stripplot(x=sted_data_16['Perimeter'], color='blue', label='STED', ax=axes[1,2], size=1)
# axes[1,2].set_title('Series016_decon', size=10)
# axes[1,2].set_xlabel('Perimeter', fontsize=10)
# axes[1,2].legend(loc='upper right')
#
# sns.stripplot(x=conf_data_17['Perimeter'], color='red', label='Confocal', ax=axes[1,3], size=1)
# sns.stripplot(x=sted_data_17['Perimeter'], color='blue', label='STED', ax=axes[1,3], size=1)
# axes[1,3].set_title('Series017_decon', size=10)
# axes[1,3].set_xlabel('Perimeter', fontsize=10)
# axes[1,3].legend(loc='upper right')


# plt.suptitle('Control (group), Confocal, STED and Synthetic STED (mod), Eccentricity (feature)')
# # plt.suptitle('Control (group), STED and Synthetic STED (mod), Eccentricity (feature) across multiple samples')
# plt.show()

# fig = ax.get_figure()
# fig.savefig('Series017_decon.png', size=(20, 16))
# plt.close()
