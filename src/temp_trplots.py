import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import numpy as np
import pdb
import os

home = os.path.expanduser('~')

# Features
# Index(['Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
#        'Orientation', 'ConvexArea', 'Circularity', 'EulerNumber', 'Perimeter',
#        'MaxFeretDiameter', 'MaxFeretAngle', 'MaxFeretCoordinates_1',
#        'MaxFeretCoordinates_2', 'MaxFeretCoordinates_3',
#        'MaxFeretCoordinates_4', 'MinFeretDiameter', 'MinFeretAngle',
#        'MinFeretCoordinates_1', 'MinFeretCoordinates_2',
#        'MinFeretCoordinates_3', 'MinFeretCoordinates_4'],
#       dtype='object')

# Fixed cell data

# prefix = '/localhome/asa420/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/'
#
# climp_conf_files = glob.glob(prefix + 'climp-features/*-conf.csv')
# climp_sted_files = glob.glob(prefix + 'climp-features/*-sted.csv')
# climp_synth_files = glob.glob(prefix + 'climp-features/*-synth.csv')
#
# exit()

#Climp replicates -
#Control

prefix = home + '/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/'
control = 'control-features/'
rtn = 'rtn-features/'
climp = 'climp-features/'

prefix_ctrl_r1 = prefix + control + 'Series004/'
# prefix_ctrl_r2 = prefix + control + 'Series009/'

prefix_climp_r1 = prefix + climp + 'Series003/'
# prefix_climp_r2 = prefix + climp + 'Series012/'

prefix_rtn_r1 = prefix + rtn + 'Series003/'
# prefix_rtn_r2 = prefix + rtn + 'Series006/'

############ Replicate 1 analysis ##########

def load_data(group):
    data = []
    if group == 'control':
        data.extend((glob.glob(prefix_ctrl_r1 + '*-conf.csv'), glob.glob(prefix_ctrl_r1 + '*-sted.csv'), glob.glob(prefix_ctrl_r1 + '*-synth.csv')))

            # ctrl_r1_conf_files = glob.glob(prefix_ctrl_r1 + '*-conf.csv')
            # ctrl_r1_sted_files = glob.glob(prefix_ctrl_r1 + '*-sted.csv')
            # ctrl_r1_synth_files = glob.glob(prefix_ctrl_r1 + '*-synth.csv')
    elif group == 'climp':
        data.extend((glob.glob(prefix_climp_r1 + '*-conf.csv'), glob.glob(prefix_climp_r1 + '*-sted.csv'), glob.glob(prefix_climp_r1 + '*-synth.csv')))

            # climp_r1_conf_files = glob.glob(prefix_climp_r1 + '*-conf.csv')
            # climp_r1_sted_files = glob.glob(prefix_climp_r1 + '*-sted.csv')
            # climp_r1_synth_files = glob.glob(prefix_climp_r1 + '*-synth.csv')
    elif group=='rtn':
        data.extend((glob.glob(prefix_rtn_r1 + '*-conf.csv'), glob.glob(prefix_rtn_r1 + '*-sted.csv'), glob.glob(prefix_rtn_r1 + '*-synth.csv')))

            # rtn_r1_conf_files = glob.glob(prefix_rtn_r1 + '*-conf.csv')
            # rtn_r1_sted_files = glob.glob(prefix_rtn_r1 + '*-sted.csv')
            # rtn_r1_synth_files = glob.glob(prefix_rtn_r1 + '*-synth.csv')
    else:
        print("Currently supported groups: control, climp and rtn. Exiting now.")
        exit()
    return

def multigroup_plots(feature):
    climp_conf_r1_feature = []
    ctrl_conf_r1_feature = []
    rtn_conf_r1_feature = []

    climp_sted_r1_feature = []
    ctrl_sted_r1_feature = []
    rtn_sted_r1_feature = []

    climp_synth_r1_feature = []
    ctrl_synth_r1_feature = []
    rtn_synth_r1_feature = []

    for climp, ctrl, rtn in zip(climp_r1_conf_files, ctrl_r1_conf_files, rtn_r1_conf_files):
        climp_conf_data = pd.read_csv(climp)[feature]
        ctrl_conf_data = pd.read_csv(ctrl)[feature]
        rtn_conf_data = pd.read_csv(rtn)[feature]

        climp_conf_r1_feature += list(climp_conf_data)
        ctrl_conf_r1_feature += list(ctrl_conf_data)
        rtn_conf_r1_feature += list(rtn_conf_data)

    for climp, ctrl, rtn in zip(climp_r1_sted_files, ctrl_r1_sted_files, rtn_r1_sted_files):
        climp_sted_data = pd.read_csv(climp)[feature]
        ctrl_sted_data = pd.read_csv(ctrl)[feature]
        rtn_sted_data = pd.read_csv(rtn)[feature]

        climp_sted_r1_feature += list(climp_sted_data)
        ctrl_sted_r1_feature += list(ctrl_sted_data)
        rtn_sted_r1_feature += list(rtn_sted_data)

    for climp, ctrl, rtn in zip(climp_r1_synth_files, ctrl_r1_synth_files, rtn_r1_synth_files):
        climp_synth_data = pd.read_csv(climp)[feature]
        ctrl_synth_data = pd.read_csv(ctrl)[feature]
        rtn_synth_data = pd.read_csv(rtn)[feature]

        climp_synth_r1_feature += list(climp_synth_data)
        ctrl_synth_r1_feature += list(ctrl_synth_data)
        rtn_synth_r1_feature += list(rtn_synth_data)

# def get_features(group, modality, feature):


conf_climp_r1_feature = []
conf_ctrl_r1_feature = []
conf_rtn_r1_feature = []

sted_climp_r1_feature = []
sted_ctrl_r1_feature = []
sted_rtn_r1_feature = []

synth_climp_r1_feature = []
synth_ctrl_r1_feature = []
synth_rtn_r1_feature = []

for conf, sted, synth in zip(climp_r1_conf_files, climp_r1_sted_files, climp_r1_synth_files):
    conf_data = pd.read_csv(conf)['MajorAxisLength']
    sted_data = pd.read_csv(sted)['MajorAxisLength']
    synth_data = pd.read_csv(synth)['MajorAxisLength']

    conf_climp_r1_feature += list(conf_data)
    sted_climp_r1_feature += list(sted_data)
    synth_climp_r1_feature += list(synth_data)

for conf, sted, synth in zip(ctrl_r1_conf_files, ctrl_r1_sted_files, ctrl_r1_synth_files):
    conf_data = pd.read_csv(conf)['MajorAxisLength']
    sted_data = pd.read_csv(sted)['MajorAxisLength']
    synth_data = pd.read_csv(synth)['MajorAxisLength']

    conf_ctrl_r1_feature += list(conf_data)
    sted_ctrl_r1_feature += list(sted_data)
    synth_ctrl_r1_feature += list(synth_data)

for conf, sted, synth in zip(rtn_r1_conf_files, rtn_r1_sted_files, rtn_r1_synth_files):
    conf_data = pd.read_csv(conf)['MajorAxisLength']
    sted_data = pd.read_csv(sted)['MajorAxisLength']
    synth_data = pd.read_csv(synth)['MajorAxisLength']

    conf_rtn_r1_feature += list(conf_data)
    sted_rtn_r1_feature += list(sted_data)
    synth_rtn_r1_feature += list(synth_data)


ctrl_r2_conf_files = glob.glob(prefix_ctrl_r2 + '*-conf.csv')
ctrl_r2_sted_files = glob.glob(prefix_ctrl_r2 + '*-sted.csv')
ctrl_r2_synth_files = glob.glob(prefix_ctrl_r2 + '*-synth.csv')

climp_r2_conf_files = glob.glob(prefix_climp_r2 + '*-conf.csv')
climp_r2_sted_files = glob.glob(prefix_climp_r2 + '*-sted.csv')
climp_r2_synth_files = glob.glob(prefix_climp_r2 + '*-synth.csv')

rtn_r2_conf_files = glob.glob(prefix_rtn_r2 + '*-conf.csv')
rtn_r2_sted_files = glob.glob(prefix_rtn_r2 + '*-sted.csv')
rtn_r2_synth_files = glob.glob(prefix_rtn_r2 + '*-synth.csv')

conf_climp_r2_feature = []
conf_ctrl_r2_feature = []
conf_rtn_r2_feature = []

sted_climp_r2_feature = []
sted_ctrl_r2_feature = []
sted_rtn_r2_feature = []

synth_climp_r2_feature = []
synth_ctrl_r2_feature = []
synth_rtn_r2_feature = []

for r1synth, r2synth in zip(climp_r1_synth_files, climp_r2_synth_files):
    r1data = pd.read_csv(r1synth)['MajorAxisLength']
    r2data = pd.read_csv(r2synth)['MajorAxisLength']

    synth_climp_r1_feature += list(r1data)
    synth_climp_r2_feature += list(r2data)

for r1conf, r2conf in zip(ctrl_r1_synth_files, ctrl_r2_synth_files):
    r1data = pd.read_csv(r1conf)['MajorAxisLength']
    r2data = pd.read_csv(r2conf)['MajorAxisLength']

    synth_ctrl_r1_feature += list(r1data)
    synth_ctrl_r2_feature += list(r2data)

for r1conf, r2conf in zip(rtn_r1_synth_files, rtn_r2_synth_files):
    r1data = pd.read_csv(r1conf)['MajorAxisLength']
    r2data = pd.read_csv(r2conf)['MajorAxisLength']

    synth_rtn_r1_feature += list(r1data)
    synth_rtn_r2_feature += list(r2data)

# print(np.count_nonzero(climp_synth_r1_feature))
# print(np.count_nonzero(ctrl_synth_r1_feature))
# print(np.count_nonzero(rtn_synth_r1_feature))

# print(sum(k > 10 for k in climp_synth_r1_feature))
# print(sum(k > 10 for k in ctrl_synth_r1_feature))
# print(sum(k > 10 for k in rtn_synth_r1_feature))

# exit()

df = pd.DataFrame()
# Feature_series = pd.concat([climp_sted_r1_feature, ctrl_sted_r1_feature, rtn_sted_r1_feature])
# Feature_list = climp_sted_r1_feature + ctrl_sted_r1_feature + rtn_sted_r1_feature

Feature_list = conf_rtn_r1_feature + sted_rtn_r1_feature + synth_rtn_r1_feature
Feature_list = conf_rtn_r1_feature + sted_rtn_r1_feature + synth_rtn_r1_feature

# Feature_list = synth_rtn_r1_feature + synth_rtn_r2_feature
df['Feature'] = pd.Series(Feature_list)

conf_list = ['Confocal'] * len(conf_rtn_r1_feature)
sted_list = ['STED'] * len(sted_rtn_r1_feature)
synth_list = ['Synthetic'] * len(synth_rtn_r1_feature)

# r1list = ['replicate 1'] * len(synth_rtn_r1_feature)
# r2list = ['replicate 2'] * len(synth_rtn_r2_feature)

df['Modality'] = pd.Series(conf_list + sted_list + synth_list)
# df['Replicate'] = pd.Series(r1list + r2list)

df.reset_index()

sns.set_theme(style="whitegrid")
plt.yscale("log") # For Area feature
sns.stripplot(x=df['Modality'], y=df['Feature'])
# plt.legend(loc='upper right')
plt.suptitle('MajorAxisLength feature values across multiple modalities in RTN group for all samples in replicate 1')
# plt.suptitle('MajorAxisLength feature values across Synthetic STED modality in RTN group for all samples in replicate 1 Vs. replicate 2')
plt.show()

exit()
# conf_data_8 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x707_y2014_coverage42-conf.csv')
# sted_data_8 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x707_y2014_coverage42-sted.csv')
# synth_data_8 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x707_y2014_coverage42-synth.csv')


ecc_conf = conf_data_8['MaxFeretDiameter']
ecc_sted = sted_data_8['Eccentricity']
ecc_synth = synth_data_8['Eccentricity']

# pdb.set_trace()

Eccentricity_series = pd.concat([ecc_conf, ecc_sted, ecc_synth])

df['Eccentricity'] = Eccentricity_series
df['Modality'] = 'Confocal'

df.reset_index()

df['Modality'].iloc[106:212] = 'STED'
df['Modality'].iloc[212:] = 'Synthetic STED'

df.reset_index()
#df['label'] = 'Conf' * len(df['conf']) + 'STED' * len(df['sted']) + 'Synth' * len(df['synth'])
sns.set_theme(style="whitegrid")
sns.swarmplot(x=df['Modality'],y=df['Eccentricity'])
plt.suptitle('#Feature across modalities for sample - ...')
plt.show()

exit()

# sns.swarmplot(x=df['Eccentricity'],y)
plt.show()


exit()

conf_data_9 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x725_y1383_coverage30-conf.csv')
sted_data_9 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x725_y1383_coverage30-sted.csv')
synth_data_9 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x725_y1383_coverage30-synth.csv')


sns.set_theme(style="whitegrid")



fig, axes = plt.subplots(1, 3)

sns.swarmplot(x=conf_data_8['Eccentricity'], color='red', label='Confocal', ax=axes[0], size=4)
axes[0].set_title('Series004_x707_y2014', size=10)
axes[0].set_xlabel('Eccentricity-Confocal', fontsize=10)
axes[0].legend(loc='upper right')

sns.swarmplot(x=sted_data_8['Eccentricity'], color='blue', label='STED', ax=axes[1], size=4)
axes[1].set_title('Series004_x707_y2014', size=10)
axes[1].set_xlabel('Eccentricity-STED', fontsize=10)
axes[1].legend(loc='upper right')

sns.swarmplot(x=synth_data_8['Eccentricity'], color='green', label='Synth', ax=axes[2], size=4)
axes[2].set_title('Series004_x707_y2014', size=10)
axes[2].set_xlabel('Eccentricity-Synth', fontsize=10)
axes[2].legend(loc='upper right')

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


plt.suptitle('Control (group), Confocal, STED and Synthetic STED (mod), Eccentricity (feature)')
#plt.suptitle('Control (group), STED and Synthetic STED (mod), Eccentricity (feature) across multiple samples')
plt.show()

# fig = ax.get_figure()
# fig.savefig('Series017_decon.png', size=(20, 16))
# plt.close()
