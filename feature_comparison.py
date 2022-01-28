import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

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

prefix = '/localhome/asa420/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/'

climp_conf_files = glob.glob(prefix + 'climp-features/*-conf.csv')
climp_sted_files = glob.glob(prefix + 'climp-features/*-sted.csv')
climp_synth_files = glob.glob(prefix + 'climp-features/*-synth.csv')

control_conf_files = glob.glob(prefix + 'control-features/*-conf.csv')
control_sted_files = glob.glob(prefix + 'control-features/*-sted.csv')
control_synth_files = glob.glob(prefix + 'control-features/*-synth.csv')

rtn_conf_files = glob.glob(prefix + 'rtn-features/*-conf.csv')
rtn_sted_files = glob.glob(prefix + 'rtn-features/*-sted.csv')
rtn_synth_files = glob.glob(prefix + 'rtn-features/*-synth.csv')

climp_conf_Area = []
control_conf_Area = []
rtn_conf_Area = []

climp_sted_Area = []
control_sted_Area = []
rtn_sted_Area = []

climp_synth_Area = []
control_synth_Area = []
rtn_synth_Area = []


for climp, control, rtn in zip(climp_conf_files, control_conf_files, rtn_conf_files):
    climp_conf_data_Area = pd.read_csv(climp)['Area']
    control_conf_data_Area = pd.read_csv(control)['Area']
    rtn_conf_data_Area = pd.read_csv(rtn)['Area']

    climp_conf_Area += list(climp_conf_data_Area)
    control_conf_Area += list(control_conf_data_Area)
    rtn_conf_Area += list(rtn_conf_data_Area)

for climp, control, rtn in zip(climp_sted_files, control_sted_files, rtn_sted_files):
    climp_sted_data_Area = pd.read_csv(climp)['Area']
    control_sted_data_Area = pd.read_csv(control)['Area']
    rtn_sted_data_Area = pd.read_csv(rtn)['Area']

    climp_sted_Area += list(climp_sted_data_Area)
    control_sted_Area += list(control_sted_data_Area)
    rtn_sted_Area += list(rtn_sted_data_Area)

for climp, control, rtn in zip(climp_synth_files, control_synth_files, rtn_synth_files):
    climp_synth_data_Area = pd.read_csv(climp)['Area']
    control_synth_data_Area = pd.read_csv(control)['Area']
    rtn_synth_data_Area = pd.read_csv(rtn)['Area']

    climp_synth_Area += list(climp_synth_data_Area)
    control_synth_Area += list(control_synth_data_Area)
    rtn_synth_Area += list(rtn_synth_data_Area)


sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(3, 3)

sns.stripplot(x=climp_conf_Area, color='red', label='Confocal', ax=axes[0,0])
axes[0,0].set_title('Climp-Confocal', size=10)
#axes[0,0].set_xlabel('Area', fontsize=10)
axes[0,0].legend(loc='upper right')

sns.stripplot(x=climp_sted_Area, color='blue', label='STED', ax=axes[0,1])
axes[0,1].set_title('Climp-STED', size=10)
#axes[0,1].set_xlabel('Area', fontsize=10)
axes[0,1].legend(loc='upper right')

sns.stripplot(x=climp_synth_Area, color='green', label='Synth', ax=axes[0,2])
axes[0,2].set_title('Climp-Synthetic', size=10)
#axes[0,2].set_xlabel('Area', fontsize=10)
axes[0,2].legend(loc='upper right')

sns.stripplot(x=control_conf_Area, color='red', label='Confocal', ax=axes[1,0])
axes[1,0].set_title('Control-Confocal', size=10)
#axes[1,0].set_xlabel('Area', fontsize=10)
axes[1,0].legend(loc='upper right')

sns.stripplot(x=control_sted_Area, color='blue', label='STED', ax=axes[1,1])
axes[1,1].set_title('Control-STED', size=10)
#axes[1,1].set_xlabel('Area', fontsize=10)
axes[1,1].legend(loc='upper right')

sns.stripplot(x=control_synth_Area, color='green', label='Synth', ax=axes[1,2])
axes[1,2].set_title('Control-Synthetic', size=10)
#axes[1,2].set_xlabel('Area', fontsize=10)
axes[1,2].legend(loc='upper right')

sns.stripplot(x=rtn_conf_Area, color='red', label='Confocal', ax=axes[2,0])
axes[2,0].set_title('RTN-Confocal', size=10)
#axes[2,0].set_xlabel('Area', fontsize=10)
axes[2,0].legend(loc='upper right')

sns.stripplot(x=rtn_sted_Area, color='blue', label='STED', ax=axes[2,1])
axes[2,1].set_title('RTN-STED', size=10)
#axes[2,1].set_xlabel('Area', fontsize=10)
axes[2,1].legend(loc='upper right')

sns.stripplot(x=rtn_synth_Area, color='green', label='Synth', ax=axes[2,2])
axes[2,2].set_title('RTN-Synthetic', size=10)
#axes[2,2].set_xlabel('Area', fontsize=10)
axes[2,2].legend(loc='upper right')

plt.suptitle('Area feature comparison across Climp, Control, RTN groups with Confocal, STED and Synthetic STED modality for all samples')
plt.show()

# sns.swarmplot(x=conf_data_8['Eccentricity'], color='red', label='Confocal', ax=axes[0], size=4)
# sns.swarmplot(x=sted_data_8['Eccentricity'], color='blue', label='STED', ax=axes[0], size=4)
# sns.swarmplot(x=synth_data_8['Eccentricity'], color='green', label='Synth', ax=axes[0], size=4)
# axes[0].set_title('Series004_x707_y2014', size=10)
# axes[0].set_xlabel('Eccentricity', fontsize=10)
# axes[0].legend(loc='upper right')

exit()

prefix = '/localhome/asa420/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/control-features/'

conf_data_8 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x707_y2014_coverage42-conf.csv')
sted_data_8 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x707_y2014_coverage42-sted.csv')
synth_data_8 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x707_y2014_coverage42-synth.csv')

conf_data_9 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x725_y1383_coverage30-conf.csv')
sted_data_9 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x725_y1383_coverage30-sted.csv')
synth_data_9 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x725_y1383_coverage30-synth.csv')

sns.set_theme(style="whitegrid")

# MFC3
# ax = sns.swarmplot(x=conf_data['MinFeretCoordinates_3'], color='red', label='Confocal')
# ax = sns.swarmplot(x=sted_data['MinFeretCoordinates_3'], color='blue', label='STED')

# Area
# ax = sns.swarmplot(x=conf_data_14['Perimeter'], color='red', label='Confocal', size=1)
# ax = sns.swarmplot(x=sted_data_14['Perimeter'], color='blue', label='STED', size=1)
#
# plt.legend(loc='upper right')
# plt.title('Series014_decon Perimeter')
# plt.show()

# exit()

fig, axes = plt.subplots(1, 2)

sns.swarmplot(x=conf_data_8['Eccentricity'], color='red', label='Confocal', ax=axes[0], size=4)
sns.swarmplot(x=sted_data_8['Eccentricity'], color='blue', label='STED', ax=axes[0], size=4)
sns.swarmplot(x=synth_data_8['Eccentricity'], color='green', label='Synth', ax=axes[0], size=4)
axes[0].set_title('Series004_x707_y2014', size=10)
axes[0].set_xlabel('Eccentricity', fontsize=10)
axes[0].legend(loc='upper right')

sns.swarmplot(x=conf_data_9['Eccentricity'], color='red', label='Confocal', ax=axes[1], size=4)
sns.swarmplot(x=sted_data_9['Eccentricity'], color='blue', label='STED', ax=axes[1], size=4)
sns.swarmplot(x=synth_data_9['Eccentricity'], color='green', label='Synth', ax=axes[1], size=4)
axes[1].set_title('Series004_x725_y1383', size=10)
axes[1].set_xlabel('Eccentricity', fontsize=10)
axes[1].legend(loc='upper right')

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


#plt.suptitle('Control (group), Confocal, STED and Synthetic STED (mod), Eccentricity (feature) across multiple samples')
plt.suptitle('Control (group), STED and Synthetic STED (mod), Eccentricity (feature) across multiple samples')
plt.show()

# fig = ax.get_figure()
# fig.savefig('Series017_decon.png', size=(20, 16))
# plt.close()
