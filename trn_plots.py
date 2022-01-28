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

# prefix = '/localhome/asa420/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/'
#
# climp_conf_files = glob.glob(prefix + 'climp-features/*-conf.csv')
# climp_sted_files = glob.glob(prefix + 'climp-features/*-sted.csv')
# climp_synth_files = glob.glob(prefix + 'climp-features/*-synth.csv')
#
# exit()

prefix = '/localhome/asa420/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/control-features/'

conf_data_8 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x707_y2014_coverage42-conf.csv')
sted_data_8 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x707_y2014_coverage42-sted.csv')
synth_data_8 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x707_y2014_coverage42-synth.csv')

conf_data_9 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x725_y1383_coverage30-conf.csv')
sted_data_9 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x725_y1383_coverage30-sted.csv')
synth_data_9 = pd.read_csv(prefix + '3_23_2021 Control COS7 Paired STED Decon_Series004_decon_ch02_densePER__x725_y1383_coverage30-synth.csv')

# conf_data_10 = pd.read_csv('/localhome/asa420/MIAL/Series010_decon_ch00.csv')
# sted_data_10 = pd.read_csv('/localhome/asa420/MIAL/Series010_decon_ch01.csv')
#
# conf_data_12 = pd.read_csv('/localhome/asa420/MIAL/Series012_decon_ch00.csv')
# sted_data_12 = pd.read_csv('/localhome/asa420/MIAL/Series012_decon_ch01.csv')
#
# conf_data_13 = pd.read_csv('/localhome/asa420/MIAL/Series013_decon_ch00.csv')
# sted_data_13 = pd.read_csv('/localhome/asa420/MIAL/Series013_decon_ch01.csv')
#
# conf_data_14 = pd.read_csv('/localhome/asa420/MIAL/Series014_decon_ch00.csv')
# sted_data_14 = pd.read_csv('/localhome/asa420/MIAL/Series014_decon_ch01.csv')
#
# conf_data_15 = pd.read_csv('/localhome/asa420/MIAL/Series015_decon_ch00.csv')
# sted_data_15 = pd.read_csv('/localhome/asa420/MIAL/Series015_decon_ch01.csv')
#
# conf_data_16 = pd.read_csv('/localhome/asa420/MIAL/Series016_decon_ch00.csv')
# sted_data_16 = pd.read_csv('/localhome/asa420/MIAL/Series016_decon_ch01.csv')
#
# conf_data_17 = pd.read_csv('/localhome/asa420/MIAL/Series017_decon_ch00.csv')
# sted_data_17 = pd.read_csv('/localhome/asa420/MIAL/Series017_decon_ch01.csv')

# Live cell avg files
# s2 = pd.read_csv('/localhome/asa420/MIAL/Live_Ctrl_STED_Series2_avg.csv')
# s3 = pd.read_csv('/localhome/asa420/MIAL/Live_Ctrl_STED_Series3_avg.csv')
# s4 = pd.read_csv('/localhome/asa420/MIAL/Live_Ctrl_STED_Series4_avg.csv')
# s5 = pd.read_csv('/localhome/asa420/MIAL/Live_Ctrl_STED_Series5_avg.csv')
# s6 = pd.read_csv('/localhome/asa420/MIAL/Live_Ctrl_STED_Series6_avg.csv')
# s7 = pd.read_csv('/localhome/asa420/MIAL/Live_Ctrl_STED_Series7_avg.csv')
# s8 = pd.read_csv('/localhome/asa420/MIAL/Live_Ctrl_STED_Series8_avg.csv')
# s9 = pd.read_csv('/localhome/asa420/MIAL/Live_Ctrl_STED_Series9_avg.csv')

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
