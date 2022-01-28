import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob

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

df = pd.DataFrame()
ecc_conf = conf_data_8['Eccentricity']
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
