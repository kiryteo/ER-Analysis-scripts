
import matplotlib.pyplot as plt
import numpy as np

# Load the images
# img1 = plt.imread('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/roi/Control/Control_6_edge_graph_projection_1.png')

img1 = plt.imread('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/graphs/ATL_4_edge_graph_projection.png')


img2 = plt.imread('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/graphs/ATL_4_edge_graph_projection_updated.png')

# Create the figure and axes
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

plt.axis('off')

# Display the first image
axs[0].imshow(img1)
# axs[0].set_title('Graph RoI')
axs[0].set_title('Initial graph')
axs[0].axis('off')
# Display the second image
axs[1].imshow(img2)
axs[1].set_title('Updated graph with near node connections')
axs[1].axis('off')
# Show the plot

plt.savefig('ATL_4_updated', bbox_inches='tight', pad_inches=0.1)
# plt.show()
plt.close()

exit()

import imageio
import numpy as np
import cv2

er_mean = np.zeros((128, 128))
for i in range(100):
    img = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A4_decon_t0{i:02d}_ch00.tif')
    img = (img - img.min())/(img.max() - img.min())
    er_mean += img

cv2.imwrite('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/er_mean/atl4_er_mean.png', (er_mean/100)*255)
exit()


import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt

# Load example data
tips = sns.load_dataset('tips')

# Create violin plot
ax = sns.violinplot(data=tips, x='day', y='total_bill')

pairs = [('Thur', 'Fri'), ('Fri', 'Sat'), ('Sat', 'Sun')]

# Add annotations
annotator = Annotator(ax=ax, data=tips, pairs=pairs, x='day', y='total_bill')
annotator.apply_test(hue='sex', test='Mann-Whitney', text_format='full', loc='outside')

# Show plot
plt.show()

exit()

import seaborn as sns


import matplotlib.pyplot as plt
from statannot import add_stat_annotation

# Load example data
tips = sns.load_dataset("tips")

# Create violin plot using catplot
g = sns.catplot(x="day", y="total_bill", kind="violin", data=tips)

# Add statistical annotations
test_results = add_stat_annotation(g.ax, x="day", y="total_bill", data=tips,
                                   box_pairs=[("Thur", "Fri"), ("Thur", "Sat"), ("Fri", "Sun")],
                                   test="Mann-Whitney", text_format="star",
                                   loc="inside", verbose=2)

# Add significance bars on the y-axis
ylim = g.ax.get_ylim()
for i, test_result in enumerate(test_results):
    y, p = test_result['pval']
    if p < 0.05:
        g.ax.plot([i-0.2, i+0.2], [ylim[1]-0.05*(ylim[1]-ylim[0]), ylim[1]-0.05*(ylim[1]-ylim[0])], lw=1.5, color='black')
        g.ax.text(i, ylim[1]-0.1*(ylim[1]-ylim[0]), "*", ha='center', va='center', color='white', fontweight='bold')

# Show plot
plt.show()

exit()


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

corr = ['RTN-Control', 'RTN-Control', 'ATL-Climp', 'RTN-Control', 'ATL-Climp', 'Climp-RTN', 'ATL-Climp', 'Climp-RTN', 'Climp-RTN', 'Climp-Control', 'Climp-Control', 'Climp-Control', 'ATL-RTN', 'ATL-RTN', 'ATL-RTN', 'ATL-Control', 'ATL-Control', 'ATL-Control']
Replicate = [3,2,3,1,1,1,2,2,3,3,1,2,1,2,3,3,1,2]
P_val = [3.597e-170, 9.346e-10, 1.069e-103, 3.027e-236, 1.286e-118, 4.904e-86, 1.09e-110, 8.106e-27, 4.131e-24, 6.354e-79, 0.0, 2.607e-55, 0.2709, 1.889e-42, 6.697e-35, 2.149e-295, 1.823e-238, 3.511e-07]
U_stat = [109500000.0, 77080000.0, 56210000.0, 60050000.0, 101900000.0, 54090000.0, 59810000.0, 68350000.0, 53920000.0, 147600000.0, 47640000.0, 46210000.0, 92260000.0, 119000000.0, 36590000.0, 94830000.0, 84080000.0, 80980000.0]

# create a dictionary of values to use in the heatmap
data = {'corr': corr,
        'Replicate': Replicate,
        'P_val': P_val,
        'U_stat': U_stat}

# create a pandas dataframe from the dictionary
df = pd.DataFrame(data)

# pivot the dataframe to create a matrix of values for the heatmap
heatmap_df = df.pivot(index='corr', columns='Replicate', values='P_val')

# create the heatmap using both P values and U_stat values
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_df, annot=True, cmap='coolwarm', cbar_kws={'label': 'P value'}, fmt='.2g', linewidths=.5)
# sns.heatmap(df.pivot(index='corr', columns='Replicate', values='U_stat'), annot=True, cmap='YlOrRd', cbar_kws={'label': 'U stat'}, fmt='.2g', linewidths=.5)
plt.title('Heatmap of P values and U stats for Replicates')
plt.show()

exit()



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a dictionary to hold the data
data = {'corr': ['RTN-Control', 'RTN-Control', 'ATL-Climp', 'RTN-Control', 'ATL-Climp', 'Climp-RTN', 'ATL-Climp', 'Climp-RTN', 'Climp-RTN', 'Climp-Control', 'Climp-Control', 'Climp-Control', 'ATL-RTN', 'ATL-RTN', 'ATL-RTN', 'ATL-Control', 'ATL-Control', 'ATL-Control'],
        'Replicate': [3,2,3,1,1,1,2,2,3,3,1,2,1,2,3,3,1,2],
        'P_val': [3.597e-170, 9.346e-10, 1.069e-103, 3.027e-236, 1.286e-118, 4.904e-86, 1.09e-110, 8.106e-27, 4.131e-24, 6.354e-79, 0.0, 2.607e-55, 0.2709, 1.889e-42, 6.697e-35, 2.149e-295, 1.823e-238, 3.511e-07],
        'U_stat': [109500000.0, 77080000.0, 56210000.0, 60050000.0, 101900000.0, 54090000.0, 59810000.0, 68350000.0, 53920000.0, 147600000.0, 47640000.0, 46210000.0, 92260000.0, 119000000.0, 36590000.0, 94830000.0, 84080000.0, 80980000.0]}

# Convert the dictionary to a pandas dataframe
df = pd.DataFrame(data)

# Create a pivot table to summarize the data
table = df.pivot_table(values='P_val', index='corr', columns='Replicate')

# Create the heatmap using seaborn
sns.heatmap(table, cmap='coolwarm', annot=True, fmt='.2g', cbar=False)
plt.show()

exit()


# text = '''R3_RTN v.s. R3_Control: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=5.313e-49 U_stat=8.101e+07
# R2_RTN v.s. R2_Control: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=7.494e-60 U_stat=6.437e+07
# R3_ATL v.s. R3_Climp: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=1.330e-10 U_stat=5.023e+07
# R1_RTN v.s. R1_Control: Mann-Whitney-Wilcoxon test two-sided with Bonferroni correction, P_val=7.675e-19 U_stat=7.373e+07'''
#
# p_val_list = []
# u_stat_list = []
#
# for line in text.split('\n'):
#     p_val = float(line.split('P_val=')[1].split()[0])
#     u_stat = float(line.split('U_stat=')[1])
#     p_val_list.append(p_val)
#     u_stat_list.append(u_stat)
#
# print("P_val list:", p_val_list)
# print("U_stat list:", u_stat_list)
#
# exit()



# Define the input text
text = '''R3_RTN v.s. R3_Control: P_val=3.597e-170 U_stat=1.095e+08
R2_RTN v.s. R2_Control: P_val=9.346e-10 U_stat=7.708e+07
R3_ATL v.s. R3_Climp: P_val=1.069e-103 U_stat=5.621e+07
R1_RTN v.s. R1_Control: P_val=3.027e-236 U_stat=6.005e+07
R1_ATL v.s. R1_Climp: P_val=1.286e-118 U_stat=1.019e+08
R1_Climp v.s. R1_RTN: P_val=4.904e-86 U_stat=5.409e+07
R2_ATL v.s. R2_Climp: P_val=1.090e-110 U_stat=5.981e+07
R2_Climp v.s. R2_RTN: P_val=8.106e-27 U_stat=6.835e+07
R3_Climp v.s. R3_RTN: P_val=4.131e-24 U_stat=5.392e+07
R3_Climp v.s. R3_Control: P_val=6.354e-79 U_stat=1.476e+08
R1_Climp v.s. R1_Control: P_val=0.000e+00 U_stat=4.764e+07
R2_Climp v.s. R2_Control: P_val=2.607e-55 U_stat=4.621e+07
R1_ATL v.s. R1_RTN: P_val=2.709e-01 U_stat=9.226e+07
R2_ATL v.s. R2_RTN: P_val=1.889e-42 U_stat=1.190e+08
R3_ATL v.s. R3_RTN: P_val=6.697e-35 U_stat=3.659e+07
R3_ATL v.s. R3_Control: P_val=2.149e-295 U_stat=9.483e+07
R1_ATL v.s. R1_Control: P_val=1.823e-238 U_stat=8.408e+07
R2_ATL v.s. R2_Control: P_val=3.511e-07 U_stat=8.098e+07'''

# Split the text into lines
lines = text.split('\n')

# Initialize empty lists to store the P_val and U_stat values
p_values = []
u_stats = []

# Loop over each line and extract the P_val and U_stat values
for line in lines:
    # Split the line by the colon character
    split_line = line.split(':')
    # Split the P_val and U_stat values by the space character and convert them to floats
    # p_val = float(split_line[1].split()[0])
    p_val = float(split_line[1].split()[0].split('=')[1])
    u_stat = float(split_line[1].split()[1].split('=')[1])
    # Append the P_val and U_stat values to their respective lists
    p_values.append(p_val)
    u_stats.append(u_stat)

# Print the P_val and U_stat lists
corr = ['RTN-Control', 'RTN-Control', 'ATL-Climp', 'RTN-Control', 'ATL-Climp', 'Climp-RTN', 'ATL-Climp', 'Climp-RTN', 'Climp-RTN', 'Climp-Control', 'Climp-Control', 'Climp-Control', 'ATL-RTN', 'ATL-RTN', 'ATL-RTN', 'ATL-Control', 'ATL-Control', 'ATL-Control']

Replicate = [3,2,3,1,1,1,2,2,3,3,1,2,1,2,3,3,1,2]



print("P_val list:", p_values)
print("U_stat list:", u_stats)

exit()




import seaborn as sns
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

# Define four lists of data
list1 = [1, 2, 3, 4, 5]
list2 = [2, 3, 4, 5, 6]
list3 = [3, 4, 5, 6, 7]
list4 = [4, 5, 6, 7, 8]

# Perform Mann-Whitney U tests across all pairs of lists
results = []
for i in range(1, 5):
    for j in range(i+1, 5):
        pval = mannwhitneyu(locals()["list" + str(i)], locals()["list" + str(j)])[1]
        results.append(pval)

# Reshape the results into a 4x4 matrix
matrix = [[0]*4 for _ in range(4)]
idx = 0
for i in range(4):
    for j in range(i+1, 4):
        matrix[i][j] = results[idx]
        matrix[j][i] = results[idx]
        idx += 1

# Create heatmap using seaborn library
sns.heatmap(matrix, annot=True)
plt.show()

exit()






import numpy as np
import matplotlib.pyplot as plt

# Create random data for four groups
group1 = np.random.rand(10, 10)
group2 = np.random.rand(10, 10)
group3 = np.random.rand(10, 10)
group4 = np.random.rand(10, 10)

# Concatenate the four groups into a single array
data = np.concatenate((group1, group2, group3, group4), axis=1)

# Set up the plot and color map
fig, ax = plt.subplots()
im = ax.imshow(data, cmap='coolwarm')

# Create color bar
cbar = ax.figure.colorbar(im, ax=ax)

# Set tick labels and positions
xtick_labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
ytick_labels = [str(i+1) for i in range(10)]
# ax.set_xticks(np.arange(data.shape[1]))
# ax.set_yticks(np.arange(data.shape[0]))
# ax.set_xticklabels(xtick_labels)
# ax.set_yticklabels(ytick_labels)

# Rotate the tick labels and set axis labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
ax.set_title("Correlation heatmap")
ax.set_xlabel("Groups")
ax.set_ylabel("Variables")

# Loop over data dimensions and create text annotations
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        text = ax.text(j, i, f'{data[i, j]:.2f}',
                       ha="center", va="center", color="black")

# Show the plot
plt.show()
