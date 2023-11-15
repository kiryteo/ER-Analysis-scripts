import seaborn as sns
import matplotlib.pyplot as plt

def plot_metrics():
    analyzer = [0.83, 0.47, 0.31]

    ernet = [0.89, 0.69, 0.52]

    erv2 = [0.86, 0.63, 0.46]

    nERdy = [0.94, 0.81, 0.69]

    nERdy_plus = [0.96, 0.85, 0.74]


    sns.set(style="whitegrid")
    sns.set_context("paper", font_scale=1.4)
    sns.set_palette("Set2")

    sns.barplot(x=["Dice score", "F1-score", "Jaccard Index"], y=analyzer, label="Analyzer")

    sns.barplot(x=["Dice score", "F1-score", "Jaccard Index"], y=ernet, label="ERnet")

    sns.barplot(x=["Dice score", "F1-score", "Jaccard Index"], y=erv2, label="ERnet-v2")

    sns.barplot(x=["Dice score", "F1-score", "Jaccard Index"], y=nERdy, label="nERdy")

    sns.barplot(x=["Dice score", "F1-score", "Jaccard Index"], y=nERdy_plus, label="nERdy+")

    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.show()

# plot_metrics()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Your lists
analyzer = [0.83, 0.47, 0.31]
ernet = [0.89, 0.69, 0.52]
erv2 = [0.86, 0.63, 0.46]
nERdy = [0.94, 0.81, 0.69]
nERdy_plus = [0.96, 0.85, 0.74]

# Combine lists into a DataFrame
df = pd.DataFrame({
    'Metric': ['Dice score', 'F1-score', 'Jaccard Index'] * 5,
    'Value': analyzer + ernet + erv2 + nERdy + nERdy_plus,
    'Method': ['AnalyzER'] * 3 + ['ERnet'] * 3 + ['ERnet-v2'] * 3 + ['nERdy'] * 3 + ['nERdy+'] * 3
})

# Set the seaborn style
sns.set(style="whitegrid")

# Create the bar plot using Seaborn
plt.figure(figsize=(8, 12))
ax = sns.barplot(x='Metric', y='Value', hue='Method', data=df, palette="muted")

yt = ax.get_yticks()
yt = [f'{y:.1f}' for y in yt]
ax.set_yticklabels(yt, fontsize=15)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)# rotation=90)

# Add labels and title
plt.xlabel('Metric', fontsize=17, fontweight='bold')
plt.ylabel('Values', fontsize=17, fontweight='bold')
plt.title('Segmentation Performance', fontsize=19)

plt.savefig('segmentation_metrics.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

# Show the plot
# plt.show()
plt.close()


exit()


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# Your lists
list1 = [0.6314750863344845, 0.6407471763683753, 12.617371335249446]
list2 = [0.44499259990133194, 0.5573414422241529, 5.491947724290345]
list3 = [0.3320177602368032, 0.4278887923544744, 3.52346660250625]
list4 = [0.139615194869265, 0.002606429192006833, 1.2411852813398745]
list5 = [0.1475086334484459, 0.2684622067767159, 0.32596236099783416]

# Number of lists
num_lists = len(list1)

# Bar width
bar_width = 0.2

# Set the positions of bars on X-axis with manual separation
r1 = np.arange(num_lists) * 2
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]

# Create the bar plot
plt.bar(r1, list1, color='b', width=bar_width, edgecolor='grey', label='List 1')
plt.bar(r2, list2, color='g', width=bar_width, edgecolor='grey', label='List 2')
plt.bar(r3, list3, color='r', width=bar_width, edgecolor='grey', label='List 3')
plt.bar(r4, list4, color='c', width=bar_width, edgecolor='grey', label='List 4')
plt.bar(r5, list5, color='m', width=bar_width, edgecolor='grey', label='List 5')

# Add xticks on the middle of the group bars
plt.xlabel('Groups', fontweight='bold')
plt.xticks([r + bar_width for r in range(0, 2*num_lists, 2)], ['Group 1', 'Group 2', 'Group 3'])

# Add a legend
plt.legend()

# Show the plot
plt.show()

exit()


import matplotlib.pyplot as plt
import numpy as np

# Your lists
analyzer = [0.83, 0.47, 0.31]

ernet = [0.89, 0.69, 0.52]

erv2 = [0.86, 0.63, 0.46]

nERdy = [0.94, 0.81, 0.69]
nERdy_plus = [0.96, 0.85, 0.74]

# Number of lists
num_lists = len(analyzer)

# Bar width
bar_width = 0.2

# Set the positions of bars on X-axis
r1 = np.arange(num_lists)
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]

# Create the bar plot
plt.bar(r1, analyzer, color='b', width=bar_width, edgecolor='grey', label='List 1')
sns.ba
plt.bar(r2, ernet, color='g', width=bar_width, edgecolor='grey', label='List 2')
plt.bar(r3, erv2, color='r', width=bar_width, edgecolor='grey', label='List 3')
plt.bar(r4, nERdy, color='c', width=bar_width, edgecolor='grey', label='List 4')
plt.bar(r5, nERdy_plus, color='m', width=bar_width, edgecolor='grey', label='List 5')

# Add xticks on the middle of the group bars
plt.xlabel('Groups', fontweight='bold')
plt.xticks([r + bar_width for r in range(num_lists)], ['Group 1', 'Group 2', 'Group 3'])

# Add a legend
plt.legend()

# Show the plot
plt.show()


