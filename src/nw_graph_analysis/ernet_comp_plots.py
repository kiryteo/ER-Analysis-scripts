import networkx as nx
import seaborn as sns
import pandas as pd
import statannot
import numpy as np
from matplotlib import pyplot as plt
import imageio
import sknw


g1 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/skel/Ct1/Ct1_decon_t000_ch00_skel.png')
g2 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/skel/Ct1/Ct1_decon_t001_ch00_skel.png')
g3 = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/skel/Ct1/Ct1_decon_t002_ch00_skel.png')

g1 = sknw.build_sknw(g1, multi=False)
g2 = sknw.build_sknw(g2, multi=False)
g3 = sknw.build_sknw(g3, multi=False)

# dist = nx.graph_edit_distance(g1, g2)
# print(dist)

# n1 = g1.nodes()
# n2 = g2.nodes()

def jaccard_similarity(g, h):
    i = set(g).intersection(h)
    return round(len(i) / (len(g) + len(h) - len(i)),3)

j = jaccard_similarity(g1.edges(), g2.edges())
print(j)
j = jaccard_similarity(g2.edges(), g3.edges())
print(j)
exit()



ernet = [147, 204, 203, 109, 200, 198, 57, 110, 106, 87, 253, 237, 95, 196, 186, 102, 203, 198, 71, 178, 168]
dynet = [84, 204, 199, 101, 218, 212, 57, 100, 95, 95, 175, 161, 73, 207, 178, 56, 219, 186, 94, 126, 124]
gt = [129, 220, 207, 65, 231, 214, 35, 221, 197, 85, 226, 214, 82, 147, 136, 61, 142, 128, 98, 232, 225]

df = pd.DataFrame()
df['Values'] = pd.Series(np.concatenate((ernet, dynet, gt)))
df['Method'] = pd.Series(np.concatenate((['ERnet'] * len(ernet), ['DyNet'] * len(dynet), ['GT'] * len(gt))))


ax = sns.boxplot(data=df, x='Method', y='Values', showfliers=False, width=0.9)#, palette=pal)

# ax.set_ylim(-0.31, 0.65)
# ax.set_xlim(-1, 3.0)

# yt = ax.get_yticks()
# yt = [f'{y:.2f}' for y in yt]
# ax.set_yticklabels(yt, fontsize=13)
# ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, rotation=90)

# region_name = 'Isolated' if region == 'iso' else 'Overlapping'

# box_pairs = [('Atlastin', 'Climp'), ('Atlastin', 'Reticulon'), ('Climp', 'Reticulon')]
box_pairs = [('ERnet', 'DyNet'), ('ERnet', 'GT'), ('DyNet', 'GT')]

statannot.add_stat_annotation(ax, x='Method', y='Values', data=df, box_pairs=box_pairs,
                              test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=15)

# plt.title(f'{region_name} CC mean cross-correlation between ERmoxGFP and mCherry over 100 frames', fontsize=20)
plt.grid(True)
plt.xlabel('Method', fontsize=15)
# plt.ylabel('Number of junctions (normalized)', fontsize=24)
plt.ylabel('Number of nodes', fontsize=15)
# plt.show()
# plt.gcf().set_size_inches(16, 4)
# plt.gcf().set_size_inches(2.5, 6)
# # plt.savefig('num_juncs_iso_norm.png', bbox_inches='tight', pad_inches=0.6)
# plt.savefig(f'CC_mean_cross_corr_{region}_v3.png', bbox_inches='tight', pad_inches=0.1)
# plt.close()
plt.show()