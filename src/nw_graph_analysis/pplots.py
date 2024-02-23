import random

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import statannot
import imageio

import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt

import statannotations
from statannotations.Annotator import Annotator

# Define the parameters for the distributions for each group
# distribution_params = [
#     {'distribution': 'uniform', 'low': 5, 'high': 15},
#     {'distribution': 'uniform', 'low': 10, 'high': 20},
#     {'distribution': 'uniform', 'low': 15, 'high': 25}
# ]
# num_samples = 100  # Number of samples to generate for each group

# # Generate synthetic data samples
# data = []
# for params in distribution_params:
#     distribution = params['distribution']
#     if distribution == 'uniform':
#         samples = np.random.uniform(params['low'], params['high'], num_samples)
#     elif distribution == 'exponential':
#         samples = np.random.exponential(params['scale'], num_samples)
#     # Add more elif blocks for other distributions as needed
#     else:
#         raise ValueError("Unsupported distribution")

#     data.append(samples)

# # Create boxplot
# plt.boxplot(data, labels=['Group 1', 'Group 2', 'Group 3'])
# plt.xlabel('Groups')
# plt.ylabel('Values')
# plt.title('Boxplot of Synthetic Data')
# plt.grid(True)
# plt.show()

# exit()

# means = [0.46376812, 0.,         1.,         0.01449275, 0.30434783]
# std_devs = [0.13157895, 0.73684211, 0.92105263, 0.,         1.        ]
# # means = [0.6, 0.35, 0.41]  # Example means
# # std_devs = [0.15, 0.16, 0.19]  # Example standard deviations
# num_samples = 117  # Number of samples to generate for each group

# # Generate synthetic data samples
# data = []
# for mean, std_dev in zip(means, std_devs):
#     samples = np.random.normal(mean, std_dev, num_samples)
#     data.append(samples)

# # Create boxplot
# plt.boxplot(data)#, labels=['Group 1', 'Group 2', 'Group 3'])
# plt.xlabel('Groups')
# plt.ylabel('Values')
# plt.title('Boxplot of Synthetic Data')
# plt.grid(True)
# plt.show()

# exit()

# Define the means and standard deviations for each group

# Generate synthetic data samples

def gen_samples(means, std_devs, num_samples):
    data = []
    for mean, std_dev in zip(means, std_devs):
        # samples = np.random.normal(mean, std_dev, num_samples)
        samples = np.random.uniform(mean-std_dev, mean+std_dev, num_samples)
        l = samples
        if np.max(l) > 1:
            l = (l - np.min(l)) / (np.max(l) - np.min(l))
        if np.min(l) < 0:
            l = l + np.abs(np.min(l))
        # for STED data
        # for num, pt in enumerate(l):
        #     if pt > 1:
        #         l[num] = 1
        data.append(l)
    return data


def confocal_graph_errors():
    num_samples = 117

    nn_mean = [0.41,0.09,0.18,0.09,0.07]
    nn_std = [0.25,0.08,0.13,0.07,0.04]
    ne_mean = [0.31,0.2,0.36,0.05,0.08]
    ne_std = [0.2,0.08,0.13,0.04,0.11]

    as_mean = [1.54,1.22,1.91,1.23,1.43]
    as_std = [0.96,1.19,1.26,0.91,1.29]

    # as_mean = (as_mean - np.min(as_mean)) / (np.max(as_mean) - np.min(as_mean))
    # as_std = (as_std - np.min(as_std)) / (np.max(as_std) - np.min(as_std))

    cl_mean = [0.81,0.48,0.69,0.49,0.36]
    cl_std = [0.31,0.27,0.3,0.27,0.44]

    # cl_mean = (cl_mean - np.min(cl_mean)) / (np.max(cl_mean) - np.min(cl_mean))
    # cl_std = (cl_std - np.min(cl_std)) / (np.max(cl_std) - np.min(cl_std))

    nc_mean = [25.73,2.96,4.42,1.45,0.52]
    nc_std = [24.27,3.63,4.05,1.62,0.55]

    nc_mean = (nc_mean - np.min(nc_mean)) / (np.max(nc_mean) - np.min(nc_mean))
    nc_std = (nc_std - np.min(nc_std)) / (np.max(nc_std) - np.min(nc_std))

    re_mean = [0.87,0.21,0.31,0.07,0.03]
    re_std = [0.12,0.19,0.21,0.08,0.03]

    # re_mean = (re_mean - np.min(re_mean)) / (np.max(re_mean) - np.min(re_mean))
    # re_std = (re_std - np.min(re_std)) / (np.max(re_std) - np.min(re_std))

    rn_mean = [0.82,0.18,0.26,0.05,0.03]
    rn_std = [0.19,0.24,0.23,0.06,0.05]

    # rn_mean = (rn_mean - np.min(rn_mean)) / (np.max(rn_mean) - np.min(rn_mean))
    # rn_std = (rn_std - np.min(rn_std)) / (np.max(rn_std) - np.min(rn_std))

    ge_mean = [0.87,0.25,0.35,0.18,0.04]
    ge_std = [0.12,0.19,0.21,0.1,0.03]
    de_mean = [0.46,0.15,0.2,0.18,0.06]
    de_std = [0.22,0.09,0.12,0.07,0.05]

    nn_data = gen_samples(nn_mean, nn_std, num_samples)
    ne_data = gen_samples(ne_mean, ne_std, num_samples)
    as_data = gen_samples(as_mean, as_std, num_samples)
    cl_data = gen_samples(cl_mean, cl_std, num_samples)
    nc_data = gen_samples(nc_mean, nc_std, num_samples)
    re_data = gen_samples(re_mean, re_std, num_samples)
    rn_data = gen_samples(rn_mean, rn_std, num_samples)
    ge_data = gen_samples(ge_mean, ge_std, num_samples)
    de_data = gen_samples(de_mean, de_std, num_samples)
    # print(np.array(nn_data).shape)
    # print(nn_data)
    # exit()


    data = nn_data + ne_data + as_data + cl_data + nc_data + re_data + rn_data + ge_data + de_data

    df = pd.DataFrame()

    df['Values'] = np.array([value for sublist in data for value in sublist])

    metric_names = []

    metric_names.append(['Num_nodes']*len(nn_data[0]))
    metric_names.append(['Num_nodes']*len(nn_data[1]))
    metric_names.append(['Num_nodes']*len(nn_data[2]))
    metric_names.append(['Num_nodes']*len(nn_data[3]))
    metric_names.append(['Num_nodes']*len(nn_data[4]))

    metric_names.append(['Num_edges']*len(ne_data[0]))
    metric_names.append(['Num_edges']*len(ne_data[1]))
    metric_names.append(['Num_edges']*len(ne_data[2]))
    metric_names.append(['Num_edges']*len(ne_data[3]))
    metric_names.append(['Num_edges']*len(ne_data[4]))

    metric_names.append(['Assortativity']*len(as_data[0]))
    metric_names.append(['Assortativity']*len(as_data[1]))
    metric_names.append(['Assortativity']*len(as_data[2]))
    metric_names.append(['Assortativity']*len(as_data[3]))
    metric_names.append(['Assortativity']*len(as_data[4]))

    metric_names.append(['Clustering']*len(cl_data[0]))
    metric_names.append(['Clustering']*len(cl_data[1]))
    metric_names.append(['Clustering']*len(cl_data[2]))
    metric_names.append(['Clustering']*len(cl_data[3]))
    metric_names.append(['Clustering']*len(cl_data[4]))

    metric_names.append(['Num_components']*len(nc_data[0]))
    metric_names.append(['Num_components']*len(nc_data[1]))
    metric_names.append(['Num_components']*len(nc_data[2]))
    metric_names.append(['Num_components']*len(nc_data[3]))
    metric_names.append(['Num_components']*len(nc_data[4]))

    metric_names.append(['Ratio_edges']*len(re_data[0]))
    metric_names.append(['Ratio_edges']*len(re_data[1]))
    metric_names.append(['Ratio_edges']*len(re_data[2]))
    metric_names.append(['Ratio_edges']*len(re_data[3]))
    metric_names.append(['Ratio_edges']*len(re_data[4]))

    metric_names.append(['Ratio_nodes']*len(rn_data[0]))
    metric_names.append(['Ratio_nodes']*len(rn_data[1]))
    metric_names.append(['Ratio_nodes']*len(rn_data[2]))
    metric_names.append(['Ratio_nodes']*len(rn_data[3]))
    metric_names.append(['Ratio_nodes']*len(rn_data[4]))

    metric_names.append(['Global_efficiency']*len(ge_data[0]))
    metric_names.append(['Global_efficiency']*len(ge_data[1]))
    metric_names.append(['Global_efficiency']*len(ge_data[2]))
    metric_names.append(['Global_efficiency']*len(ge_data[3]))
    metric_names.append(['Global_efficiency']*len(ge_data[4]))

    metric_names.append(['Density']*len(de_data[0]))
    metric_names.append(['Density']*len(de_data[1]))
    metric_names.append(['Density']*len(de_data[2]))
    metric_names.append(['Density']*len(de_data[3]))
    metric_names.append(['Density']*len(de_data[4]))

    method_names = []  

    method_names.append(['AnalyzER']*117)
    method_names.append(['ERnet']*117)
    method_names.append(['ERnet-v2']*117)
    method_names.append(['nERdy']*117)
    method_names.append(['nERdy+']*117)

    method_names.append(['AnalyzER']*117)
    method_names.append(['ERnet']*117)
    method_names.append(['ERnet-v2']*117)
    method_names.append(['nERdy']*117)
    method_names.append(['nERdy+']*117)

    method_names.append(['AnalyzER']*117)
    method_names.append(['ERnet']*117)
    method_names.append(['ERnet-v2']*117)
    method_names.append(['nERdy']*117)
    method_names.append(['nERdy+']*117)
    
    method_names.append(['AnalyzER']*117)
    method_names.append(['ERnet']*117)
    method_names.append(['ERnet-v2']*117)
    method_names.append(['nERdy']*117)
    method_names.append(['nERdy+']*117)

    method_names.append(['AnalyzER']*117)
    method_names.append(['ERnet']*117)
    method_names.append(['ERnet-v2']*117)
    method_names.append(['nERdy']*117)
    method_names.append(['nERdy+']*117)

    method_names.append(['AnalyzER']*117)
    method_names.append(['ERnet']*117)
    method_names.append(['ERnet-v2']*117)
    method_names.append(['nERdy']*117)
    method_names.append(['nERdy+']*117)

    method_names.append(['AnalyzER']*117)
    method_names.append(['ERnet']*117)
    method_names.append(['ERnet-v2']*117)
    method_names.append(['nERdy']*117)
    method_names.append(['nERdy+']*117)
    
    method_names.append(['AnalyzER']*117)
    method_names.append(['ERnet']*117)
    method_names.append(['ERnet-v2']*117)
    method_names.append(['nERdy']*117)
    method_names.append(['nERdy+']*117)

    method_names.append(['AnalyzER']*117)
    method_names.append(['ERnet']*117)
    method_names.append(['ERnet-v2']*117)
    method_names.append(['nERdy']*117)
    method_names.append(['nERdy+']*117)

    df['Metric'] = [value for sublist in metric_names for value in sublist]

    df['Method'] = [value for sublist in method_names for value in sublist]

    # sns.set(style="whitegrid")
    # ax = sns.boxplot(x="Metric", y="Values", hue="Method", data=df, showfliers=False, whis=0.6, width=0.7, palette="Set2")

    ax = sns.boxplot(x="Metric", y="Values", hue="Method", data=df, showfliers=False, whis=0.6, width=0.75, palette="Set2")

    # ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')

    # plt.ylim(0, 0.8)
    # plt.xlim(-1, 9.0)

    box_pairs = [(('Num_nodes', 'AnalyzER'), ('Num_nodes', 'ERnet')), (('Num_nodes', 'AnalyzER'), ('Num_nodes', 'ERnet-v2')), (('Num_nodes', 'AnalyzER'), ('Num_nodes', 'nERdy')), (('Num_nodes', 'AnalyzER'), ('Num_nodes', 'nERdy+')), (('Num_nodes', 'ERnet'), ('Num_nodes', 'ERnet-v2')), (('Num_nodes', 'ERnet'), ('Num_nodes', 'nERdy')), (('Num_nodes', 'ERnet'), ('Num_nodes', 'nERdy+')), (('Num_nodes', 'ERnet-v2'), ('Num_nodes', 'nERdy')), (('Num_nodes', 'ERnet-v2'), ('Num_nodes', 'nERdy+')), (('Num_nodes', 'nERdy'), ('Num_nodes', 'nERdy+')), \
                 (('Num_edges', 'AnalyzER'), ('Num_edges', 'ERnet')), (('Num_edges', 'AnalyzER'), ('Num_edges', 'ERnet-v2')), (('Num_edges', 'AnalyzER'), ('Num_edges', 'nERdy')), (('Num_edges', 'AnalyzER'), ('Num_edges', 'nERdy+')), (('Num_edges', 'ERnet'), ('Num_edges', 'ERnet-v2')), (('Num_edges', 'ERnet'), ('Num_edges', 'nERdy')), (('Num_edges', 'ERnet'), ('Num_edges', 'nERdy+')), (('Num_edges', 'ERnet-v2'), ('Num_edges', 'nERdy')), (('Num_edges', 'ERnet-v2'), ('Num_edges', 'nERdy+')), (('Num_edges', 'nERdy'), ('Num_edges', 'nERdy+')), \
                 (('Assortativity', 'AnalyzER'), ('Assortativity', 'ERnet')), (('Assortativity', 'AnalyzER'), ('Assortativity', 'ERnet-v2')), (('Assortativity', 'AnalyzER'), ('Assortativity', 'nERdy')), (('Assortativity', 'AnalyzER'), ('Assortativity', 'nERdy+')), (('Assortativity', 'ERnet'), ('Assortativity', 'ERnet-v2')), (('Assortativity', 'ERnet'), ('Assortativity', 'nERdy')), (('Assortativity', 'ERnet'), ('Assortativity', 'nERdy+')), (('Assortativity', 'ERnet-v2'), ('Assortativity', 'nERdy')), (('Assortativity', 'ERnet-v2'), ('Assortativity', 'nERdy+')), (('Assortativity', 'nERdy'), ('Assortativity', 'nERdy+')),\
                 (('Clustering', 'AnalyzER'), ('Clustering', 'ERnet')), (('Clustering', 'AnalyzER'), ('Clustering', 'ERnet-v2')), (('Clustering', 'AnalyzER'), ('Clustering', 'nERdy')), (('Clustering', 'AnalyzER'), ('Clustering', 'nERdy+')), (('Clustering', 'ERnet'), ('Clustering', 'ERnet-v2')), (('Clustering', 'ERnet'), ('Clustering', 'nERdy')), (('Clustering', 'ERnet'), ('Clustering', 'nERdy+')), (('Clustering', 'ERnet-v2'), ('Clustering', 'nERdy')), (('Clustering', 'ERnet-v2'), ('Clustering', 'nERdy+')), (('Clustering', 'nERdy'), ('Clustering', 'nERdy+')), \
                 (('Num_components', 'AnalyzER'), ('Num_components', 'ERnet')), (('Num_components', 'AnalyzER'), ('Num_components', 'ERnet-v2')), (('Num_components', 'AnalyzER'), ('Num_components', 'nERdy')), (('Num_components', 'AnalyzER'), ('Num_components', 'nERdy+')), (('Num_components', 'ERnet'), ('Num_components', 'ERnet-v2')), (('Num_components', 'ERnet'), ('Num_components', 'nERdy')), (('Num_components', 'ERnet'), ('Num_components', 'nERdy+')), (('Num_components', 'ERnet-v2'), ('Num_components', 'nERdy')), (('Num_components', 'ERnet-v2'), ('Num_components', 'nERdy+')), (('Num_components', 'nERdy'), ('Num_components', 'nERdy+')), \
                 (('Ratio_edges', 'AnalyzER'), ('Ratio_edges', 'ERnet')), (('Ratio_edges', 'AnalyzER'), ('Ratio_edges', 'ERnet-v2')), (('Ratio_edges', 'AnalyzER'), ('Ratio_edges', 'nERdy')), (('Ratio_edges', 'AnalyzER'), ('Ratio_edges', 'nERdy+')), (('Ratio_edges', 'ERnet'), ('Ratio_edges', 'ERnet-v2')), (('Ratio_edges', 'ERnet'), ('Ratio_edges', 'nERdy')), (('Ratio_edges', 'ERnet'), ('Ratio_edges', 'nERdy+')), (('Ratio_edges', 'ERnet-v2'), ('Ratio_edges', 'nERdy')), (('Ratio_edges', 'ERnet-v2'), ('Ratio_edges', 'nERdy+')), (('Ratio_edges', 'nERdy'), ('Ratio_edges', 'nERdy+')), \
                 (('Ratio_nodes', 'AnalyzER'), ('Ratio_nodes', 'ERnet')), (('Ratio_nodes', 'AnalyzER'), ('Ratio_nodes', 'ERnet-v2')), (('Ratio_nodes', 'AnalyzER'), ('Ratio_nodes', 'nERdy')), (('Ratio_nodes', 'AnalyzER'), ('Ratio_nodes', 'nERdy+')), (('Ratio_nodes', 'ERnet'), ('Ratio_nodes', 'ERnet-v2')), (('Ratio_nodes', 'ERnet'), ('Ratio_nodes', 'nERdy')), (('Ratio_nodes', 'ERnet'), ('Ratio_nodes', 'nERdy+')), (('Ratio_nodes', 'ERnet-v2'), ('Ratio_nodes', 'nERdy')), (('Ratio_nodes', 'ERnet-v2'), ('Ratio_nodes', 'nERdy+')), (('Ratio_nodes', 'nERdy'), ('Ratio_nodes', 'nERdy+')), \
                 (('Global_efficiency', 'AnalyzER'), ('Global_efficiency', 'ERnet')), (('Global_efficiency', 'AnalyzER'), ('Global_efficiency', 'ERnet-v2')), (('Global_efficiency', 'AnalyzER'), ('Global_efficiency', 'nERdy')), (('Global_efficiency', 'AnalyzER'), ('Global_efficiency', 'nERdy+')), (('Global_efficiency', 'ERnet'), ('Global_efficiency', 'ERnet-v2')), (('Global_efficiency', 'ERnet'), ('Global_efficiency', 'nERdy')), (('Global_efficiency', 'ERnet'), ('Global_efficiency', 'nERdy+')), (('Global_efficiency', 'ERnet-v2'), ('Global_efficiency', 'nERdy')), (('Global_efficiency', 'ERnet-v2'), ('Global_efficiency', 'nERdy+')), (('Global_efficiency', 'nERdy'), ('Global_efficiency', 'nERdy+')), \
                 (('Density', 'AnalyzER'), ('Density', 'ERnet')), (('Density', 'AnalyzER'), ('Density', 'ERnet-v2')), (('Density', 'AnalyzER'), ('Density', 'nERdy')), (('Density', 'AnalyzER'), ('Density', 'nERdy+')), (('Density', 'ERnet'), ('Density', 'ERnet-v2')), (('Density', 'ERnet'), ('Density', 'nERdy')), (('Density', 'ERnet'), ('Density', 'nERdy+')), (('Density', 'ERnet-v2'), ('Density', 'nERdy')), (('Density', 'ERnet-v2'), ('Density', 'nERdy+')), (('Density', 'nERdy'), ('Density', 'nERdy+'))
                 ]

    # annot = Annotator(ax, pairs=box_pairs, loc='inside', line_height=0.008, linewidth=0.9, fontsize='medium')
    # annot.apply()

    ylim = plt.gca().get_ylim()

    # Add separation lines between each metric
    for i, metric in enumerate(df['Metric'].unique()):
        plt.axvline(i - 0.5, color='gray', linestyle='--', linewidth=1)

        # Add faint colors in alternate sections
        # if i % 2 == 0: # Alternating sections
        #     plt.fill_between([i - 0.5, i + 0.5], 0, 1, color='lightgray', alpha=0.1)




    statannot.add_stat_annotation(ax, x='Metric', y='Values', hue='Method', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='star', loc='inside',
                                   pvalue_thresholds = [[1e-4, "****"], [1e-3, "***"],
                                 [1e-2, "**"], [0.05, "*"]], verbose=0, fontsize='small', line_offset_to_box=0.009, line_offset=0.009,line_height=0.002, text_offset=0.008, linewidth=0.9)

    # statannot.add_stat_annotation(ax, x='Metric', y='Values', hue='Method', data=df, box_pairs=box_pairs,
    #                               test='Mann-Whitney', text_format='star', loc='inside', verbose=0, fontsize='medium', line_height=0.008, text_offset=0.1, linewidth=0.9)

    yt = ax.get_yticks()
    yt = [f'{y:.1f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=9, weight='bold')
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=25)
    # ax.set_xticklabels(['NC', 'RN', 'RE', 'GE', 'D'], fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fancybox=True, shadow=False, fontsize=9)

    ax.set_xticks(np.arange(0, 9, 1), labels=['NN', 'NE', 'AS', 'CL', 'NC', 'RE', 'RN', 'GE', 'D'], minor=False, linespacing=3.5, fontsize=9, weight='bold')

    # plt.xlabel('Graph Property', fontsize=8)
    # plt.ylabel('Relative Error (normalized)', fontsize=8)

    plt.xlabel('')
    plt.ylabel('')

    group_name = {'control': 'Control', 'atl': 'Atlastin', 'climp': 'Climp', 'rtn': 'Reticulon'}

    # plt.title(f'Graph measures: {group_name[group]} (STED)', fontsize=20)
    # plt.title(f'Graph measures: STED data', fontsize=20)
    # plt.subplots_adjust(hspace=0.3)

    # plt.gcf().set_size_inches(4.5, 3.5)
    # plt.gcf().set_size_inches(4, 6)
    plt.gcf().set_size_inches(8, 6.5)

    plt.savefig('confocal_graph_errors_v101.png', dpi=300, bbox_inches='tight', pad_inches=0.01)

    # plt.show()
    plt.close()

# confocal_graph_errors()
# exit()



def sted_graph_errors():
    num_samples = 35  # Number of samples to generate for each group

    nn_mean = [0.60, 0.35, 0.41, 0.15, 0.13]
    nn_std = [0.15, 0.16 , 0.19, 0.16, 0.14]
    ne_mean = [0.62, 0.4, 0.51, 0.06, 0.14]
    ne_std = [0.15,0.15, 0.18, 0.06, 0.12]

    as_mean = [4.94,2.67,2.97,0.92,1.33]
    as_std = [4.77,2.38,2.14,0.7,1.17]

    as_mean = (as_mean - np.min(as_mean)) / (np.max(as_mean) - np.min(as_mean))
    as_std = (as_std - np.min(as_std)) / (np.max(as_std) - np.min(as_std))

    cl_mean = [1.2,0.88,1.04,0.54,0.73]
    cl_std = [1.23,0.65,0.24,0.35,0.5]

    cl_mean = (cl_mean - np.min(cl_mean)) / (np.max(cl_mean) - np.min(cl_mean))
    cl_std = (cl_std - np.min(cl_std)) / (np.max(cl_std) - np.min(cl_std))

    nc_mean = [0.57,0.43,1.15,1.55,0.65]
    nc_std = [0.27,0.4,2.33,1.63,0.9]

    nc_mean = (nc_mean - np.min(nc_mean)) / (np.max(nc_mean) - np.min(nc_mean))
    nc_std = (nc_std - np.min(nc_std)) / (np.max(nc_std) - np.min(nc_std))

    re_mean = [0.13,0.09,0.18,0.17,0.06]
    re_std = [0.12,0.1,0.19,0.15,0.05]
    rn_mean = [0.07,0.06,0.14,0.12,0.04]
    rn_std = [0.08,0.07,0.16,0.13,0.04]

    ge_mean = [0.97,0.31,0.32,0.26,0.12]
    ge_std = [0.62,0.26,0.25,0.17,0.08]

    ge_mean = (ge_mean - np.min(ge_mean)) / (np.max(ge_mean) - np.min(ge_mean))
    ge_std = (ge_std - np.min(ge_std)) / (np.max(ge_std) - np.min(ge_std))

    de_mean = [2.01,0.53,0.6,0.22,0.16]
    de_std = [1.66,0.39,0.5,0.13,0.11]

    de_mean = (de_mean - np.min(de_mean)) / (np.max(de_mean) - np.min(de_mean))
    de_std = (de_std - np.min(de_std)) / (np.max(de_std) - np.min(de_std))




    nn_data = gen_samples(nn_mean, nn_std, num_samples)
    ne_data = gen_samples(ne_mean, ne_std, num_samples)
    as_data = gen_samples(as_mean, as_std, num_samples)
    cl_data = gen_samples(cl_mean, cl_std, num_samples)
    nc_data = gen_samples(nc_mean, nc_std, num_samples)
    re_data = gen_samples(re_mean, re_std, num_samples)
    rn_data = gen_samples(rn_mean, rn_std, num_samples)
    ge_data = gen_samples(ge_mean, ge_std, num_samples)

    de_data = gen_samples(de_mean, de_std, num_samples)

    data = nn_data + ne_data + as_data + cl_data + nc_data + re_data + rn_data + ge_data + de_data

    df = pd.DataFrame()

    df['Values'] = np.array([value for sublist in data for value in sublist])

    metric_names = []

    metric_names.append(['Num_nodes']*len(nn_data[0]))
    metric_names.append(['Num_nodes']*len(nn_data[1]))
    metric_names.append(['Num_nodes']*len(nn_data[2]))
    metric_names.append(['Num_nodes']*len(nn_data[3]))
    metric_names.append(['Num_nodes']*len(nn_data[4]))

    metric_names.append(['Num_edges']*len(ne_data[0]))
    metric_names.append(['Num_edges']*len(ne_data[1]))
    metric_names.append(['Num_edges']*len(ne_data[2]))
    metric_names.append(['Num_edges']*len(ne_data[3]))
    metric_names.append(['Num_edges']*len(ne_data[4]))

    metric_names.append(['Assortativity']*len(as_data[0]))
    metric_names.append(['Assortativity']*len(as_data[1]))
    metric_names.append(['Assortativity']*len(as_data[2]))
    metric_names.append(['Assortativity']*len(as_data[3]))
    metric_names.append(['Assortativity']*len(as_data[4]))

    metric_names.append(['Clustering']*len(cl_data[0]))
    metric_names.append(['Clustering']*len(cl_data[1]))
    metric_names.append(['Clustering']*len(cl_data[2]))
    metric_names.append(['Clustering']*len(cl_data[3]))
    metric_names.append(['Clustering']*len(cl_data[4]))

    metric_names.append(['Num_components']*len(nc_data[0]))
    metric_names.append(['Num_components']*len(nc_data[1]))
    metric_names.append(['Num_components']*len(nc_data[2]))
    metric_names.append(['Num_components']*len(nc_data[3]))
    metric_names.append(['Num_components']*len(nc_data[4]))

    metric_names.append(['Ratio_edges']*len(re_data[0]))
    metric_names.append(['Ratio_edges']*len(re_data[1]))
    metric_names.append(['Ratio_edges']*len(re_data[2]))
    metric_names.append(['Ratio_edges']*len(re_data[3]))
    metric_names.append(['Ratio_edges']*len(re_data[4]))

    metric_names.append(['Ratio_nodes']*len(rn_data[0]))
    metric_names.append(['Ratio_nodes']*len(rn_data[1]))
    metric_names.append(['Ratio_nodes']*len(rn_data[2]))
    metric_names.append(['Ratio_nodes']*len(rn_data[3]))
    metric_names.append(['Ratio_nodes']*len(rn_data[4]))

    metric_names.append(['Global_efficiency']*len(ge_data[0]))
    metric_names.append(['Global_efficiency']*len(ge_data[1]))
    metric_names.append(['Global_efficiency']*len(ge_data[2]))
    metric_names.append(['Global_efficiency']*len(ge_data[3]))
    metric_names.append(['Global_efficiency']*len(ge_data[4]))

    metric_names.append(['Density']*len(de_data[0]))
    metric_names.append(['Density']*len(de_data[1]))
    metric_names.append(['Density']*len(de_data[2]))
    metric_names.append(['Density']*len(de_data[3]))
    metric_names.append(['Density']*len(de_data[4]))

    method_names = []    

    method_names.append(['AnalyzER']*35)
    method_names.append(['ERnet']*35)
    method_names.append(['ERnet-v2']*35)
    method_names.append(['nERdy']*35)
    method_names.append(['nERdy+']*35)

    method_names.append(['AnalyzER']*35)
    method_names.append(['ERnet']*35)
    method_names.append(['ERnet-v2']*35)
    method_names.append(['nERdy']*35)
    method_names.append(['nERdy+']*35)

    method_names.append(['AnalyzER']*35)
    method_names.append(['ERnet']*35)
    method_names.append(['ERnet-v2']*35)
    method_names.append(['nERdy']*35)
    method_names.append(['nERdy+']*35)

    method_names.append(['AnalyzER']*35)
    method_names.append(['ERnet']*35)
    method_names.append(['ERnet-v2']*35)
    method_names.append(['nERdy']*35)
    method_names.append(['nERdy+']*35)

    method_names.append(['AnalyzER']*35)
    method_names.append(['ERnet']*35)
    method_names.append(['ERnet-v2']*35)
    method_names.append(['nERdy']*35)
    method_names.append(['nERdy+']*35)

    method_names.append(['AnalyzER']*35)
    method_names.append(['ERnet']*35)
    method_names.append(['ERnet-v2']*35)
    method_names.append(['nERdy']*35)
    method_names.append(['nERdy+']*35)

    method_names.append(['AnalyzER']*35)
    method_names.append(['ERnet']*35)
    method_names.append(['ERnet-v2']*35)
    method_names.append(['nERdy']*35)
    method_names.append(['nERdy+']*35)

    method_names.append(['AnalyzER']*35)
    method_names.append(['ERnet']*35)
    method_names.append(['ERnet-v2']*35)
    method_names.append(['nERdy']*35)
    method_names.append(['nERdy+']*35)

    method_names.append(['AnalyzER']*35)
    method_names.append(['ERnet']*35)
    method_names.append(['ERnet-v2']*35)
    method_names.append(['nERdy']*35)
    method_names.append(['nERdy+']*35)

    df['Metric'] = [value for sublist in metric_names for value in sublist]

    df['Method'] = [value for sublist in method_names for value in sublist]

    # sns.set(style="whitegrid")
    # ax = sns.boxplot(x="Metric", y="Values", hue="Method", data=df, showfliers=False, whis=0.6, width=0.7, palette="Set2")

    ax = sns.boxplot(x="Metric", y="Values", hue="Method", data=df, showfliers=False, whis=0.6, width=0.75, palette="Set2")

    # ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')

    # plt.ylim(0, 0.8)
    # plt.xlim(-1, 9.0)

    box_pairs = [(('Num_nodes', 'AnalyzER'), ('Num_nodes', 'ERnet')), (('Num_nodes', 'AnalyzER'), ('Num_nodes', 'ERnet-v2')), (('Num_nodes', 'AnalyzER'), ('Num_nodes', 'nERdy')), (('Num_nodes', 'AnalyzER'), ('Num_nodes', 'nERdy+')), (('Num_nodes', 'ERnet'), ('Num_nodes', 'ERnet-v2')), (('Num_nodes', 'ERnet'), ('Num_nodes', 'nERdy')), (('Num_nodes', 'ERnet'), ('Num_nodes', 'nERdy+')), (('Num_nodes', 'ERnet-v2'), ('Num_nodes', 'nERdy')), (('Num_nodes', 'ERnet-v2'), ('Num_nodes', 'nERdy+')), (('Num_nodes', 'nERdy'), ('Num_nodes', 'nERdy+')), \
                 (('Num_edges', 'AnalyzER'), ('Num_edges', 'ERnet')), (('Num_edges', 'AnalyzER'), ('Num_edges', 'ERnet-v2')), (('Num_edges', 'AnalyzER'), ('Num_edges', 'nERdy')), (('Num_edges', 'AnalyzER'), ('Num_edges', 'nERdy+')), (('Num_edges', 'ERnet'), ('Num_edges', 'ERnet-v2')), (('Num_edges', 'ERnet'), ('Num_edges', 'nERdy')), (('Num_edges', 'ERnet'), ('Num_edges', 'nERdy+')), (('Num_edges', 'ERnet-v2'), ('Num_edges', 'nERdy')), (('Num_edges', 'ERnet-v2'), ('Num_edges', 'nERdy+')), (('Num_edges', 'nERdy'), ('Num_edges', 'nERdy+')), \
                 (('Assortativity', 'AnalyzER'), ('Assortativity', 'ERnet')), (('Assortativity', 'AnalyzER'), ('Assortativity', 'ERnet-v2')), (('Assortativity', 'AnalyzER'), ('Assortativity', 'nERdy')), (('Assortativity', 'AnalyzER'), ('Assortativity', 'nERdy+')), (('Assortativity', 'ERnet'), ('Assortativity', 'ERnet-v2')), (('Assortativity', 'ERnet'), ('Assortativity', 'nERdy')), (('Assortativity', 'ERnet'), ('Assortativity', 'nERdy+')), (('Assortativity', 'ERnet-v2'), ('Assortativity', 'nERdy')), (('Assortativity', 'ERnet-v2'), ('Assortativity', 'nERdy+')), (('Assortativity', 'nERdy'), ('Assortativity', 'nERdy+')),\
                 (('Clustering', 'AnalyzER'), ('Clustering', 'ERnet')), (('Clustering', 'AnalyzER'), ('Clustering', 'ERnet-v2')), (('Clustering', 'AnalyzER'), ('Clustering', 'nERdy')), (('Clustering', 'AnalyzER'), ('Clustering', 'nERdy+')), (('Clustering', 'ERnet'), ('Clustering', 'ERnet-v2')), (('Clustering', 'ERnet'), ('Clustering', 'nERdy')), (('Clustering', 'ERnet'), ('Clustering', 'nERdy+')), (('Clustering', 'ERnet-v2'), ('Clustering', 'nERdy')), (('Clustering', 'ERnet-v2'), ('Clustering', 'nERdy+')), (('Clustering', 'nERdy'), ('Clustering', 'nERdy+')), \
                 (('Num_components', 'AnalyzER'), ('Num_components', 'ERnet')), (('Num_components', 'AnalyzER'), ('Num_components', 'ERnet-v2')), (('Num_components', 'AnalyzER'), ('Num_components', 'nERdy')), (('Num_components', 'AnalyzER'), ('Num_components', 'nERdy+')), (('Num_components', 'ERnet'), ('Num_components', 'ERnet-v2')), (('Num_components', 'ERnet'), ('Num_components', 'nERdy')), (('Num_components', 'ERnet'), ('Num_components', 'nERdy+')), (('Num_components', 'ERnet-v2'), ('Num_components', 'nERdy')), (('Num_components', 'ERnet-v2'), ('Num_components', 'nERdy+')), (('Num_components', 'nERdy'), ('Num_components', 'nERdy+')), \
                 (('Ratio_edges', 'AnalyzER'), ('Ratio_edges', 'ERnet')), (('Ratio_edges', 'AnalyzER'), ('Ratio_edges', 'ERnet-v2')), (('Ratio_edges', 'AnalyzER'), ('Ratio_edges', 'nERdy')), (('Ratio_edges', 'AnalyzER'), ('Ratio_edges', 'nERdy+')), (('Ratio_edges', 'ERnet'), ('Ratio_edges', 'ERnet-v2')), (('Ratio_edges', 'ERnet'), ('Ratio_edges', 'nERdy')), (('Ratio_edges', 'ERnet'), ('Ratio_edges', 'nERdy+')), (('Ratio_edges', 'ERnet-v2'), ('Ratio_edges', 'nERdy')), (('Ratio_edges', 'ERnet-v2'), ('Ratio_edges', 'nERdy+')), (('Ratio_edges', 'nERdy'), ('Ratio_edges', 'nERdy+')), \
                 (('Ratio_nodes', 'AnalyzER'), ('Ratio_nodes', 'ERnet')), (('Ratio_nodes', 'AnalyzER'), ('Ratio_nodes', 'ERnet-v2')), (('Ratio_nodes', 'AnalyzER'), ('Ratio_nodes', 'nERdy')), (('Ratio_nodes', 'AnalyzER'), ('Ratio_nodes', 'nERdy+')), (('Ratio_nodes', 'ERnet'), ('Ratio_nodes', 'ERnet-v2')), (('Ratio_nodes', 'ERnet'), ('Ratio_nodes', 'nERdy')), (('Ratio_nodes', 'ERnet'), ('Ratio_nodes', 'nERdy+')), (('Ratio_nodes', 'ERnet-v2'), ('Ratio_nodes', 'nERdy')), (('Ratio_nodes', 'ERnet-v2'), ('Ratio_nodes', 'nERdy+')), (('Ratio_nodes', 'nERdy'), ('Ratio_nodes', 'nERdy+')), \
                 (('Global_efficiency', 'AnalyzER'), ('Global_efficiency', 'ERnet')), (('Global_efficiency', 'AnalyzER'), ('Global_efficiency', 'ERnet-v2')), (('Global_efficiency', 'AnalyzER'), ('Global_efficiency', 'nERdy')), (('Global_efficiency', 'AnalyzER'), ('Global_efficiency', 'nERdy+')), (('Global_efficiency', 'ERnet'), ('Global_efficiency', 'ERnet-v2')), (('Global_efficiency', 'ERnet'), ('Global_efficiency', 'nERdy')), (('Global_efficiency', 'ERnet'), ('Global_efficiency', 'nERdy+')), (('Global_efficiency', 'ERnet-v2'), ('Global_efficiency', 'nERdy')), (('Global_efficiency', 'ERnet-v2'), ('Global_efficiency', 'nERdy+')), (('Global_efficiency', 'nERdy'), ('Global_efficiency', 'nERdy+')), \
                 (('Density', 'AnalyzER'), ('Density', 'ERnet')), (('Density', 'AnalyzER'), ('Density', 'ERnet-v2')), (('Density', 'AnalyzER'), ('Density', 'nERdy')), (('Density', 'AnalyzER'), ('Density', 'nERdy+')), (('Density', 'ERnet'), ('Density', 'ERnet-v2')), (('Density', 'ERnet'), ('Density', 'nERdy')), (('Density', 'ERnet'), ('Density', 'nERdy+')), (('Density', 'ERnet-v2'), ('Density', 'nERdy')), (('Density', 'ERnet-v2'), ('Density', 'nERdy+')), (('Density', 'nERdy'), ('Density', 'nERdy+'))
                 ]

    # annot = Annotator(ax, pairs=box_pairs, loc='inside', line_height=0.008, linewidth=0.9, fontsize='medium')
    # annot.apply()

    for i, metric in enumerate(df['Metric'].unique()):
        plt.axvline(i - 0.5, color='gray', linestyle='--', linewidth=1)


    statannot.add_stat_annotation(ax, x='Metric', y='Values', hue='Method', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='star', loc='inside',
                                   pvalue_thresholds = [[1e-4, "****"], [1e-3, "***"],
                                 [1e-2, "**"], [0.05, "*"]], verbose=0, fontsize='small', line_offset_to_box=0.009, line_offset=0.009,line_height=0.002, text_offset=0.008, linewidth=0.9)
    # statannot.add_stat_annotation(ax, x='Metric', y='Values', hue='Method', data=df, box_pairs=box_pairs,
    #                               test='Mann-Whitney', text_format='star', loc='inside', verbose=0, fontsize='medium', line_height=0.008, text_offset=0.1, linewidth=0.9)

    yt = ax.get_yticks()
    yt = [f'{y:.1f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=9, weight='bold')
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=25)
    # ax.set_xticklabels(['NC', 'RN', 'RE', 'GE', 'D'], fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fancybox=True, shadow=False, fontsize=9)

    ax.set_xticks(np.arange(0, 9, 1), labels=['NN', 'NE', 'AS', 'CL', 'NC', 'RE', 'RN', 'GE', 'D'], minor=False, linespacing=3.5, fontsize=9, weight='bold')

    # plt.xlabel('Graph Property', fontsize=8)
    # plt.ylabel('Relative Error (normalized)', fontsize=8)

    plt.xlabel('')
    plt.ylabel('')

    group_name = {'control': 'Control', 'atl': 'Atlastin', 'climp': 'Climp', 'rtn': 'Reticulon'}

    # plt.title(f'Graph measures: {group_name[group]} (STED)', fontsize=20)
    # plt.title(f'Graph measures: STED data', fontsize=20)
    # plt.subplots_adjust(hspace=0.3)

    # plt.gcf().set_size_inches(4.5, 3.5)
    # plt.gcf().set_size_inches(4, 6)
    plt.gcf().set_size_inches(8, 6.5)
    # plt.gcf().set_size_inches(4, 6)

    # plt.show()
    plt.savefig('sted_graph_errors_v101.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()

sted_graph_errors()
exit()


# Create boxplot
plt.boxplot(data, labels=['Group 1', 'Group 2', 'Group 3'])
plt.xlabel('Groups')
plt.ylabel('Values')
plt.title('Boxplot of Synthetic Data')
plt.grid(True)
plt.show()

exit()



data = [0.022447953154858896, 0.8190502274213899, 0.018270662210985417, 0.008808847926000486, 0.3923956392961265, 0.44190512167854906, 0.0594714100599783, 0.09939470030604589, 0.028434378553302886, 0.09180884180118701, 0.02442037218229184, 0.07658327762593538, 0.6737366002133117, 0.05018980863181603, 0.07911505925634524, 0.02814173583288727, 0.14867782995409415, 0.12961001579357367, 0.04127257389377112, 0.179417077642488, 0.029510804976781567, 0.04532313732280303, 0.21770880100370826, 0.034200200037120196, 0.40782050484967924, 0.8962627441895266, 1.0, 0.037215236357333176, 0.03506884431910056, 0.0, 0.03369743845423027, 0.02350729143034727, 0.028929219188141488, 0.045879200141242145, 0.03792185936341046]

plt.boxplot(data)
plt.title('Boxplot of the Provided Data')
plt.xlabel('Data')
plt.ylabel('Values')
plt.show()
exit()


# l = [0.041666666666666664, 0.015503875968992248, 0.05555555555555555, 0.07692307692307691, 0.02702702702702703, 0.02564102564102564, 0.02702702702702703, 0.01818181818181818, 0.03125, 0.023809523809523808, 0.015723270440251572, 0.025, 0.050925925925925916, 0.019230769230769232]

# print(np.mean(l))
# exit()

# def runner(writer):
#     pref = '/localhome/asa420/MIAL/data/confocal-data/ATL/A9_junc_viz/'
#     for frame in range(100):
#         filename = f'{pref}A9_junc_viz_t{frame:02d}.png'
#         image = imageio.imread(filename)
#         writer.append_data(image)

# def create_sequence():
#     with imageio.get_writer(f'atl9_junc_viz.gif', mode='I', duration=0.5) as writer:
#         runner(writer)

# data = pkl.load(open('nw_graph_analysis/sted_climp_analyzer_graph_err.pkl', 'rb'))

def repair_clust_coeff_data(method):
    with open(f'nw_graph_analysis/sted_climp_{method}_graph_err.pkl', 'rb') as f:
        climp_data = pkl.load(f)

    with open(f'nw_graph_analysis/sted_control_{method}_graph_err.pkl', 'rb') as f:
        control_data = pkl.load(f)

    with open(f'nw_graph_analysis/sted_rtn_{method}_graph_err.pkl', 'rb') as f:
        rtn_data = pkl.load(f)

    with open(f'nw_graph_analysis/sted_{method}_graph_err.pkl', 'rb') as f:
        data = pkl.load(f)

    climp_analyzer_cc = climp_data[3]
    control_analyzer_cc = control_data[3]
    rtn_analyzer_cc = rtn_data[3]

    analyzer_cc = data[3]

    # print(len(climp_data[0]))
    # print(len(climp_analyzer_cc))
    # print(len(control_data[0]))
    # print(len(control_analyzer_cc))
    # print(len(rtn_data[0]))
    # print(len(rtn_analyzer_cc))


    for i in range(4):
        climp_data[3].append(np.mean(climp_analyzer_cc))

    for i in range(4):
        control_data[3].append(np.mean(control_analyzer_cc))

    for i in range(6):
        rtn_data[3].append(np.mean(rtn_analyzer_cc))

    analyzer_cc = climp_analyzer_cc + control_analyzer_cc + rtn_analyzer_cc

    data[3] = analyzer_cc

    return data


def std_data(data):
    return list((data - min(data)) / (max(data) - min(data)))

def get_graph_perf():
    # sted_analyzer = pkl.load(open('nw_graph_analysis/sted_analyzer.pkl', 'rb'))
    # sted_ernet = pkl.load(open('nw_graph_analysis/sted_ernet.pkl', 'rb'))
    # sted_ernet_v2 = pkl.load(open('nw_graph_analysis/sted_ernet_v2.pkl', 'rb'))
    # sted_nerdy = pkl.load(open('nw_graph_analysis/sted_nerdy.pkl', 'rb'))
    # sted_p4m = pkl.load(open('nw_graph_analysis/sted_p4m.pkl', 'rb'))

    # df = pd.DataFrame()

    # val_data = sted_analyzer + sted_ernet + sted_ernet_v2 + sted_nerdy + sted_p4m

    # l1 = [value for sublist in val_data for value in sublist]
    
    # df['Values'] = l1

    # metric_names = []
    # method_names = []

    # sted_analyzer = repair_clust_coeff_data('analyzer')
    # sted_ernet = repair_clust_coeff_data('ernet')
    # sted_ernet_v2 = repair_clust_coeff_data('erv2')
    # sted_nerdy = repair_clust_coeff_data('nerdy')
    # sted_nerdy_p4m = repair_clust_coeff_data('p4m')

    # sted_analyzer = pkl.load(open(f'nw_graph_analysis/sted_analyzer_graph_err.pkl', 'rb'))
    # sted_ernet = pkl.load(open(f'nw_graph_analysis/sted_ernet_graph_err.pkl', 'rb'))
    # sted_ernet_v2 = pkl.load(open(f'nw_graph_analysis/sted_erv2_graph_err.pkl', 'rb'))
    # sted_nerdy = pkl.load(open(f'nw_graph_analysis/sted_nerdy_graph_err.pkl', 'rb'))
    # sted_nerdy_p4m = pkl.load(open(f'nw_graph_analysis/sted_p4m_graph_err.pkl', 'rb'))

    # sted_analyzer[3] = sted_analyzer[3] + [np.mean(sted_analyzer[3])]*14
    # sted_ernet[3] = sted_ernet[3] + [np.mean(sted_ernet[3])]*14
    # sted_ernet_v2[3] = sted_ernet_v2[3] + [np.mean(sted_ernet_v2[3])]*14
    # sted_nerdy[3] = sted_nerdy[3] + [np.mean(sted_nerdy[3])]*14
    # sted_nerdy_p4m[3] = sted_nerdy_p4m[3] + [np.mean(sted_nerdy_p4m[3])]*14

    sted_analyzer = pkl.load(open(f'nw_graph_analysis/sted_control_analyzer_graph_err.pkl', 'rb'))
    sted_ernet = pkl.load(open(f'nw_graph_analysis/sted_control_ernet_graph_err.pkl', 'rb'))
    sted_ernet_v2 = pkl.load(open(f'nw_graph_analysis/sted_control_erv2_graph_err.pkl', 'rb'))
    sted_nerdy = pkl.load(open(f'nw_graph_analysis/sted_control_nerdy_graph_err.pkl', 'rb'))
    sted_nerdy_p4m = pkl.load(open(f'nw_graph_analysis/sted_control_p4m_graph_err.pkl', 'rb'))

    sted_analyzer[3] = sted_analyzer[3] + [np.mean(sted_analyzer[3])]*4
    sted_ernet[3] = sted_ernet[3] + [np.mean(sted_ernet[3])]*4
    sted_ernet_v2[3] = sted_ernet_v2[3] + [np.mean(sted_ernet_v2[3])]*4
    sted_nerdy[3] = sted_nerdy[3] + [np.mean(sted_nerdy[3])]*4
    sted_nerdy_p4m[3] = sted_nerdy_p4m[3] + [np.mean(sted_nerdy_p4m[3])]*4


    sted_climp_analyzer = np.array(sted_analyzer).T
    sted_climp_ernet = np.array(sted_ernet).T
    sted_climp_ernet_v2 = np.array(sted_ernet_v2).T
    sted_climp_nerdy = np.array(sted_nerdy).T
    sted_climp_nerdy_p4m = np.array(sted_nerdy_p4m).T

    # sted_climp_analyzer = pkl.load(open(f'nw_graph_analysis/sted_{group}_analyzer_graph_err.pkl', 'rb'))
    # sted_climp_ernet = pkl.load(open(f'nw_graph_analysis/sted_{group}_ernet_graph_err.pkl', 'rb'))
    # sted_climp_ernet_v2 = pkl.load(open(f'nw_graph_analysis/sted_{group}_erv2_graph_err.pkl', 'rb'))
    # sted_climp_nerdy = pkl.load(open(f'nw_graph_analysis/sted_{group}_nerdy_graph_err.pkl', 'rb'))
    # sted_climp_nerdy_p4m = pkl.load(open(f'nw_graph_analysis/sted_{group}_p4m_graph_err.pkl', 'rb'))

    # sted_climp_analyzer = np.array(sted_climp_analyzer).T
    # sted_climp_ernet = np.array(sted_climp_ernet).T
    # sted_climp_ernet_v2 = np.array(sted_climp_ernet_v2).T
    # sted_climp_nerdy = np.array(sted_climp_nerdy).T
    # sted_climp_nerdy_p4m = np.array(sted_climp_nerdy_p4m).T


    num_nodes_climp_analyzer = sted_climp_analyzer[0]
    num_nodes_climp_analyzer = std_data(num_nodes_climp_analyzer)
    num_nodes_climp_ernet = sted_climp_ernet[0]
    num_nodes_climp_ernet = std_data(num_nodes_climp_ernet)
    num_nodes_climp_ernet_v2 = sted_climp_ernet_v2[0]
    num_nodes_climp_ernet_v2 = std_data(num_nodes_climp_ernet_v2)
    num_nodes_climp_nerdy = sted_climp_nerdy[0]
    num_nodes_climp_nerdy = std_data(num_nodes_climp_nerdy)
    num_nodes_climp_nerdy_p4m = sted_climp_nerdy_p4m[0]
    num_nodes_climp_nerdy_p4m = std_data(num_nodes_climp_nerdy_p4m)

    num_edges_climp_analyzer = sted_climp_analyzer[1]
    num_edges_climp_analyzer = std_data(num_edges_climp_analyzer)
    num_edges_climp_ernet = sted_climp_ernet[1]
    num_edges_climp_ernet = std_data(num_edges_climp_ernet)
    num_edges_climp_ernet_v2 = sted_climp_ernet_v2[1]
    num_edges_climp_ernet_v2 = std_data(num_edges_climp_ernet_v2)
    num_edges_climp_nerdy = sted_climp_nerdy[1]
    num_edges_climp_nerdy = std_data(num_edges_climp_nerdy)
    num_edges_climp_nerdy_p4m = sted_climp_nerdy_p4m[1]
    num_edges_climp_nerdy_p4m = std_data(num_edges_climp_nerdy_p4m)

    assortativity_climp_analyzer = sted_climp_analyzer[2]
    assortativity_climp_analyzer = std_data(assortativity_climp_analyzer)
    assortativity_climp_ernet = sted_climp_ernet[2]
    assortativity_climp_ernet = std_data(assortativity_climp_ernet)
    assortativity_climp_ernet_v2 = sted_climp_ernet_v2[2]
    assortativity_climp_ernet_v2 = std_data(assortativity_climp_ernet_v2)
    assortativity_climp_nerdy = sted_climp_nerdy[2]
    assortativity_climp_nerdy = std_data(assortativity_climp_nerdy)
    assortativity_climp_nerdy_p4m = sted_climp_nerdy_p4m[2]
    assortativity_climp_nerdy_p4m = std_data(assortativity_climp_nerdy_p4m)

    clustering_climp_analyzer = sted_climp_analyzer[3]
    clustering_climp_analyzer = std_data(clustering_climp_analyzer)
    clustering_climp_ernet = sted_climp_ernet[3]
    clustering_climp_ernet = std_data(clustering_climp_ernet)
    clustering_climp_ernet_v2 = sted_climp_ernet_v2[3]
    clustering_climp_ernet_v2 = std_data(clustering_climp_ernet_v2)
    clustering_climp_nerdy = sted_climp_nerdy[3]
    clustering_climp_nerdy = std_data(clustering_climp_nerdy)
    clustering_climp_nerdy_p4m = sted_climp_nerdy_p4m[3]
    clustering_climp_nerdy_p4m = std_data(clustering_climp_nerdy_p4m)

    num_components_climp_analyzer = sted_climp_analyzer[4]
    num_components_climp_analyzer = std_data(num_components_climp_analyzer)
    num_components_climp_ernet = sted_climp_ernet[4]
    num_components_climp_ernet = std_data(num_components_climp_ernet)
    num_components_climp_ernet_v2 = sted_climp_ernet_v2[4]
    num_components_climp_ernet_v2 = std_data(num_components_climp_ernet_v2)
    num_components_climp_nerdy = sted_climp_nerdy[4]
    num_components_climp_nerdy = std_data(num_components_climp_nerdy)
    num_components_climp_nerdy_p4m = sted_climp_nerdy_p4m[4]
    num_components_climp_nerdy_p4m = std_data(num_components_climp_nerdy_p4m)

    ratio_nodes_climp_analyzer = sted_climp_analyzer[5]
    ratio_nodes_climp_analyzer = std_data(ratio_nodes_climp_analyzer)
    ratio_nodes_climp_ernet = sted_climp_ernet[5]
    ratio_nodes_climp_ernet = std_data(ratio_nodes_climp_ernet)
    ratio_nodes_climp_ernet_v2 = sted_climp_ernet_v2[5]
    ratio_nodes_climp_ernet_v2 = std_data(ratio_nodes_climp_ernet_v2)
    ratio_nodes_climp_nerdy = sted_climp_nerdy[5]
    ratio_nodes_climp_nerdy = std_data(ratio_nodes_climp_nerdy)
    ratio_nodes_climp_nerdy_p4m = sted_climp_nerdy_p4m[5]
    ratio_nodes_climp_nerdy_p4m = std_data(ratio_nodes_climp_nerdy_p4m)

    ratio_edges_climp_analyzer = sted_climp_analyzer[6]
    ratio_edges_climp_analyzer = std_data(ratio_edges_climp_analyzer)
    ratio_edges_climp_ernet = sted_climp_ernet[6]
    ratio_edges_climp_ernet = std_data(ratio_edges_climp_ernet)
    ratio_edges_climp_ernet_v2 = sted_climp_ernet_v2[6]
    ratio_edges_climp_ernet_v2 = std_data(ratio_edges_climp_ernet_v2)
    ratio_edges_climp_nerdy = sted_climp_nerdy[6]
    ratio_edges_climp_nerdy = std_data(ratio_edges_climp_nerdy)
    ratio_edges_climp_nerdy_p4m = sted_climp_nerdy_p4m[6]
    ratio_edges_climp_nerdy_p4m = std_data(ratio_edges_climp_nerdy_p4m)

    global_efficiency_climp_analyzer = sted_climp_analyzer[7]
    global_efficiency_climp_analyzer = std_data(global_efficiency_climp_analyzer)
    global_efficiency_climp_ernet = sted_climp_ernet[7]
    global_efficiency_climp_ernet = std_data(global_efficiency_climp_ernet)
    global_efficiency_climp_ernet_v2 = sted_climp_ernet_v2[7]
    global_efficiency_climp_ernet_v2 = std_data(global_efficiency_climp_ernet_v2)
    global_efficiency_climp_nerdy = sted_climp_nerdy[7]
    global_efficiency_climp_nerdy = std_data(global_efficiency_climp_nerdy)
    global_efficiency_climp_nerdy_p4m = sted_climp_nerdy_p4m[7]
    global_efficiency_climp_nerdy_p4m = std_data(global_efficiency_climp_nerdy_p4m)

    density_climp_analyzer = sted_climp_analyzer[8]
    density_climp_analyzer = std_data(density_climp_analyzer)
    density_climp_ernet = sted_climp_ernet[8]
    density_climp_ernet = std_data(density_climp_ernet)
    density_climp_ernet_v2 = sted_climp_ernet_v2[8]
    density_climp_ernet_v2 = std_data(density_climp_ernet_v2)
    density_climp_nerdy = sted_climp_nerdy[8]
    density_climp_nerdy = std_data(density_climp_nerdy)
    density_climp_nerdy_p4m = sted_climp_nerdy_p4m[8]
    density_climp_nerdy_p4m = std_data(density_climp_nerdy_p4m)

    df = pd.DataFrame()

    # val_data = assortativity_climp_analyzer + assortativity_climp_ernet + assortativity_climp_ernet_v2 + assortativity_climp_nerdy + assortativity_climp_nerdy_p4m +

    val_data = num_nodes_climp_analyzer + num_nodes_climp_ernet + num_nodes_climp_ernet_v2 + num_nodes_climp_nerdy + num_nodes_climp_nerdy_p4m + num_edges_climp_analyzer + num_edges_climp_ernet + num_edges_climp_ernet_v2 + num_edges_climp_nerdy + num_edges_climp_nerdy_p4m +   clustering_climp_analyzer + clustering_climp_ernet + clustering_climp_ernet_v2 + clustering_climp_nerdy + clustering_climp_nerdy_p4m + assortativity_climp_analyzer + assortativity_climp_ernet + assortativity_climp_ernet_v2 + assortativity_climp_nerdy + assortativity_climp_nerdy_p4m + num_components_climp_analyzer + num_components_climp_ernet + num_components_climp_ernet_v2 + num_components_climp_nerdy + num_components_climp_nerdy_p4m + ratio_nodes_climp_analyzer + ratio_nodes_climp_ernet + ratio_nodes_climp_ernet_v2 + ratio_nodes_climp_nerdy + ratio_nodes_climp_nerdy_p4m + ratio_edges_climp_analyzer + ratio_edges_climp_ernet + ratio_edges_climp_ernet_v2 + ratio_edges_climp_nerdy + ratio_edges_climp_nerdy_p4m +   global_efficiency_climp_analyzer + global_efficiency_climp_ernet + global_efficiency_climp_ernet_v2 + global_efficiency_climp_nerdy + global_efficiency_climp_nerdy_p4m + density_climp_analyzer + density_climp_ernet + density_climp_ernet_v2 + density_climp_nerdy + density_climp_nerdy_p4m

    # val_data = num_components_climp_analyzer + num_components_climp_ernet + num_components_climp_ernet_v2 + num_components_climp_nerdy + num_components_climp_nerdy_p4m + ratio_nodes_climp_analyzer + ratio_nodes_climp_ernet + ratio_nodes_climp_ernet_v2 + ratio_nodes_climp_nerdy + ratio_nodes_climp_nerdy_p4m + ratio_edges_climp_analyzer + ratio_edges_climp_ernet + ratio_edges_climp_ernet_v2 + ratio_edges_climp_nerdy + ratio_edges_climp_nerdy_p4m +   global_efficiency_climp_analyzer + global_efficiency_climp_ernet + global_efficiency_climp_ernet_v2 + global_efficiency_climp_nerdy + global_efficiency_climp_nerdy_p4m + density_climp_analyzer + density_climp_ernet + density_climp_ernet_v2 + density_climp_nerdy + density_climp_nerdy_p4m

    # l1 = [value for sublist in val_data for value in sublist]

    df['Values'] = val_data

    metric_names = []

    metric_names.append(['Num_nodes']*len(num_nodes_climp_analyzer))
    metric_names.append(['Num_nodes']*len(num_nodes_climp_ernet))
    metric_names.append(['Num_nodes']*len(num_nodes_climp_ernet_v2))
    metric_names.append(['Num_nodes']*len(num_nodes_climp_nerdy))
    metric_names.append(['Num_nodes']*len(num_nodes_climp_nerdy_p4m))

    metric_names.append(['Num_edges']*len(num_edges_climp_analyzer))
    metric_names.append(['Num_edges']*len(num_edges_climp_ernet))
    metric_names.append(['Num_edges']*len(num_edges_climp_ernet_v2))
    metric_names.append(['Num_edges']*len(num_edges_climp_nerdy))
    metric_names.append(['Num_edges']*len(num_edges_climp_nerdy_p4m))

    metric_names.append(['Assortativity']*len(assortativity_climp_analyzer))
    metric_names.append(['Assortativity']*len(assortativity_climp_ernet))
    metric_names.append(['Assortativity']*len(assortativity_climp_ernet_v2))
    metric_names.append(['Assortativity']*len(assortativity_climp_nerdy))
    metric_names.append(['Assortativity']*len(assortativity_climp_nerdy_p4m))

    metric_names.append(['Clustering']*len(clustering_climp_analyzer))
    metric_names.append(['Clustering']*len(clustering_climp_ernet))
    metric_names.append(['Clustering']*len(clustering_climp_ernet_v2))
    metric_names.append(['Clustering']*len(clustering_climp_nerdy))
    metric_names.append(['Clustering']*len(clustering_climp_nerdy_p4m))

    metric_names.append(['Num_components']*len(num_components_climp_analyzer))
    metric_names.append(['Num_components']*len(num_components_climp_ernet))
    metric_names.append(['Num_components']*len(num_components_climp_ernet_v2))
    metric_names.append(['Num_components']*len(num_components_climp_nerdy))
    metric_names.append(['Num_components']*len(num_components_climp_nerdy_p4m))

    metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_analyzer))
    metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_ernet))
    metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_ernet_v2))
    metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_nerdy))
    metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_nerdy_p4m))

    metric_names.append(['ratio_edges']*len(ratio_edges_climp_analyzer))
    metric_names.append(['ratio_edges']*len(ratio_edges_climp_ernet))
    metric_names.append(['ratio_edges']*len(ratio_edges_climp_ernet_v2))
    metric_names.append(['ratio_edges']*len(ratio_edges_climp_nerdy))
    metric_names.append(['ratio_edges']*len(ratio_edges_climp_nerdy_p4m))

    metric_names.append(['Global_efficiency']*len(global_efficiency_climp_analyzer))
    metric_names.append(['Global_efficiency']*len(global_efficiency_climp_ernet))
    metric_names.append(['Global_efficiency']*len(global_efficiency_climp_ernet_v2))
    metric_names.append(['Global_efficiency']*len(global_efficiency_climp_nerdy))
    metric_names.append(['Global_efficiency']*len(global_efficiency_climp_nerdy_p4m))

    metric_names.append(['Density']*len(density_climp_analyzer))
    metric_names.append(['Density']*len(density_climp_ernet))
    metric_names.append(['Density']*len(density_climp_ernet_v2))
    metric_names.append(['Density']*len(density_climp_nerdy))
    metric_names.append(['Density']*len(density_climp_nerdy_p4m))


    method_names = []    
    
    method_names.append(['AnalyzER']*len(sted_climp_analyzer[0]))
    method_names.append(['ERnet']*len(sted_climp_ernet[0]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[0]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[0]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[0]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[1]))
    method_names.append(['ERnet']*len(sted_climp_ernet[1]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[1]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[1]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[1]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[2]))
    method_names.append(['ERnet']*len(sted_climp_ernet[2]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[2]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[2]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[2]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[3]))
    method_names.append(['ERnet']*len(sted_climp_ernet[3]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[3]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[3]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[3]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[4]))
    method_names.append(['ERnet']*len(sted_climp_ernet[4]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[4]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[4]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[4]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[5]))
    method_names.append(['ERnet']*len(sted_climp_ernet[5]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[5]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[5]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[5]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[6]))
    method_names.append(['ERnet']*len(sted_climp_ernet[6]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[6]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[6]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[6]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[7]))
    method_names.append(['ERnet']*len(sted_climp_ernet[7]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[7]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[7]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[7]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[8]))
    method_names.append(['ERnet']*len(sted_climp_ernet[8]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[8]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[8]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[8]))

    df['Metric'] = [value for sublist in metric_names for value in sublist]

    df['Method'] = [value for sublist in method_names for value in sublist]

    # sns.set(style="whitegrid")
    ax = sns.boxplot(x="Metric", y="Values", hue="Method", data=df, showfliers=False, whis=0.6, width=0.7, palette="Set2")

    # ax.set_aspect(0.8/ax.get_data_ratio(), adjustable='box')

    # plt.ylim(0, 0.8)
    # plt.xlim(-1, 9.0)

    yt = ax.get_yticks()
    yt = [f'{y:.1f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=12)
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=25)
    # ax.set_xticklabels(['NC', 'RN', 'RE', 'GE', 'D'], fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.005), ncol=5, fancybox=True, shadow=True, fontsize=12)

    ax.set_xticks(np.arange(0, 9, 1), labels=['NN', 'NE', 'AS', 'CL', 'NC', 'RN', 'RE', 'GE', 'D'], minor=False, linespacing=3.5, fontsize=13)

    plt.xlabel('Graph Property', fontsize=16)
    plt.ylabel('Relative Error (normalized)', fontsize=16)

    group_name = {'control': 'Control', 'atl': 'Atlastin', 'climp': 'Climp', 'rtn': 'Reticulon'}

    # plt.title(f'Graph measures: {group_name[group]} (STED)', fontsize=20)
    plt.title(f'Graph measures: STED data', fontsize=20)
    # plt.subplots_adjust(hspace=0.3)

    plt.gcf().set_size_inches(12, 12)
    # plt.gcf().set_size_inches(4, 6)

    plt.show()
    # plt.savefig(f'nw_graph_analysis/sted_graph_err_v3.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    # plt.close()

# get_graph_perf('control')
# get_graph_perf('atl')
# get_graph_perf('climp')
# get_graph_perf('rtn')
# get_graph_perf()
# exit()


def get_conf_graph_perf():
    # sted_analyzer = pkl.load(open('nw_graph_analysis/sted_analyzer.pkl', 'rb'))
    # sted_ernet = pkl.load(open('nw_graph_analysis/sted_ernet.pkl', 'rb'))
    # sted_ernet_v2 = pkl.load(open('nw_graph_analysis/sted_ernet_v2.pkl', 'rb'))
    # sted_nerdy = pkl.load(open('nw_graph_analysis/sted_nerdy.pkl', 'rb'))
    # sted_p4m = pkl.load(open('nw_graph_analysis/sted_p4m.pkl', 'rb'))

    # df = pd.DataFrame()

    # val_data = sted_analyzer + sted_ernet + sted_ernet_v2 + sted_nerdy + sted_p4m

    # l1 = [value for sublist in val_data for value in sublist]
    
    # df['Values'] = l1

    # metric_names = []
    # method_names = []

    analyzer = pkl.load(open(f'nw_graph_analysis/conf_analyzer_graph_err.pkl', 'rb'))
    ernet = pkl.load(open(f'nw_graph_analysis/conf_ernet_graph_err.pkl', 'rb'))
    ernet_v2 = pkl.load(open(f'nw_graph_analysis/conf_erv2_graph_err.pkl', 'rb'))
    nerdy = pkl.load(open(f'nw_graph_analysis/conf_nerdy_graph_err.pkl', 'rb'))
    nerdy_p4m = pkl.load(open(f'nw_graph_analysis/conf_p4m_graph_err.pkl', 'rb'))

    sted_climp_analyzer = np.array(analyzer)
    sted_climp_ernet = np.array(ernet)
    sted_climp_ernet_v2 = np.array(ernet_v2)
    sted_climp_nerdy = np.array(nerdy)
    sted_climp_nerdy_p4m = np.array(nerdy_p4m)

    # climp_analyzer = pkl.load(open(f'nw_graph_analysis/{group}_analyzer_graph_err.pkl', 'rb'))
    # climp_ernet = pkl.load(open(f'nw_graph_analysis/{group}_ernet_graph_err.pkl', 'rb'))
    # climp_ernet_v2 = pkl.load(open(f'nw_graph_analysis/{group}_erv2_graph_err.pkl', 'rb'))
    # climp_nerdy = pkl.load(open(f'nw_graph_analysis/{group}_nerdy_graph_err.pkl', 'rb'))
    # climp_nerdy_p4m = pkl.load(open(f'nw_graph_analysis/{group}_p4m_graph_err.pkl', 'rb'))

    # sted_climp_analyzer = np.array(climp_analyzer)
    # sted_climp_ernet = np.array(climp_ernet)
    # sted_climp_ernet_v2 = np.array(climp_ernet_v2)
    # sted_climp_nerdy = np.array(climp_nerdy)
    # sted_climp_nerdy_p4m = np.array(climp_nerdy_p4m)


    num_nodes_climp_analyzer = sted_climp_analyzer[0]
    num_nodes_climp_analyzer = std_data(num_nodes_climp_analyzer)
    num_nodes_climp_ernet = sted_climp_ernet[0]
    num_nodes_climp_ernet = std_data(num_nodes_climp_ernet)
    num_nodes_climp_ernet_v2 = sted_climp_ernet_v2[0]
    num_nodes_climp_ernet_v2 = std_data(num_nodes_climp_ernet_v2)
    num_nodes_climp_nerdy = sted_climp_nerdy[0]
    num_nodes_climp_nerdy = std_data(num_nodes_climp_nerdy)
    num_nodes_climp_nerdy_p4m = sted_climp_nerdy_p4m[0]
    num_nodes_climp_nerdy_p4m = std_data(num_nodes_climp_nerdy_p4m)

    num_edges_climp_analyzer = sted_climp_analyzer[1]
    num_edges_climp_analyzer = std_data(num_edges_climp_analyzer)
    num_edges_climp_ernet = sted_climp_ernet[1]
    num_edges_climp_ernet = std_data(num_edges_climp_ernet)
    num_edges_climp_ernet_v2 = sted_climp_ernet_v2[1]
    num_edges_climp_ernet_v2 = std_data(num_edges_climp_ernet_v2)
    num_edges_climp_nerdy = sted_climp_nerdy[1]
    num_edges_climp_nerdy = std_data(num_edges_climp_nerdy)
    num_edges_climp_nerdy_p4m = sted_climp_nerdy_p4m[1]
    num_edges_climp_nerdy_p4m = std_data(num_edges_climp_nerdy_p4m)

    assortativity_climp_analyzer = sted_climp_analyzer[2]
    assortativity_climp_analyzer = std_data(assortativity_climp_analyzer)
    assortativity_climp_ernet = sted_climp_ernet[2]
    assortativity_climp_ernet = std_data(assortativity_climp_ernet)
    assortativity_climp_ernet_v2 = sted_climp_ernet_v2[2]
    assortativity_climp_ernet_v2 = std_data(assortativity_climp_ernet_v2)
    assortativity_climp_nerdy = sted_climp_nerdy[2]
    assortativity_climp_nerdy = std_data(assortativity_climp_nerdy)
    assortativity_climp_nerdy_p4m = sted_climp_nerdy_p4m[2]
    assortativity_climp_nerdy_p4m = std_data(assortativity_climp_nerdy_p4m)

    clustering_climp_analyzer = sted_climp_analyzer[3]
    clustering_climp_analyzer = std_data(clustering_climp_analyzer)
    clustering_climp_ernet = sted_climp_ernet[3]
    clustering_climp_ernet = std_data(clustering_climp_ernet)
    clustering_climp_ernet_v2 = sted_climp_ernet_v2[3]
    clustering_climp_ernet_v2 = std_data(clustering_climp_ernet_v2)
    clustering_climp_nerdy = sted_climp_nerdy[3]
    clustering_climp_nerdy = std_data(clustering_climp_nerdy)
    clustering_climp_nerdy_p4m = sted_climp_nerdy_p4m[3]
    clustering_climp_nerdy_p4m = std_data(clustering_climp_nerdy_p4m)

    num_components_climp_analyzer = sted_climp_analyzer[4]
    num_components_climp_analyzer = std_data(num_components_climp_analyzer)
    num_components_climp_ernet = sted_climp_ernet[4]
    num_components_climp_ernet = std_data(num_components_climp_ernet)
    num_components_climp_ernet_v2 = sted_climp_ernet_v2[4]
    num_components_climp_ernet_v2 = std_data(num_components_climp_ernet_v2)
    num_components_climp_nerdy = sted_climp_nerdy[4]
    num_components_climp_nerdy = std_data(num_components_climp_nerdy)
    num_components_climp_nerdy_p4m = sted_climp_nerdy_p4m[4]
    num_components_climp_nerdy_p4m = std_data(num_components_climp_nerdy_p4m)

    ratio_nodes_climp_analyzer = sted_climp_analyzer[5]
    ratio_nodes_climp_analyzer = std_data(ratio_nodes_climp_analyzer)
    ratio_nodes_climp_ernet = sted_climp_ernet[5]
    ratio_nodes_climp_ernet = std_data(ratio_nodes_climp_ernet)
    ratio_nodes_climp_ernet_v2 = sted_climp_ernet_v2[5]
    ratio_nodes_climp_ernet_v2 = std_data(ratio_nodes_climp_ernet_v2)
    ratio_nodes_climp_nerdy = sted_climp_nerdy[5]
    ratio_nodes_climp_nerdy = std_data(ratio_nodes_climp_nerdy)
    ratio_nodes_climp_nerdy_p4m = sted_climp_nerdy_p4m[5]
    ratio_nodes_climp_nerdy_p4m = std_data(ratio_nodes_climp_nerdy_p4m)

    ratio_edges_climp_analyzer = sted_climp_analyzer[6]
    ratio_edges_climp_analyzer = std_data(ratio_edges_climp_analyzer)
    ratio_edges_climp_ernet = sted_climp_ernet[6]
    ratio_edges_climp_ernet = std_data(ratio_edges_climp_ernet)
    ratio_edges_climp_ernet_v2 = sted_climp_ernet_v2[6]
    ratio_edges_climp_ernet_v2 = std_data(ratio_edges_climp_ernet_v2)
    ratio_edges_climp_nerdy = sted_climp_nerdy[6]
    ratio_edges_climp_nerdy = std_data(ratio_edges_climp_nerdy)
    ratio_edges_climp_nerdy_p4m = sted_climp_nerdy_p4m[6]
    ratio_edges_climp_nerdy_p4m = std_data(ratio_edges_climp_nerdy_p4m)

    global_efficiency_climp_analyzer = sted_climp_analyzer[7]
    global_efficiency_climp_analyzer = std_data(global_efficiency_climp_analyzer)
    global_efficiency_climp_ernet = sted_climp_ernet[7]
    global_efficiency_climp_ernet = std_data(global_efficiency_climp_ernet)
    global_efficiency_climp_ernet_v2 = sted_climp_ernet_v2[7]
    global_efficiency_climp_ernet_v2 = std_data(global_efficiency_climp_ernet_v2)
    global_efficiency_climp_nerdy = sted_climp_nerdy[7]
    global_efficiency_climp_nerdy = std_data(global_efficiency_climp_nerdy)
    global_efficiency_climp_nerdy_p4m = sted_climp_nerdy_p4m[7]
    global_efficiency_climp_nerdy_p4m = std_data(global_efficiency_climp_nerdy_p4m)

    density_climp_analyzer = sted_climp_analyzer[8]
    density_climp_analyzer = std_data(density_climp_analyzer)
    density_climp_ernet = sted_climp_ernet[8]
    density_climp_ernet = std_data(density_climp_ernet)
    density_climp_ernet_v2 = sted_climp_ernet_v2[8]
    density_climp_ernet_v2 = std_data(density_climp_ernet_v2)
    density_climp_nerdy = sted_climp_nerdy[8]
    density_climp_nerdy = std_data(density_climp_nerdy)
    density_climp_nerdy_p4m = sted_climp_nerdy_p4m[8]
    density_climp_nerdy_p4m = std_data(density_climp_nerdy_p4m)

    df = pd.DataFrame()

    # val_data = num_components_climp_analyzer + num_components_climp_ernet + num_components_climp_ernet_v2 + num_components_climp_nerdy + num_components_climp_nerdy_p4m + ratio_nodes_climp_analyzer + ratio_nodes_climp_ernet + ratio_nodes_climp_ernet_v2 + ratio_nodes_climp_nerdy + ratio_nodes_climp_nerdy_p4m + ratio_edges_climp_analyzer + ratio_edges_climp_ernet + ratio_edges_climp_ernet_v2 + ratio_edges_climp_nerdy + ratio_edges_climp_nerdy_p4m +   global_efficiency_climp_analyzer + global_efficiency_climp_ernet + global_efficiency_climp_ernet_v2 + global_efficiency_climp_nerdy + global_efficiency_climp_nerdy_p4m + density_climp_analyzer + density_climp_ernet + density_climp_ernet_v2 + density_climp_nerdy + density_climp_nerdy_p4m

    
    val_data = num_nodes_climp_analyzer + num_nodes_climp_ernet + num_nodes_climp_ernet_v2 + num_nodes_climp_nerdy + num_nodes_climp_nerdy_p4m + num_edges_climp_analyzer + num_edges_climp_ernet + num_edges_climp_ernet_v2 + num_edges_climp_nerdy + num_edges_climp_nerdy_p4m + clustering_climp_analyzer + clustering_climp_ernet + clustering_climp_ernet_v2 + clustering_climp_nerdy + clustering_climp_nerdy_p4m + assortativity_climp_analyzer + assortativity_climp_ernet + assortativity_climp_ernet_v2 + assortativity_climp_nerdy + assortativity_climp_nerdy_p4m + num_components_climp_analyzer + num_components_climp_ernet + num_components_climp_ernet_v2 + num_components_climp_nerdy + num_components_climp_nerdy_p4m + ratio_nodes_climp_analyzer + ratio_nodes_climp_ernet + ratio_nodes_climp_ernet_v2 + ratio_nodes_climp_nerdy + ratio_nodes_climp_nerdy_p4m + ratio_edges_climp_analyzer + ratio_edges_climp_ernet + ratio_edges_climp_ernet_v2 + ratio_edges_climp_nerdy + ratio_edges_climp_nerdy_p4m +   global_efficiency_climp_analyzer + global_efficiency_climp_ernet + global_efficiency_climp_ernet_v2 + global_efficiency_climp_nerdy + global_efficiency_climp_nerdy_p4m + density_climp_analyzer + density_climp_ernet + density_climp_ernet_v2 + density_climp_nerdy + density_climp_nerdy_p4m

    # l1 = [value for sublist in val_data for value in sublist]

    df['Values'] = val_data

    metric_names = []

    metric_names.append(['Num_nodes']*len(num_nodes_climp_analyzer))
    metric_names.append(['Num_nodes']*len(num_nodes_climp_ernet))
    metric_names.append(['Num_nodes']*len(num_nodes_climp_ernet_v2))
    metric_names.append(['Num_nodes']*len(num_nodes_climp_nerdy))
    metric_names.append(['Num_nodes']*len(num_nodes_climp_nerdy_p4m))

    metric_names.append(['Num_edges']*len(num_edges_climp_analyzer))
    metric_names.append(['Num_edges']*len(num_edges_climp_ernet))
    metric_names.append(['Num_edges']*len(num_edges_climp_ernet_v2))
    metric_names.append(['Num_edges']*len(num_edges_climp_nerdy))
    metric_names.append(['Num_edges']*len(num_edges_climp_nerdy_p4m))

    metric_names.append(['Assortativity']*len(assortativity_climp_analyzer))
    metric_names.append(['Assortativity']*len(assortativity_climp_ernet))
    metric_names.append(['Assortativity']*len(assortativity_climp_ernet_v2))
    metric_names.append(['Assortativity']*len(assortativity_climp_nerdy))
    metric_names.append(['Assortativity']*len(assortativity_climp_nerdy_p4m))

    metric_names.append(['Clustering']*len(clustering_climp_analyzer))
    metric_names.append(['Clustering']*len(clustering_climp_ernet))
    metric_names.append(['Clustering']*len(clustering_climp_ernet_v2))
    metric_names.append(['Clustering']*len(clustering_climp_nerdy))
    metric_names.append(['Clustering']*len(clustering_climp_nerdy_p4m))

    metric_names.append(['Num_components']*len(num_components_climp_analyzer))
    metric_names.append(['Num_components']*len(num_components_climp_ernet))
    metric_names.append(['Num_components']*len(num_components_climp_ernet_v2))
    metric_names.append(['Num_components']*len(num_components_climp_nerdy))
    metric_names.append(['Num_components']*len(num_components_climp_nerdy_p4m))

    metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_analyzer))
    metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_ernet))
    metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_ernet_v2))
    metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_nerdy))
    metric_names.append(['ratio_nodes']*len(ratio_nodes_climp_nerdy_p4m))

    metric_names.append(['ratio_edges']*len(ratio_edges_climp_analyzer))
    metric_names.append(['ratio_edges']*len(ratio_edges_climp_ernet))
    metric_names.append(['ratio_edges']*len(ratio_edges_climp_ernet_v2))
    metric_names.append(['ratio_edges']*len(ratio_edges_climp_nerdy))
    metric_names.append(['ratio_edges']*len(ratio_edges_climp_nerdy_p4m))

    metric_names.append(['Global_efficiency']*len(global_efficiency_climp_analyzer))
    metric_names.append(['Global_efficiency']*len(global_efficiency_climp_ernet))
    metric_names.append(['Global_efficiency']*len(global_efficiency_climp_ernet_v2))
    metric_names.append(['Global_efficiency']*len(global_efficiency_climp_nerdy))
    metric_names.append(['Global_efficiency']*len(global_efficiency_climp_nerdy_p4m))

    metric_names.append(['Density']*len(density_climp_analyzer))
    metric_names.append(['Density']*len(density_climp_ernet))
    metric_names.append(['Density']*len(density_climp_ernet_v2))
    metric_names.append(['Density']*len(density_climp_nerdy))
    metric_names.append(['Density']*len(density_climp_nerdy_p4m))

    method_names = []    
    
    method_names.append(['AnalyzER']*len(sted_climp_analyzer[0]))
    method_names.append(['ERnet']*len(sted_climp_ernet[0]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[0]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[0]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[0]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[1]))
    method_names.append(['ERnet']*len(sted_climp_ernet[1]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[1]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[1]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[1]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[2]))
    method_names.append(['ERnet']*len(sted_climp_ernet[2]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[2]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[2]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[2]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[3]))
    method_names.append(['ERnet']*len(sted_climp_ernet[3]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[3]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[3]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[3]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[4]))
    method_names.append(['ERnet']*len(sted_climp_ernet[4]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[4]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[4]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[4]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[5]))
    method_names.append(['ERnet']*len(sted_climp_ernet[5]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[5]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[5]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[5]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[6]))
    method_names.append(['ERnet']*len(sted_climp_ernet[6]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[6]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[6]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[6]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[7]))
    method_names.append(['ERnet']*len(sted_climp_ernet[7]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[7]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[7]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[7]))

    method_names.append(['AnalyzER']*len(sted_climp_analyzer[8]))
    method_names.append(['ERnet']*len(sted_climp_ernet[8]))
    method_names.append(['ERnet-v2']*len(sted_climp_ernet_v2[8]))
    method_names.append(['nERdy']*len(sted_climp_nerdy[8]))
    method_names.append(['nERdy+']*len(sted_climp_nerdy_p4m[8]))

    df['Metric'] = [value for sublist in metric_names for value in sublist]

    df['Method'] = [value for sublist in method_names for value in sublist]

    # sns.set(style="whitegrid")
    ax = sns.boxplot(x="Metric", y="Values", hue="Method", data=df, showfliers=False, whis=0.55, width=0.7, palette="Set2")

    # plt.ylim(0, 0.8)
    # plt.xlim(-1, 4.0)

    yt = ax.get_yticks()
    yt = [f'{y:.1f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=12)
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=25)
    # ax.set_xticklabels(['NC', 'RN', 'RE', 'GE', 'D'], fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.005), ncol=5, fancybox=True, shadow=True, fontsize=12)

    ax.set_xticks(np.arange(0, 9, 1), labels=['NN', 'NE', 'AS', 'CL', 'NC', 'RN', 'RE', 'GE', 'D'], minor=False, linespacing=3.5, fontsize=13)

    plt.xlabel('Graph Property', fontsize=16)
    plt.ylabel('Relative Error (normalized)', fontsize=16)

    group_name = {'control': 'Control', 'atl': 'Atlastin', 'climp': 'Climp', 'rtn': 'Reticulon'}

    # plt.title(f'Graph measures: {group_name[group]} (Confocal)', fontsize=20)
    plt.title(f'Graph measures: Confocal data', fontsize=20)
    # plt.subplots_adjust(hspace=0.3)

    plt.gcf().set_size_inches(12, 12)
    # plt.gcf().set_size_inches(4, 6)

    # plt.show()
    plt.savefig(f'nw_graph_analysis/confocal_graph_err_v3.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.close()


# get_conf_graph_perf()
# exit()
# get_conf_graph_perf('atl')
# get_conf_graph_perf('climp')
# get_conf_graph_perf('rtn')
# get_conf_graph_perf('control')
# exit()


def get_group_seg_perf(group):
    conf_dice = pkl.load(open(f'nw_graph_analysis/conf_{group}_dice.pkl', 'rb'))
    conf_dice_p4m = pkl.load(open(f'nw_graph_analysis/conf_{group}_dice_p4m.pkl', 'rb'))
    conf_f1 = pkl.load(open(f'nw_graph_analysis/conf_{group}_f1.pkl', 'rb'))
    conf_f1_p4m = pkl.load(open(f'nw_graph_analysis/conf_{group}_f1_p4m.pkl', 'rb'))
    conf_jaccard = pkl.load(open(f'nw_graph_analysis/conf_{group}_jacc.pkl', 'rb'))
    conf_jaccard_p4m = pkl.load(open(f'nw_graph_analysis/conf_{group}_jacc_p4m.pkl', 'rb'))

    # sted_dice = pkl.load(open(f'nw_graph_analysis/sted_{group}_dice.pkl', 'rb'))
    # sted_f1 = pkl.load(open(f'nw_graph_analysis/sted_{group}_f1.pkl', 'rb'))
    # sted_jaccard = pkl.load(open(f'nw_graph_analysis/sted_{group}_iou.pkl', 'rb'))

    df = pd.DataFrame()

    val_data = conf_dice + [conf_dice_p4m] + conf_f1 + [conf_f1_p4m] + conf_jaccard + [conf_jaccard_p4m]

    # val_data = conf_dice + [conf_dice_p4m] + conf_f1 + [conf_f1_p4m] + conf_jaccard + [conf_jaccard_p4m] + sted_dice + sted_f1 + sted_jaccard

    l1 = [value for sublist in val_data for value in sublist]

    df['Values'] = l1

    metric_names = []
    method_names = []
    col_name = []

    for method in conf_dice:
        metric_names.append(['Dice score']*len(method))
    metric_names.append(['Dice score']*len(conf_dice_p4m))

    method_names.append(['ERnet']*len(conf_dice[0]))
    method_names.append(['ERnet-v2']*len(conf_dice[1]))
    method_names.append(['AnalyzER']*len(conf_dice[2]))
    method_names.append(['nERdy']*len(conf_dice[3]))
    method_names.append(['nERdy+']*len(conf_dice_p4m))

    for method in conf_f1:
        metric_names.append(['F1 score']*len(method))
    metric_names.append(['F1 score']*len(conf_f1_p4m))

    method_names.append(['ERnet']*len(conf_f1[0]))
    method_names.append(['ERnet-v2']*len(conf_f1[1]))
    method_names.append(['AnalyzER']*len(conf_f1[2]))
    method_names.append(['nERdy']*len(conf_f1[3]))
    method_names.append(['nERdy+']*len(conf_f1_p4m))

    for method in conf_jaccard:
        metric_names.append(['Jaccard Index']*len(method))
    metric_names.append(['Jaccard Index']*len(conf_jaccard_p4m))

    method_names.append(['ERnet']*len(conf_jaccard[0]))
    method_names.append(['ERnet-v2']*len(conf_jaccard[1]))
    method_names.append(['AnalyzER']*len(conf_jaccard[2]))
    method_names.append(['nERdy']*len(conf_jaccard[3]))
    method_names.append(['nERdy+']*len(conf_jaccard_p4m))

    col_name.append(['Confocal']*(len(conf_dice[0]) + len(conf_dice[1]) + len(conf_dice[2]) + len(conf_dice[3]) + len(conf_dice_p4m) + len(conf_f1[0]) + len(conf_f1[1]) + len(conf_f1[2]) + len(conf_f1[3]) + len(conf_f1_p4m) + len(conf_jaccard[0]) + len(conf_jaccard[1]) + len(conf_jaccard[2]) + len(conf_jaccard[3]) + len(conf_jaccard_p4m)))

    #########################################

    # for method in sted_dice:
    #     metric_names.append(['Dice score']*len(method))
    # for method in sted_f1:
    #     metric_names.append(['F1 score']*len(method))
    # for method in sted_jaccard:
    #     metric_names.append(['Jaccard Index']*len(method))
    
    # method_names.append(['AnalyzER']*len(sted_dice[0]))
    # method_names.append(['ERnet-v2']*len(sted_dice[1]))
    # method_names.append(['ERnet']*len(sted_dice[2]))
    # method_names.append(['nERdy']*len(sted_dice[3]))
    # method_names.append(['nERdy+']*len(sted_dice[4]))

    # method_names.append(['AnalyzER']*len(sted_f1[0]))
    # method_names.append(['ERnet-v2']*len(sted_f1[1]))
    # method_names.append(['ERnet']*len(sted_f1[2]))
    # method_names.append(['nERdy']*len(sted_f1[3]))
    # method_names.append(['nERdy+']*len(sted_f1[4]))

    # method_names.append(['AnalyzER']*len(sted_jaccard[0]))
    # method_names.append(['ERnet-v2']*len(sted_jaccard[1]))
    # method_names.append(['ERnet']*len(sted_jaccard[2]))
    # method_names.append(['nERdy']*len(sted_jaccard[3]))
    # method_names.append(['nERdy+']*len(sted_jaccard[4]))

    # col_name.append(['STED']*(len(sted_dice[0]) + len(sted_dice[1]) + len(sted_dice[2]) + len(sted_dice[3]) + len(sted_dice[4]) + len(sted_f1[0]) + len(sted_f1[1]) + len(sted_f1[2]) + len(sted_f1[3]) + len(sted_f1[4]) + len(sted_jaccard[0]) + len(sted_jaccard[1]) + len(sted_jaccard[2]) + len(sted_jaccard[3]) + len(sted_jaccard[4])))

    #########################################

    l2 = [value for sublist in metric_names for value in sublist]

    df['Metric'] = l2

    df['Method'] = [value for sublist in method_names for value in sublist]

    df['Modality'] = [value for sublist in col_name for value in sublist]


    ax = sns.catplot(kind='box', x="Metric", y="Values", hue="Method", col='Modality', data=df, showfliers=False, width=0.9, palette="Set2", legend=False)

    plt.ylim(0.1, 1.0)
    # plt.xlim(-1, 4.0)

    # yt = ax.get_yticks()
    # yt = [f'{y:.1f}' for y in yt]
    # ax.set_yticklabels(yt, fontsize=14)
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)#, rotation=45)
    ax.set_xticklabels(['Dice score', 'F1 score', 'Jaccard Index'], fontsize=13)
    ax.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], fontsize=13)

    # plt.legend(loc='lower right', bbox_to_anchor=(0.5, 1.005), ncol=5, fancybox=True, shadow=True, fontsize=11)
    # plt.legend(loc='lower left', fontsize=12)

    # plt.legend(loc='lower left', bbox_to_anchor=(-0.95, 0.05), frameon=True, fontsize=12)

    plt.legend(loc='lower left', bbox_to_anchor=(0.05, 0.05), frameon=True, fontsize=12) # For ATL

    # sns.move_legend(ax, 'lower right', bbox_to_anchor=(0.25, .15), fontsize=11)

    # ax.set_xlabel('Metric', fontsize=16)
    # ax.set_ylabel('Value', fontsize=16)

    ax.set_axis_labels('Metric', 'Values', fontsize=16)

    # ax.grid(axis='y')#, linestyle='-', linewidth=0.5, color='white')
    # plt.xlabel('Metric', fontsize=16)
    # plt.ylabel('Value', fontsize=16)

    plt.gcf().set_size_inches(5, 8)

    # plt.show()
    plt.savefig(f'nw_graph_analysis/combined_{group}_segmentation.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.close()
     
# get_group_seg_perf('atl')
# exit()

def get_segmentation_perf():
    conf_dice = pkl.load(open('nw_graph_analysis/conf_dice.pkl', 'rb'))
    conf_dice_p4m = pkl.load(open('nw_graph_analysis/conf_dice_p4m.pkl', 'rb'))
    conf_f1 = pkl.load(open('nw_graph_analysis/conf_f1.pkl', 'rb'))
    conf_f1_p4m = pkl.load(open('nw_graph_analysis/conf_f1_p4m.pkl', 'rb'))
    conf_jaccard = pkl.load(open('nw_graph_analysis/conf_jacc.pkl', 'rb'))
    conf_jaccard_p4m = pkl.load(open('nw_graph_analysis/conf_jacc_p4m.pkl', 'rb'))


    sted_dice = pkl.load(open('nw_graph_analysis/sted_dice.pkl', 'rb'))
    sted_f1 = pkl.load(open('nw_graph_analysis/sted_f1.pkl', 'rb'))
    sted_jaccard = pkl.load(open('nw_graph_analysis/sted_iou.pkl', 'rb'))


    df = pd.DataFrame()

    # val_data = conf_dice + [conf_dice_p4m] + conf_f1 + [conf_f1_p4m] + conf_jaccard + [conf_jaccard_p4m]

    val_data = conf_dice + [conf_dice_p4m] + conf_f1 + [conf_f1_p4m] + conf_jaccard + [conf_jaccard_p4m] + sted_dice + sted_f1 + sted_jaccard

    l1 = [value for sublist in val_data for value in sublist]


    # sted_data = sted_dice + sted_f1 + sted_jaccard

    # l1 = [value for sublist in sted_data for value in sublist]


    # df['Values'] = conf_dice + [conf_dice_p4m] + conf_f1 + [conf_f1_p4m] + conf_jaccard + [conf_jaccard_p4m]# + sted_dice + sted_f1 + sted_jaccard

    df['Values'] = l1

    # print(df['Values'])

    metric_names = []
    method_names = []
    col_name = []


    # Conf

    for method in conf_dice:
        metric_names.append(['Dice score']*len(method))
    metric_names.append(['Dice score']*len(conf_dice_p4m))

    method_names.append(['AnalyzER']*len(conf_dice[0]))
    method_names.append(['ERnet']*len(conf_dice[1]))
    method_names.append(['ERnet-v2']*len(conf_dice[2]))
    method_names.append(['nERdy']*len(conf_dice[3]))
    method_names.append(['nERdy+']*len(conf_dice_p4m))

    for method in conf_f1:
        metric_names.append(['F1 score']*len(method))
    metric_names.append(['F1 score']*len(conf_f1_p4m))

    method_names.append(['AnalyzER']*len(conf_f1[0]))
    method_names.append(['ERnet']*len(conf_f1[1]))
    method_names.append(['ERnet-v2']*len(conf_f1[2]))
    method_names.append(['nERdy']*len(conf_f1[3]))
    method_names.append(['nERdy+']*len(conf_f1_p4m))

    for method in conf_jaccard:
        metric_names.append(['Jaccard Index']*len(method))
    metric_names.append(['Jaccard Index']*len(conf_jaccard_p4m))

    method_names.append(['AnalyzER']*len(conf_jaccard[0]))
    method_names.append(['ERnet']*len(conf_jaccard[1]))
    method_names.append(['ERnet-v2']*len(conf_jaccard[2]))
    method_names.append(['nERdy']*len(conf_jaccard[3]))
    method_names.append(['nERdy+']*len(conf_jaccard_p4m))

    col_name.append(['Confocal']*(len(conf_dice[0]) + len(conf_dice[1]) + len(conf_dice[2]) + len(conf_dice[3]) + len(conf_dice_p4m) + len(conf_f1[0]) + len(conf_f1[1]) + len(conf_f1[2]) + len(conf_f1[3]) + len(conf_f1_p4m) + len(conf_jaccard[0]) + len(conf_jaccard[1]) + len(conf_jaccard[2]) + len(conf_jaccard[3]) + len(conf_jaccard_p4m)))


    #########################################

    for method in sted_dice:
        metric_names.append(['Dice score']*len(method))
    for method in sted_f1:
        metric_names.append(['F1 score']*len(method))
    for method in sted_jaccard:
        metric_names.append(['Jaccard Index']*len(method))

    method_names.append(['AnalyzER']*len(sted_dice[0]))
    method_names.append(['ERnet']*len(sted_dice[1]))
    method_names.append(['ERnet-v2']*len(sted_dice[2]))
    method_names.append(['nERdy']*len(sted_dice[3]))
    method_names.append(['nERdy+']*len(sted_dice[4]))

    method_names.append(['AnalyzER']*len(sted_f1[0]))
    method_names.append(['ERnet']*len(sted_f1[1]))
    method_names.append(['ERnet-v2']*len(sted_f1[2]))
    method_names.append(['nERdy']*len(sted_f1[3]))
    method_names.append(['nERdy+']*len(sted_f1[4]))

    method_names.append(['AnalyzER']*len(sted_jaccard[0]))
    method_names.append(['ERnet']*len(sted_jaccard[1]))
    method_names.append(['ERnet-v2']*len(sted_jaccard[2]))
    method_names.append(['nERdy']*len(sted_jaccard[3]))
    method_names.append(['nERdy+']*len(sted_jaccard[4]))

    col_name.append(['STED']*(len(sted_dice[0]) + len(sted_dice[1]) + len(sted_dice[2]) + len(sted_dice[3]) + len(sted_dice[4]) + len(sted_f1[0]) + len(sted_f1[1]) + len(sted_f1[2]) + len(sted_f1[3]) + len(sted_f1[4]) + len(sted_jaccard[0]) + len(sted_jaccard[1]) + len(sted_jaccard[2]) + len(sted_jaccard[3]) + len(sted_jaccard[4])))

    ###########################################

    l2 = [value for sublist in metric_names for value in sublist]

    df['Metric'] = l2

    df['Method'] = [value for sublist in method_names for value in sublist]

    df['Modality'] = [value for sublist in col_name for value in sublist]

    # print(df)

    # sns.set(style="whitegrid")
    # ax = sns.boxplot(x="Metric", y="Values", hue="Method", data=df, showfliers=False, width=0.9, palette="Set2")

    ax = sns.catplot(kind='box', x="Metric", y="Values", hue="Method", col='Modality', data=df, showfliers=False, width=0.9, palette="Set2", legend=False)

    plt.ylim(0.1, 1.0)
    # plt.xlim(-1, 4.0)

    # yt = ax.get_yticks()
    # yt = [f'{y:.1f}' for y in yt]
    # ax.set_yticklabels(yt, fontsize=14)
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)#, rotation=45)

    ax.set_xticklabels(['Dice score', 'F1 score', 'Jaccard Index'], fontsize=13)
    ax.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], fontsize=13)
    
    plt.legend(loc='lower left', bbox_to_anchor=(-0.95, 0.05), frameon=True, fontsize=12)

    ax.set_axis_labels('Metric', 'Values', fontsize=16)
    plt.gcf().set_size_inches(8, 8)

    # plt.show()
    plt.savefig('nw_graph_analysis/combined_segmentation.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.close()

# get_segmentation_perf()
# exit()
