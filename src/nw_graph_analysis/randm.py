import random

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import statannot
import imageio


# def runner(writer):
#     pref = '/localhome/asa420/MIAL/data/confocal-data/ATL/A9_junc_viz/'
#     for frame in range(100):
#         filename = f'{pref}A9_junc_viz_t{frame:02d}.png'
#         image = imageio.imread(filename)
#         writer.append_data(image)

# def create_sequence():
#     with imageio.get_writer(f'atl9_junc_viz.gif', mode='I', duration=0.5) as writer:
#         runner(writer)



def get_graph_perf():
    sted_analyzer = pkl.load(open('nw_graph_analysis/sted_analyzer.pkl', 'rb'))
    sted_ernet = pkl.load(open('nw_graph_analysis/sted_ernet.pkl', 'rb'))
    sted_ernet_v2 = pkl.load(open('nw_graph_analysis/sted_ernet_v2.pkl', 'rb'))
    sted_nerdy = pkl.load(open('nw_graph_analysis/sted_nerdy.pkl', 'rb'))
    sted_p4m = pkl.load(open('nw_graph_analysis/sted_p4m.pkl', 'rb'))

    df = pd.DataFrame()

    val_data = sted_analyzer + sted_ernet + sted_ernet_v2 + sted_nerdy + sted_p4m

    l1 = [value for sublist in val_data for value in sublist]
    
    df['Values'] = l1

    metric_names = []
    method_names = []
    


def get_group_seg_perf(group):
    conf_dice = pkl.load(open(f'nw_graph_analysis/conf_{group}_dice.pkl', 'rb'))
    conf_dice_p4m = pkl.load(open(f'nw_graph_analysis/conf_{group}_dice_p4m.pkl', 'rb'))
    conf_f1 = pkl.load(open(f'nw_graph_analysis/conf_{group}_f1.pkl', 'rb'))
    conf_f1_p4m = pkl.load(open(f'nw_graph_analysis/conf_{group}_f1_p4m.pkl', 'rb'))
    conf_jaccard = pkl.load(open(f'nw_graph_analysis/conf_{group}_jacc.pkl', 'rb'))
    conf_jaccard_p4m = pkl.load(open(f'nw_graph_analysis/conf_{group}_jacc_p4m.pkl', 'rb'))

    sted_dice = pkl.load(open(f'nw_graph_analysis/sted_{group}_dice.pkl', 'rb'))
    sted_f1 = pkl.load(open(f'nw_graph_analysis/sted_{group}_f1.pkl', 'rb'))
    sted_jaccard = pkl.load(open(f'nw_graph_analysis/sted_{group}_iou.pkl', 'rb'))

    df = pd.DataFrame()

    # val_data = conf_dice + [conf_dice_p4m] + conf_f1 + [conf_f1_p4m] + conf_jaccard + [conf_jaccard_p4m]

    val_data = sted_dice + sted_f1 + sted_jaccard

    l1 = [value for sublist in val_data for value in sublist]

    df['Values'] = l1

    metric_names = []
    method_names = []

    # for method in conf_dice:
    #     metric_names.append(['Dice score']*len(method))
    # metric_names.append(['Dice score']*len(conf_dice_p4m))

    # method_names.append(['ERnet']*len(conf_dice[0]))
    # method_names.append(['ERnet-v2']*len(conf_dice[1]))
    # method_names.append(['AnalyzER']*len(conf_dice[2]))
    # method_names.append(['nERdy']*len(conf_dice[3]))
    # method_names.append(['nERdy+']*len(conf_dice_p4m))

    # for method in conf_f1:
    #     metric_names.append(['F1 score']*len(method))
    # metric_names.append(['F1 score']*len(conf_f1_p4m))

    # method_names.append(['ERnet']*len(conf_f1[0]))
    # method_names.append(['ERnet-v2']*len(conf_f1[1]))
    # method_names.append(['AnalyzER']*len(conf_f1[2]))
    # method_names.append(['nERdy']*len(conf_f1[3]))
    # method_names.append(['nERdy+']*len(conf_f1_p4m))

    # for method in conf_jaccard:
    #     metric_names.append(['Jaccard Index']*len(method))
    # metric_names.append(['Jaccard Index']*len(conf_jaccard_p4m))

    # method_names.append(['ERnet']*len(conf_jaccard[0]))
    # method_names.append(['ERnet-v2']*len(conf_jaccard[1]))
    # method_names.append(['AnalyzER']*len(conf_jaccard[2]))
    # method_names.append(['nERdy']*len(conf_jaccard[3]))
    # method_names.append(['nERdy+']*len(conf_jaccard_p4m))

    #########################################

    for method in sted_dice:
        metric_names.append(['Dice score']*len(method))
    for method in sted_f1:
        metric_names.append(['F1 score']*len(method))
    for method in sted_jaccard:
        metric_names.append(['Jaccard Index']*len(method))
    
    method_names.append(['AnalyzER']*len(sted_dice[0]))
    method_names.append(['ERnet-v2']*len(sted_dice[1]))
    method_names.append(['ERnet']*len(sted_dice[2]))
    method_names.append(['nERdy']*len(sted_dice[3]))
    method_names.append(['nERdy+']*len(sted_dice[4]))

    method_names.append(['AnalyzER']*len(sted_f1[0]))
    method_names.append(['ERnet-v2']*len(sted_f1[1]))
    method_names.append(['ERnet']*len(sted_f1[2]))
    method_names.append(['nERdy']*len(sted_f1[3]))
    method_names.append(['nERdy+']*len(sted_f1[4]))

    method_names.append(['AnalyzER']*len(sted_jaccard[0]))
    method_names.append(['ERnet-v2']*len(sted_jaccard[1]))
    method_names.append(['ERnet']*len(sted_jaccard[2]))
    method_names.append(['nERdy']*len(sted_jaccard[3]))
    method_names.append(['nERdy+']*len(sted_jaccard[4]))

    l2 = [value for sublist in metric_names for value in sublist]

    df['Metric'] = l2

    df['Method'] = [value for sublist in method_names for value in sublist]

    # print(df)

    # sns.set(style="whitegrid")
    ax = sns.boxplot(x="Metric", y="Values", hue="Method", data=df, showfliers=False, width=0.9, palette="Set2")

    plt.ylim(0.1, 1.0)
    # plt.xlim(-1, 4.0)

    yt = ax.get_yticks()
    yt = [f'{y:.1f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)#, rotation=45)

    ax.grid(axis='y')#, linestyle='-', linewidth=0.5, color='white')
    plt.xlabel('Metric', fontsize=16)
    plt.ylabel('Value', fontsize=16)

    plt.gcf().set_size_inches(5, 8)

    # plt.show()
    plt.savefig(f'nw_graph_analysis/sted_{group}_segmentation.png', dpi=300, bbox_inches='tight')

    plt.close()
     
# get_group_seg_perf('climp')
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

    # l1 = [value for sublist in val_data for value in sublist]


    sted_data = sted_dice + sted_f1 + sted_jaccard

    l1 = [value for sublist in sted_data for value in sublist]


    # df['Values'] = conf_dice + [conf_dice_p4m] + conf_f1 + [conf_f1_p4m] + conf_jaccard + [conf_jaccard_p4m]# + sted_dice + sted_f1 + sted_jaccard

    df['Values'] = l1

    # print(df['Values'])

    metric_names = []
    method_names = []


    # Conf

    # for method in conf_dice:
    #     metric_names.append(['Dice score']*len(method))
    # metric_names.append(['Dice score']*len(conf_dice_p4m))

    # method_names.append(['AnalyzER']*len(conf_dice[0]))
    # method_names.append(['ERnet']*len(conf_dice[1]))
    # method_names.append(['ERnet-v2']*len(conf_dice[2]))
    # method_names.append(['nERdy']*len(conf_dice[3]))
    # method_names.append(['nERdy+']*len(conf_dice_p4m))

    # for method in conf_f1:
    #     metric_names.append(['F1 score']*len(method))
    # metric_names.append(['F1 score']*len(conf_f1_p4m))

    # method_names.append(['AnalyzER']*len(conf_f1[0]))
    # method_names.append(['ERnet']*len(conf_f1[1]))
    # method_names.append(['ERnet-v2']*len(conf_f1[2]))
    # method_names.append(['nERdy']*len(conf_f1[3]))
    # method_names.append(['nERdy+']*len(conf_f1_p4m))

    # for method in conf_jaccard:
    #     metric_names.append(['Jaccard Index']*len(method))
    # metric_names.append(['Jaccard Index']*len(conf_jaccard_p4m))

    # method_names.append(['AnalyzER']*len(conf_jaccard[0]))
    # method_names.append(['ERnet']*len(conf_jaccard[1]))
    # method_names.append(['ERnet-v2']*len(conf_jaccard[2]))
    # method_names.append(['nERdy']*len(conf_jaccard[3]))
    # method_names.append(['nERdy+']*len(conf_jaccard_p4m))


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

    ###########################################

    l2 = [value for sublist in metric_names for value in sublist]

    df['Metric'] = l2

    df['Method'] = [value for sublist in method_names for value in sublist]



    # print(df)

    # sns.set(style="whitegrid")
    ax = sns.boxplot(x="Metric", y="Values", hue="Method", data=df, showfliers=False, width=0.9, palette="Set2")

    plt.ylim(0.2, 1.0)
    # plt.xlim(-1, 4.0)

    yt = ax.get_yticks()
    yt = [f'{y:.1f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)#, rotation=45)

    # statannot.add_stat_annotation(ax, data=df, x="Metric", y="Values", hue="Method",
    #                     box_pairs=[(("Dice score", "ERnet"), ("Dice score", "ERnet-v2")),
    #                                     (("Dice score", "ERnet"), ("Dice score", "AnalyzER")),
    #                                     (("Dice score", "ERnet"), ("Dice score", "nERdy")),
    #                                     (("Dice score", "ERnet"), ("Dice score", "nERdy+")),
    #                                     (("Dice score", "ERnet-v2"), ("Dice score", "AnalyzER")),
    #                                     (("Dice score", "ERnet-v2"), ("Dice score", "nERdy")),
    #                                     (("Dice score", "ERnet-v2"), ("Dice score", "nERdy+")),
    #                                     (("Dice score", "AnalyzER"), ("Dice score", "nERdy")),
    #                                     (("Dice score", "AnalyzER"), ("Dice score", "nERdy+")),
    #                                     (("Dice score", "nERdy"), ("Dice score", "nERdy+")),
    #                                     (("F1 score", "ERnet"), ("F1 score", "ERnet-v2")),
    #                                     (("F1 score", "ERnet"), ("F1 score", "AnalyzER")),
    #                                     (("F1 score", "ERnet"), ("F1 score", "nERdy")),
    #                                     (("F1 score", "ERnet"), ("F1 score", "nERdy+")),
    #                                     (("F1 score", "ERnet-v2"), ("F1 score", "AnalyzER")),
    #                                     (("F1 score", "ERnet-v2"), ("F1 score", "nERdy")),
    #                                     (("F1 score", "ERnet-v2"), ("F1 score", "nERdy+")),
    #                                     (("F1 score", "AnalyzER"), ("F1 score", "nERdy")),
    #                                     (("F1 score", "AnalyzER"), ("F1 score", "nERdy+")),
    #                                     (("F1 score", "nERdy"), ("F1 score", "nERdy+")),
    #                                     (("Jaccard Index", "ERnet"), ("Jaccard Index", "ERnet-v2")),
    #                                     (("Jaccard Index", "ERnet"), ("Jaccard Index", "AnalyzER")),
    #                                     (("Jaccard Index", "ERnet"), ("Jaccard Index", "nERdy")),
    #                                     (("Jaccard Index", "ERnet"), ("Jaccard Index", "nERdy+")),
    #                                     (("Jaccard Index", "ERnet-v2"), ("Jaccard Index", "AnalyzER")),
    #                                     (("Jaccard Index", "ERnet-v2"), ("Jaccard Index", "nERdy")),
    #                                     (("Jaccard Index", "ERnet-v2"), ("Jaccard Index", "nERdy+")),
    #                                     (("Jaccard Index", "AnalyzER"), ("Jaccard Index", "nERdy")),
    #                                     (("Jaccard Index", "AnalyzER"), ("Jaccard Index", "nERdy+")),
    #                                     (("Jaccard Index", "nERdy"), ("Jaccard Index", "nERdy+"))],
    #                     test='Mann-Whitney', text_format='star', loc='inside', verbose=2)


    ax.grid(axis='y')#, linestyle='-', linewidth=0.5, color='white')
    plt.xlabel('Metric', fontsize=16)
    plt.ylabel('Value', fontsize=16)

    plt.gcf().set_size_inches(5, 8)

    # plt.show()
    plt.savefig('nw_graph_analysis/sted_segmentation.png', dpi=300, bbox_inches='tight')

    plt.close()


