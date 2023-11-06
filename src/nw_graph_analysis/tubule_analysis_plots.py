import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import statannot
import itertools
import matplotlib.pyplot as plt
import imageio
import sknw

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies'


def get_group_box_pairs(channel):
    groups = (
        ['ATL', 'Climp', 'RTN']
        if channel == 'mch'
        else ['ATL', 'Climp', 'RTN', 'Control']
    )
    return list(itertools.combinations(groups, 2))


def get_box_pairs(channel):
    # Get group names
    if channel == 'mch':
        groups = ['ATL', 'Climp', 'RTN']
    else:
        groups = ['ATL', 'Climp', 'RTN', 'Control']

    # Get region names
    regions = ['R1', 'R2', 'R3']

    # Get all pairs of groups across regions
    box_pairs = []
    for r in regions:
        for i, m1 in enumerate(groups):
            for m2 in groups[i+1:]:
                pair = ((r, m1), (r, m2))
                box_pairs.append(pair)

    return box_pairs


def get_correlation_data_per_replicate(data_egfp, data_mch) -> object:
    """
    Function to get the correlation data for each replicate.
    :param data_egfp: Data from eGFP
    :param data_mch: Data from mCherry
    :return: Correlation data for each replicate
    """

    # Get correlation data for each replicate
    correlation_data_r1 = []
    for tub_eg, tub_mch in zip(data_egfp[:10], data_mch[:10]):
        correlation_data_r1.extend(np.corrcoef(i, j)[0][1] for i, j in zip(tub_eg, tub_mch))

    correlation_data_r2 = []
    for tub_eg, tub_mch in zip(data_egfp[10:20], data_mch[10:20]):
        correlation_data_r2.extend(np.corrcoef(i, j)[0][1] for i, j in zip(tub_eg, tub_mch))

    correlation_data_r3 = []
    for tub_eg, tub_mch in zip(data_egfp[20:], data_mch[20:]):
        correlation_data_r3.extend(np.corrcoef(i, j)[0][1] for i, j in zip(tub_eg, tub_mch))

    return correlation_data_r1, correlation_data_r2, correlation_data_r3


def get_correlation_data_per_replicate(data_egfp, data_mch):
    correlation_data = []
    for tub_eg, tub_mch in zip(data_egfp, data_mch):
        correlation_data.append(np.corrcoef(tub_eg, tub_mch)[0][1])
    return correlation_data


def get_channel_corr(group, connection):
    with open(f'{group.lower()}_{connection}.pkl', 'rb') as f:
        data_egfp = pkl.load(f)

    with open(f'{group.lower()}_{connection}_mch.pkl', 'rb') as f:
        data_mch = pkl.load(f)

    if connection != 'None':
        return get_correlation_data_per_replicate(data_egfp, data_mch)


# def get_pickle_data(group, conn, measure, channel):
#     with open(
#             f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/pickles/{measure}/{group.lower()}_{conn}_{measure}_{channel}.pkl',
#             'rb') as f:
#         data = pkl.load(f)
#     return data


def filter_data(data):
    new_list = [arr for arr in data if arr is not None and not np.all(arr == None)]
    return new_list


def get_per_pixel_variation_over_sequence(group, connection, channel, variation):
    """
    Function to get the per-pixel variation over sequence for a given group, connection, channel and variation.
    :param group: Group name
    :param connection: Connection type
    :param channel: Channel name
    :param variation: Variation type
    :return: Per-pixel variation over sequence
    """

    # data: All tubule intensity data over 100 frames for all movies in the group.

    # Load the data from the pickle file
    data = pkl.load(open(f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/pickles/{group.lower()}_{connection}_tubules_{channel}.pkl', 'rb'))

    # Filter the data
    data = filter_data(data)

    # Extract the variation values
    variation_vals = []
    for series in data:
        for tubule in series:
            transposed_list = list(map(list, zip(*tubule)))
            for pixel_seq_vals in transposed_list:
                if variation == 'mean':
                    variation_vals.append(np.mean(pixel_seq_vals))
                else:
                    variation_vals.append(np.std(pixel_seq_vals))
    return variation_vals


def get_per_tubule_variation_over_sequence(group, connection, channel, variation):
    data = pkl.load(open(f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/pickles/{group.lower()}_{connection}_tubules_{channel}.pkl', 'rb'))

    data = filter_data(data)

    variation_vals = []
    for series in data:
        # print(len(series))
        for tubule in series:
            frame_data = [np.mean(frame) for frame in tubule]
            if variation == 'mean':
                variation_vals.append(np.mean(frame_data))
            else:
                variation_vals.append(np.std(frame_data))

    return variation_vals


def get_per_tubule_correlation_over_sequence(group, connection, channel):
    data = pkl.load(open(f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/pickles/{group.lower()}_{connection}_tubules_{channel}.pkl', 'rb'))

    data = filter_data(data)

    variation_vals = []
    for series in data:
        for tubule in series:
            frame_data = [np.mean(frame) for frame in tubule]
            variation_vals.append(frame_data)
    return variation_vals


def plot_per_tubule_correlation(connection):
    atl_egfp = get_per_tubule_correlation_over_sequence('ATL', connection, 'egfp')
    atl_mch = get_per_tubule_correlation_over_sequence('ATL', connection, 'mch')

    climp_egfp = get_per_tubule_correlation_over_sequence('Climp', connection, 'egfp')
    climp_mch = get_per_tubule_correlation_over_sequence('Climp', connection, 'mch')

    rtn_egfp = get_per_tubule_correlation_over_sequence('RTN', connection, 'egfp')
    rtn_mch = get_per_tubule_correlation_over_sequence('RTN', connection, 'mch')

    df = pd.DataFrame()
    atl_correlation_vals = [
        np.corrcoef(egfp, mch)[0][1] for egfp, mch in zip(atl_egfp, atl_mch)
    ]
    climp_correlation_vals = [
        np.corrcoef(egfp, mch)[0][1]
        for egfp, mch in zip(climp_egfp, climp_mch)
    ]
    rtn_correlation_vals = [
        np.corrcoef(egfp, mch)[0][1] for egfp, mch in zip(rtn_egfp, rtn_mch)
    ]
    df['Per-tubule-correlation'] = pd.Series(np.concatenate((rtn_correlation_vals, climp_correlation_vals, atl_correlation_vals)))

    df['Group'] = pd.Series(np.concatenate((['Reticulon']*len(rtn_correlation_vals), ['Climp']*len(climp_correlation_vals), ['Atlastin']*len(atl_correlation_vals))))

    colors = sns.color_palette(n_colors=4)

    pal = {'Reticulon': colors[1], 'Climp': colors[2], 'Atlastin': colors[3]}

    ax = sns.boxplot(data=df, x='Group', y='Per-tubule-correlation', showfliers=False, width=0.95, palette=pal)
    # ax.set_aspect(1.5)
    # ax.set_ylim(-0.28, 0.62)
    ax.set_ylim(-0.29, 0.75)
    ax.set_xlim(-1, 3.0)

    yt = ax.get_yticks()
    yt = [f'{y:.2f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15, rotation=90)

    # box_pairs = [('Atlastin', 'Climp'), ('Atlastin', 'Reticulon'), ('Climp', 'Reticulon'), ('Control', 'Reticulon'), ('Control', 'Climp'), ('Control', 'Atlastin')]

    # box_pairs = [('Atlastin', 'Climp'), ('Atlastin', 'Reticulon'), ('Climp', 'Reticulon')]

    # statannot.add_stat_annotation(ax, x='Group', y='Per-tubule-correlation', data=df, box_pairs=box_pairs,
    #                               test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=15)

    # ch_name = 'ERmoxGFP' if channel == 'egfp' else 'mCherry'

    # plt.title(f'Per-pixel {variation_name} over sequence \n for {connection} tubules in {ch_name}', fontsize=24)

    plt.grid(True)
    # plt.subplots_adjust(hspace = 1, wspace = 0)
    plt.xlabel('Group', fontsize=18)
    # plt.ylabel(f'Tubular {variation}, log scale', fontsize=18)
    # plt.ylabel(f'Tubular {variation}', fontsize=18)
    plt.ylabel('Cross-correlation over sequence', fontsize=18)
    plt.gcf().set_size_inches(3, 6)
    # plt.savefig('num_juncs_iso_norm.png', bbox_inches='tight', pad_inches=0.6)

    # plt.show()
    plt.savefig(f'Seq_Correlation_Tubule_mean_{connection}_v3.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # plt.savefig(f'Seq_{variation}_Tubule_mean_{connection}_{channel}_v2.png', bbox_inches='tight', pad_inches=0.1)
    # plt.close()

# plot_per_tubule_correlation('iso-iso')
# exit()


def plot_per_tubule_variation_over_sequence(connection, channel, variation):

    df = pd.DataFrame()

    atl_variation = get_per_tubule_variation_over_sequence('ATL', connection, channel, variation)
    rtn_variation = get_per_tubule_variation_over_sequence('RTN', connection, channel, variation)
    climp_variation = get_per_tubule_variation_over_sequence('Climp', connection, channel, variation)
    if channel == 'egfp':
        control_variation = get_per_tubule_variation_over_sequence('Control', connection, channel, variation)
        
        df['Tubule-mean'] = pd.Series(np.concatenate((control_variation, rtn_variation, climp_variation, atl_variation)))

        df['Group'] = pd.Series(np.concatenate((['Control']*len(control_variation), ['Reticulon']*len(rtn_variation), ['Climp']*len(climp_variation), ['Atlastin']*len(atl_variation))))
    else:
        df['Tubule-mean'] = pd.Series(np.concatenate((rtn_variation, climp_variation, atl_variation)))

        df['Group'] = pd.Series(np.concatenate((['Reticulon']*len(rtn_variation), ['Climp']*len(climp_variation), ['Atlastin']*len(atl_variation))))

    # colors = sns.color_palette(n_colors=4)

    # pal = {'Reticulon': colors[1], 'Climp': colors[2], 'Atlastin': colors[3]}

    ax = sns.boxplot(data=df, x='Group', y='Tubule-mean', showfliers=False, width=0.9)#, palette=pal)

    # ax.set_ylim(0, 0.164)
    ax.set_ylim(0, 0.13)

    # ax.set_xlim(-1, 3.0)
    ax.set_xlim(-1, 4.0)
    # box_pairs = get_group_box_pairs(channel)

    # statannot.add_stat_annotation(ax, x='Group', y='Per-pixel-variation', data=df, box_pairs=box_pairs, test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize=20)

    yt = ax.get_yticks()
    yt = [f'{y:.2f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=13)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, rotation=90)

    # box_pairs = [('Atlastin', 'Climp'), ('Atlastin', 'Reticulon'), ('Climp', 'Reticulon'), ('Control', 'Reticulon'), ('Control', 'Climp'), ('Control', 'Atlastin')]

    # box_pairs = [('Atlastin', 'Climp'), ('Atlastin', 'Reticulon'), ('Climp', 'Reticulon')]

    # statannot.add_stat_annotation(ax, x='Group', y='Tubule-mean', data=df, box_pairs=box_pairs,
    #                               test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=15)

    # ch_name = 'ERmoxGFP' if channel == 'egfp' else 'mCherry'
    variation_name = 'standard deviation' if variation == 'std' else 'mean'

    # plt.title(f'Per-pixel {variation_name} over sequence \n for {connection} tubules in {ch_name}', fontsize=24)

    ax.grid(axis='y')
    # plt.grid(True)
    # plt.subplots_adjust(hspace = 1, wspace = 0)
    plt.xlabel('Group', fontsize=15)
    # plt.ylabel(f'Tubular {variation}, log scale', fontsize=18)
    # plt.ylabel(f'Tubular {variation}', fontsize=18)
    plt.ylabel(f'{variation_name} over sequence', fontsize=15)
    plt.gcf().set_size_inches(2.2, 6)
    # plt.savefig('num_juncs_iso_norm.png', bbox_inches='tight', pad_inches=0.6)

    # plt.show()

    plt.savefig(f'Seq_{variation}_Tubule_mean_{connection}_{channel}_v4.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()


# plot_per_tubule_variation_over_sequence('iso-iso', 'mch', 'mean')
# plot_per_tubule_variation_over_sequence('iso-iso', 'mch', 'std')

# plot_per_tubule_variation_over_sequence('iso-iso', 'egfp', 'mean')
# plot_per_tubule_variation_over_sequence('iso-iso', 'egfp', 'std')

# exit()



def plot_per_pixel_variation_over_sequence(connection, channel, variation):
    
    df = pd.DataFrame()
    
    atl_variation = get_per_pixel_variation_over_sequence('ATL', connection, channel, variation)
    rtn_variation = get_per_pixel_variation_over_sequence('RTN', connection, channel, variation)
    climp_variation = get_per_pixel_variation_over_sequence('Climp', connection, channel, variation)
    if channel == 'egfp':
        control_variation = get_per_pixel_variation_over_sequence('Control', connection, channel, variation)
        
        df['Per-pixel-variation'] = pd.Series(np.concatenate((control_variation, rtn_variation, climp_variation, atl_variation)))

        df['Group'] = pd.Series(np.concatenate((['Control']*len(control_variation), ['Reticulon']*len(rtn_variation), ['Climp']*len(climp_variation), ['Atlastin']*len(atl_variation))))
    else:
        df['Per-pixel-variation'] = pd.Series(np.concatenate((rtn_variation, climp_variation, atl_variation)))

        df['Group'] = pd.Series(np.concatenate((['Reticulon']*len(rtn_variation), ['Climp']*len(climp_variation), ['Atlastin']*len(atl_variation))))

    colors = sns.color_palette(n_colors=4)

    pal = {'Reticulon': colors[1], 'Climp': colors[2], 'Atlastin': colors[3]}

    ax = sns.boxplot(data=df, x='Group', y='Per-pixel-variation', showfliers=False, width=0.5, palette=pal)
    # ax.set_ylim(0, 0.3)
    # box_pairs = get_group_box_pairs(channel)

    # statannot.add_stat_annotation(ax, x='Group', y='Per-pixel-variation', data=df, box_pairs=box_pairs, test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize=20)

    yt = ax.get_yticks()
    yt = [f'{y:.2f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=12)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=90)

    ch_name = 'ERmoxGFP' if channel == 'egfp' else 'mCherry'
    variation_name = 'standard deviation' if variation == 'std' else 'mean'

    plt.title(f'Per-pixel {variation_name} over sequence \n for {connection} tubules in {ch_name}', fontsize=15)

    plt.grid(True)
    # plt.subplots_adjust(hspace = 1, wspace = 0)
    plt.xlabel('Group', fontsize=12)
    # plt.ylabel(f'Tubular {variation}, log scale', fontsize=18)
    # plt.ylabel(f'Tubular {variation}', fontsize=18)
    plt.ylabel(f'{variation_name} over sequence', fontsize=12)
    plt.gcf().set_size_inches(2, 6)
    plt.savefig(f'Seq_{variation}_Per_pixel_{connection}_{channel}_v3.png', bbox_inches='tight', pad_inches=0.1)
    # plt.show()
    plt.close()



# ylim 0, 1

# plot_per_pixel_variation_over_sequence('iso-iso', 'egfp', 'std')
# plot_per_pixel_variation_over_sequence('iso-fuz', 'egfp', 'mean')
# plot_per_pixel_variation_over_sequence('fuz-fuz', 'egfp', 'mean')

# exit()

# ylim 0, 0.9
# plot_per_pixel_variation_over_sequence('iso-iso', 'mch', 'mean')
# plot_per_pixel_variation_over_sequence('iso-iso', 'mch', 'std')
# plot_per_pixel_variation_over_sequence('iso-fuz', 'mch', 'mean')
# plot_per_pixel_variation_over_sequence('fuz-fuz', 'mch', 'mean')


# exit()

# ylim 0, 0.35

# plot_per_pixel_variation_over_sequence('iso-iso', 'egfp', 'std')
# plot_per_pixel_variation_over_sequence('iso-fuz', 'egfp', 'std')
# plot_per_pixel_variation_over_sequence('fuz-fuz', 'egfp', 'std')

# exit()

# ylim 0, 0.3

# plot_per_pixel_variation_over_sequence('iso-iso', 'mch', 'std')
# plot_per_pixel_variation_over_sequence('iso-fuz', 'mch', 'std')
# plot_per_pixel_variation_over_sequence('fuz-fuz', 'mch', 'std')
# exit()



def get_per_pixel_correlation_over_sequence(group, connection):
    # egfp_data = pkl.load(open(f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/corrected_pickles/{group.lower()}_{connection}_tubules_egfp.pkl', 'rb'))
    # mch_data = pkl.load(open(f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/corrected_pickles/{group.lower()}_{connection}_tubules_mch.pkl', 'rb'))

    egfp_data = pkl.load(open(f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/{group.lower()}_{connection}_tubules_egfp.pkl', 'rb'))
    mch_data = pkl.load(open(f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/{group.lower()}_{connection}_tubules_mch.pkl', 'rb'))

    egfp_data = filter_data(egfp_data)
    mch_data = filter_data(mch_data)

    correlation_vals = []

    for egfp, mch in zip(egfp_data, mch_data):
        for egfp_tubule, mch_tubule in zip(egfp, mch):
            egfp_transposed_list = list(map(list, zip(*egfp_tubule)))
            mch_transposed_list = list(map(list, zip(*mch_tubule)))

            for egfp_pixel_seq_vals, mch_pixel_seq_vals in zip(egfp_transposed_list, mch_transposed_list):
                correlation_vals.append(np.corrcoef(egfp_pixel_seq_vals, mch_pixel_seq_vals)[0][1])
    
    return correlation_vals


def plot_per_pixel_correlation_over_sequence(connection):
    atl_corr_vals = get_per_pixel_correlation_over_sequence('ATL', connection)
    climp_corr_vals = get_per_pixel_correlation_over_sequence('Climp', connection)
    rtn_corr_vals = get_per_pixel_correlation_over_sequence('RTN', connection)
    # control_corr_vals = get_per_pixel_correlation_over_sequence('Control', connection)

    # sns.distplot(atl_corr_vals, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3}, label='ATL')
    # sns.distplot(climp_corr_vals, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3}, label='Climp')
    # sns.distplot(rtn_corr_vals, hist=False, kde=True, kde_kws={'shade': True, 'linewidth': 3}, label='RTN')
    # plt.show()

    # exit()

    df = pd.DataFrame()
    
    df['Per-pixel-corr'] = pd.Series(np.concatenate((rtn_corr_vals, climp_corr_vals, atl_corr_vals)))
    
    df['Group'] = pd.Series(np.concatenate((['RTN']*len(rtn_corr_vals), ['Climp']*len(climp_corr_vals), ['ATL']*len(atl_corr_vals))))

    ax = sns.violinplot(data=df, x='Group', y='Per-pixel-corr')
    ax.set_ylim(-0.4, 0.85)

    # box_pairs = get_group_box_pairs('mch')
    # statannot.add_stat_annotation(ax, x='Group', y='Per-pixel-corr', data=df, box_pairs=box_pairs, test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize=20)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)
    yt = ax.get_yticks()
    yt = [f'{y:.2f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=18)

    # plt.yscale('log')


    # ch_name = 'ERmoxGFP' if channel == 'egfp' else 'mCherry'
    # variation_name = 'standard deviation' if variation == 'std' else 'mean'

    plt.title(f'Per pixel cross-correlation over the sequence in {connection} tubules', fontsize=24)

    plt.grid(True)
    plt.subplots_adjust(hspace = 1, wspace = 0)
    plt.xlabel('Group', fontsize=24)
    # plt.ylabel(f'Tubular {variation}, log scale', fontsize=18)
    # plt.ylabel(f'Tubular {variation}', fontsize=18)
    plt.ylabel('Cross-correlation value', fontsize=24)
    plt.show()

# plot_per_pixel_correlation_over_sequence('iso-iso')
# plot_per_pixel_correlation_over_sequence('iso-fuz')
# plot_per_pixel_correlation_over_sequence('fuz-fuz')

# exit()


# plot_per_pixel_variation_over_sequence('iso-iso', 'egfp', 'mean')
# plot_per_pixel_variation_over_sequence('iso-fuz', 'egfp', 'mean')
# plot_per_pixel_variation_over_sequence('fuz-fuz', 'egfp', 'mean')

# plot_per_pixel_variation_over_sequence('iso-iso', 'egfp', 'std')
# plot_per_pixel_variation_over_sequence('iso-fuz', 'egfp', 'std')
# plot_per_pixel_variation_over_sequence('fuz-fuz', 'egfp', 'std')

# plot_per_pixel_variation_over_sequence('iso-iso', 'mch', 'mean')
# plot_per_pixel_variation_over_sequence('iso-fuz', 'mch', 'mean')
# plot_per_pixel_variation_over_sequence('fuz-fuz', 'mch', 'mean')

# plot_per_pixel_variation_over_sequence('iso-iso', 'mch', 'std')
# plot_per_pixel_variation_over_sequence('iso-fuz', 'mch', 'std')
# plot_per_pixel_variation_over_sequence('fuz-fuz', 'mch', 'std')


def per_pixel_correlation(egfp_data, mch_data):
    egfp = filter_data(egfp_data)
    mch = filter_data(mch_data)

    corr_vals = []

    for s1, s2 in zip(egfp, mch):
        for t1, t2 in zip(s1, s2):
            transposed_egfp = list(map(list, zip(*t1)))
            transposed_mch = list(map(list, zip(*t2)))
            corr_vals.extend(np.corrcoef(e1, e2)[0, 1] for e1, e2 in zip(transposed_egfp, transposed_mch))

    return corr_vals


def get_group_correlation_data(group, connection):
    egfp = pkl.load(open(f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/{group.lower()}_{connection}_tubules_egfp.pkl', 'rb'))
    mch = pkl.load(open(f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/{group.lower()}_{connection}_tubules_mch.pkl', 'rb'))

    return egfp, mch


def get_replicate_data(egfp, mch):
    r1e, r1m = egfp[:10], mch[:10]

    r2e, r2m = egfp[10:20], mch[10:20]
    r3e, r3m = egfp[20:], mch[20:]

    r1 = per_pixel_correlation(r1e, r1m)
    r2 = per_pixel_correlation(r2e, r2m)
    r3 = per_pixel_correlation(r3e, r3m)

    return r1, r2, r3



def plot_per_pixel_correlation(connection, plottype):
    # Get data for each group
    atl_egfp, atl_mch = get_group_correlation_data('ATL', connection)
    climp_egfp, climp_mch = get_group_correlation_data('Climp', connection)
    rtn_egfp, rtn_mch = get_group_correlation_data('RTN', connection)

    df = pd.DataFrame()

    if plottype == 'all':
        atl = per_pixel_correlation(atl_egfp, atl_mch)
        climp = per_pixel_correlation(climp_egfp, climp_mch)
        rtn = per_pixel_correlation(rtn_egfp, rtn_mch)

        per_pixel_data = np.concatenate((atl, climp, rtn))
        group_labels = np.concatenate((['ATL'] * len(atl), ['Climp'] * len(climp), ['RTN'] * len(rtn)))

        df['Per-pixel-variation'] = pd.Series(per_pixel_data)
        df['Group'] = pd.Series(group_labels)

        box_pairs = get_group_box_pairs('mch')

        ax = sns.boxenplot(data=df, x='Group', y='Per-pixel-variation')

        statannot.add_stat_annotation(ax, x='Group', y='Per-pixel-variation', data=df, box_pairs=box_pairs,
                                      test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    else:
        ar1, ar2, ar3 = get_replicate_data(atl_egfp, atl_mch)
        cr1, cr2, cr3 = get_replicate_data(climp_egfp, climp_mch)
        rr1, rr2, rr3 = get_replicate_data(rtn_egfp, rtn_mch)

        df['Per-pixel-variation'] = pd.Series(np.concatenate((ar1, ar2, ar3, cr1, cr2, cr3, rr1, rr2, rr3)))

        df['Group'] = pd.Series(np.concatenate((
            ['ATL'] * len(ar1), ['ATL'] * len(ar2), ['ATL'] * len(ar3), ['Climp'] * len(cr1), ['Climp'] * len(cr2),
            ['Climp'] * len(cr3), ['RTN'] * len(rr1), ['RTN'] * len(rr2), ['RTN'] * len(rr3))))

        df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(ar1), ['R2'] * len(ar2), ['R3'] * len(ar3), ['R1'] * len(cr1),
                                                    ['R2'] * len(cr2), ['R3'] * len(cr3), ['R1'] * len(rr1), ['R2'] * len(rr2),
                                                    ['R3'] * len(rr3))))

        ax = sns.boxenplot(data=df, x='Replicate', y='Per-pixel-variation', hue='Group', dodge=True)

        box_pairs = get_box_pairs('mch')

        statannot.add_stat_annotation(ax, x='Replicate', y='Per-pixel-variation', hue='Group', data=df, box_pairs=box_pairs, test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    plt.grid(True)
    plt.xlabel('Replicates', fontsize=18)
    plt.ylabel('Cross-correlation values', fontsize=18)

    plt.title(f'Per pixel cross-correlation over 100 frames between ERmoxGFP and mCherry in each {connection} connection tubule', fontsize=18)

    plt.show()


# plot_per_pixel_correlation('fuz-fuz', 'replicate')
# exit()

def plot_per_pixel_variation(connection, channel, variation, plottype):
    atl = get_per_pixel_variation_over_sequence('ATL', connection, channel, variation)
    climp = get_per_pixel_variation_over_sequence('Climp', connection, channel, variation)
    rtn = get_per_pixel_variation_over_sequence('RTN', connection, channel, variation)

    df = pd.DataFrame()

    # ar1, ar2, ar3 = atl[:10], atl[10:20], atl[20:]
    # cr1, cr2, cr3 = climp[:10], climp[10:20], climp[20:]
    # rr1, rr2, rr3 = rtn[:10], rtn[10:20], rtn[20:]

    if channel == 'egfp':
        ctrl = get_per_pixel_variation_over_sequence('Control', connection, channel, variation)
        if plottype == 'all':

            df['Per-pixel-variation'] = pd.Series(np.concatenate((atl, climp, rtn, ctrl)))

            df['Group'] = pd.Series(np.concatenate((['ATL']*len(atl), ['Climp']*len(climp), ['RTN']*len(rtn), ['Control']*len(ctrl))))

        else:
            ctr1, ctr2, ctr3 = ctrl[:10], ctrl[10:20], ctrl[20:]
            df['Per-pixel-variation'] = pd.Series(np.concatenate((ar1, ar2, ar3, cr1, cr2, cr3, rr1, rr2, rr3, ctr1, ctr2, ctr3)))

            df['Group'] = pd.Series(np.concatenate((
                ['ATL'] * len(ar1), ['ATL'] * len(ar2), ['ATL'] * len(ar3), ['Climp'] * len(cr1), ['Climp'] * len(cr2),
                ['Climp'] * len(cr3), ['RTN'] * len(rr1), ['RTN'] * len(rr2), ['RTN'] * len(rr3), ['Control']*len(ctr1), ['Control']*len(ctr2), ['Control']*len(ctr3))))

            df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(ar1), ['R2'] * len(ar2), ['R3'] * len(ar3), ['R1'] * len(cr1),
                                                        ['R2'] * len(cr2), ['R3'] * len(cr3), ['R1'] * len(rr1), ['R2'] * len(rr2),
                                                        ['R3'] * len(rr3), ['R1'] * len(ctr1), ['R2'] * len(ctr2), ['R3'] * len(ctr3))))

    else:
        if plottype == 'all':
            df['Per-pixel-variation'] = pd.Series(np.concatenate((atl, climp, rtn)))

            df['Group'] = pd.Series(np.concatenate((['ATL']*len(atl), ['Climp']*len(climp), ['RTN']*len(rtn))))

        else:
            df['Per-pixel-variation'] = pd.Series(np.concatenate((ar1, ar2, ar3, cr1, cr2, cr3, rr1, rr2, rr3)))

            df['Group'] = pd.Series(np.concatenate((['ATL'] * len(ar1), ['ATL'] * len(ar2), ['ATL'] * len(ar3), ['Climp'] * len(cr1), ['Climp'] * len(cr2), ['Climp'] * len(cr3), ['RTN'] * len(rr1), ['RTN'] * len(rr2), ['RTN'] * len(rr3))))

            df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(ar1), ['R2'] * len(ar2), ['R3'] * len(ar3), ['R1'] * len(cr1), ['R2'] * len(cr2), ['R3'] * len(cr3), ['R1'] * len(rr1), ['R2'] * len(rr2), ['R3'] * len(rr3))))

    if plottype == 'all':
        ax = sns.boxenplot(data=df, x='Group', y='Per-pixel-variation')
        box_pairs = get_group_box_pairs(channel)
        statannot.add_stat_annotation(ax, x='Group', y='Per-pixel-variation', data=df, box_pairs=box_pairs,
                                      test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')
    else:
        ax = sns.boxenplot(data=df, x='Replicate', y='Per-pixel-variation', hue='Group', dodge=True)
        box_pairs = get_box_pairs(channel)
        statannot.add_stat_annotation(ax, x='Replicate', y='Per-pixel-variation', hue='Group', data=df, box_pairs=box_pairs, test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')


    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    # plt.yscale('log')


    ch_name = 'ERmoxGFP' if channel == 'egfp' else 'mCherry'
    variation_name = 'standard deviation' if variation == 'std' else 'Mean'

    plt.title(f'Per-pixel {variation_name} over sequence for {connection} tubules in {ch_name}', fontsize=20)

    plt.grid(True)
    plt.xlabel('Group', fontsize=18)
    # plt.ylabel(f'Tubular {variation}, log scale', fontsize=18)
    # plt.ylabel(f'Tubular {variation}', fontsize=18)
    plt.ylabel(f'Per pixel {variation_name} over 100 frames in each tubule', fontsize=18)
    plt.show()

# plot_per_pixel_variation('iso-iso', 'egfp', 'mean', 'replicates')
# exit()



def get_egfp_plots(connection, channel, variation):

    def get_variation(data, variation):
        if variation == 'mean':
            func = np.mean
        elif variation == 'std':
            func = np.std
        else:
            raise ValueError('Invalid variation')

        result = []

        for subset in [data[:10], data[10:20], data[20:]]:
            subset = filter_data(subset)
            subset_result = [func(i) for each in subset for i in each if len(i) > 0]
            result.append(subset_result)

        return tuple(result)

    atl = get_pickle_data('ATL', connection, 'tub_mean', channel)
    climp = get_pickle_data('Climp', connection, 'tub_mean', channel)
    rtn = get_pickle_data('RTN', connection, 'tub_mean', channel)

    ar1, ar2, ar3 = get_variation(atl, 'mean')
    cr1, cr2, cr3 = get_variation(climp, 'mean')
    rr1, rr2, rr3 = get_variation(rtn, 'mean')

    df = pd.DataFrame()


    if channel == 'egfp':
        ctrl = get_pickle_data('Control', connection, 'tub_mean', channel)
        ctr1, ctr2, ctr3 = get_variation(ctrl, 'mean')

        df['tub-mean'] = pd.Series(np.concatenate((ar1, ar2, ar3, cr1, cr2, cr3, rr1, rr2, rr3, ctr1, ctr2, ctr3)))

        df['Group'] = pd.Series(np.concatenate((
            ['ATL'] * len(ar1), ['ATL'] * len(ar2), ['ATL'] * len(ar3), ['Climp'] * len(cr1),
            ['Climp'] * len(cr2), ['Climp'] * len(cr3), ['RTN'] * len(rr1),
            ['RTN'] * len(rr2), ['RTN'] * len(rr3), ['Control'] * len(ctr1),
            ['Control'] * len(ctr2), ['Control'] * len(ctr3))))

        df['Replicate'] = pd.Series(
            np.concatenate((['R1'] * len(ar1), ['R2'] * len(ar2), ['R3'] * len(ar3), ['R1'] * len(cr1),
                            ['R2'] * len(cr2), ['R3'] * len(cr3), ['R1'] * len(rr1), ['R2'] * len(rr2),
                            ['R3'] * len(rr3), ['R1'] * len(ctr1), ['R2'] * len(ctr2),
                            ['R3'] * len(ctr3))))

    else:
        df['tub-mean'] = pd.Series(np.concatenate((ar1, ar2, ar3, cr1, cr2, cr3, rr1, rr2, rr3)))

        df['Group'] = pd.Series(np.concatenate((
            ['ATL'] * len(ar1), ['ATL'] * len(ar2), ['ATL'] * len(ar3), ['Climp'] * len(cr1),
            ['Climp'] * len(cr2), ['Climp'] * len(cr3), ['RTN'] * len(rr1),
            ['RTN'] * len(rr2), ['RTN'] * len(rr3))))

        df['Replicate'] = pd.Series(
            np.concatenate((['R1'] * len(ar1), ['R2'] * len(ar2), ['R3'] * len(ar3), ['R1'] * len(cr1),
                            ['R2'] * len(cr2), ['R3'] * len(cr3), ['R1'] * len(rr1), ['R2'] * len(rr2),
                            ['R3'] * len(rr3))))

    ax = sns.boxenplot(data=df, x='Replicate', y='tub-mean', hue='Group', dodge=True)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    # plt.yscale('log')

    box_pairs = get_box_pairs(channel)

    statannot.add_stat_annotation(ax, x='Replicate', y='tub-mean', hue='Group', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    ch_name = 'ERmoxGFP' if channel == 'egfp' else 'mCherry'

    if variation == 'std':
        plt.title(f'Standard deviation of sequence for tubule intensity mean ({ch_name}) in {connection} edges',
                  fontsize=18)
    else:
        plt.title(f'Mean of sequence for tubule intensity mean ({ch_name}) in {connection} edges',
                  fontsize=18)
    # plt.suptitle(f'{region_name} CC area across conditions', fontsize=20)
    # plt.title('CC area denotes the total movement of each junction', fontsize=18)
    plt.grid(True)
    plt.xlabel('Replicate', fontsize=18)
    # plt.ylabel(f'Tubular {variation}, log scale', fontsize=18)
    plt.ylabel(f'Tubular {variation}', fontsize=18)
    plt.show()



# get_egfp_plots('fuz-fuz', 'mch', 'std')

# exit()

# tubule_data = get_pickle_data('ATL', 'iso-iso', measure, channel)


def load_corrected_pickles(group, connection, channel):
    return pkl.load(open(f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/corrected_pickles/{group.lower()}_{connection}_tubules_{channel}.pkl', 'rb'))


def load_annot_tub_pickles(group, connection, channel):
    return pkl.load(open(f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/pickles/{group.lower()}_{connection}_tubules_{channel}.pkl', 'rb'))


def get_length_per_tubule(data):
    all_tubules = []
    data = filter_data(data)
    for series in data:
        lengths = [len(tubule_seq[0]) for tubule_seq in series if len(tubule_seq[0]) > 3]
        all_tubules.extend(lengths)
    return all_tubules

def get_length_per_tubule_variation(data, measure):
    all_tubules = []
    data = filter_data(data)
    for series in data:
        if measure == 'var':
            lengths_variance = np.std([len(tubule_seq[0]) for tubule_seq in series if len(tubule_seq[0]) > 3]) ** 2
            all_tubules.append(lengths_variance)
        else:
            lengths_mean = np.mean([len(tubule_seq[0]) for tubule_seq in series if len(tubule_seq[0]) > 3])
            all_tubules.append(lengths_mean)
    return all_tubules


def plot_tubule_length_distribution(connection, channel, measure):

    atl_conn = load_annot_tub_pickles('ATL', connection, channel)
    climp_conn = load_annot_tub_pickles('Climp', connection, channel)
    rtn_conn = load_annot_tub_pickles('RTN', connection, channel)
    ctrl_conn = load_annot_tub_pickles('Control', connection, channel)

    atl_conn_variation = get_length_per_tubule_variation(atl_conn, measure)
    climp_conn_variation = get_length_per_tubule_variation(climp_conn, measure)
    rtn_conn_variation = get_length_per_tubule_variation(rtn_conn, measure)
    ctrl_conn_variation = get_length_per_tubule_variation(ctrl_conn, measure)


    # atl_iso_iso = load_annot_tub_pickles('ATL', 'iso-iso', channel)
    # atl_iso_fuz = load_annot_tub_pickles('ATL', 'iso-fuz', channel)
    # atl_fuz_fuz = load_annot_tub_pickles('ATL', 'fuz-fuz', channel)

    # climp_iso_iso = load_annot_tub_pickles('Climp', 'iso-iso', channel)
    # climp_iso_fuz = load_annot_tub_pickles('Climp', 'iso-fuz', channel)
    # climp_fuz_fuz = load_annot_tub_pickles('Climp', 'fuz-fuz', channel)

    # rtn_iso_iso = load_annot_tub_pickles('RTN', 'iso-iso', channel)
    # rtn_iso_fuz = load_annot_tub_pickles('RTN', 'iso-fuz', channel)
    # rtn_fuz_fuz = load_annot_tub_pickles('RTN', 'fuz-fuz', channel)

    # ctr_iso_iso = load_annot_tub_pickles('Control', 'iso-iso', channel)
    # ctr_iso_fuz = load_annot_tub_pickles('Control', 'iso-fuz', channel)
    # ctr_fuz_fuz = load_annot_tub_pickles('Control', 'fuz-fuz', channel)

    # atl_lengths = get_length_per_tubule(atl_data)
    # climp_lengths = get_length_per_tubule(climp_data)
    # rtn_lengths = get_length_per_tubule(rtn_data)
    # ctr_lengths = get_length_per_tubule(ctr_data)

    # atl_iso_iso_mean = get_length_per_tubule_variation(atl_iso_iso, measure)
    # atl_iso_fuz_mean = get_length_per_tubule_variation(atl_iso_fuz, measure)
    # atl_fuz_fuz_mean = get_length_per_tubule_variation(atl_fuz_fuz, measure)

    # climp_iso_iso_mean = get_length_per_tubule_variation(climp_iso_iso, measure)
    # climp_iso_fuz_mean = get_length_per_tubule_variation(climp_iso_fuz, measure)
    # climp_fuz_fuz_mean = get_length_per_tubule_variation(climp_fuz_fuz, measure)

    # rtn_iso_iso_mean = get_length_per_tubule_variation(rtn_iso_iso, measure)
    # rtn_iso_fuz_mean = get_length_per_tubule_variation(rtn_iso_fuz, measure)
    # rtn_fuz_fuz_mean = get_length_per_tubule_variation(rtn_fuz_fuz, measure)

    # ctr_iso_iso_mean = get_length_per_tubule_variation(ctr_iso_iso, measure)
    # ctr_iso_fuz_mean = get_length_per_tubule_variation(ctr_iso_fuz, measure)
    # ctr_fuz_fuz_mean = get_length_per_tubule_variation(ctr_fuz_fuz, measure)

    # climp_mean = get_length_per_tubule_variation(climp_data, measure)
    # rtn_mean = get_length_per_tubule_variation(rtn_data, measure)
    # ctr_mean = get_length_per_tubule_variation(ctr_data, measure)

    # ymin = min(min(atl_iso_iso_mean), min(atl_iso_fuz_mean), min(climp_iso_iso_mean), min(climp_iso_fuz_mean), min(rtn_iso_iso_mean), min(rtn_iso_fuz_mean), min(ctr_iso_iso_mean), min(ctr_iso_fuz_mean))
    # ymax = max(max(atl_iso_iso_mean), max(atl_iso_fuz_mean), max(climp_iso_iso_mean), max(climp_iso_fuz_mean), max(rtn_iso_iso_mean), max(rtn_iso_fuz_mean), max(ctr_iso_iso_mean), max(ctr_iso_fuz_mean))
    
    
    # ymin = min(min(atl_iso_iso_mean), min(atl_iso_fuz_mean), min(atl_fuz_fuz_mean), min(climp_iso_iso_mean), min(climp_iso_fuz_mean), min(climp_fuz_fuz_mean), min(rtn_iso_iso_mean), min(rtn_iso_fuz_mean), min(rtn_fuz_fuz_mean), min(ctr_iso_iso_mean), min(ctr_iso_fuz_mean), min(ctr_fuz_fuz_mean))
    # ymax = max(max(atl_iso_iso_mean), max(atl_iso_fuz_mean), max(atl_fuz_fuz_mean), max(climp_iso_iso_mean), max(climp_iso_fuz_mean), max(climp_fuz_fuz_mean), max(rtn_iso_iso_mean), max(rtn_iso_fuz_mean), max(rtn_fuz_fuz_mean), max(ctr_iso_iso_mean), max(ctr_iso_fuz_mean), max(ctr_fuz_fuz_mean))

    # ymin = min(min(atl_iso_iso_mean), min(climp_iso_iso_mean), min(rtn_iso_iso_mean), min(ctr_iso_iso_mean))
    # ymax = max(max(atl_iso_iso_mean), max(climp_iso_iso_mean), max(rtn_iso_iso_mean), max(ctr_iso_iso_mean))

    # for mean
    # ymin = 3.0
    # ymax = 17.5

    # for iso-iso
    ymin = 6.0
    ymax = 15.0

    xmin = 6.0
    xmax = 15.0

    # for variation
    # ymin = 0.0
    # ymax = 100.0

    # ymin = 0.0
    # ymax = 60.0

    # xmin = 0.0
    # xmax = 60.0

    # print(atl_conn_variation)
    # print(climp_conn_variation)
    # print(rtn_conn_variation)
    # print(ctrl_conn_variation)

    # mean
    # print(np.median(atl_conn_variation)) # 1
    # print(np.median(climp_conn_variation)) # 30
    # print(np.median(rtn_conn_variation)) # 11
    # print(np.median(ctrl_conn_variation)) # 7

    # variance
    # print(np.median(atl_conn_variation)) # 8
    # print(np.median(climp_conn_variation)) # 12
    # print(np.median(rtn_conn_variation)) # 25
    # print(np.median(ctrl_conn_variation)) # 13

    # exit()


    df = pd.DataFrame()
    df['tub-length'] = pd.Series(np.concatenate((ctrl_conn_variation, rtn_conn_variation, climp_conn_variation, atl_conn_variation)))
    df['Group'] = pd.Series(np.concatenate((['Control'] * len(ctrl_conn_variation), ['Reticulon'] * len(rtn_conn_variation), ['Climp'] * len(climp_conn_variation), ['Atlastin'] * len(atl_conn_variation))))

    # if connection == 'iso-iso':
    #     df['tub-length'] = pd.Series(np.concatenate((ctr_iso_iso_mean, rtn_iso_iso_mean, climp_iso_iso_mean, atl_iso_iso_mean)))
    #     df['Group'] = pd.Series(np.concatenate((['Control'] * len(ctr_iso_iso_mean), ['RTN'] * len(rtn_iso_iso_mean), ['Climp'] * len(climp_iso_iso_mean), ['ATL'] * len(atl_iso_iso_mean))))
    # elif connection == 'iso-fuz':
    #     df['tub-length'] = pd.Series(np.concatenate((ctr_iso_fuz_mean, rtn_iso_fuz_mean, climp_iso_fuz_mean, atl_iso_fuz_mean)))
    #     df['Group'] = pd.Series(np.concatenate((['Control'] * len(ctr_iso_fuz_mean), ['RTN'] * len(rtn_iso_fuz_mean), ['Climp'] * len(climp_iso_fuz_mean), ['ATL'] * len(atl_iso_fuz_mean))))
    # else:
    #     df['tub-length'] = pd.Series(np.concatenate((ctr_fuz_fuz_mean, rtn_fuz_fuz_mean, climp_fuz_fuz_mean, atl_fuz_fuz_mean)))
    #     df['Group'] = pd.Series(np.concatenate((['Control'] * len(ctr_fuz_fuz_mean), ['RTN'] * len(rtn_fuz_fuz_mean), ['Climp'] * len(climp_fuz_fuz_mean), ['ATL'] * len(atl_fuz_fuz_mean))))

    # df['tub-length'] = pd.Series(np.concatenate((ctr_mean, rtn_mean, climp_mean, atl_mean)))
    # df['Group'] = pd.Series(np.concatenate((['Control'] * len(ctr_mean), ['RTN'] * len(rtn_mean), ['Climp'] * len(climp_mean), ['ATL'] * len(atl_mean))))

    # df['tub-length'] = pd.Series(np.concatenate((atl_lengths, climp_lengths, rtn_lengths, ctr_lengths)))
    # df['Group'] = pd.Series(np.concatenate((
            # ['ATL'] * len(atl_lengths), ['Climp'] * len(climp_lengths), ['RTN'] * len(rtn_lengths), ['Control'] * len(ctr_lengths))))
    
    ax = sns.boxplot(data=df, x='Group', y='tub-length', showfliers=False, width=0.9)#, dodge=True)
    # sns.swarmplot(data=df, x='Group', y='tub-length', size=8, dodge=True)

    # ax.set_ylim(6, 18)
    ax.set_xlim(-1, 4.0)
    ax.set_ylim(0, 81)

    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, rotation=90)
    yt = ax.get_yticks()
    yt = [f'{y}' for y in yt]
    ax.set_yticklabels(yt, fontsize=13)

    # plt.yscale('log')

    # box_pairs = [('Atlastin', 'Climp'), ('Atlastin', 'Reticulon'), ('Atlastin', 'Control'), ('Climp', 'Reticulon'), ('Climp', 'Control'),
                #  ('Reticulon', 'Control')]
    # box_pairs = get_group_box_pairs(channel)

    # statannot.add_stat_annotation(ax, x='Group', y='tub-length', data=df, box_pairs=box_pairs,
                                    # test='Mann-Whitney', text_format='star', loc='inside', verbose=2, fontsize=12)
    
    # ch_name = 'ERmoxGFP' if channel == 'egfp' else 'mCherry'
    measure_name = 'variance' if measure == 'var' else 'mean (pixels)'
    # plt.title(f'Tubule length in {connection} connections', fontsize=24)

    if connection == 'iso-iso':
        conn_name = 'isolated-isolated'
    elif connection == 'iso-fuz':
        conn_name = 'isolated-overlapping'
    else:
        conn_name = 'overlapping-overlapping'

    # plt.title(f'Tubule length {measure_name} per sequence \n in {conn_name} connections', fontsize=20)

    # plt.suptitle(f'{region_name} CC area across conditions', fontsize=20)
    # plt.title('CC area denotes the total movement of each junction', fontsize=18)
    plt.grid(True)
    # plt.subplots_adjust(hspace = 1, wspace = 0)
    plt.xlabel('Group', fontsize=15)
    # plt.ylabel('Tubule length (pixels)', fontsize=24)
    plt.ylabel(f'Tubule length {measure_name}', fontsize=15)

    plt.gcf().set_size_inches(2.2, 6)

    # plt.savefig(f'Seq_{measure}_tub_length_{connection}.png', bbox_inches='tight', pad_inches=0.6)
    # plt.show()
    # plt.savefig(f'Seq_{measure}_tub_length_{connection}_annot.png', bbox_inches='tight', pad_inches=0.4)
    plt.savefig(f'Seq_{measure}_tub_length_{connection}_v2.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()


# plot_tubule_length_distribution('iso-iso', 'egfp', 'mean')
# plot_tubule_length_distribution('iso-fuz', 'egfp', 'mean')
# plot_tubule_length_distribution('fuz-fuz', 'egfp', 'mean')

# plot_tubule_length_distribution('iso-iso', 'egfp', 'var')
# plot_tubule_length_distribution('iso-fuz', 'egfp',  'var')
# plot_tubule_length_distribution('fuz-fuz', 'egfp', 'var')
# exit()


def plot_tubule_length_distribution_all():
    atl_ii = load_annot_tub_pickles('ATL', 'iso-iso', 'egfp')
    # atl_if = load_annot_tub_pickles('ATL', 'iso-fuz', 'egfp')
    # atl_ff = load_annot_tub_pickles('ATL', 'fuz-fuz', 'egfp')

    climp_ii = load_annot_tub_pickles('Climp', 'iso-iso', 'egfp')
    # climp_if = load_annot_tub_pickles('Climp', 'iso-fuz', 'egfp')
    # climp_ff = load_annot_tub_pickles('Climp', 'fuz-fuz', 'egfp')

    rtn_ii = load_annot_tub_pickles('RTN', 'iso-iso', 'egfp')
    # rtn_if = load_annot_tub_pickles('RTN', 'iso-fuz', 'egfp')
    # rtn_ff = load_annot_tub_pickles('RTN', 'fuz-fuz', 'egfp')

    ctr_ii = load_annot_tub_pickles('Control', 'iso-iso', 'egfp')
    # ctr_if = load_annot_tub_pickles('Control', 'iso-fuz', 'egfp')
    # ctr_ff = load_annot_tub_pickles('Control', 'fuz-fuz', 'egfp')

    atl_lengths = get_length_per_tubule(atl_ii)# + get_length_per_tubule(atl_if) + get_length_per_tubule(atl_ff)
    climp_lengths = get_length_per_tubule(climp_ii)# + get_length_per_tubule(climp_if) + get_length_per_tubule(climp_ff)
    rtn_lengths = get_length_per_tubule(rtn_ii)# + get_length_per_tubule(rtn_if) + get_length_per_tubule(rtn_ff)
    ctr_lengths = get_length_per_tubule(ctr_ii)# + get_length_per_tubule(ctr_if) + get_length_per_tubule(ctr_ff)

    df = pd.DataFrame()
    df['tub-length'] = pd.Series(np.concatenate((atl_lengths, climp_lengths, rtn_lengths, ctr_lengths)))
    df['Group'] = pd.Series(np.concatenate((
            ['ATL'] * len(atl_lengths), ['Climp'] * len(climp_lengths), ['RTN'] * len(rtn_lengths), ['Control'] * len(ctr_lengths))))
    
    # ax = sns.boxenplot(data=df, x='Group', y='tub-length', dodge=True)
    ax = sns.boxenplot(data=df, x='Group', y='tub-length', showfliers=False, linewidth=2)#, dodge=True)
    # ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    yt = ax.get_yticks()
    yt = [f'{y:.2f}' for y in yt]
    ax.set_yticklabels(yt, fontsize=18)

    # plt.yscale('log')
    channel = 'egfp'

    # box_pairs = get_group_box_pairs(channel)

    # statannot.add_stat_annotation(ax, x='Group', y='tub-length', data=df, box_pairs=box_pairs,
    #                                 test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize=20)
    
    # ch_name = 'ERmoxGFP' if channel == 'egfp' else 'mCherry'
    plt.title('Tubule length in all tubules across groups', fontsize=24)
    # plt.suptitle
    # plt.suptitle(f'{region_name} CC area across conditions', fontsize=20)
    # plt.title('CC area denotes the total movement of each junction', fontsize=18)
    plt.grid(True)
    plt.xlabel('Group', fontsize=24)
    plt.ylabel('Tubule length (pixels)', fontsize=24)
    plt.show()

plot_tubule_length_distribution_all()
exit()


def get_edge_length(group, connection, measure, channel):

    tubule_data = get_pickle_data(group, connection, measure, channel)

    # len(tubule[0]) < 61 and
    def get_length_per_tubule(data):
        all_tubules = []
        data = filter_data(data)
        for series in data:
            lengths = [len(tubule[0]) for tubule in series if len(tubule[0]) > 3]
            all_tubules.extend(lengths)
        return all_tubules

    def get_mean_length_per_series(data):
        mean_lengths = []
        data = filter_data(data)
        for series in data:
            lengths = [len(tubule[0]) for tubule in series]
            mean_lengths.append(np.mean(lengths))
        return mean_lengths

    # r1_mean_lengths = get_mean_length_per_series(tubule_data[:10])
    # r2_mean_lengths = get_mean_length_per_series(tubule_data[10:20])
    # r3_mean_lengths = get_mean_length_per_series(tubule_data[20:])

    # return r1_mean_lengths, r2_mean_lengths, r3_mean_lengths
    r1_length = get_length_per_tubule(tubule_data[:10])
    r2_length = get_length_per_tubule(tubule_data[10:20])
    r3_length = get_length_per_tubule(tubule_data[20:])

    return r1_length, r2_length, r3_length


# ar1_ii, ar2_ii, ar3_ii = get_edge_length('ATL', 'iso-iso', 'tubules', 'egfp')
# cr1_ii, cr2_ii, cr3_ii = get_edge_length('Climp', 'iso-iso', 'tubules', 'egfp')
# rr1_ii, rr2_ii, rr3_ii = get_edge_length('RTN', 'iso-iso', 'tubules', 'egfp')
# ctr1_ii, ctr2_ii, ctr3_ii = get_edge_length('Control', 'iso-iso', 'tubules', 'egfp')
#
# ar1_if, ar2_if, ar3_if = get_edge_length('ATL', 'iso-fuz', 'tubules', 'egfp')
# cr1_if, cr2_if, cr3_if = get_edge_length('Climp', 'iso-fuz', 'tubules', 'egfp')
# rr1_if, rr2_if, rr3_if = get_edge_length('RTN', 'iso-fuz', 'tubules', 'egfp')
# ctr1_if, ctr2_if, ctr3_if = get_edge_length('Control', 'iso-fuz', 'tubules', 'egfp')
#
# ar1_ff, ar2_ff, ar3_ff = get_edge_length('ATL', 'fuz-fuz', 'tubules', 'egfp')
# cr1_ff, cr2_ff, cr3_ff = get_edge_length('Climp', 'fuz-fuz', 'tubules', 'egfp')
# rr1_ff, rr2_ff, rr3_ff = get_edge_length('RTN', 'fuz-fuz', 'tubules', 'egfp')
# ctr1_ff, ctr2_ff, ctr3_ff = get_edge_length('Control', 'fuz-fuz', 'tubules', 'egfp')

# ar1 = ar1_ii + ar1_if + ar1_ff
# cr1 = cr1_ii + cr1_if + cr1_ff
# rr1 = rr1_ii + rr1_if + rr1_ff
# ctr1 = ctr1_ii + ctr1_if + ctr1_ff
#
# ar2 = ar2_ii + ar2_if + ar2_ff
# cr2 = cr2_ii + cr2_if + cr2_ff
# rr2 = rr2_ii + rr2_if + rr2_ff
# ctr2 = ctr2_ii + ctr2_if + ctr2_ff
#
# ar3 = ar3_ii + ar3_if + ar3_ff
# cr3 = cr3_ii + cr3_if + cr3_ff
# rr3 = rr3_ii + rr3_if + rr3_ff
# ctr3 = ctr3_ii + ctr3_if + ctr3_ff

def compare_groups(connection, plottype):

    ar1, ar2, ar3 = get_edge_length('ATL', connection, 'tubules', 'egfp')
    cr1, cr2, cr3 = get_edge_length('Climp', connection, 'tubules', 'egfp')
    rr1, rr2, rr3 = get_edge_length('RTN', connection, 'tubules', 'egfp')
    ctr1, ctr2, ctr3 = get_edge_length('Control', connection, 'tubules', 'egfp')

    df = pd.DataFrame()

    if plottype == 'group':
        a1 = ar1 + ar2 + ar3
        c1 = cr1 + cr2 + cr3
        r1 = rr1 + rr2 + rr3
        ct1 = ctr1 + ctr2 + ctr3

        df['tub-mean'] = pd.Series(np.concatenate((a1, c1, r1, ct1)))
        df['Group'] = pd.Series(np.concatenate((
            ['ATL'] * len(a1), ['Climp'] * len(c1), ['RTN'] * len(r1), ['Control'] * len(ct1))))

        ax = sns.boxenplot(data=df, x='Group', y='tub-mean')

        box_pairs = get_group_box_pairs('egfp')

        statannot.add_stat_annotation(ax, x='Group', y='tub-mean', data=df, box_pairs=box_pairs,
                                      test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')
        plt.xlabel('Group', fontsize=18)

    else:
        df['tub-mean'] = pd.Series(np.concatenate((ar1, ar2, ar3, cr1, cr2, cr3, rr1, rr2, rr3, ctr1, ctr2, ctr3)))

        df['Group'] = pd.Series(np.concatenate((
            ['ATL'] * len(ar1), ['ATL'] * len(ar2), ['ATL'] * len(ar3), ['Climp'] * len(cr1),
            ['Climp'] * len(cr2), ['Climp'] * len(cr3), ['RTN'] * len(rr1),
            ['RTN'] * len(rr2), ['RTN'] * len(rr3), ['Control'] * len(ctr1),
            ['Control'] * len(ctr2), ['Control'] * len(ctr3))))

        df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(ar1), ['R2'] * len(ar2), ['R3'] * len(ar3), ['R1'] * len(cr1), ['R2'] * len(cr2), ['R3'] * len(cr3), ['R1'] * len(rr1), ['R2'] * len(rr2), ['R3'] * len(rr3), ['R1'] * len(ctr1), ['R2'] * len(ctr2), ['R3'] * len(ctr3))))

        ax = sns.boxenplot(data=df, x='Replicate', y='tub-mean', hue='Group', dodge=True)

        box_pairs = get_box_pairs('egfp')

        statannot.add_stat_annotation(ax, x='Replicate', y='tub-mean', hue='Group', data=df, box_pairs=box_pairs,
                                      test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

        plt.xlabel('Replicates', fontsize=18)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    plt.title(f'Tubule length in {connection} connections for all sequences', fontsize=18)
    plt.grid(True)
    plt.ylabel('Tubule length (pixels)', fontsize=18)
    plt.show()

compare_groups('iso-iso', 'group')