import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import statannot
import matplotlib.pyplot as plt


def get_correlation_data_per_replicate(data_egfp, data_mch) -> object:
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


def get_channel_corr(group, connection):
    with open(f'{group.lower()}_{connection}.pkl', 'rb') as f:
        data_egfp = pkl.load(f)

    with open(f'{group.lower()}_{connection}_mch.pkl', 'rb') as f:
        data_mch = pkl.load(f)

    if connection != 'None':
        return get_correlation_data_per_replicate(data_egfp, data_mch)
    correlation_data = []
    for tub_eg, tub_mch in zip(data_egfp, data_mch):
        correlation_data.extend(np.corrcoef(i, j)[0][1] for i, j in zip(tub_eg, tub_mch))
    return correlation_data


def get_pickle_data(group, conn, measure, channel):
    with open(
            f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/pickles/{measure}/{group.lower()}_{conn}_{measure}_{channel}.pkl',
            'rb') as f:
        data = pkl.load(f)
    return data


def get_box_pairs(channel):
    if channel == 'mch':
        groups = ['ATL', 'Climp', 'RTN']
    else:
        groups = ['ATL', 'Climp', 'RTN', 'Control']
    regions = ['R1', 'R2', 'R3']
    box_pairs = []

    for r in regions:
        for i, m1 in enumerate(groups):
            for m2 in groups[i+1:]:
                pair = ((r, m1), (r, m2))
                box_pairs.append(pair)

    return box_pairs


def filter_data(data):
    new_list = [arr for arr in data if arr is not None and not np.all(arr == None)]
    return new_list


# egfp = pkl.load(open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/pickles/tubules/climp_iso-iso_tubules_egfp.pkl', 'rb'))
# mch = pkl.load(open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/pickles/tubules/climp_iso-iso_tubules_mch.pkl', 'rb'))
#
# for e, m in zip(egfp[0], mch[0]):
#     if np.array_equal(e[0], m[0]):
#         print(e[0])
#
#
# # print(egfp[0][0][2])
# # print(mch[0][0][2])
# exit()


# data = get_pickle_data('ATL', 'iso-iso', 'tubule', 'egfp')
# data_mch = get_pickle_data('ATL', 'iso-iso', 'tubule', 'mch')
# print(data[0].shape)

# print(data[0][0][0])
# print(data_mch[0][0][0])


# atl_mch = pkl.load(open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/atl_fuz-fuz_tubules_mch.pkl', 'rb'))
# atl_egfp = pkl.load(open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/atl_fuz-fuz_tubules_egfp.pkl', 'rb'))
# rtn_egfp = pkl.load(open('/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/rtn_fuz-fuz_tubules_egfp.pkl', 'rb'))


# def load_data(group, connection):
#     data_egfp = pkl.load(open(f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/{group.lower()}_{connection}_tubules_egfp.pkl', 'rb'))
#     data_mch = pkl.load(open(f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/{group.lower()}_{connection}_tubules_mch.pkl', 'rb'))
#     return data_egfp, data_mch


# def get_per_pixel_mean_over_sequence(egfp, mch):
#     e_mean = []
#     m_mean = []
#     # l_std = []
#     for e_series, m_series in zip(egfp, mch):
#         for e_tubule, m_tubule in zip(e_series, m_series):
#             e_transposed_list = list(map(list, zip(*e_tubule)))
#             m_transposed_list = list(map(list, zip(*m_tubule)))
#             for e, m in zip(e_transposed_list, m_transposed_list):
#                 e_mean.append(np.mean(e))
#                 m_mean.append(np.mean(m))
#     return e_mean, m_mean


def get_per_pixel_variation_over_sequence(group, connection, channel, variation):
    # data: All tubule intensity data over 100 frames for all movies in the group.

    data = pkl.load(open(f'/localhome/asa420/ER-Analysis-scripts/src/nw_graph_analysis/{group.lower()}_{connection}_tubules_{channel}.pkl', 'rb'))
    data = filter_data(data)
    # print(len(data))
    # print(data[21].shape)
    # print(data[21][53])
    # print(data[21][53][1].shape)
    # exit()
    variation_vals = []
    for series in data:
        for tubule in series:
            transposed_list = list(map(list, zip(*tubule)))
            for each in transposed_list:
                if variation == 'mean':
                    variation_vals.append(np.mean(each))
                else:
                    variation_vals.append(np.std(each))
    return variation_vals


# get_per_pixel_mean_over_sequence('ATL', 'iso-fuz', 'egfp')
# exit()

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

        df['Per-pixel-mean'] = pd.Series(per_pixel_data)
        df['Group'] = pd.Series(group_labels)

        box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('Climp', 'RTN')]

        ax = sns.boxenplot(data=df, x='Group', y='Per-pixel-mean')

        statannot.add_stat_annotation(ax, x='Group', y='Per-pixel-mean', data=df, box_pairs=box_pairs,
                                      test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    else:
        ar1, ar2, ar3 = get_replicate_data(atl_egfp, atl_mch)
        cr1, cr2, cr3 = get_replicate_data(climp_egfp, climp_mch)
        rr1, rr2, rr3 = get_replicate_data(rtn_egfp, rtn_mch)

        df['Per-pixel-mean'] = pd.Series(np.concatenate((ar1, ar2, ar3, cr1, cr2, cr3, rr1, rr2, rr3)))

        df['Group'] = pd.Series(np.concatenate((
            ['ATL'] * len(ar1), ['ATL'] * len(ar2), ['ATL'] * len(ar3), ['Climp'] * len(cr1), ['Climp'] * len(cr2),
            ['Climp'] * len(cr3), ['RTN'] * len(rr1), ['RTN'] * len(rr2), ['RTN'] * len(rr3))))

        df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(ar1), ['R2'] * len(ar2), ['R3'] * len(ar3), ['R1'] * len(cr1),
                                                    ['R2'] * len(cr2), ['R3'] * len(cr3), ['R1'] * len(rr1), ['R2'] * len(rr2),
                                                    ['R3'] * len(rr3))))

        ax = sns.boxenplot(data=df, x='Replicate', y='Per-pixel-mean', hue='Group', dodge=True)

        box_pairs = get_box_pairs('mch')

        statannot.add_stat_annotation(ax, x='Replicate', y='Per-pixel-mean', hue='Group', data=df, box_pairs=box_pairs, test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    plt.grid(True)
    plt.xlabel('Replicates', fontsize=18)
    plt.ylabel('Cross-correlation values', fontsize=18)

    plt.title(f'Per pixel cross-correlation over 100 frames between ERmoxGFP and mCherry in each {connection} connection tubule', fontsize=18)

    plt.show()


plot_per_pixel_correlation('fuz-fuz', 'replicate')
exit()

def plot_per_pixel_variation(connection, channel, variation, plottype):
    atl = get_per_pixel_variation_over_sequence('ATL', connection, channel, variation)
    climp = get_per_pixel_variation_over_sequence('Climp', connection, channel, variation)
    rtn = get_per_pixel_variation_over_sequence('RTN', connection, channel, variation)

    df = pd.DataFrame()

    ar1, ar2, ar3 = atl[:10], atl[10:20], atl[20:]
    cr1, cr2, cr3 = climp[:10], climp[10:20], climp[20:]
    rr1, rr2, rr3 = rtn[:10], rtn[10:20], rtn[20:]

    if channel == 'egfp':
        ctrl = get_per_pixel_variation_over_sequence('Control', connection, channel, variation)
        if plottype == 'all':

            df['Per-pixel-mean'] = pd.Series(np.concatenate((atl, climp, rtn, ctrl)))

            df['Group'] = pd.Series(np.concatenate((['ATL']*len(atl), ['Climp']*len(climp), ['RTN']*len(rtn), ['Control']*len(ctrl))))

        else:
            ctr1, ctr2, ctr3 = ctrl[:10], ctrl[10:20], ctrl[20:]
            df['Per-pixel-mean'] = pd.Series(np.concatenate((ar1, ar2, ar3, cr1, cr2, cr3, rr1, rr2, rr3, ctr1, ctr2, ctr3)))

            df['Group'] = pd.Series(np.concatenate((
                ['ATL'] * len(ar1), ['ATL'] * len(ar2), ['ATL'] * len(ar3), ['Climp'] * len(cr1), ['Climp'] * len(cr2),
                ['Climp'] * len(cr3), ['RTN'] * len(rr1), ['RTN'] * len(rr2), ['RTN'] * len(rr3), ['Control']*len(ctr1), ['Control']*len(ctr2), ['Control']*len(ctr3))))

            df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(ar1), ['R2'] * len(ar2), ['R3'] * len(ar3), ['R1'] * len(cr1),
                                                        ['R2'] * len(cr2), ['R3'] * len(cr3), ['R1'] * len(rr1), ['R2'] * len(rr2),
                                                        ['R3'] * len(rr3), ['R1'] * len(ctr1), ['R2'] * len(ctr2), ['R3'] * len(ctr3))))

    else:
        if plottype == 'all':
            df['Per-pixel-mean'] = pd.Series(np.concatenate((atl, climp, rtn)))

            df['Group'] = pd.Series(np.concatenate((['ATL']*len(atl), ['Climp']*len(climp), ['RTN']*len(rtn))))

        else:
            df['Per-pixel-mean'] = pd.Series(np.concatenate((ar1, ar2, ar3, cr1, cr2, cr3, rr1, rr2, rr3)))

            df['Group'] = pd.Series(np.concatenate((['ATL'] * len(ar1), ['ATL'] * len(ar2), ['ATL'] * len(ar3), ['Climp'] * len(cr1), ['Climp'] * len(cr2), ['Climp'] * len(cr3), ['RTN'] * len(rr1), ['RTN'] * len(rr2), ['RTN'] * len(rr3))))

            df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(ar1), ['R2'] * len(ar2), ['R3'] * len(ar3), ['R1'] * len(cr1), ['R2'] * len(cr2), ['R3'] * len(cr3), ['R1'] * len(rr1), ['R2'] * len(rr2), ['R3'] * len(rr3))))

    if plottype == 'all':
        ax = sns.boxenplot(data=df, x='Group', y='Per-pixel-mean')
        box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('ATL', 'Control'), ('Climp', 'Control'), ('Climp', 'RTN'),
                                 ('Control', 'RTN')] if channel == 'egfp' else [('ATL', 'Climp'), ('ATL', 'RTN'), ('Climp', 'RTN')]
        statannot.add_stat_annotation(ax, x='Group', y='Per-pixel-mean', data=df, box_pairs=box_pairs,
                                      test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')
    else:
        ax = sns.boxenplot(data=df, x='Replicate', y='Per-pixel-mean', hue='Group', dodge=True)
        box_pairs = get_box_pairs(channel)
        statannot.add_stat_annotation(ax, x='Replicate', y='Per-pixel-mean', hue='Group', data=df, box_pairs=box_pairs, test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')


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



# def get_edge_length():
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

        box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('ATL', 'Control'), ('Climp', 'RTN'), ('Climp', 'Control'), ('RTN', 'Control')]

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