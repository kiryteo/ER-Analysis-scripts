import imageio
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import statannot
import matplotlib.pyplot as plt
import itertools
from junction_analysis_modules import JunctionAnalysis as JA

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'

junc_analysis = JA(confocal_data_path)


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


def get_per_frame_ref_junc_data(region_data, group_pref, dirname, channel, group, num):
    data = []
    for ref_junc in region_data:
        # store 100 values per ref junc
        junc_signal_vals = []

        for i in range(100):
            input_data = imageio.imread(f'{confocal_data_path}/{group}/new_op_jul/{dirname}/{group_pref[group]}{num}_decon_t0{i:02d}_ch{channel:02d}_std.png')
            junc_signal_vals.append(input_data[ref_junc[0], ref_junc[1]])

        data.append(junc_signal_vals)
    return data


def ref_junc_data_per_group(group, num_series, channel, region):
    """
    @param group: group to be analyzed
    @param num_series: sequence number
    @param channel: channel to be analyzed
    @param region: region to be analyzed
    @return: group_data (list) - provides all reference junctions per movie with 100 values in a group
    """
    group_data = []

    dirname, ch = ('std_egfp', 0) if channel == 'egfp' else ('std_mch', 1)
    group_pref = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}

    for num in range(1, num_series+1):
        ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, num)
        label_ids, unassigned_cc_dict = junc_analysis.separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)
        iso, fuz, unk = junc_analysis.get_junction_areas(label_ids, unassigned_cc_dict)

        if region == 'iso':
            region_data = iso
        elif region == 'fuz':
            region_data = fuz

        data = get_per_frame_ref_junc_data(region_data, group_pref, dirname, ch, group, num)

        group_data.append(data)

    return group_data


def ref_junc_variation(data, measure):
    measure_func = np.mean if measure == 'mean' else np.std
    measure_vals = [measure_func(np.array(junc)/255) for proj_frame in data for junc in proj_frame]
    return measure_vals


def create_ref_junc_pickles(channel, region):
    groups = {'ATL':26, 'Climp':31, 'RTN':29, 'Control':31}

    for group in groups:
        if channel == 'mch' and group == 'Control':
            continue
        data = ref_junc_data_per_group(group, groups[group], channel, region)
        with open(f'{group.lower()}_ref_junc_{channel}_{region}.pkl', 'wb') as fl:
            pkl.dump(data, fl)


def get_data_per_replicate(data, measure):
    r1, r2, r3 = data[:10], data[10:20], data[20:]
    r1, r2, r3 = ref_junc_variation(r1, measure), ref_junc_variation(r2, measure), ref_junc_variation(r3, measure)
    return r1, r2, r3


def plot_ref_junc_variation(channel, region, measure, plot_type):

    atl_data = pkl.load(open(f'atl_ref_junc_{channel}_{region}.pkl', 'rb'))
    climp_data = pkl.load(open(f'climp_ref_junc_{channel}_{region}.pkl', 'rb'))
    rtn_data = pkl.load(open(f'rtn_ref_junc_{channel}_{region}.pkl', 'rb'))
    ctrl_data = pkl.load(open(f'control_ref_junc_{channel}_{region}.pkl', 'rb'))

    region_name = 'isolated' if region == 'iso' else 'fuzzy'
    ch_name = 'ERmoxGFP' if channel == 'egfp' else 'mCherry'

    if plot_type == 'all':
        atl = ref_junc_variation(atl_data, measure)
        climp = ref_junc_variation(climp_data, measure)
        rtn = ref_junc_variation(rtn_data, measure)

        df = pd.DataFrame()

        if channel == 'egfp':
            ctrl = ref_junc_variation(ctrl_data, measure)
        
            df['data_tubule_mean'] = pd.Series(np.concatenate((atl, climp, rtn, ctrl)))

            df['Group'] = pd.Series(np.concatenate((['ATL'] * len(atl), ['Climp'] * len(climp), ['RTN'] * len(rtn), ['Control'] * len(ctrl))))

        else:
            df['data_tubule_mean'] = pd.Series(np.concatenate((atl, climp, rtn)))

            df['Group'] = pd.Series(np.concatenate((['ATL'] * len(atl), ['Climp'] * len(climp), ['RTN'] * len(rtn))))

        ax = sns.boxenplot(data=df, x='Group', y='data_tubule_mean')

        box_pairs = get_group_box_pairs(channel)

        statannot.add_stat_annotation(ax, x='Group', y='data_tubule_mean', data=df, box_pairs=box_pairs,
                                    test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')
        plt.title(f'Reference junction {measure} over sequence for {region_name} CCs in {ch_name} channel', fontsize=20)
        plt.xlabel('Group', fontsize=18)

    else:
        a1, a2, a3 = get_data_per_replicate(atl_data, measure)
        c1, c2, c3 = get_data_per_replicate(climp_data, measure)
        r1, r2, r3 = get_data_per_replicate(rtn_data, measure)
        ct1, ct2, ct3 = get_data_per_replicate(ctrl_data, measure)

        if channel == 'egfp':
            df['data_tubule_mean'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3, ct1, ct2, ct3)))

            df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1), ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2), ['R3'] * len(r3), ['R1'] * len(ct1), ['R2'] * len(ct2), ['R3'] * len(ct3))))
            
            df['Group'] = pd.Series(np.concatenate((
            ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1), ['Climp'] * len(c2),
            ['Climp'] * len(c3), ['RTN'] * len(r1), ['RTN'] * len(r2), ['RTN'] * len(r3), ['Control'] * len(ct1), ['Control'] * len(ct2), ['Control'] * len(ct3))))

        else:
            df['data_tubule_mean'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3)))
            
            df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
                                                                                                ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
                                                                                                ['R3'] * len(r3))))
            
            df['Group'] = pd.Series(np.concatenate((
            ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1), ['Climp'] * len(c2),
            ['Climp'] * len(c3), ['RTN'] * len(r1), ['RTN'] * len(r2), ['RTN'] * len(r3))))

        ax = sns.boxenplot(data=df, x='Replicate', y='data_tubule_mean', hue='Group', dodge=True)  # , yscale='log')

        replicates = ['R1', 'R2', 'R3']
        
        box_pairs = get_box_pairs(channel, replicates)

        statannot.add_stat_annotation(ax, x='Replicate', y='data_tubule_mean', hue='Group', data=df, box_pairs=box_pairs,
                                      test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')
        plt.title(f'Reference junction {measure} over sequence for {region_name} CCs in {ch_name} channel across replicates', fontsize=20)
        plt.xlabel('Replicate', fontsize=20)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    plt.grid(True)
    plt.ylabel('Mean over sequence per reference junction', fontsize=18)

    plt.show()