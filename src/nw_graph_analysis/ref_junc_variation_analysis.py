import imageio
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import statannot
import matplotlib.pyplot as plt


confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies'

def ref_junc_data_per_group(group, num_series, channel):
    group_data = []
    # grp_data_egfp, grp_data_mch = [], []

    dirname, ch = ('std_egfp', 0) if channel == 'egfp' else ('std_mch', 1)
    group_pref = {'ATL': 'A', 'Climp': 'C', 'Control': 'Ct', 'RTN': 'R'}

    for num in range(1, num_series+1):
        ref_junctions, per_frame_junctions, labelled_img = junc_analysis.label_junctions(group, num)
        label_ids, unassigned_cc_dict = separate_junc_cc(ref_junctions, per_frame_junctions, labelled_img)
        iso, fuz, unk = get_junction_areas(label_ids, unassigned_cc_dict)

        # store per movie list of per ref junc list with 100 values
        data = []
        for each in iso:
            # store 100 values per ref junc
            junc_signal_vals = []
            # junc_sig_egfp = []
            # junc_sig_mch = []
            for i in range(100):
                input = imageio.imread(f'{confocal_data_path}/{group}/new_op_jul/{dirname}/{group_pref[group]}{num}_decon_t0{i:02d}_ch{ch:02d}_std.png')
                junc_signal_vals.append(input[each[0], each[1]])
            data.append(junc_signal_vals)

        # store list of per movie data list
        group_data.append(data)

    return group_data


def ref_junc_variation(data, measure):
    # sourcery skip: inline-immediately-returned-variable
    measure_func = np.mean if measure == 'mean' else np.std
    measure_vals = [measure_func(np.array(junc)/255) for proj_frame in data for junc in proj_frame]
    return measure_vals


def create_ref_junc_pickles(channel):
    groups = {'ATL':26, 'Climp':31, 'RTN':29, 'Control':31}

    for group in groups:
        data = ref_junc_data_per_group(group, groups[group], channel)
        with open(f'{group.lower()}_ref_junc_{channel}.pkl', 'wb') as fl:
            pkl.dump(data, fl)


def get_data_per_replicate(data):
    r1, r2, r3 = data[:10], data[10:20], data[20:]
    return r1, r2, r3


def plot_ref_junc_variation(channel):

    atl_data = pkl.load(open(f'atl_ref_junc_{channel}.pkl', 'rb'))

    # a1, a2, a3 = atl_data[:10], atl_data[10:20], atl_data[20:]
    # a1, a2, a3 = ref_junc_variation(a1, 'std'), ref_junc_variation(a2, 'std'), ref_junc_variation(a3, 'std')

    atl = ref_junc_variation(atl_data, 'mean')

    climp_data = pkl.load(open('climp_ref_junc_mch.pkl', 'rb'))
    # c1, c2, c3 = climp_data[:10], climp_data[10:20], climp_data[20:]
    # c1, c2, c3 = ref_junc_variation(c1, 'std'), ref_junc_variation(c2, 'std'), ref_junc_variation(c3, 'std')
    climp = ref_junc_variation(climp_data, 'mean')

    rtn_data = pkl.load(open('rtn_ref_junc_mch.pkl', 'rb'))
    # r1, r2, r3 = rtn_data[:10], rtn_data[10:20], rtn_data[20:]
    # r1, r2, r3 = ref_junc_variation(r1, 'std'), ref_junc_variation(r2, 'std'), ref_junc_variation(r3, 'std')
    rtn = ref_junc_variation(rtn_data, 'mean')

    # ctrl_data = get_region_areas_per_group('Control', 31, 'mch')
    # write_pickle('Control', 'mch', ctrl_data)

    # ctrl_data = pkl.load(open('control_ref_junc_egfp.pkl', 'rb'))
    # ct1, ct2, ct3 = ctrl_data[:10], ctrl_data[10:20], ctrl_data[20:]
    # ct1, ct2, ct3 = ref_junc_variation(ct1, 'mean'), ref_junc_variation(ct2, 'std'), ref_junc_variation(ct3, 'std')
    # ctrl = ref_junc_variation(ctrl_data, 'mean')

    # print(ct1)
    # exit()
    df = pd.DataFrame()

    df['data_tubule_mean'] = pd.Series(np.concatenate((atl, climp, rtn)))

    # df['data_tubule_mean'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3, ct1, ct2, ct3)))
    # df['data_tubule_mean'] = pd.Series(np.concatenate((a1, a2, a3, c1, c2, c3, r1, r2, r3)))

    # df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
    #                                             ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
    #                                             ['R3'] * len(r3), ['R1'] * len(ct1), ['R2'] * len(ct2),
    #                                             ['R3'] * len(ct3))))

    # df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(a1), ['R2'] * len(a2), ['R3'] * len(a3), ['R1'] * len(c1),
    #                                                                                         ['R2'] * len(c2), ['R3'] * len(c3), ['R1'] * len(r1), ['R2'] * len(r2),
    #                                                                                         ['R3'] * len(r3))))

    df['Group'] = pd.Series(np.concatenate((['ATL']*len(atl), ['Climp']*len(climp), ['RTN']*len(rtn))))

    # df['Group'] = pd.Series(np.concatenate((
    #     ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1), ['Climp'] * len(c2),
    #     ['Climp'] * len(c3), ['RTN'] * len(r1), ['RTN'] * len(r2), ['RTN'] * len(r3), ['Control'] * len(ct1), ['Control'] * len(ct2), ['Control'] * len(ct3))))
    #

    # df['Group'] = pd.Series(np.concatenate((
    #     ['ATL'] * len(a1), ['ATL'] * len(a2), ['ATL'] * len(a3), ['Climp'] * len(c1), ['Climp'] * len(c2),
    #     ['Climp'] * len(c3), ['RTN'] * len(r1), ['RTN'] * len(r2), ['RTN'] * len(r3))))

    ax = sns.boxenplot(data=df, x='Group', y='data_tubule_mean')
    # ax = sns.boxenplot(data=df, x='Replicate', y='data_tubule_mean', hue='Group', dodge=True)  # , yscale='log')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

    box_pairs = [('ATL', 'Climp'), ('ATL', 'RTN'), ('Climp', 'RTN')]

    # replicates = ['R1', 'R2', 'R3']
    #
    # if channel == 'egfp':
    #     groups = ['ATL', 'Climp', 'RTN', 'Control']
    # else:
    #     groups = ['ATL', 'Climp', 'RTN']
    # box_pairs = [((x, y), (x, z)) for i, x in enumerate(replicates) for j, y in enumerate(groups) for z in groups[j + 1 :]]

    # statannot.add_stat_annotation(ax, x='Replicate', y='data_tubule_mean', hue='Group', data=df, box_pairs=box_pairs,
    #                               test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    statannot.add_stat_annotation(ax, x='Group', y='data_tubule_mean', data=df, box_pairs=box_pairs,
                                  test='Mann-Whitney', text_format='simple', loc='inside', verbose=2, fontsize='large')

    # plt.title('Cross-correlation between ERmoxGFP and mCherry over sequence for tubule intensity mean in all tubules',
    #           fontsize=18)
    # plt.title('CC area for Fuzzy CCs across conditions', fontsize=20)
    plt.title('Reference junction mean over sequence for isolated CCs in mCherry channel', fontsize=20)
    plt.grid(True)
    # plt.xlabel('Replicate', fontsize=20)
    plt.xlabel('Group', fontsize=20)
    plt.ylabel('Mean over sequence per reference junction', fontsize=18)

    plt.show()