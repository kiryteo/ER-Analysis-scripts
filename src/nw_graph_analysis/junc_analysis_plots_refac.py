import ...




def get_egfp_plots(connection, channel, variation):

    def filter_data(data):
        new_list = [arr for arr in data if arr is not None and not np.all(arr == None)]
        return new_list

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

    def get_group_replicate_data(data, group_name):
        ar1, ar2, ar3 = get_variation(data['ATL'], variation)
        cr1, cr2, cr3 = get_variation(data['Climp'], variation)
        rr1, rr2, rr3 = get_variation(data['RTN'], variation)
        ctr1, ctr2, ctr3 = get_variation(data['Control'], variation)

        df = pd.DataFrame()

        df['tub-mean'] = pd.Series(np.concatenate((ar1, ar2, ar3, cr1, cr2, cr3, rr1, rr2, rr3, ctr1, ctr2, ctr3)))
        df['Group'] = pd.Series(np.concatenate((['ATL'] * len(ar1), ['ATL'] * len(ar2), ['ATL'] * len(ar3), ['Climp'] * len(cr1), ['Climp'] * len(cr2), ['Climp'] * len(cr3), ['RTN'] * len(rr1), ['RTN'] * len(rr2), ['RTN'] * len(rr3), ['Control'] * len(ctr1), ['Control'] * len(ctr2), ['Control'] * len(ctr3))))
        df['Replicate'] = pd.Series(np.concatenate((['R1'] * len(ar1), ['R2'] * len(ar2), ['R3'] * len(ar3), ['R1'] * len(cr1), ['R2'] * len(cr2), ['R3'] * len(cr3), ['R1'] * len(rr1), ['R2'] * len(rr2), ['R3'] * len(rr3), ['R1'] * len(ctr1), ['R2'] * len(ctr2), ['R3'] * len(ctr3))))

        return df[df['Group'] == group_name]

    data = {
        'ATL': channel_corr_all('ATL', connection, 'tub_mean', channel),
        'Climp': channel_corr_all('Climp', connection, 'tub_mean', channel),
        'RTN': channel_corr_all('RTN', connection, 'tub_mean', channel),
        'Control': channel_corr_all('Control', connection, 'tub_mean', channel),
    }

    df_list = [get_group_replicate_data(data, group_name) for group_name in ['ATL', 'Climp', 'RTN', 'Control']]
    df = pd.concat(df_list)

    ax = sns.boxenplot(data=df, x='Replicate', y='tub-mean', hue='Group', dodge=True)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)

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



# Define the groups and replicates
groups = ['ATL', 'Climp', 'RTN', 'Control']
replicates = ['R1', 'R2', 'R3']

# Calculate the tub-mean based on the channel
if channel == 'egfp':
    # Calculate the control variation
    ctrl = channel_corr_all('Control', connection, 'tub_mean', channel)
    ctrl_mean_variation = get_variation(ctrl, 'mean')

    # Add control variation to tub-mean
    tub_mean_variation = ctrl_mean_variation + [ar1, ar2, ar3, cr1, cr2, cr3, rr1, rr2, rr3]
else:
    tub_mean_variation = [ar1, ar2, ar3, cr1, cr2, cr3, rr1, rr2, rr3]

# Combine the tub-mean and create the Group and Replicate columns
df['tub-mean'] = pd.Series(np.concatenate(tub_mean_variation))
df['Group'] = pd.Series(np.concatenate([([group] * len(variation)) for group, variation in zip(groups, tub_mean_variation)]))
df['Replicate'] = pd.Series(np.concatenate([([replicate] * len(variation)) for replicate in replicates for variation in tub_mean_variation]))
