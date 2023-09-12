# all the pickle creation routines

import pickle as pkl

from junction_analysis import cc_signal
import tubule_analysis as ta

def cc_mean_variation_pickles(channel, region):
    """
    Create pickle files for the mean variation of the CC signal
    @param channel: EGFP or mCherry
    @param region: iso or fuz
    """
    atl = cc_signal('ATL', channel, region)
    pkl.dump(atl, open(f'ATL_{channel}_{region}_CC_mean.pkl', 'wb'))
    climp = cc_signal('Climp', channel, region)
    pkl.dump(climp, open(f'Climp_{channel}_{region}_CC_mean.pkl', 'wb'))
    rtn = cc_signal('RTN', channel, region)
    pkl.dump(rtn, open(f'RTN_{channel}_{region}_CC_mean.pkl', 'wb'))
    control = cc_signal('Control', channel, region)
    pkl.dump(control, open(f'Control_{channel}_{region}_CC_mean.pkl', 'wb'))


def create_tub_data_pickles(group, total_series, connection, channel, measure):
    group_data = []
    for i in range(1, total_series + 1):
        seq_data = np.array(ta.tubule_sequence_data(group, i, connection, channel, measure))
        group_data.append(seq_data)

    with open(f'{group.lower()}_{connection}_{measure}_{channel}.pkl', 'wb') as fl:
        pkl.dump(group_data, fl)


# groups = {'ATL': 26, 'Climp': 31, 'Control': 31, 'RTN': 29}
# connections = ['iso-iso', 'iso-fuz', 'fuz-fuz']
# channels = ['egfp', 'mch']
#
# create_tub_data_pickles('ATL', 26, 'iso-iso', 'egfp', 'tubules')
# create_tub_data_pickles('ATL', 26, 'iso-iso', 'mch', 'tubules')
# create_tub_data_pickles('ATL', 26, 'iso-fuz', 'egfp', 'tubules')
# create_tub_data_pickles('ATL', 26, 'iso-fuz', 'mch', 'tubules')
# create_tub_data_pickles('ATL', 26, 'fuz-fuz', 'egfp', 'tubules')
# create_tub_data_pickles('ATL', 26, 'fuz-fuz', 'mch', 'tubules')
#
# create_tub_data_pickles('Climp', 31, 'iso-iso', 'egfp', 'tubules')
# create_tub_data_pickles('Climp', 31, 'iso-iso', 'mch', 'tubules')
# create_tub_data_pickles('Climp', 31, 'iso-fuz', 'egfp', 'tubules')
# create_tub_data_pickles('Climp', 31, 'iso-fuz', 'mch', 'tubules')
# create_tub_data_pickles('Climp', 31, 'fuz-fuz', 'egfp', 'tubules')
# create_tub_data_pickles('Climp', 31, 'fuz-fuz', 'mch', 'tubules')
#
# create_tub_data_pickles('RTN', 29, 'iso-iso', 'egfp', 'tubules')
# create_tub_data_pickles('RTN', 29, 'iso-iso', 'mch', 'tubules')
# create_tub_data_pickles('RTN', 29, 'iso-fuz', 'egfp', 'tubules')
# create_tub_data_pickles('RTN', 29, 'iso-fuz', 'mch', 'tubules')
# create_tub_data_pickles('RTN', 29, 'fuz-fuz', 'egfp', 'tubules')
# create_tub_data_pickles('RTN', 29, 'fuz-fuz', 'mch', 'tubules')
#
# create_tub_data_pickles('Control', 31, 'iso-iso', 'egfp', 'tubules')
# create_tub_data_pickles('Control', 31, 'iso-fuz', 'egfp', 'tubules')
# create_tub_data_pickles('Control', 31, 'fuz-fuz', 'egfp', 'tubules')
#
# exit()


def create_pickles(groups: dict, connections: list, channels: list, measure: list) -> None:
    """
    Creates pickles of tubule data for the given groups, connections, channels, and measure.

    Args:
        groups (dict): A dictionary of group names and the number of series in each group.
        connections (list): A list of connection types.
        channels (list): A list of channel names.
        measure (str): The name of the measure to be pickled.

    Returns:
        None
    """

    for group, num_series in groups.items():
        for connection in connections:
            if group != 'Control':
                for channel in channels:
                    # print(group, num_series, connection, channel, measure)
                    create_tub_data_pickles(group, num_series, connection, channel, measure)
            else:
                # print(group, num_series, connection, 'egfp', measure)
                create_tub_data_pickles(group, num_series, connection, 'egfp', measure)


def pickle_creation_runner():
    groups = {'ATL': 26, 'Climp': 31, 'Control': 31, 'RTN': 29}
    connections = ['iso-iso', 'iso-fuz', 'fuz-fuz']
    channels = ['egfp', 'mch']
    measure = 'tubules'
    # measure = VALID_MEASURES
    # create_pickles(groups, VALID_CONNECTIONS, VALID_CHANNELS, VALID_MEASURES)
    create_pickles(groups, connections, channels, measure)

pickle_creation_runner()
exit()