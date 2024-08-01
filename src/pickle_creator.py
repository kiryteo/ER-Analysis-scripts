import pickle as pkl
from junction_analysis import cc_signal
import tubule_analysis as ta
import numpy as np
from typing import Dict, List

def cc_mean_variation_pickles(channel: str, region: str) -> None:
    """
    Create pickle files for the mean variation of the CC signal.
    
    Args:
        channel (str): EGFP or mCherry.
        region (str): iso or fuz.
    """
    for group in ['ATL', 'Climp', 'RTN', 'Control']:
        data = cc_signal(group, channel, region)
        with open(f'{group}_{channel}_{region}_CC_mean.pkl', 'wb') as file:
            pkl.dump(data, file)

def create_tub_data_pickles(group: str, total_series: int, connection: str, channel: str, measure: str) -> None:
    """
    Create pickle files for tubule data.
    
    Args:
        group (str): Group name.
        total_series (int): Total number of series.
        connection (str): Connection type.
        channel (str): Channel name.
        measure (str): Measure name.
    """
    group_data = [np.array(ta.tubule_sequence_data(group, i, connection, channel, measure)) for i in range(1, total_series + 1)]
    with open(f'{group.lower()}_{connection}_{measure}_{channel}.pkl', 'wb') as file:
        pkl.dump(group_data, file)

def create_pickles(groups: Dict[str, int], connections: List[str], channels: List[str], measure: str) -> None:
    """
    Creates pickles of tubule data for the given groups, connections, channels, and measure.
    
    Args:
        groups (Dict[str, int]): A dictionary of group names and the number of series in each group.
        connections (List[str]): A list of connection types.
        channels (List[str]): A list of channel names.
        measure (str): The name of the measure to be pickled.
    """
    for group, num_series in groups.items():
        for connection in connections:
            if group != 'Control':
                for channel in channels:
                    create_tub_data_pickles(group, num_series, connection, channel, measure)
            else:
                create_tub_data_pickles(group, num_series, connection, 'egfp', measure)

def pickle_creation_runner() -> None:
    """
    Runner function to create pickles for predefined groups, connections, channels, and measure.
    """
    groups = {'ATL': 26, 'Climp': 31, 'Control': 31, 'RTN': 29}
    connections = ['iso-iso', 'iso-fuz', 'fuz-fuz']
    channels = ['egfp', 'mch']
    measure = 'tubules'
    create_pickles(groups, connections, channels, measure)

# Uncomment the following line to run the pickle creation process
# pickle_creation_runner()