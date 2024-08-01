import os
import pickle as pkl

home = os.path.expanduser('~')

def load_pickle_data(filepath):
    """
    Load data from a pickle file.

    Args:
        filepath (str): Path to the pickle file.

    Returns:
        data: Data loaded from the pickle file, or None if an error occurs.
    """
    try:
        with open(filepath, 'rb') as f:
            return pkl.load(f)
    except (FileNotFoundError, pkl.UnpicklingError) as e:
        print(f"Error loading file {filepath}: {e}")
        return None

def get_pickle_data(group, conn, measure, channel):
    """
    Get data from a pickle file based on group, connection, measure, and channel.

    Args:
        group (str): Group name.
        conn (str): Connection type.
        measure (str): Measure type.
        channel (str): Channel name.

    Returns:
        data: Data loaded from the pickle file, or None if an error occurs.
    """
    filepath = f'{home}/ER-Analysis-scripts/src/nw_graph_analysis/pickles/{measure}/{group.lower()}_{conn}_{measure}_{channel}.pkl'
    return load_pickle_data(filepath)

def load_corrected_pickles(group, connection, channel):
    """
    Load corrected pickle data based on group, connection, and channel.

    Args:
        group (str): Group name.
        connection (str): Connection type.
        channel (str): Channel name.

    Returns:
        data: Data loaded from the pickle file, or None if an error occurs.
    """
    filepath = f'{home}/ER-Analysis-scripts/src/nw_graph_analysis/corrected_pickles/{group.lower()}_{connection}_tubules_{channel}.pkl'
    return load_pickle_data(filepath)

def load_annot_tub_pickles(group, connection, channel):
    """
    Load annotated tubule pickle data based on group, connection, and channel.

    Args:
        group (str): Group name.
        connection (str): Connection type.
        channel (str): Channel name.

    Returns:
        data: Data loaded from the pickle file, or None if an error occurs.
    """
    filepath = f'{home}/ER-Analysis-scripts/src/nw_graph_analysis/pickles/{group.lower()}_{connection}_tubules_{channel}.pkl'
    return load_pickle_data(filepath)