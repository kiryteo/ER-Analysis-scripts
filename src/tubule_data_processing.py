import numpy as np

def filter_data(data):
    """
    Filter out None values and arrays with all None elements from the data.

    Args:
        data (list): List of data arrays.

    Returns:
        list: Filtered data.
    """
    return [arr for arr in data if arr is not None and not np.all(arr == None)]

def get_correlation_data(data_egfp, data_mch):
    """
    Get the correlation data for the entire dataset.

    Args:
        data_egfp (list): Data from eGFP.
        data_mch (list): Data from mCherry.

    Returns:
        list: Correlation data for the entire dataset.
    """
    return [np.corrcoef(tub_eg, tub_mch)[0][1] for tub_eg, tub_mch in zip(data_egfp, data_mch)]

def get_variation(data, variation):
    """
    Calculate the specified variation (mean or std) for the data.

    Args:
        data (list): List of data arrays.
        variation (str): Type of variation ('mean' or 'std').

    Returns:
        tuple: Variation data for the dataset.
    """
    if variation == 'mean':
        func = np.mean
    elif variation == 'std':
        func = np.std
    else:
        raise ValueError('Invalid variation')

    filtered_data = filter_data(data)
    subsets = [filtered_data[:10], filtered_data[10:20], filtered_data[20:]]
    result = [[func(i) for i in each if len(i) > 0] for subset in subsets for each in subset]
    return tuple(result)

def get_length_per_tubule(data):
    """
    Get the length of each tubule sequence in the data.

    Args:
        data (list): List of data arrays.

    Returns:
        list: Lengths of tubule sequences.
    """
    filtered_data = filter_data(data)
    return [len(tubule_seq[0]) for series in filtered_data for tubule_seq in series if len(tubule_seq[0]) > 3]

def get_mean_length_per_series(data):
    """
    Get the mean length of tubule sequences for each series in the data.

    Args:
        data (list): List of data arrays.

    Returns:
        list: Mean lengths of tubule sequences per series.
    """
    filtered_data = filter_data(data)
    return [np.mean([len(tubule[0]) for tubule in series]) for series in filtered_data]