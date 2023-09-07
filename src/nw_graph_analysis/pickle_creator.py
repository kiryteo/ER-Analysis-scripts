# all the pickle creation routines

import pickle as pkl

from junction_analysis import cc_signal

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


