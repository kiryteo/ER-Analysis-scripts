import imageio
import numpy as np
import itertools
import matplotlib.pyplot as plt


confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'


def get_std_img(path):
    img = imageio.imread(path)
    return (img - img.min()) / (img.max() - img.min())


def seq_fourier_analysis(group, num_series):
    l = []
    for i in range(100):
        if group == 'Control':
            path = f'{confocal_data_path}{group}/files/img_{num_series}_decon_t0{i:02d}.tif'


        else:
            path = f'{confocal_data_path}{group}/files/{group[0]}{num_series}_decon_t0{i:02d}_ch00.tif'


        img = get_std_img(path)
        l.extend(img)

    l = np.reshape(l, (100, 128, 128))

    f = np.fft.fftn(l - np.mean(l))
    fabs = np.abs(f)
    fviz = np.fft.fftshift(fabs)

    return [np.sum(each) for each in fviz]


def seq_movie_fourier_analysis(group, num_series):
    l = []
    for num, i in itertools.product(range(1, num_series+1), range(100)):
        path = f'{confocal_data_path}{group}/files/img_{num}_decon_t0{i:02d}.tif' if group == 'Control' else f'{confocal_data_path}{group}/files/{group[0]}{num}_decon_t0{i:02d}_ch00.tif'


        img = get_std_img(path)
        l.extend(img)

    # if group == 'ATL':
    #     l = np.reshape(l, (2600, 128, 128))
    # elif group == 'Climp' or group == 'Control':
    #     l = np.reshape(l, (3100, 128, 128))
    # else:
    #     l = np.reshape(l, (2900, 128, 128))
    l = np.reshape(l, (2600, 128, 128))

    f = np.fft.fftn(l - np.mean(l))
    fabs = np.abs(f)
    fviz = np.fft.fftshift(fabs)

    return [np.sum(each) for each in fviz]


def seq_movie_fourier_analysis_runner():
    atl = seq_movie_fourier_analysis('ATL', 26)
    climp = seq_movie_fourier_analysis('Climp', 26)
    control = seq_movie_fourier_analysis('Control', 26)
    rtn = seq_movie_fourier_analysis('RTN', 26)

    # import scipy.io
    # from scipy.io import savemat
    #
    # atldt = {}
    # atldt['atl'] = atl
    # cldt = {}
    # cldt['climp'] = climp
    # ctdt = {}
    # ctdt['control'] = control
    # rtndt = {}
    # rtndt['rtn'] = rtn
    #
    # savemat('atl.mat', atldt)
    # savemat('climp.mat', cldt)
    # savemat('control.mat', ctdt)
    # savemat('rtn.mat', rtndt)


    # climp = []
    # control = []
    # rtn = []

    # for i in range(1, 27):
    #     atl_k = seq_fourier_analysis('ATL', i)
    #     atl.extend(atl_k)

    # print(len(atl))
    # print(atl[0])
    # exit()


    # atl = list(itertools.chain.from_iterable(atl))

    # for i in range(1, 32):
    #     climp_k = seq_fourier_analysis('Climp', i)
    #     climp.extend(climp_k)
    #
    # # climp = list(itertools.chain.from_iterable(climp))
    #
    # for i in range(1, 32):
    #     ctrl_k = seq_fourier_analysis('Control', i)
    #     control.extend(ctrl_k)
    #
    # # control = list(itertools.chain.from_iterable(control))
    #
    # for i in range(1, 30):
    #     rtn_k = seq_fourier_analysis('RTN', i)
    #     rtn.extend(rtn_k)

    # rtn = list(itertools.chain.from_iterable(rtn))

    # atl = sorted(atl)
    # climp = sorted(climp)
    # control = sorted(control)
    # rtn = sorted(rtn)

    plt.plot(atl, label='ATL')
    plt.plot(climp, label='Climp')
    plt.plot(control, label='Control')
    plt.plot(rtn, label='RTN')
    plt.legend()
    plt.title('EGFP frequency analysis across conditions')
    plt.show()