import imageio
from plantcv import plantcv as pcv
import skimage
from skimage import measure
import glob
import numpy as np
import cv2
import pickle
import pandas as pd
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.signal import chirp, find_peaks, peak_widths
import seaborn as sns
from scipy.optimize import curve_fit
import os

home = os.path.expanduser('~')

dirs_path = home + '/MIAL/data/selective_analysis/'

def get_props(enh):
    """
    Multiple steps of the pipeline to obtain
    the tubules and morphological properties

    @param enh: Vessel enhanced image input
    @return: dilated tubules and morphological props
    """
    skel = pcv.morphology.skeletonize(mask=enh)
    brpts = pcv.morphology.find_branch_pts(skel_img=skel)
    dil_brpts = skimage.morphology.binary_dilation(brpts).astype('int')
    tubules = skel - dil_brpts
    tubules = (tubules == 255).astype('int')
    lab_tubules = skimage.measure.label(tubules)
    dil_tubules = skimage.morphology.dilation(lab_tubules)
    props = skimage.measure.regionprops(dil_tubules)

    return dil_tubules, props


def get_prop_dicts():
    for grp in ['climp', 'ctrl', 'rtn']:
        data = {}
        series_names = glob.glob(dirs_path + grp + '/*')
        for series in series_names:
            ser_files = glob.glob(series + '/match_enh/*')
            for file in ser_files:
                enh = imageio.imread(file)
                dil_tubules, props = get_props(enh)
                data[file] = props

        area_data = []
        area_convex_data = []
        axis_major_length_data = []
        axis_minor_length_data = []
        eccentricity_data = []

        for series in series_names:
            ser_files = glob.glob(series + '/match_enh/*')
            for file in ser_files:
                features = data[file][:]
                for area_val in features:
                    area_data.append((area_val['area']))
                for ax_major_val in features:
                    axis_major_length_data.append((ax_major_val['axis_major_length']))
                for ax_min_val in features:
                    axis_minor_length_data.append((ax_min_val['axis_major_length']))
                for area_convex_val in features:
                    area_convex_data.append((area_convex_val['area_convex']))
                for eccentricity_val in features:
                    eccentricity_data.append((eccentricity_val['eccentricity']))

        # print(np.mean(area_data))
        # print(np.mean(axis_major_length_data))
        # print(np.mean(axis_minor_length_data))
        # print(np.mean(area_convex_data))
        # print(np.mean(eccentricity_data))
        # print(len(area_data))


def get_intensity_features():
    """
    Get intensity values along the tubules.
    @return: None
    """
    for grp in ['climp', 'ctrl', 'rtn']:
        series_names = glob.glob(dirs_path + grp + '/*')
        # intensity_features = {}
        intensity_vals = {}
        for series in series_names:

            ser_files = glob.glob(series + '/match_enh/*')
            for file in ser_files:
                enh = imageio.imread(file)
                dil_tubules, props = get_props(enh)
                # intensity_features[file] = {}
                er_name = series + '/matching/' + file.split('/')[-1][:-12] + '.png'
                # print(er_name)
                er_file = imageio.imread(er_name)
                intensity_vals[er_name] = {}
                for i in range(1, len(np.unique(dil_tubules))):
                    a, b = np.where(dil_tubules == i)[0], np.where(dil_tubules == i)[1]
                    if len(a) > 10:
                        # intensity_features[file][i] = {}
                        # intensity_features[file][i]['X'] = a
                        # intensity_features[file][i]['Y'] = b

                        intensity_vals[er_name][i] = er_file[a, b]
                        # print(er_file[a, b])
        with open(dirs_path + grp + '_match_intensity.pkl', 'wb') as fl:
            op = pickle.dump(intensity_vals, fl)


def fwhm_analysis(grp):
    """

    @param grp:
    @return:
    """
    with open(dirs_path + '%s_intensity.pkl'%(f'{grp}'), 'rb') as fl:
        intensity_data = pickle.load(fl)

    # print(intensity_data.keys())
    width_list = []
    for k, v in intensity_data.items():
        # print(len(v))
        for i in range(1, len(v)):
            try:
                peaks, _ = find_peaks(v[i])
                res_half = peak_widths(v[i], peaks, rel_height=0.5)
                width_list.extend(res_half[0])
            except:
                pass

            # print(res_half[0])
            # plt.plot(res_half[0])
            # plt.show()
            # break

        # print(v[23])
        # peaks, _ = find_peaks(v[23])
        # reshalf = peak_widths(v[23], peaks, rel_height=0.5)
        # print(reshalf[0])
        # resfull = peak_widths(v[23], peaks, rel_height=1.0)
        # print(resfull[0])
        # mx = max(v[23])
        # xs = [x for x in range(len(v[23])) if v[23][x] > mx/2.0 ]
        # print(min(xs), max(xs))
        # plt.plot(v[23])
        # plt.show()
        # break
    # print(width_list[0])

    return width_list




def fwhm_analysis_fr(grp):
    with open(dirs_path + '%s_intensity.pkl'%(f'{grp}'), 'rb') as fl:
        intensity_data = pickle.load(fl)

    width_list = []
    for k, v in intensity_data.items():
        width_l = []
        for i in range(len(v)):
            try:
                peaks, _ = find_peaks(v[i])
                res_half = peak_widths(v[i], peaks, rel_height=0.5)

                width_l.extend(list(res_half)[0])
            except:
                pass

        # sns.displot(width_l)
        # plt.show()

        # width_l = (width_l - min(width_l)) / (max(width_l) - min(width_l))
        width_list.extend(width_l)

        # df['FWHM'] = width_l
        # df['grp'] = pd.Series('climp' * len(width_l))
        # sns.boxplot(x=df['FWHM'], y=df['grp'])
        # plt.hist(width_l)
        # plt.show()
        # break

    return width_list

def get_boxplot(climp, ctrl, rtn):
    df = pd.DataFrame()
    df['FWHM'] = pd.Series(climp + ctrl + rtn)
    climplen = ['Climp'] * len(climp)
    ctrllen = ['Ctrl'] * len(ctrl)
    rtnlen = ['RTN'] * len(rtn)
    #
    df['Group'] = pd.Series(climplen + ctrllen + rtnlen)

    sns.boxplot(x=df['FWHM'], y=df['Group'])
    plt.show()


def get_distplot(climp, ctrl, rtn, fit, hist):
    sns.distplot(l1, fit=fit, hist=hist, label='climp')
    sns.distplot(l2, fit=fit, hist=hist, label='ctrl')
    sns.distplot(l3, fit=fit, hist=hist, label='rtn')
    plt.legend()
    plt.show()

def combined_analysis():

    climp_width_list = fwhm_analysis_fr('climp')
    ctrl_width_list = fwhm_analysis_fr('ctrl')
    rtn_width_list = fwhm_analysis_fr('rtn')

    climp_log = np.log(climp_width_list)
    ctrl_log = np.log(ctrl_width_list)
    rtn_log = np.log(rtn_width_list)

    get_distplot(climp_log, ctrl_log, rtn_log)



def remove_parts(feature_list, part):
    """

    @param feature_list: Data
    @param part: threshold
    @return: Data with values above/ below the threshold
    """
    return [val for val in feature_list if val > part]


# print(stats.skew(climp_width_list))
# print(stats.kurtosis(climp_width_list))

# sns.distplot((climp_width_list), fit=norm, hist=False)
# sns.distplot((ctrl_width_list), fit=norm, hist=False)
# sns.distplot((rtn_width_list), fit=norm, hist=False)

sns.distplot(np.log(climp_width_list), label='Climp')
sns.distplot(np.log(ctrl_width_list), label='Control')
sns.distplot(np.log(rtn_width_list), label='RTN')
plt.legend()
plt.show()


def remove_ones(feature_list):
    return [val for val in feature_list if val > 3]


df = pd.DataFrame()
widths = climp_width_list + ctrl_width_list + rtn_width_list
df['FWHM'] = pd.Series(widths)

climplen = ['Climp'] * len(climp_width_list)
ctrllen = ['Ctrl'] * len(ctrl_width_list)
rtnlen = ['RTN'] * len(rtn_width_list)

df['Group'] = pd.Series(climplen + ctrllen + rtnlen)

sns.set_theme(style='whitegrid')
# sns.stripplot(x=df['FWHM'], y=df['Group'])
sns.boxplot(x=df['FWHM'], y=df['Group'])
# sns.swarmplot(x=df['FWHM'], y=df['Group'])
plt.suptitle('FWHM values for blobs in three groups')
plt.show()

# numbins=20
# min_limit = min(min(ctrl_width_list), min(climp_width_list), min(rtn_width_list))
# max_limit = max(max(ctrl_width_list), max(climp_width_list), max(rtn_width_list))
# bins = np.linspace(min_limit, max_limit, numbins+1)
# plt.hist(ctrl_width_list, bins, alpha=0.33, color='Blue', label='Control')
# plt.hist(climp_width_list, bins, alpha=0.33, color='Red', label='Climp')
# plt.hist(rtn_width_list, bins, alpha=0.34, color='Green', label='RTN')
# plt.legend(loc='upper right')
# #plt.suptitle('Junction Area analysis', size=15)
# #plt.title('Mean intensity within specified radius around the reference frame junctions (radius/ euc. dist = 2)', size=12)
# plt.show()
# plt.close()

# plt.show()

def exp_func(x, a, b, c):
    return a* np.exp(-b*x) + c


def ncurve(width_l):
    x = np.arange(len(width_l))
    ffunc = lambda x, a, x0, s: a*np.exp(-0.5*(x-x0)**2/s**2)
    p, _ = curve_fit(ffunc, x, width_l)
    x0 = p[1]
    plt.plot(x, width_l)
    plt.plot(x, ffunc(x, *p))
    plt.show()