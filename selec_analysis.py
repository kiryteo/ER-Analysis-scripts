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
from scipy.stats import shapiro, normaltest, pearsonr, mannwhitneyu, kruskal

home = os.path.expanduser('~')

dirs_path = home + '/MIAL/data/selective_analysis/'
plos_data_path = home + '/MIAL/data/CROP-n/Plos_data_crops/'


def get_len_list(dil_tubules):
    """
    Get length for individual tubules
    @param dil_tubules: dilated tubules
    @return: list of lengths for all tubules
    """
    tub_len_list = []
    for i in range(1, len(np.unique(dil_tubules))):
        X, Y = np.where(dil_tubules == i)[0], np.where(dil_tubules == i)[1]
        tub_len_list.append(len(X))
    return tub_len_list


def refine_dil_tubules(dil_tubules, img):
    """
    Refine the dilated tubules shape to
    match the original ER shape [get rid
    of unnecessary signals created via
    dilation].
    @type dil_tubules: object
    @param img: Original ER input
    @param dil_tubules: dilated tubules
    @return: refined tubules
    """
    for i in range(1, len(np.unique(dil_tubules))):
        X, Y = np.where(dil_tubules == i)[0], np.where(dil_tubules == i)[1]
        for x, y in zip(X, Y):
            if img[x, y] == 0:
                dil_tubules[x, y] = 0
    return dil_tubules


def get_props(enh, samples=False) -> object:
    """
    Multiple steps of the pipeline to obtain
    the tubules and morphological properties

    @param samples: Create outputs for the multiple stages
    @param enh: Vessel enhanced image input
    @return: dilated tubules and morphological props
    """
    skel = pcv.morphology.skeletonize(mask=enh)
    brpts = pcv.morphology.find_branch_pts(skel_img=skel)
    dil_brpts = skimage.morphology.binary_dilation(brpts).astype('int') * 255
    tubules = skel - dil_brpts

    # tubules = (tubules == 255).astype('int') * 255 # for viz
    tubules = (tubules == 255).astype('int')
    lab_tubules = skimage.measure.label(tubules)
    dil_tubules = skimage.morphology.dilation(lab_tubules)
    # print(dil_tubules.shape)
    # print(np.unique(dil_tubules))

    # if samples==True:
        # pcv.print_image(brpts, 'rtn_t62_brpts.png')
        # pcv.print_image(dil_brpts, 'rtn_t62_dilbr.png')
        # pcv.print_image(tubules, 'rtn_t62_tub.png')
        # pcv.print_image(tubules, 'rtn_t62_tub2.png')
        # pcv.print_image(lab_tubules, 'rtn_t62_lab_tub.png')
        # pcv.print_image(dil_tubules, '/localhome/asa420/MIAL/data/CROP-n/Plos_data_crops/Climp_crops/Series038_decon_ch02-3_dil_tub.png')
        # pcv.print_image(refined_tubules, 'dfdf')


    # tub_len_list = get_len_list(dil_tubules)

    # erip = imageio.imread('/localhome/asa420/MIAL/data/selective_analysis/ctrl/ctrl_s7/std/Series007_decon_converted_t00_ch00_std.png')
    # erip[a, b] = 255

    props = skimage.measure.regionprops(dil_tubules)
    # props = skimage.measure.regionprops(lab_tubules)

    return dil_tubules, props


# img, _, _ = pcv.readimage('/localhome/asa420/MIAL/data/CROP-n/Plos_data_crops/ATL_crops/ATL_enh/Series037_decon_ch02-1_enhance.png')
# dt, _, tlist = get_props(img)
#
# erfile = imageio.imread('/localhome/asa420/MIAL/data/CROP-n/Plos_data_crops/ATL_crops/samples/Series037_decon_ch02-1.tif')
#
# # dil_tubules = refine_dil_tubules(dt, erfile)
# pcv.print_image(dt, '/localhome/asa420/MIAL/data/CROP-n/Plos_data_crops/ATL_crops/Series037_decon_ch02-1_lab.png')
#
#
# exit()


def get_intensity_features_plos():
    """
    Get intensity values along the tubules.
    @return: None
    """
    for grp in ['Climp', 'CTRL', 'RTN', 'ATL']:
        enh_samples = glob.glob(plos_data_path + grp + '_crops/' + grp + '_enh/*')
        # intensity_features = {}
        intensity_vals = {}
        for sample in enh_samples:
            enh = imageio.imread(sample)
            dil_tubules, props = get_props(enh)

            er_name = plos_data_path + grp + '_crops/samples/' + sample.split('/')[-1][:-12] + '.tif'
            er_file = imageio.imread(er_name)

            er_file = (er_file - er_file.min()) / (er_file.max() - er_file.min())

            er_file = er_file * 255

            dil_tubules = refine_dil_tubules(dil_tubules, er_file)

            # tub_len_list = get_len_list(dil_tubules)

            intensity_vals[er_name] = {}
            for i in range(1, len(np.unique(dil_tubules))):
                a, b = np.where(dil_tubules == i)[0], np.where(dil_tubules == i)[1]
                if len(a) > 10 and len(a) < 200:
                        # intensity_features[file][i] = {}
                        # intensity_features[file][i]['X'] = a
                        # intensity_features[file][i]['Y'] = b

                    intensity_vals[er_name][i] = er_file[a, b]
                        # print(er_file[a, b])
        with open(plos_data_path + grp + '_plos_intensity.pkl', 'wb') as fl:
            op = pickle.dump(intensity_vals, fl)



def normalize_er_samples():
    files = glob.glob('/localhome/asa420/MIAL/data/CROP-n/Plos_data_crops/RTN_crops/RTN_enh/*')

    intensity_vals = {}
    for each in files:
        enh = imageio.imread(each)
        dt, pr, tl = get_props(enh)

        fname = '/localhome/asa420/MIAL/data/CROP-n/Plos_data_crops/RTN_crops/' + each.split('/')[-1].split('.')[0][:-8] + '.tif'
        er_file = imageio.imread(fname)

        er_file = (er_file - np.min(er_file)) / (np.max(er_file) - np.min(er_file))
        er_file = er_file * 255

        intensity_vals[fname] = {}
        for i in range(1, len(np.unique(dt))):
            a, b = np.where(dt == i)[0], np.where(dt == i)[1]
            if len(a) > 10:
                # intensity_features[file][i] = {}
                # intensity_features[file][i]['X'] = a
                # intensity_features[file][i]['Y'] = b

                intensity_vals[fname][i] = er_file[a, b]

    for k, v in intensity_vals.items():
        print(k, v)

    return intensity_vals


def remove_outliers(x):
    # return [a for a in x if a < 200]
    return [a for a in x if a < 200 and a > 10]


def get_tub_len_plos():
    data = {}
    for grp in ['Climp', 'CTRL', 'RTN', 'ATL']:
        data[grp] = []
        enh_samples = glob.glob(plos_data_path + grp + '_crops/' + grp + '_enh/*')
        # intensity_features = {}
        intensity_vals = {}
        for sample in enh_samples:
            enh = imageio.imread(sample)
            dil_tubules, props = get_props(enh)

            er_name = plos_data_path + grp + '_crops/samples/' + sample.split('/')[-1][:-12] + '.tif'
            er_file = imageio.imread(er_name)

            er_file = (er_file - er_file.min()) / (er_file.max() - er_file.min())

            dil_tubules = refine_dil_tubules(dil_tubules, er_file)

            tub_len_list = get_len_list(dil_tubules)
            data[grp].extend(tub_len_list)

    at = data['ATL']
    cl = data['Climp']
    ct = data['CTRL']
    rt = data['RTN']

    # print(at)

    # stat, p_atcl = mannwhitneyu(at, cl)
    # stat, p_atct = mannwhitneyu(at, ct)
    # stat, p_atrt = mannwhitneyu(at, rt)
    # stat, p_clct = mannwhitneyu(cl, ct)
    # stat, p_clrt = mannwhitneyu(cl, rt)
    # stat, p_ctrt = mannwhitneyu(ct, rt)
    # print(p_atcl)
    # print(p_atct)
    # print(p_atrt)
    # print(p_clct)
    # print(p_clrt)
    # print(p_ctrt)

    # print("/n")

    # stat, p_atcl = kruskal(at, cl, ct, rt)
    # # stat, p_atct = kruskal(at, ct)
    # # stat, p_atrt = kruskal(at, rt)
    # # stat, p_clct = kruskal(cl, ct)
    # # stat, p_clrt = kruskal(cl, rt)
    # # stat, p_ctrt = kruskal(ct, rt)
    # print(p_atcl)
    # print(p_atct)
    # print(p_atrt)
    # print(p_clct)
    # print(p_clrt)
    # print(p_ctrt)

    # at = remove_outliers(at)
    # cl = remove_outliers(cl)
    # ct = remove_outliers(ct)
    # rt = remove_outliers(rt)

    at = np.log(remove_outliers(at))
    cl = np.log(remove_outliers(cl))
    ct = np.log(remove_outliers(ct))
    rt = np.log(remove_outliers(rt))

    sns.distplot(at, hist=False, label='ATL')
    sns.distplot(cl, hist=False, label='Climp')
    sns.distplot(ct, hist=False, label='Control')
    sns.distplot(rt, hist=False, label='RTN')
    plt.xlabel('Tubules length (in pixels, log scale)')
    # plt.xlabel('FWHM values (log scale)')
    plt.legend()
    # # plt.suptitle('Pipeline: std->vess->skel->brpts->tubules->dilated tubules->intensity analysis, log scale')
    # # plt.suptitle('Pipeline: std->hist_matching->vess->skel->brpts->tubules->dilated tubules->intensity analysis')
    plt.title('Tubule length values per group (PLOS data analysis)')
    plt.show()


def get_tubule_length():
    data = {}
    for grp in ['climp', 'ctrl', 'rtn']:
        data[grp] = []
        series_names = glob.glob(dirs_path + grp + '/*')
        for series in series_names:
            ser_files = glob.glob(series + '/match_enh/*')
            for file in ser_files:
                enh = imageio.imread(file)
                dil_tubules, props, tlist = get_props(enh)
                data[grp].extend(tlist)

    cl = data['climp']
    ct = data['ctrl']
    rt = data['rtn']

    cl = np.log(remove_outliers(cl))
    ct = np.log(remove_outliers(ct))
    rt = np.log(remove_outliers(rt))

    sns.distplot(cl, label='Climp')
    sns.distplot(ct, label='Control')
    sns.distplot(rt, label='RTN')
    plt.xlabel('Tubules length (in pixels)')
    # plt.xlabel('FWHM values (log scale)')
    # plt.legend()
    # plt.suptitle('Pipeline: std->vess->skel->brpts->tubules->dilated tubules->intensity analysis, log scale')
    # plt.suptitle('Pipeline: std->hist_matching->vess->skel->brpts->tubules->dilated tubules->intensity analysis')
    # plt.title('Tubule length values per group')
    plt.show()


def get_prop_dicts():
    for grp in ['climp', 'ctrl', 'rtn']:
        data = {}
        series_names = glob.glob(dirs_path + grp + '/*')
        for series in series_names:
            ser_files = glob.glob(series + '/match_enh/*')
            for file in ser_files:
                enh = imageio.imread(file)
                dil_tubules, props, tlist = get_props(enh)
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
                # print(features[6]['axis_major_length'])
                # exit()
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
        # print(area_convex_data[0])
        # exit()
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
            # ser_files = glob.glob(series + '/skel/*')
            for file in ser_files:
                enh = imageio.imread(file)
                # skel = imageio.imread(file)
                dil_tubules, props = get_props(enh)
                # dil_tubules, props = get_props(skel)
                # intensity_features[file] = {}
                er_name = series + '/matching/' + file.split('/')[-1][:-12] + '.png'
                # er_name = series + '/std/' + file.split('/')[-1][:-17] + '.png'
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
        with open(dirs_path + grp + '_match_dil_intensity.pkl', 'wb') as fl:
            op = pickle.dump(intensity_vals, fl)


def fwhm_analysis(grp):
    """

    @param grp:
    @return:
    """
    with open(dirs_path + '%s_intensity.pkl' % (f'{grp}'), 'rb') as fl:
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


def rtn_intensity_profiles_plos():
    with open(plos_data_path + 'CTRL_plos_intensity.pkl', 'rb') as fl:
        intensity_data = pickle.load(fl)

    ser221 = intensity_data['/localhome/asa420/MIAL/data/CROP-n/Plos_data_crops/CTRL_crops/samples/Series002_decon_ch02-3.tif']

    tubule = ser221[29]
    plt.xlabel('Tubule length')
    plt.ylabel('Intensity value')
    plt.title('Intensity profile, CTRL-Series002-3, tubule ID 29')
    plt.plot(tubule)
    plt.show()
    # for k, v in intensity_data.items():
    #     print(k)



rtn_intensity_profiles_plos()
exit()


def fwhm_analysis_plos(grp):
    with open(plos_data_path + '%s_plos_intensity.pkl' % (f'{grp}'), 'rb') as fl:
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

        width_list.extend(width_l)

    return width_list


atl_list = fwhm_analysis_plos('ATL')
climp_list = fwhm_analysis_plos('Climp')
ctrl_list = fwhm_analysis_plos('CTRL')
rtn_list = fwhm_analysis_plos('RTN')

# stat, atcl = mannwhitneyu(atl_list, climp_list)
# stat, atct = mannwhitneyu(atl_list, ctrl_list)
# stat, atrt = mannwhitneyu(atl_list, rtn_list)
# stat, clct = mannwhitneyu(climp_list, ctrl_list)
# stat, clrt = mannwhitneyu(climp_list, rtn_list)
# stat, ctrt = mannwhitneyu(ctrl_list, rtn_list)


# print(atcl)
# print(atct)
# print(atrt)
# print(clct)
# print(clrt)
# print(ctrt)

# st, p = kruskal(atl_list, climp_list, ctrl_list, rtn_list)
# print(p)
#
# exit()

# atl_list = np.log(atl_list)
# climp_list = np.log(climp_list)
# ctrl_list = np.log(ctrl_list)
# rtn_list = np.log(rtn_list)

sns.distplot(atl_list, hist=False, label='ATL')
sns.distplot(climp_list, hist=False, label='Climp')
sns.distplot(ctrl_list, hist=False, label='Control')
sns.distplot(rtn_list, hist=False, label='RTN')
plt.xlabel('FWHM values')
plt.legend()
# plt.suptitle('Pipeline: std->vess->skel->brpts->tubules->dilated tubules->intensity analysis, log scale')
# plt.suptitle('Pipeline: std->hist_matching->vess->skel->brpts->tubules->dilated tubules->intensity analysis')
plt.title('FWHM values per group (PLOS data analysis)')
plt.show()

exit()


def fwhm_analysis_fr(grp):
    with open(dirs_path + '%s_match_dil_intensity.pkl' % (f'{grp}'), 'rb') as fl:
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

        width_list.extend(width_l)

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


def get_distplot(climp, ctrl, rtn):
    sns.distplot(climp, label='Climp')
    sns.distplot(ctrl, label='Control')
    sns.distplot(rtn, label='RTN')
    plt.xlabel('FWHM values')
    # plt.xlabel('FWHM values (log scale)')
    plt.legend()
    # plt.suptitle('Pipeline: std->vess->skel->brpts->tubules->dilated tubules->intensity analysis, log scale')
    plt.suptitle('Pipeline: std->hist_matching->vess->skel->brpts->tubules->dilated tubules->intensity analysis')
    plt.show()


def combined_analysis(log=False):
    climp_width_list = fwhm_analysis_fr('climp')
    ctrl_width_list = fwhm_analysis_fr('ctrl')
    rtn_width_list = fwhm_analysis_fr('rtn')

    climp_log = np.log(climp_width_list)
    ctrl_log = np.log(ctrl_width_list)
    rtn_log = np.log(rtn_width_list)

    if log:
        get_distplot(climp_log, ctrl_log, rtn_log)
    else:
        get_distplot(climp_width_list, ctrl_width_list, rtn_width_list)


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
    return a * np.exp(-b * x) + c


def ncurve(width_l):
    x = np.arange(len(width_l))
    ffunc = lambda x, a, x0, s: a * np.exp(-0.5 * (x - x0) ** 2 / s ** 2)
    p, _ = curve_fit(ffunc, x, width_l)
    x0 = p[1]
    plt.plot(x, width_l)
    plt.plot(x, ffunc(x, *p))
    plt.show()
