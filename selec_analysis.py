import imageio
from plantcv import plantcv as pcv
import skimage
from skimage import measure
import glob
import numpy as np
import cv2
import pickle

# er_files = glob.glob('/localhome/asa420/MIAL/data/selective_analysis/climp/climp_s1/files/*')
# skel_files = glob.glob('/localhome/asa420/MIAL/data/selective_analysis/climp/climp_s1/skel/*')


def get_props(skel):
    brpts = pcv.morphology.find_branch_pts(skel_img=skel)
    dil_brpts = skimage.morphology.binary_dilation(brpts).astype('int')
    tubules = skel - dil_brpts
    tubules = (tubules == 255).astype('int')
    lab_tubules = skimage.measure.label(tubules)
    dil_tubules = skimage.morphology.dilation(lab_tubules)
    props = skimage.measure.regionprops(dil_tubules)

    return dil_tubules, props

dirs_path = '/localhome/asa420/MIAL/data/selective_analysis/'

def get_prop_dicts():
    for grp in ['climp', 'ctrl', 'rtn']:
        data = {}
        series_names = glob.glob(dirs_path + grp + '/*')
        for series in series_names:
            ser_files = glob.glob(series + '/skel/*')
            for file in ser_files:
                skel = imageio.imread(file)
                dil_tubules, props = get_props(skel)
                data[file] = props

        area_data = []
        area_convex_data = []
        axis_major_length_data = []
        axis_minor_length_data = []
        eccentricity_data = []

        for series in series_names:
            ser_files = glob.glob(series + '/skel/*')
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

        print(np.mean(area_data))
        print(np.mean(axis_major_length_data))
        print(np.mean(axis_minor_length_data))
        print(np.mean(area_convex_data))
        print(np.mean(eccentricity_data))


def get_intensity_features():
    for grp in ['climp', 'ctrl', 'rtn']:
        series_names = glob.glob(dirs_path + grp + '/*')

        # intensity_features = {}
        intensity_vals = {}
        for series in series_names:
            ser_files = glob.glob(series + '/skel/*')
            for file in ser_files:
                skel = imageio.imread(file)
                dil_tubules, props = get_props(skel)
                # intensity_features[file] = {}
                er_name = series + '/std/' + file.split('/')[-1][:-17] + '.png'
                # print(er_name)
                er_file = imageio.imread(er_name)
                intensity_vals[er_name] = {}
                for i in range(1, len(np.unique(dil_tubules))):
                    a, b = np.where(dil_tubules==i)[0], np.where(dil_tubules==i)[1]
                    if len(a) > 10:
                        # intensity_features[file][i] = {}
                        # intensity_features[file][i]['X'] = a
                        # intensity_features[file][i]['Y'] = b

                        intensity_vals[er_name][i] = er_file[a, b]
                        # print(er_file[a, b])
        with open(dirs_path + grp + '_intensity.pkl', 'wb') as fl:
            op = pickle.dump(intensity_vals, fl)


# get_intensity_features()

def process_props():
    with open(dirs_path + 'climp.pkl', 'rb') as fl:
        data = pickle.load(fl)

    print(fl)


def fwhm_analysis():
    with open(dirs_path + 'climp_intensity.pkl', 'rb') as fl:
        intensity_data = pickle.load(fl)



