"""
Initial method to obtain individual tubules and label them.
NOT USED FOR ANALYSIS ANYMORE.
"""

from plantcv import plantcv as pcv
import subprocess
import imageio
import numpy as np
import copy
from skimage.transform import resize
# import matlab.engine
import scipy
import matplotlib.pyplot as plt

import os
import glob
import scipy
import cv2

home = os.path.expanduser('~')


def standardize_image(img):
    return img - np.min(img) / (np.max(img) - np.min(img))


def thresh_image(img, threshold=12, max_value=255):
    return pcv.threshold.binary(gray_img=img, threshold=threshold, max_value=max_value)


def get_skel(img):
    return pcv.morphology.skeletonize(mask=img)


def get_brpts(img):
    return pcv.morphology.find_branch_pts(skel_img=img)


def dilate_brpts(img):
    return pcv.dilate(gray_img=img, ksize=3, i=1)


def get_tubules(img):
    thr = thresh_image(img)
    skel = get_skel(thr)
    brpts = get_brpts(skel)
    return skel - brpts


def create_image(img, fname):
    pcv.print_image(img=img, filename=fname)


def network_extraction(file):
    img, path, filename = pcv.readimage(file)
    std_img = standardize_image(img)
    # pcv.print_image(img=std_img, filename=file.split('.')[0] + '_std.png')
    thresh = thresh_image(std_img)
    skeleton = get_skel(thresh)
    branchpts = get_brpts(skeleton)

    

    # pcv.print_image(img=branchpts, filename= home + '/Desktop/Climp/brpts/' + file.split('.')[0].split('/')[-1] + '_brpts.png')

    # fname = file.split('.')[0] + '_skel.png'
    # pcv.print_image(img=skel, filename=fname)


def adapt_hist(file):
    img, path, filename = pcv.readimage(file)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    op = clahe.apply(img)
    pcv.print_image(img=op, filename=file.split('.')[0] + '_adj.png')


def adapt_hist_runner():
    files = glob.glob('/localhome/asa420/Desktop/Climp/files/*_ch01_std.png')

    # for each in files:
    #     network_extraction(each)

    # for each in files:
    #     adapt_hist(each)


def vessel_enhance(fname):
    return subprocess.run(['matlab', '-nodesktop', '-nosplash', '-nodisplay',
                           '-r "Vessel2d(\'/localhome/asa420/Desktop/RTN/std/Series%s_decon_converted/std/Series%s_decon_converted_t%s_ch00_std.png\');exit;" ' % (
                           f'{i:03d}', f'{i:03d}', f'{j:02d}')], stdout=subprocess.PIPE)


# def enhance_image():
#     for i in range(11, 12):
#         for j in range(13, 100):
#             enhanced_image = subprocess.run(['matlab', '-nodesktop', '-nosplash', '-nodisplay', '-r "Vessel2d(\'/localhome/asa420/Desktop/RTN/std/Series%s_decon_converted/std/Series%s_decon_converted_t%s_ch00_std.png\');exit;" '%(f'{i:03d}', f'{i:03d}', f'{j:02d}')], stdout=subprocess.PIPE)

# enhanced_image = subprocess.run(['matlab', '-nodesktop', '-nosplash', '-nodisplay', '-r "Vessel2d(\'/localhome/asa420/MIAL/std_img0.png\');exit;" '], stdout=subprocess.PIPE)

# Get skeleton
def skeleton_processing(file):
    img, path, filename = pcv.readimage(file)

    thresh = pcv.threshold.binary(gray_img=img, threshold=12, max_value=255)

    skel = pcv.morphology.skeletonize(mask=thresh)

    pcv.print_image(img=skel, filename=file.split('.')[0] + '_skel.png')
    branchpts = pcv.morphology.find_branch_pts(skel_img=skel)
    pcv.print_image(img=branchpts, filename=file.split('.')[0] + '_brpts.png')
    # dilate_brpts = pcv.dilate(gray_img=branchpts, ksize=3, i=1)


# pcv.print_image(img=dilate_brpts, filename='/home/ashwin/MIAL/blob-analysis/sted/1_Series012_decon_merged_patch10_sted_enh_dil.png')

# tubules = skel - branchpts
# tubules = skel - dilate_brpts
# pcv.print_image(img=tubules, filename= file.split('.')[0] + '_tubules.png')

# enh1 = glob.glob('/localhome/asa420/MIAL/live-cell-movies/COSKDELCLIMP/COSKDELCLIMP/Decon/Series004_decon_converted/enh/*')

# enh1 = glob.glob('/localhome/asa420/MIAL/live-cell-movies/COSKDEL/COSKDEL/Decon/Series004_decon_converted/enh/*')

# skeleton_processing('/localhome/asa420/MIAL/aggregation-with-median/Climp/avg/ClimpSeries1-avg_enhance.png')


def label_tubules():
    tubules = subprocess.run(['/usr/local/MATLAB/R2018a/bin/matlab', '-nodesktop', '-nosplash', '-nodisplay',
                              '-r "label_tubules(\'/home/ashwin/MIAL/KV_paired_ER_tubules/STED/RTN/tub_dil/3_23_2021 RTN4ACOS7 Paired STED Decon_Series001_decon_ch02_std_enhance_tubules.png\');exit;" '],
                             stdout=subprocess.PIPE)


# Label tubules
# tubules = subprocess.run(['/usr/local/MATLAB/R2018a/bin/matlab', '-nodesktop', '-nosplash', '-nodisplay', '-r "label_tubules(\'/home/ashwin/MIAL/KV_paired_ER_tubules/STED/RTN/tub_dil/3_23_2021 RTN4ACOS7 Paired STED Decon_Series001_decon_ch02_std_enhance_tubules.png\');exit;" '], stdout=subprocess.PIPE)
# tubules = subprocess.run(['/usr/local/MATLAB/R2018a/bin/matlab', '-nodesktop', '-nosplash', '-nodisplay', '-r "label_tubules(\'/home/ashwin/MIAL/KV_paired_ER_tubules/STED/RTN/tub_dil/3_23_2021 RTN4ACOS7 Paired STED Decon_Series002_decon_ch02_std_enhance_tubules.png\');exit;" '], stdout=subprocess.PIPE)
# tubules = subprocess.run(['/usr/local/MATLAB/R2018a/bin/matlab', '-nodesktop', '-nosplash', '-nodisplay', '-r "label_tubules(\'/home/ashwin/MIAL/KV_paired_ER_tubules/STED/RTN/tub_dil/3_23_2021 RTN4ACOS7 Paired STED Decon_Series003_decon_ch02_std_enhance_tubules.png\');exit;" '], stdout=subprocess.PIPE)

Climpdir = glob.glob('/home/ashwin/MIAL/KV_paired_ER_tubules/STED/Climp/lab_dil/*')
Ctrldir = glob.glob('/home/ashwin/MIAL/KV_paired_ER_tubules/STED/Control/lab_dil/*')
RTNdir = glob.glob('/home/ashwin/MIAL/KV_paired_ER_tubules/STED/RTN/lab_dil/*')



def filter_tubules(file):
    file1 = imageio.imread(file)
    # l_exc = []
    l_tub = []

    for i in range(1, len(np.unique(file1))):
        a, b = np.where(file1 == i)[0], np.where(file1 == i)[1]

        if len(a) > 36:
            l_tub.append(i)

        # if len(a) < 160:
        #     l_exc.append(i)
        # else:
        #     # print(i, len(a))
        #     l_tub.append(i)

    fnew = copy.deepcopy(file1)
    # for each in l_exc:
    # 	fnew[np.where(file1==each)] = 200
    for each in l_tub:
        fnew[np.where(file1 == each)] = 254

    fnew[np.where(fnew != 254)] = 0
    imageio.imwrite(file.split('.')[0] + '_selected.png', fnew)

    # new = copy.deepcopy(fnew)

    # new[np.where(fnew==254)] = 255

    # imageio.imwrite('/home/ashwin/MIAL/KV_paired_ER_tubules/STED/Climp/3_23_2021 CLIMP COS7 Paired STED Decon_Series001_decon_ch02_res_std_tubules_labeled_selected.png', new)

# for each in Climpdir:
#     filter_tubules(each)
# for each in Ctrldir:
#     filter_tubules(each)
# for each in RTNdir:
#     filter_tubules(each)

# label these again

# relab = subprocess.run(['/usr/local/MATLAB/R2018a/bin/matlab', '-nodesktop', '-nosplash', '-nodisplay', '-r "label_tubules(\'/home/ashwin/MIAL/KV_paired_ER_tubules/STED/RTN/selection_dil/3_23_2021 RTN4ACOS7 Paired STED Decon_Series001_decon_ch02_std_enhance_tubules_labeled_selected.png\');exit;" '], stdout=subprocess.PIPE)
