import scipy
import scipy.signal
import scipy.ndimage
import imageio
import cv2
import sep
import matplotlib.pyplot as plt

from plantcv import plantcv as pcv
import glob
import skimage

from skimage.exposure import match_histograms
import shutil
import os

import numpy as np
import skimage.io as io
from scipy.signal import correlate


img = imageio.imread('/localhome/asa420/MIAL/data/live-cell-movies/annotations/Control/Ctrl_Series2/Series002_decon_converted_t01_ch00_std.png')

med = scipy.signal.medfilt2d(img)

clahe = cv2.createCLAHE()
medcl = clahe.apply(med).astype('float')

# print(type(medcl))
# exit()
obj = sep.extract(medcl, 15)
plt.imshow(obj)
plt.show()


def get_tubules():
    files = glob.glob('/localhome/asa420/MIAL/data/CROP-n/Plos_data_crops/RTN_crops/RTN_enh/*')

    for each in files:
        img = imageio.imread(each)
        skel = pcv.morphology.skeletonize(mask=img)
        brpts = pcv.morphology.find_branch_pts(skel_img=skel)
        dil_brpts = skimage.morphology.binary_dilation(brpts).astype('int') * 255
        tubules = skel - dil_brpts
        name = '/localhome/asa420/MIAL/data/CROP-n/Plos_data_crops/RTN_crops/' + each.split('/')[-1][:-12] + '_tubules.png'
        pcv.print_image(tubules, name)


def run_matching(grp, ser_num):
    if grp == 'ctrl':
        dir_pref = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL'
    elif grp == 'rtn':
        dir_pref = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL' + 'RTN'
    else:
        dir_pref = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL' + 'CLIMP'

    ref_img_name = dir_pref + '/Decon/Series%s_decon_converted/std/Series%s_decon_converted_t00_ch00_std.png'%(f'{ser_num:03d}', f'{ser_num:03d}')
    ref_img = imageio.imread(ref_img_name)

    img_path = dir_pref + '/Decon/Series%s_decon_converted/std/'%f'{ser_num:03d}'

    matched_dir = dir_pref + '/Decon/Series%s_decon_converted/matching/'%f'{ser_num:03d}'
    os.makedirs(matched_dir)

    shutil.copy(ref_img_name, matched_dir)

    for i in range(1, 100):
        imgname = img_path + 'Series%s_decon_converted_t%s_ch00_std.png' % (f'{ser_num:03d}', f'{i:02d}')
        img = imageio.imread(imgname)
        op = match_histograms(img, ref_img).astype('uint8')
        imageio.imsave(matched_dir + imgname.split('/')[-1], op)


def interfr_corr():
    imdir = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/Decon/Series002_decon_converted/std/'
    #
    # l = []
    #
    # im1 = io.imread(imdir + 'Series002_decon_converted_t05_ch00_std.png')
    # im2 = io.imread(imdir + 'Series002_decon_converted_t08_ch00_std.png')
    #
    # print(np.corrcoef(im1.flat, im2.flat)[0, 1])
    #
    # # num1 = im1.flat - np.mean(im1.flat)
    # # num2 = im2.flat - np.mean(im2.flat)
    # #
    # # print(np.mean(im1.flat))
    # # print(num1)
    #
    # exit()


    l = []
    for i in range(2,3):
        im1 = io.imread(imdir + 'Series002_decon_converted_t%s_ch00_std.png'%f'{i:02d}')
        for j in range(i+1, 100):
            im2 = io.imread(imdir + 'Series002_decon_converted_t%s_ch00_std.png'%f'{j:02d}')
            cm = np.corrcoef(im1.flat, im2.flat)
            l.append(cm[0, 1])
            # print(j, cm[0, 1])

        #l.append(cm[0, 1])

    #plt.hist(l)
    plt.plot(l)
    plt.show()


def skimg_filters():
    img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/preproc/A4/A4_decon_t002_ch00_proc.png')

    # op = skimage.filters.threshold_local(img)
    # op = skimage.filters.sobel(img)
    # op = skimage.filters.hessian(img, (1,3,1))

    # op = skimage.filters.frangi(img, (1,2,1))
    # op = skimage.filters.threshold_local(img, 3, method='mean')

    fd, op = skimage.feature.hog(img, visualize=True)

    import matplotlib.pyplot as plt

    plt.imshow(op)
    plt.show()


"""
Hausdorff distance calculation
"""

import medpy
import numpy as np
import imageio
import medpy.metric


def get_hausdorff_dist(grp, num, frame):
    pref = '/localhome/asa420/MIAL/data/live-cell-movies/annotations/'
    if grp == 'climp':
        gt = imageio.imread(pref + 'Climp/Climp_Series%s/Series0%s_t%s.png'%(f'{num}', f'{num}', f'{frame:02d}'))
        pred = imageio.imread(pref + 'Climp/Climp_Series%s/Series0%s_decon_converted_t%s_ch00_std_enhance_skel.png'%(f'{num}', f'{num}', f'{frame:02d}'))
    elif grp == 'ctrl':
        gt = imageio.imread(pref + 'Control/Ctrl_Series%s/Series0%s_t%s.png'%(f'{num}', f'{num}', f'{frame:02d}'))
        pred = imageio.imread(pref + 'Control/Ctrl_Series%s/Series0%s_decon_converted_t%s_ch00_std_enhance_skel.png'%(f'{num}', f'{num}', f'{frame:02d}'))
    else:
        gt = imageio.imread(pref + 'RTN/RTN_Series%s/Series00%s_t%s.png'%(f'{num}', f'{num}', f'{frame:02d}'))
        pred = imageio.imread(pref + 'RTN/RTN_Series%s/Series00%s_decon_converted_t%s_ch00_std_enhance_skel.png'%(f'{num}', f'{num}', f'{frame:02d}'))

    hdd = medpy.metric.binary.hd(pred, gt)
    print(grp, num)
    print(str(frame))
    print(hdd)


# get_hausdorff_dist('rtn', 9, 35)
