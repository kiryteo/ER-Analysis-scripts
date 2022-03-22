import skimage
import skimage.io as io
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import numpy as np
import vigra
import cv2
import mahotas
import diplib as dip
from skimage.filters import threshold_niblack, threshold_local, threshold_sauvola


from sklearn.metrics import jaccard_score

# img0 = io.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/COSKDEL/Decon/Series002_decon_converted/std/Series002_decon_converted_t79_ch00_std.png')
#
# stdimg = io.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/COSKDEL/Decon/Series002_decon_converted/std/Series002_decon_converted_t65_ch00_std.png')
# img = io.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/COSKDEL/Decon/Series002_decon_converted/enh/Series002_decon_converted_t65_ch00_std_enhance.png')

# img0 =

def get_image(img):
    return io.imread(img)


def vigra_skel(img):
    cc = vigra.analysis.labelImageWithBackground(img.astype('uint32'))
    skel = vigra.filters.skeletonizeImage(cc, 'PruneLength', 0.5)
    featdict = vigra.analysis.extractSkeletonFeatures(cc)
    return skel


def pcv_skel(img):
    return pcv.morphology.skeletonize(mask=img)


def skimg_skel(img):
    if np.max(img) == 255:
        img = img / 255
    return skimage.morphology.skeletonize(img)


def skimg_thin(img):
    if np.max(img) == 255:
        img = img / 255
    return skimage.morphology.thin(img)


def skimg_medaxis(img):
    if np.max(img) == 255:
        img = img / 255
    return skimage.morphology.medial_axis(img)


def cv_skel(img, type):
    if type == 'gh':
        return cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    else:
        return cv2.ximgproc.thinning(img)


def mahotas_thin(img):
    return mahotas.thin(img)


def dip_skel(img):
    dipimg = (img>0)
    skel_dip_three = dip.EuclideanSkeleton(dipimg, endPixelCondition='three neighbors')
    skel_dip_two = dip.EuclideanSkeleton(dipimg, endPixelCondition='two neighbors')
    skel_dip_one = dip.EuclideanSkeleton(dipimg, endPixelCondition='one neighbor')
    skel_dip_nat = dip.EuclideanSkeleton(dipimg, endPixelCondition='natural')
    return skel_dip_nat, skel_dip_one, skel_dip_two, skel_dip_three

def thresh_image(img, threshold=12, max_value=255):
    return pcv.threshold.binary(gray_img=img, threshold=threshold, max_value=max_value)

def thresh_niblack(img):
    """Grayscale image"""
    return threshold_niblack(img, window_size=5)


def thresh_loc(img, block=5, method='mean', offset=0, mode='nearest'):
    return threshold_local(img, block_size=block, method=method, offset=offset, mode=mode)


def skimage_dilate_img(img):
    return skimage.morphology.dilation(img)


def skimage_erode_img(img):
    return skimage.morphology.erosion(img.astype('int'))


def plot_samples():
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(dilsk)
    ax[1].imshow(dilavg)

    plt.show()

    # fig, ax = plt.subplots(2, 6)
    #
    # ax[0][0].imshow(stdimg)
    # # ax[0][1].imshow(er1)
    # # ax[0][2].imshow(loc)
    # # ax[0][3].imshow(er2)
    # ax[0][4].imshow(er3)
    # ax[0][5].imshow(mh_thin)
    # ax[1][0].imshow(skcv)
    # ax[1][1].imshow(skghcv)
    # ax[1][2].imshow(skpcv)
    # ax[1][3].imshow(sksk)
    # ax[1][4].imshow(skth)
    # ax[1][5].imshow(skmed)
    # plt.show()

    # ax[0].imshow(dipimg)
    # ax[1].imshow(thrlocal3)
    # ax[2].imshow(thrlocal5)
    # ax[3].imshow(thrlocal7)
    #
    # # ax[0].imshow(skel_dip_three)
    # # # ax[1].imshow(thin)
    # # # ax[2].imshow(medax)
    # # # ax[3].imshow(skel_pcv)
    # # # ax[4].imshow(skel_cv)
    # # # ax[5].imshow(skel_cv_gh)
    # # # ax[6].imshow(skel_maho)
    # # ax[1].imshow(skel_dip_two)
    # # ax[2].imshow(skel_dip_one)
    # # ax[3].imshow(skel_dip_nat)
    #
    # plt.show()

# stdimg = io.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/COSKDEL/Decon/Series002_decon_converted/std/Series002_decon_converted_t00_ch00_std.png')

def exp():
    er1 = skimage_erode_img(stdimg)
    loc = thresh_loc(er1)
    er2 = skimage_erode_img(loc)
    er3 = skimage_erode_img(er2)

    mh_thin = mahotas_thin(er3)
    skcv = cv_skel(er3.astype('uint8'), 'e')

    # fig, ax = plt.subplots(1, 4)
    # ax[0].imshow(er1)
    # ax[1].imshow(loc)
    # ax[2].imshow(er2)
    # ax[3].imshow(er3)

    # biner3 = threshold_niblack(er3)
    # skghcv = cv_skel(biner3.astype('uint8'), 'gh')
    # skpcv = pcv_skel(biner3)
    # sksk = skimg_skel(biner3.astype('int')/255)
    # skth = skimg_thin(er3.astype('int')/255)
    # skmed = skimg_medaxis(er3.astype('int')/255)

    # fsk = io.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDELCLIMP/COSKDELCLIMP/Decon/Series007_decon_converted/skel/Series007_decon_converted_t00_ch00_std_enhance_skel.png')
    # favg = io.imread('/localhome/asa420/MIAL/data/aggregation-with-median/Climp/avg/skel/ClimpSeries7-avg_skel.png')
    #
    # dilsk = skimage_dilate_img(fsk)
    # dilavg = skimage_dilate_img(favg)




    # timg = io.imread('/localhome/asa420/ER-Analysis-scripts/ero5.png')
    #
    op = skimage.filters.butterworth(er3, cutoff_frequency_ratio=0.02, order=3.0)#, high_pass=False)
    # k = np.where(op < 0)
    # op[k]=0
    plt.imshow(op)
    plt.colorbar()
    plt.show()

# exp()

# skel_dip_away = dip.EuclideanSkeleton(dipimg, endPixelCondition='loose ends away') ####### not useful


