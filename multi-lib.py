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
import phasepack
# import matlab.engine as m_engine
#
# Engine = m_engine.start_matlab()


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


def dip_skel(dipimg):
    # dipimg = (img>0)
    skel_dip_three = dip.EuclideanSkeleton(dipimg, endPixelCondition='three neighbors')
    skel_dip_two = dip.EuclideanSkeleton(dipimg, endPixelCondition='two neighbors')
    skel_dip_one = dip.EuclideanSkeleton(dipimg, endPixelCondition='one neighbor')
    skel_dip_nat = dip.EuclideanSkeleton(dipimg, endPixelCondition='natural')
    return skel_dip_nat, skel_dip_one, skel_dip_two,skel_dip_three

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



stdimg = io.imread('/localhome/asa420/MIAL/multi-lib-op/Series002_decon_converted_t00_ch00_std.png')
# enh = io.imread('/localhome/asa420/MIAL/multi-lib-op/Series002_decon_converted_t00_ch00_std_enhance.png')


def cv2_medax(img):
    img = cv2.imread(img,0)
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)

    ret,img = cv2.threshold(img,10,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True

    return skel


# cvm = cv2_medax('/localhome/asa420/ER-Analysis-scripts/er1_enhance.png')
#
# plt.imshow(cvm)
# plt.show()
#
# exit()

def newskels(imgpath):
    files = []
    skelmean = np.zeros((128, 128))
    img = io.imread(imgpath)
    sksk = skimg_skel(img)
    files.append(sksk)
    skmedax = skimg_medaxis(img)
    files.append(skmedax)
    skthin = skimg_thin(img)
    files.append(skthin)
    cvth = cv_skel(img, 'e')
    files.append(cvth)
    cvghth = cv_skel(img, 'gh')
    files.append(cvghth)
    thmaho = mahotas_thin(img)
    files.append(thmaho)

    t1, t2, t3, tnat = dip_skel(img.astype('bool'))
    files.append(t1)
    files.append(t2)
    files.append(t3)
    files.append(tnat)
    cvm = cv2_medax(imgpath)
    files.append(cvm)

    # matskel = Engine.bwskel(Engine.imread(imgpath))
    # matmorph = Engine.bwmorph(Engine.imread(imgpath), 'skel', 'Inf')

    for each in files:
        # img = imageio.imread(each)
        skelmean += each
        # imgstack.append(img)
    # pcv.print_image(img=skelmean, filename='er1-mean-skel.png')

    op = (skelmean > 0).astype('int') * 255
    pref = '/localhome/asa420/MIAL/data/live-cell-movies/unet-exp/images/control/'

    fn = imgpath.split('/')[-1].split('.')[0] + '_projection.png'
    pcv.print_image(img=op, filename=pref + fn)
    # plt.imshow(op)
    # plt.show()

    # fig, ax = plt.subplots(2, 6)
    # ax[0][0].imshow(img)
    # ax[0][1].imshow(sksk)
    # ax[0][2].imshow(skmedax)
    # ax[0][3].imshow(skthin)
    # ax[0][4].imshow(cvth)
    # ax[0][5].imshow(cvm)
    # ax[1][0].imshow(cvghth)
    # ax[1][1].imshow(thmaho)
    # ax[1][2].imshow(t1)
    # ax[1][3].imshow(t2)
    # ax[1][4].imshow(t3)
    # ax[1][5].imshow(tnat)
    #
    # plt.show()

import glob

dirs = glob.glob('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/Decon/*')
for each in dirs:
    dirend = each.split('/')[-1].split('_')[-1]
    sernum = each.split('/')[-1].split('_')[0][-3:]
    if dirend == 'converted':
        files = glob.glob(each + '/erenh/*')
        for file in files:
            newskels(file)

exit()
# newskels('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/Decon/Series002_decon_converted/erenh/Series002_decon_converted_t01_ch00_std_erode_enhance.png')
# exit()

# dipimg = io.imread('/localhome/asa420/ER-Analysis-scripts/er1_enhance.png').astype('bool')
# skel_dip_three = dip.EuclideanSkeleton(dipimg, endPixelCondition='three neighbors')
#
# plt.imshow(skel_dip_three)
# plt.show()

def high_pass_filter(img):
    er3 = io.imread('/localhome/asa420/ER-Analysis-scripts/ero5.png')
    op = skimage.filters.butterworth(er3, cutoff_frequency_ratio=0.002, order=3.0)#, high_pass=False)
    # # k = np.where(op < 0)
    # # op[k]=0
    plt.imshow(op)
    plt.colorbar()
    plt.show()

def exp():

    ### Dont use clahe
    # clahe = cv2.createCLAHE()
    # cl1 = clahe.apply(stdimg)
    loc = thresh_loc(stdimg, 3)
    er1 = skimage_erode_img(loc)
    # loc = thresh_loc(er1)

    # ph = phasepack.phasecong(er1)

    # phloc = thresh_loc(ph[1])
    # phadapt = threshold_adaptive(ph, 10, offset=10)

    # erenh = skimage_erode_img(enh)
    # er2 = skimage_erode_img(er1)
    # loc = thresh_loc(er1)
    # er2 = skimage_erode_img(loc)
    # loc2 = thresh_loc(er2)
    # er3 = skimage_erode_img(loc2)

    # plt.imshow(er1)
    # plt.show()

    # io.imsave('locer1.png', er1)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(cl1)
    ax[1].imshow(er1)
    ax[2].imshow(loc)
    # ax[0].imshow(stdimg)
    # ax[1].imshow(er1)
    # ax[2].imshow(loc)
    # ax[3].imshow(ph[3])
    # ax[4].imshow(phloc)
    # ax[3].imshow(er2)
    # ax[4].imshow(loc2)
    # ax[5].imshow(er3)
    plt.show()

    # mh_thin = mahotas_thin(er3)
    # skcv = cv_skel(er3.astype('uint8'), 'e')

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

# exp()

def create_erode_img():
    for i in range(7, 17):
        dirname = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/COSKDELRTN/Decon/Series0%s_decon_converted/std/*'%(f'{i:02d}')
        files = glob.glob(dirname)
        for each in files:
            img = io.imread(each)
            erode = skimage_erode_img(img)
            fname = each.split('.')[0] + '_erode.png'
            cv2.imwrite(fname, erode)


# skel_dip_away = dip.EuclideanSkeleton(dipimg, endPixelCondition='loose ends away') ####### not useful