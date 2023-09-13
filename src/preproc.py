import imageio
import skimage
from skimage.filters import threshold_local, threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
import copy
from skimage.color import rgb2gray
import cv2
from skimage import measure
from skimage.morphology import area_closing
import scipy.ndimage
import sknw
import networkx as nx
import os
from plantcv import plantcv as pcv

from skimage import exposure
from skimage.exposure import match_histograms
import mclahe as mc



for i in range(1, 17):
    try:
        enh = imageio.imread(f'/localhome/asa420/MIAL/data/live-cell-movies/rtn_mean/er_mean_proc/Series0{i:02d}_decon_converted_mean_proc_v2_enhance.png')
        skel = pcv.morphology.skeletonize(mask=enh)

        imageio.imsave(f'/localhome/asa420/MIAL/data/live-cell-movies/rtn_mean/er_mean_proc/Series0{i:02d}_decon_converted_mean_proc_v2_enhance_skel.png', skel)
    except Exception:
        pass


exit()

# group = 'Control'

# for i in range(2, 17):
#     try:
#         os.makedirs(f'/localhome/asa420/MIAL/data/live-cell-movies/{group}/preproc/Series0{i:02d}_decon_converted')
#     except Exception:
#         pass

# import shutil

# sted_data_path = '/localhome/asa420/MIAL/data/live-cell-movies/'




# for i in range(2, 17):
#     try:
#         for num in range(100):
#             shutil.move(f'{sted_data_path}{group}/Series0{i:02d}_decon_converted/preproc/Series0{i:02d}_decon_converted_t{num:02d}_ch00_proc_filt.png', f'/localhome/asa420/MIAL/data/live-cell-movies/{group}/preproc/Series0{i:02d}_decon_converted/Series0{i:02d}_decon_converted_t{num:02d}_ch00_proc_filt.png')
#     except Exception:
#         pass

# exit()


# img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_CC_Contours_grayscale.png')
# print(img.shape)
# exit()

# img = imageio.imread('mv50_gray.png')
# ot = threshold_local(img, 75)
# img = img > ot
# sk = pcv.morphology.skeletonize(mask=img)
# cv2.imwrite('mv50_gray_bin_local.png', sk)
# exit()

# img = imageio.imread('/localhome/asa420/Downloads/AnalyzER_Simon_Fraser_University/some_GTs/GFP-HDEL_mv_50_Airyscan Processing_t1.jpg')
# # print(img.shape)
# img_gr = rgb2gray(img)
# cv2.imwrite('mv50_gray.png', img_gr*255)
# exit()

# for i in range(10, 12):
#     os.makedirs(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/Climp/Series0{i}_decon_converted/enh')

# for i in range(1, 6):
#     os.makedirs(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/rtn_skel/Series0{i:02d}_decon_converted/')

# for i in range(7, 11):
#     os.makedirs(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/rtn_skel/Series0{i:02d}_decon_converted/')

# for i in range(12, 17):
#     os.makedirs(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/RTN/Series0{i:02d}_decon_converted/align')

# import shutil
# for num in range(12, 17):
#     for i in range(100):
#         shutil.move(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/RTN/Series0{num:02d}_decon_converted/Series0{num:02d}_decon_converted_t{i:02d}_ch00_proc_align.png', f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/RTN/Series0{num:02d}_decon_converted/align/Series0{num:02d}_decon_converted_t{i:02d}_ch00_proc_align.png')






# ref = imageio.imread('/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/Climp/Series001_decon_converted/files/Series001_decon_converted_t00_ch00.tif')
# img = imageio.imread('/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/Climp/Series001_decon_converted/files/Series001_decon_converted_t55_ch00.tif')
# img = (img - img.min()) / (img.max() - img.min())

# matched = match_histograms(img, ref)

# # Our hyperparameters of choice were the kernel size (20, 20, 10, 25), 256 bins in the histogram, and a clipping limit of 0.25. We used a global histogram range


# # mcl1 = mc.mclahe(img)
# mcl1 = mc.mclahe(img, adaptive_hist_range=True)
# l = threshold_local(mcl1, 5)

# # mcl1 = mc.mclahe(matched, adaptive_hist_range=True)
# # l = threshold_local(mcl1, 5)

# plt.subplot(141)
# plt.imshow(img)
# plt.subplot(142)
# plt.imshow(matched)
# plt.subplot(143)
# plt.imshow(mcl1)
# plt.subplot(144)
# plt.imshow(l)

# plt.show()

# exit()


# mcl2 = mc.mclahe(img, kernel_size=(32, 32), n_bins=256, clip_limit=0.25)
# mcl2 = mc.mclahe(img, kernel_size=(32, 32), n_bins=256)
# mcl3 = mc.mclahe(img, kernel_size=(64, 64), n_bins=256, clip_limit=0.25)

# plt.imshow(mcl)
# plt.show()

# plt.subplot(131)
# plt.imshow(mcl1)
# plt.subplot(132)
# plt.imshow(mcl2)
# plt.subplot(133)
# plt.imshow(mcl3)

# plt.show()

# exit()

# imgh = img.flatten()
# # print(img.shape)

# # plt.hist(imgh)
# # plt.show()

# plt.hist(mcl.flatten())
# plt.show()

# exit()

import numpy as np
import cv2
from im2dhisteq import im2dhisteq

# def imresize(img, wr=500, hr=None): # This is just for imshow-ing images with titles
#     [ h, w] = img.shape
#     hr = (h*wr)//w if not hr else hr
#     img_resized = cv2.resize(img, dsize=(wr, hr))
#     return img_resized

# def main():
#     fullname = '/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/Climp/Series001_decon_converted/files/Series001_decon_converted_t00_ch00.tif'
#     image = cv2.imread(fullname)
#     # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # w_neighboring=6 is generally an adequate value, drived by a lot of experimenting.
#     # w_neighboring=6 corresponds to a 13*13 square
#     gray_image_2DHisteq = im2dhisteq(image, w_neighboring=5)

#     # This is just for imshow-ing images with titles
#     # gray_Image_resized = imresize(gray_image)
#     # gray_Image_2DHisteq_resized = imresize(gray_image_2DHisteq)

#     # cv2.imshow('Original Image', gray_Image_resized)
#     # cv2.imshow('2DHeq Image', gray_Image_2DHisteq_resized)
#     cv2.imshow('Original Image', image)
#     cv2.imshow('2DHeq Image', gray_image_2DHisteq)
#     cv2.waitKey(0)

# if __name__ == '__main__': main()


# exit()

import imageio

# def histogram_equalization_non_zero(image):
#     # Calculate the histogram of non-zero values
#     non_zero_values = [value for value in image.ravel() if value != 0]  # Flatten the image and filter non-zero values
#     histogram = [0] * 256
#     total_non_zero_pixels = len(non_zero_values)

#     for pixel_value in non_zero_values:
#         histogram[pixel_value] += 1

#     # Calculate the cumulative distribution function (CDF) of the histogram
#     cdf = [0] * 256
#     cdf[0] = histogram[0] / total_non_zero_pixels

#     for i in range(1, 256):
#         cdf[i] = cdf[i - 1] + (histogram[i] / total_non_zero_pixels)

#     # Scale the CDF to the range [0, 255]
#     cdf_scaled = [int(255 * val + 0.5) for val in cdf]

#     # Apply the equalization to the input image
#     equalized_image = [0 if pixel_value == 0 else cdf_scaled[pixel_value] for pixel_value in image.ravel()]

#     return equalized_image

# # Load a grayscale image using imageio
# input_image = imageio.imread('your_image.png', as_gray=True)

# # Perform histogram equalization based on non-zero values
# equalized_image = histogram_equalization_non_zero(input_image)

# # Display or save the result as needed
# imageio.imsave('equalized_image.png', equalized_image.astype('uint8'))



# data = []
# def get_intensity_variation():
#     for i in range(100):
#         # img = imageio.imread(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/Climp/Series001_decon_converted/files/Series001_decon_converted_t{i:02d}_ch00.tif')
#         # img = (img - img.min()) / (img.max() - img.min())
#         img = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/std_egfp/A2_decon_t0{i:02d}_ch00_std.png')
#         m = np.mean(img)
#         data.append(m)

#     plt.plot(data)
#     plt.show()

# get_intensity_variation()

# exit()



# exit()

# import itk
# input_image = itk.imread('/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/Climp/Series001_decon_converted/files/Series001_decon_converted_t55_ch00.tif')
# hessian_image = itk.hessian_recursive_gaussian_image_filter(
#     input_image, sigma=1.0
# )

# vesselness_filter = itk.Hessian3DToVesselnessMeasureImageFilter[
#     itk.ctype("float")
# ].New()
# vesselness_filter.SetInput(hessian_image)
# vesselness_filter.SetAlpha1(0.5)
# vesselness_filter.SetAlpha2(2.0)

# itk.imwrite(vesselness_filter, '/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/Climp/Series001_decon_converted/Series001_decon_converted_t55_ch00_vessel.png')

# exit()


# img = imageio.imread('/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/Climp/Series001_decon_converted/files/Series001_decon_converted_t55_ch00.tif')

# img = (img - img.min()) / (img.max() - img.min())
# op = np.invert(img)

# imageio.imsave('/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/Climp/Series001_decon_converted/Series001_decon_converted_t55_ch00_inv.png', op)

# exit()


# import itk

# input_image = '/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/Climp/Series001_decon_converted/files/Series001_decon_converted_t55_ch00.tif'

# input_image = itk.imread(input_image, itk.F)

# ImageType = type(input_image)
# Dimension = input_image.GetImageDimension()
# HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
# HessianImageType = itk.Image[HessianPixelType, Dimension]

# objectness_filter = itk.HessianToObjectnessMeasureImageFilter[
#     HessianImageType, ImageType
# ].New()
# objectness_filter.SetBrightObject(True)
# objectness_filter.SetScaleObjectnessMeasure(False)
# objectness_filter.SetAlpha(0.5)
# objectness_filter.SetBeta(1.0)
# objectness_filter.SetGamma(5.0)

# multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[
#     ImageType, HessianImageType, ImageType
# ].New()
# multi_scale_filter.SetInput(input_image)
# multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
# multi_scale_filter.SetSigmaStepMethodToLogarithmic()
# multi_scale_filter.SetSigmaMinimum(1.0)
# multi_scale_filter.SetSigmaMaximum(10.0)
# multi_scale_filter.SetNumberOfSigmaSteps(10)

# OutputPixelType = itk.UC
# OutputImageType = itk.Image[OutputPixelType, Dimension]

# rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
# rescale_filter.SetInput(multi_scale_filter)

# itk.imwrite(rescale_filter.GetOutput(), '/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/Climp/Series001_decon_converted/Series001_decon_converted_t55_ch00_vessel.png')


# exit()




def mean_proj_proc():
    for i in range(16, 17):
        img = imageio.imread(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/control_mean/Series0{i:02d}_decon_converted_mean.png')

        thr = threshold_otsu(img)
        thr = thr / 5

        aop = skimage.morphology.area_opening(img, area_threshold=2)
        erod = skimage.morphology.erosion(aop)
        aop = skimage.morphology.area_opening(erod, area_threshold=2)
        cl = skimage.morphology.area_closing(aop, area_threshold=32)
        aop = skimage.morphology.area_opening(cl, area_threshold=2)

        loc = threshold_local(aop, 3)

        loc = skimage.morphology.dilation(loc)

        bloc = copy.deepcopy(loc)
        bloc[np.where(loc<=thr)] = 0

        imageio.imsave(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/control_mean/Series0{i:02d}_decon_converted_mean_proc.png', bloc)

# mean_proj_proc()
# exit()



def align_images(group):
    for num in range(1, 17):
        try:
            mean_proc = imageio.imread(f'/localhome/asa420/MIAL/data/live-cell-movies/{group.lower()}_mean/Series0{num:02d}_decon_converted_mean_proc.png')
            mean_proc_fl = mean_proc.flatten()
            # for i in range(100):
            #     img = imageio.imread(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/{group}/Series0{num:02d}_decon_converted/files/Series0{num:02d}_decon_converted_t{i:02d}_ch00.tif')
            #     img = (img - img.min()) / (img.max() - img.min())
            #     img = img * 255

            #     thr = threshold_otsu(img)
            #     thr = thr / 5
            #     img[np.where(img<thr)] = 0

            #     img_fl = img.flatten()

            #     mask = img_fl > 0
            #     # res = np.where(mask, np.maximum(img_fl, mean_proc_fl), mean_proc_fl)
            #     res = np.where(mask, np.maximum(img_fl, mean_proc_fl), 0)

            #     res[np.where(mean_proc_fl==0)] = 0

            #     res = res.reshape(img.shape)
            #     # plt.imshow(res)
            #     # plt.imshow(res.reshape(img.shape))
            #     # plt.show()
            #     imageio.imsave(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/{group}/Series0{num:02d}_decon_converted/Series0{num:02d}_decon_converted_t{i:02d}_ch00_proc_align.png', res)
            for i in range(100):
                img = imageio.imread(f'/localhome/asa420/MIAL/data/live-cell-movies/{group}/Series0{num:02d}_decon_converted/preproc/Series0{num:02d}_decon_converted_t{i:02d}_ch00_proc_enhance.png')
                img_fl = img.flatten()
                img_fl[np.where(mean_proc_fl==0)] = 0

                # plt.imshow(img_fl.reshape(img.shape))
                # plt.show()
                imageio.imsave(f'/localhome/asa420/MIAL/data/live-cell-movies/{group}/Series0{num:02d}_decon_converted/preproc/Series0{num:02d}_decon_converted_t{i:02d}_ch00_proc_filt.png', img_fl.reshape(img.shape))
        except Exception:
            pass

align_images('RTN')
exit()




# for i in range(100):
#     plt.subplot(121)
#     img = imageio.imread(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/Climp/Series001_decon_converted/files/Series001_decon_converted_t{i:02d}_ch00.tif')
#     img = (img - img.min()) / (img.max() - img.min())
#     img = img * 255
#     thr = threshold_otsu(img)
#     thr = thr / 4
#     img[np.where(img<thr)] = 0
#     plt.imshow(img)
#     plt.subplot(122)
#     plt.imshow(imageio.imread(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/Climp/Series001_decon_converted/aligned/Series001_decon_converted_t{i:02d}_ch00_proc_align.png'))

#     plt.show()
# exit()

# img = imageio.imread('/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/Climp/Series001_decon_converted/align-v2/Series001_decon_converted_t60_ch00_proc_align.png')
# for i in range(1, 17):
#     try:
#         img = imageio.imread(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/rtn_mean/Series0{i:02d}_decon_converted_mean_proc_v2_gt.png')

#         op = copy.deepcopy(img)
#         op[np.where(img!=255)] = 0

#         imageio.imsave(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/rtn_mean/Series0{i:02d}_decon_converted_mean_proc_skel.png', op)
#     except Exception:
#         pass

# exit()


# for i in range(1, 17):
#     try:
#         img = imageio.imread(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/rtn_mean/Series0{i:02d}_decon_converted_mean_proc.png')
#         aop = skimage.morphology.area_opening(img, area_threshold=2)
#         erod = skimage.morphology.erosion(aop)
#         aop = skimage.morphology.area_opening(erod, area_threshold=2)
#         cl = skimage.morphology.area_closing(aop, area_threshold=32)
#         aop = skimage.morphology.area_opening(cl, area_threshold=2)

#         loc = threshold_local(aop, 3)

#         imageio.imsave(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/rtn_mean/Series0{i:02d}_decon_converted_mean_proc_v2.png', loc)
#     except Exception:
#         pass


exit()


def preprocess_samples(group):
    for num in range(3, 11):
        # ref = imageio.imread(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/{group}/Series0{num:02d}_decon_converted/files/Series0{num:02d}_decon_converted_t00_ch00.tif')
        # ref = (ref - ref.min()) / (ref.max() - ref.min())
        for i in range(100):
            img = imageio.imread(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/{group}/Series0{num:02d}_decon_converted/files/Series0{num:02d}_decon_converted_t{i:02d}_ch00.tif')

            img = (img - img.min()) / (img.max() - img.min())

            img = mc.mclahe(img)
            # img = match_histograms(img, ref)

            aop = skimage.morphology.area_opening(img, area_threshold=2)
            erod = skimage.morphology.erosion(aop)
            aop = skimage.morphology.area_opening(erod, area_threshold=2)
            cl = skimage.morphology.area_closing(aop, area_threshold=32)
            aop = skimage.morphology.area_opening(cl, area_threshold=2)

            loc = threshold_local(aop, 3)
            # loc = threshold_local(loc, 3)

            plt.imshow(loc)
            plt.show()

            # imageio.imsave(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/{group}/Series0{num:02d}_decon_converted/preproc/Series0{num:02d}_decon_converted_t{i:02d}_ch00_proc.png', loc)
            # cv2.imwrite(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/Control/Series0{num:02d}_decon_converted/preproc/Series0{num:02d}_decon_converted_t{i:02d}_ch00_proc.png', loc)


# preprocess_samples('Climp')


def get_skels(group):
    for i in range(12, 17):
        for frame in range(35, 100):
            img = imageio.imread(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/{group}/Series0{i:02d}_decon_converted/preproc/Series0{i:02d}_decon_converted_t{frame:02d}_ch00_proc_enhance.png')
            sk = pcv.morphology.skeletonize(mask=img)
            # imageio.imsave(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/{group.lower()}_skel/Series0{i:02d}_decon_converted/Series0{i:02d}_decon_converted_t{frame:02d}_ch00_proc_filt_skel.png', sk)
            op = skimage.morphology.white_tophat(sk)
            plt.imshow(op)
            plt.show()


# get_skels('RTN')

def mean_skels(group):
    try:
        for i in range(1, 11):
            skel = imageio.imread(f'/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/climp_mean/Series0{i:02d}_decon_converted_mean_proc.png')

    except:
        pass


exit()


img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_t0_ch0.png')

aop = skimage.morphology.area_opening(img, area_threshold=2)
erod = skimage.morphology.erosion(aop)
aop = skimage.morphology.area_opening(erod, area_threshold=2)
cl = skimage.morphology.area_closing(aop, area_threshold=32)
aop = skimage.morphology.area_opening(cl, area_threshold=2)

cv2.imwrite('morph-ops_c12_t0.png', aop)

loc = threshold_local(aop, 3)
loc = threshold_local(loc, 3)

cv2.imwrite('thresh_ops_c12_t0.png', loc)

# cv2.imwrite('AnalyzER-mv50.png', loc*255)

exit()


skel = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/er_mean_proc/climp12_er_mean_proc_enhance_skel.png')


def skel_to_graph(skel):
    g = sknw.build_sknw(skel, iso=False)
    G = nx.Graph()

    node_set = g.nodes()
    edge_set = g.edges()

    G.add_nodes_from(node_set)
    G.add_edges_from(edge_set)

    degree_list = G.degree
    return g, node_set, edge_set, degree_list

graph, node_set, edge_set, d = skel_to_graph(skel)

node_coords = np.array([node_set[node]['o'] for node in node_set])
newps = [node_coords[j] for j, val in enumerate(d) if val[1] > 2]
nps = [[each[0], each[1]] for each in newps]
nps = np.array(nps)

er_mean = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/er_mean/climp12_er_mean.png')
er_proj = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/er_mean_proc/climp12_er_mean_proc.png')


# er_proj[59, 52] = 258
#
# ot = threshold_otsu(er_proj)
# e = np.where(er_proj>ot)
#
# er_proj[e] = ot


# exit()

# er_proj[e] = er_proj[e] - 5
# ref_junc = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/Climp_12_proj_junctions.png')

# plt.imshow(er_proj)
# plt.imshow(contours)
# plt.imshow(ref_junc)
# plt.show()
# exit()


img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_junc_projection.png')

op = area_closing(img, area_threshold=3)

contours = measure.find_contours(op)



fig, ax = plt.subplots()
# s = np.where(img>0)
# img[s] = 0
ax.imshow(er_mean, cmap=plt.cm.gray)

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], 'cyan', linewidth=1)

ax.plot(nps[:, 1], nps[:, 0], 'o', markerfacecolor='magenta', markeredgecolor='magenta', mew=0.75, markersize=3)

# ax.imshow(ref_junc)

plt.axis('off')
# plt.show()
plt.savefig('C12_er_proj_Contours_ref_junc.png', bbox_inches='tight', pad_inches=0, dpi=700)
# plt.close()
# ax.axis('image')
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()
exit()

# er_mean = np.zeros((128,128))
# for i in range(100):
#     img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/junctions/C12/C12_decon_t0%s_ch00_junc.png'%(f'{i:02d}'))
#     # img = (img - img.min())/(img.max() - img.min())
#     er_mean += img
# cv2.imwrite('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/junctions/C12_junc_projection.png', (er_mean)*255)

# exit()

img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/files/C12_decon_t099_ch00.tif')
op = (img - img.min()) / (img.max() - img.min())
op = op * 255
cv2.imwrite('C12_t99_ch0.png', op)
exit()
#
# img = imageio.imread('/localhome/asa420/Downloads/Oligos_PLP1_abcam_MBP_DAPI_40x_z0_ch00.tif')
# op = (img - img.min()) / (img.max() - img.min())
# op = op * 255
# cv2.imwrite('/localhome/asa420/Downloads/Oligos_PLP1_abcam_MBP_DAPI_40x_z0_ch00_std.png', op)
# exit()

def preprocess_samples():
    img = imageio.imread('/localhome/asa420/Downloads/O4+ Day 81 on IBIDI_1_std.png')
    # std_img = ((img) / (img.max() - img.min())) * 255
    # plt.imshow(img)
    # plt.show()
    # exit()
    # img = rgb2gray(img)
    aop = skimage.morphology.area_opening(img, area_threshold=2)
    # plt.imshow(aop)
    # plt.show()
    # exit()

    # erod = skimage.morphology.erosion(aop)
    # nerod = copy.deepcopy(erod)
    # # plt.imshow(erod)
    # # plt.show()
    # nerod[np.where(erod)]= img[np.where(erod)]
    #
    # nimg = img - nerod
    #
    # plt.imshow(nimg)
    # plt.show()
    #
    # exit()
    # aop = skimage.morphology.area_opening(nimg, area_threshold=2)

    # plt.imshow(aop)
    # plt.show()
    #
    # exit()

    cl = skimage.morphology.area_closing(aop, area_threshold=32)
    # plt.imshow(cl)
    # plt.show()
    #
    # exit()
    aop = skimage.morphology.area_opening(cl, area_threshold=2)
    # plt.imshow(aop)
    # plt.show()

    # print(aop.shape)
    aop = rgb2gray(aop)
    # plt.imshow(aop)
    # plt.show()
    #
    # exit()
    loc = threshold_local(aop, 3)
    # plt.imshow(loc)
    # plt.show()
    # exit()

    loc = threshold_local(loc, 3)
    # plt.axis('off')
    # plt.imshow(loc, cmap='gray')
    # plt.show()
    # exit()

    imageio.imsave('/localhome/asa420/Downloads/O4+ Day 81 on IBIDI_1_std_proc.png', loc)
    # plt.savefig('/localhome/asa420/Downloads/Oligos_PLP1_abcam_MBP_DAPI_40x_z0_ch00_std_proc.png', bbox_inches='tight', pad_inches=0, dpi=400)
    # cv2.imwrite('/localhome/asa420/Downloads/Oligos_PLP1_abcam_MBP_DAPI_40x_z0_ch00_std_proc.png', loc*255)

# preprocess_samples()
#
# exit()

# img = imageio.imread('/localhome/asa420/Downloads/Oligos_PLP1_abcam_MBP_DAPI_40x_z0_Composite.png')
# print(img.min())
# print(img.max())

# img = imageio.imread('/localhome/asa420/Downloads/Oligos_preproc_enhance.png')
# aop = skimage.morphology.area_opening(img, area_threshold=8)

from skimage.filters.rank import median
from skimage.morphology import disk

# aop = scipy.ndimage.median_filter(img, size=4)

# fig, axes = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
# ax = axes.ravel()

# ax[0].imshow(skimage.morphology.area_opening(img, area_threshold=12))
# ax[1].imshow(median(img, disk(2)))

# aop = median(img, disk(1))
# aop = threshold_local(aop, 3)

# cv2.imwrite('/localhome/asa420/Downloads/Oligos_preproc_enh_proc.png', aop)

# plt.imshow(aop)
# plt.show()

from plantcv import plantcv as pcv

nimg = imageio.imread('/localhome/asa420/Downloads/Oligos_PLP1abcam_MBP_DAPI_40x_z0_ch00_std_proc_enhance.png')
sk = pcv.morphology.skeletonize(nimg)

# op = median(sk, disk(1))
# op = skimage.morphology.area_opening(sk, area_threshold=1)
# op = skimage.morphology.area_opening(op, area_threshold=1)
# op = skimage.morphology.area_closing(op, area_threshold=12)

cv2.imwrite('/localhome/asa420/Downloads/Oligos_PLP1abcam_MBP_DAPI_40x_z0_ch00_std_proc_enhance_skel.png', sk)
# plt.imshow(op)
# plt.show()

