import skimage
# from skimage.filters import threshold_local, threshold_otsu, unsharp_mask, frangi, meijering, sato, butterworth
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from plantcv import plantcv as pcv
import glob
from skimage import restoration
import seaborn as sns
import scipy.fftpack
import scipy.stats
from skimage import measure
from scipy import ndimage
from PIL import Image



def get_fuzzy_correlation(group):
    if group == 'ATL':
        img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_fuzzy/A1_mean_fuzz.png')
    elif group == 'Climp':
        img = imageio.imread('')
    elif group == 'Control':
        img = imageio.imread('')
    else:
        img = imageio.imread('')

    op = ndimage.median_filter(img, size=3)
    op = skimage.morphology.area_opening(op, area_threshold=9)
    flop = op.flatten()
    q = np.quantile(flop, 0.95)
    qt_list = np.where(op > q)
    # nqt = np.where(op <= q)
    # op[qt_list] = 255.
    # op[nqt] = 0.
    return qt_list


corr_vals_atl = []
for j in range(1, 27):
    mean_frame = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_fuzzy/A%s_mean_fuzz.png'%f'{j}')
    op = ndimage.median_filter(mean_frame, size=3)
    op = skimage.morphology.area_opening(op, area_threshold=9)
    flop = op.flatten()
    q = np.quantile(flop, 0.95)
    qt_list = np.where(op > q)
    for i in range(99):
        er_input = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A%s_decon_t0%s_ch00.tif'%(f'{j}', f'{i:02d}'))
        er_next = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/A%s_decon_t0%s_ch00.tif'%(f'{j}',f'{i+1:02d}'))

        std_er = (er_input - er_input.min()) / (er_input.max() - er_input.min())
        std_er_next = (er_next - er_next.min()) / (er_next.max() - er_next.min())

        fuz_patch_a = std_er[qt_list]
        fuz_patch_b = std_er_next[qt_list]

        valnum = np.corrcoef(fuz_patch_a, fuz_patch_b)
        corr_vals_atl.append(valnum[0, 1])

corr_vals_climp = []
for j in range(1, 32):
    mean_frame = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/Climp_fuzzy/C%s_mean_fuzz.png'%f'{j}')
    op = ndimage.median_filter(mean_frame, size=3)
    op = skimage.morphology.area_opening(op, area_threshold=9)
    flop = op.flatten()
    q = np.quantile(flop, 0.95)
    qt_list = np.where(op > q)
    for i in range(99):
        er_input = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/files/C%s_decon_t0%s_ch00.tif'%(f'{j}', f'{i:02d}'))
        er_next = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/files/C%s_decon_t0%s_ch00.tif'%(f'{j}',f'{i+1:02d}'))

        std_er = (er_input - er_input.min()) / (er_input.max() - er_input.min())
        std_er_next = (er_next - er_next.min()) / (er_next.max() - er_next.min())

        fuz_patch_a = std_er[qt_list]
        fuz_patch_b = std_er_next[qt_list]

        valnum = np.corrcoef(fuz_patch_a, fuz_patch_b)
        corr_vals_climp.append(valnum[0, 1])

corr_vals_ctrl = []
for j in range(1, 32):
    mean_frame = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/ctrl_fuzzy/Ct%s_mean_fuzz.png'%f'{j}')
    op = ndimage.median_filter(mean_frame, size=3)
    op = skimage.morphology.area_opening(op, area_threshold=9)
    flop = op.flatten()
    q = np.quantile(flop, 0.95)
    qt_list = np.where(op > q)
    for i in range(99):
        er_input = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_%s_decon_t0%s.tif'%(f'{j}', f'{i:02d}'))
        er_next = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Control/files/img_%s_decon_t0%s.tif'%(f'{j}',f'{i+1:02d}'))

        std_er = (er_input - er_input.min()) / (er_input.max() - er_input.min())
        std_er_next = (er_next - er_next.min()) / (er_next.max() - er_next.min())

        fuz_patch_a = std_er[qt_list]
        fuz_patch_b = std_er_next[qt_list]

        valnum = np.corrcoef(fuz_patch_a, fuz_patch_b)
        corr_vals_ctrl.append(valnum[0, 1])

corr_vals_rtn = []
for j in range(1, 30):
    mean_frame = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/RTN_fuzzy/R%s_mean_fuzz.png'%f'{j}')
    op = ndimage.median_filter(mean_frame, size=3)
    op = skimage.morphology.area_opening(op, area_threshold=9)
    flop = op.flatten()
    q = np.quantile(flop, 0.95)
    qt_list = np.where(op > q)
    for i in range(99):
        er_input = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/RTN/files/R%s_decon_t0%s_ch00.tif'%(f'{j}', f'{i:02d}'))
        er_next = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/RTN/files/R%s_decon_t0%s_ch00.tif'%(f'{j}',f'{i+1:02d}'))

        std_er = (er_input - er_input.min()) / (er_input.max() - er_input.min())
        std_er_next = (er_next - er_next.min()) / (er_next.max() - er_next.min())

        fuz_patch_a = std_er[qt_list]
        fuz_patch_b = std_er_next[qt_list]

        valnum = np.corrcoef(fuz_patch_a, fuz_patch_b)
        corr_vals_atl.append(valnum[0, 1])

sns.distplot(corr_vals_atl, label='ATL')
sns.distplot(corr_vals_climp, label='Climp')
sns.distplot(corr_vals_ctrl, label='Control')
sns.distplot(corr_vals_rtn, label='RTN')
plt.legend()
plt.show()

exit()



# for i in range(1, 27):
#     a = Image.open('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A%s_mean.png'%f'{i}')
#     b = Image.open('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/fuzzy_bin/A%s_fuzz_bin.png'%f'{i}')
#     c = Image.blend(a, b, 0.5)
#     imageio.imwrite('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A%s_fuzz_bin_overlay.png'%f'{i}', c)

# exit()

def frame_creator():
    fig = plt.figure(figsize=(8, 3))
    plt.axis('off')
    # plt.title('Final fuzzy : small elements removal + binarization')
    # plt.suptitle('RTN Series %s'%f'{series_num}')
    plt.title('ATL Series 2')
    r, c = 1, 4


    fig.add_subplot(r, c, 1)
    plt.imshow(imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A2_mean.png'))
    plt.title('Input')
    plt.axis('off')

    fig.add_subplot(r, c, 2)
    plt.imshow(imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_fuzzy/A2_mean_fuzz.png'))
    plt.title('Rolling ball (RB)')
    plt.axis('off')

    fig.add_subplot(r, c, 3)
    plt.imshow(imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/fuzzy_bin/A2_fuzz_bin.png'))
    plt.title('Final fuzzy')
    plt.axis('off')

    fig.add_subplot(r, c, 4)
    plt.imshow(imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/A2_fuzz_bin_overlay.png'))
    plt.title('Input + Fuzzy')
    plt.axis('off')

    # plt.show()
    plt.savefig('ATL_Series2_fuz', bbox_inches='tight')
    plt.close()



def get_fuzzy_binary():
    for i in range(1, 2):
        img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/ctrl_fuzzy/Ct%s_mean_fuzz.png'%f'{i}')
        op = ndimage.median_filter(img, size=3)
        op = skimage.morphology.area_opening(op, area_threshold=9)
        flop = op.flatten()
        q = np.quantile(flop, 0.95)
        qt_list = np.where(op > q)
        nqt = np.where(op <= q)
        op[qt_list] = 255.
        op[nqt] = 0.
        cv2.imwrite('/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/fuzzy_bin/Ct%s_fuzz_bin.png'%f'{i}', op)
        # plt.imshow(op)
        # plt.show()

# get_fuzzy_binary()
# exit()


def remove_small_components(series_num):
    mean_proj = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/RTN_mean_proj/R%s_mean.png'%f'{series_num}')
    img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/RTN_fuzzy/R%s_mean_fuzz.png'%f'{series_num}')
    # aop = skimage.morphology.area_opening(img, area_threshold=5)
    op = ndimage.median_filter(img, size=3)
    op = skimage.morphology.area_opening(op, area_threshold=9)

    flop = op.flatten()
    # plt.hist(flop)
    # plt.show()
    q = np.quantile(flop, 0.95)
    qt_list = np.where(op > q)
    op[qt_list] = 255.

    # plt.imshow(op)
    # plt.show()

    fig = plt.figure(figsize=(6, 4))
    plt.axis('off')
    plt.title('Final fuzzy : small elements removal + binarization')
    plt.suptitle('RTN Series %s'%f'{series_num}')
    r, c = 1, 3

    fig.add_subplot(r, c, 1)
    plt.imshow(mean_proj)
    plt.title('Input')
    plt.axis('off')

    fig.add_subplot(r, c, 2)
    plt.imshow(img)
    plt.title('Rolling ball (RB)')
    plt.axis('off')
    #
    fig.add_subplot(r, c, 3)
    plt.imshow(op)
    plt.title('Final fuzzy')
    plt.axis('off')

    plt.show()
    # plt.savefig('RTN_Series_%s_fuzzy_bin '%f'{series_num}', bbox_inches='tight')
    # plt.close()

# remove_small_components(15)
# exit()

def interframe_corr(group, num_series):
    pref = '/localhome/asa420/MIAL/data/confocal_movies/' + group + '/new_op_jul/skel/'
    corr_vals = []
    if group == 'ATL':
        ser_name = 'A'
    elif group == 'Climp':
        ser_name = 'C'
    elif group == 'Control':
        ser_name = 'Ct'
    else:
        ser_name = 'R'

    for i in range(1, num_series + 1):
        for j in range(99):
            im1 = imageio.imread(pref + '%s/%s_decon_t0%s_ch00_skel.png' % (f'{ser_name}{i}', f'{ser_name}{i}', f'{j:02d}')).flatten()
            im2 = imageio.imread(pref + '%s/%s_decon_t0%s_ch00_skel.png' % (f'{ser_name}{i}', f'{ser_name}{i}', f'{j + 1:02d}')).flatten()
            valnum = np.corrcoef(im1, im2)
            corr_vals.append(valnum[0, 1])

    return corr_vals


# corr_vals_atl = interframe_corr('ATL', 26)
# corr_vals_climp = interframe_corr('Climp', 31)
# corr_vals_ctrl = interframe_corr('Control', 31)
# corr_vals_rtn = interframe_corr('RTN', 29)
#
# sns.distplot(corr_vals_atl, label='ATL')
# sns.distplot(corr_vals_climp, label='Climp')
# sns.distplot(corr_vals_ctrl, label='Control')
# sns.distplot(corr_vals_rtn, label='RTN')
# plt.legend()
# plt.title('Interframe correlation across different groups')
# plt.show()
#
# exit()


def cc_fuzzy_regions():
    fuz = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_fuzzy/A2_mean_fuzz.png')
    # aop = skimage.morphology.area_opening(fuz, area_threshold=2)
    blobs = measure.label(fuz)
    plt.imshow(blobs)
    plt.show()

cc_fuzzy_regions()
exit()


def detect_fuzzy_win():
    # img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/window-5/C6_decon_ch00_skel_mean_1_win5.png')
    img = imageio.imread(
        '/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/window-10/C6_decon_ch00_skel_mean_1_win10.png')
    fuz = restoration.rolling_ball(img, radius=1)

    op = img - fuz

    # import copy
    # op = copy.deepcopy(img)
    #
    # tl = np.where(fuz > 0)
    # op[tl] = 0

    fig = plt.figure(figsize=(8, 4))
    plt.axis('off')
    plt.title('Climp_series6: Window 10 mean')
    r, c = 1, 3

    fig.add_subplot(r, c, 1)
    plt.imshow(img)
    plt.title('Input')
    plt.axis('off')
    #
    fig.add_subplot(r, c, 2)
    plt.imshow(fuz)
    plt.title('Rolling ball (RB)')
    plt.axis('off')

    fig.add_subplot(r, c, 3)
    plt.imshow(op)
    plt.title('Input - RB')
    plt.axis('off')

    # plt.show()
    plt.savefig('Climp_fuzzy_frame_series6-window10', bbox_inches='tight')
    plt.close()


detect_fuzzy_win()
exit()


def blur_detect_fft():
    image = cv2.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A1_mean.png')
    size = 60
    thresh = 10

    (cX, cY) = (int(128 / 2.0), int(128 / 2.0))

    fft = np.fft.fftn(image)
    fftShift = np.fft.fftshift(fft)

    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifftn(fftShift)

    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    op = (mean <= thresh)

    plt.imshow(op)
    plt.show()


def mean_proj_interval(group, num_series, interval):
    path = '/localhome/asa420/MIAL/data/confocal_movies/' + group + '/new_op_jul/'

    for series in range(1, num_series):
        cnt = 0
        for frame in range(0, 100, interval):
            cnt += 1
            img_mean = np.zeros((128, 128))
            for fr_stack in range(frame, frame + interval):
                each = path + 'skel/C%s/C%s_decon_t0%s_ch00_skel.png' % (f'{series}', f'{series}', f'{fr_stack:02d}')
                img = imageio.imread(each)
                img_mean += img
            cv2.imwrite(path + 'window-%s/C%s_decon_ch00_skel_mean_%s_win%s.png' % (
                f'{interval}', f'{series}', f'{cnt}', f'{interval}'), img_mean / interval)


mean_proj_interval('Climp', 20)
exit()


# for i in range(1, 5):
#     img = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A%s_mean.png'%f'{i}')
#     # kernel = restoration.ellipsoid_kernel((2, 2), 2)
#     # fuz = restoration.rolling_ball(img, kernel=kernel)
#     fuz = restoration.rolling_ball(img, radius=1)
#     op1 = skimage.morphology.erosion(fuz)
#     op2 = skimage.morphology.erosion(op1)
#
#     fig = plt.figure(figsize=(8, 4))
#     # plt.title('R%s_fuzzy'%f'{i}')
#     plt.axis('off')
#     r, c = 1, 4
#
#     fig.add_subplot(r, c, 1)
#     plt.imshow(img)
#     plt.title('ER input')
#     plt.axis('off')
#
#     fig.add_subplot(r, c, 2)
#     plt.imshow(fuz)
#     plt.title('Rolling ball')
#     plt.axis('off')
#
#     fig.add_subplot(r, c, 3)
#     plt.imshow(op1)
#     plt.title('RB + Erosion 1')
#     plt.axis('off')
#
#     fig.add_subplot(r, c, 4)
#     plt.imshow(op2)
#     plt.title('RB + Erosion 2')
#     plt.axis('off')
#
#     # plt.show()
#     plt.savefig('ATL_fuzzy_frame_%s'%f'{i}', bbox_inches='tight')
#     plt.close()
#
# exit()

def fuzzy_analysis_freq():
    # atl = []
    #
    # path_pref = '/localhome/asa420/MIAL/data/confocal_movies/Control/files/'
    #
    # img = imageio.imread(path_pref + 'C%s_decon_t0%s_ch00.tif' % (f'{i}', f'{j:02d}'))

    # for i in range(1, 27):
    #     img = imageio.imread(
    #         '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A%s_mean.png' % f'{i}')
    # kernel = restoration.ellipsoid_kernel((2, 2), 2)
    # fuz = restoration.rolling_ball(img, kernel=kernel)

    diff_list = []
    for i in range(0, 100):
        img = imageio.imread(
            '/localhome/asa420/MIAL/data/confocal_movies/Climp/files/C1_decon_t0%s_ch00.tif' % f'{i:02d}')
        std_img = ((img) / (img.max() - img.min())) * 255
        im2 = imageio.imread(
            '/localhome/asa420/MIAL/data/confocal_movies/Climp/files/C1_decon_t0%s_ch00.tif' % f'{i + 1:02d}')
        std_im2 = ((im2) / (im2.max() - im2.min())) * 255

        nz_img = np.where(std_img != 0)

        d1 = scipy.fftpack.dctn(std_img)
        d2 = scipy.fftpack.dctn(std_im2)

        # print(d1)
        # print(d2)

        diff = d2 - d1
        diff_m = diff.mean()
        diff_inv = scipy.fftpack.idctn(diff_m)

        diff_list.append(diff_inv)
        # print(diff_inv)

    plt.plot(diff_list)
    plt.xlabel('Difference values over sequence')
    plt.title('Subsequent frame frequency space difference (mean error)')
    # plt.legend()
    plt.show()

    # fuz = restoration.rolling_ball(std_img, radius=1)
    # op = skimage.morphology.erosion(fuz)
    # op = skimage.morphology.erosion(op)
    # cv2.imwrite('/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/RTN_mean_proj/R%s_mean_fuzz.png'%f'{i}', op)

    # print(np.unique(img))
    # print(np.unique(op))

    # nz_img = np.where(img != 0)
    # nz_op = np.where(op != 0)


# fuzzy_analysis_freq()
# exit()


def extract_fuzzy_ten():
    atl_ten = []
    for i in range(1):
        img = imageio.imread(
            '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A%s_mean.png' % f'{i}')


def cut_fuzzy_region():
    for i in range(1, 27):
        img = imageio.imread(
            '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A%s_mean.png' % f'{i}')

        # kernel = restoration.ellipsoid_kernel((2, 2), 2)
        # fuz = restoration.rolling_ball(img, kernel=kernel)

        fuz = restoration.rolling_ball(img, radius=1)
        # op1 = skimage.morphology.erosion(fuz)

        # print(np.mean(fuz))

        # print(fuz.max())
        # print(fuz.min())

        # fuz = fuz[np.where(fuz < 10) == 0]

        # tl = np.unique(fuz)
        # # plt.hist(tl)
        # sns.distplot(tl)
        # plt.show()

        reg1 = img - fuz

        # op2 = img - op1

        # op2 = skimage.morphology.erosion(reg1)
        # op2 = skimage.filters.threshold_local(reg1, block_size=3, method='median')
        # op2_skel = pcv.morphology.skeletonize(mask=op2)

        # img_bin = (img > 0).astype('int')
        # fuz_bin = (op1 > 0).astype('int')
        #
        # assert isinstance(fuz_bin, object)
        # reg_f = img_bin - fuz_bin

        # fuz2 = restoration.rolling_ball(reg1, radius=1)
        # op2 = reg1 - fuz2

        # reg_skel = pcv.morphology.skeletonize(mask=reg1)
        # reg1 = img[fuz==1]
        # cut_region = img - op1

        # plt.imshow(reg_skel)
        # plt.show()

        fig = plt.figure(figsize=(8, 4))
        plt.title('A%s_decon_mean_frame' % f'{i:02d}')
        plt.axis('off')
        r, c = 1, 3
        #
        fig.add_subplot(r, c, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Input')

        fig.add_subplot(r, c, 2)
        plt.imshow(fuz)
        plt.axis('off')
        plt.title('Rolling_ball (RB)')

        fig.add_subplot(r, c, 3)
        plt.imshow(reg1)
        plt.axis('off')
        plt.title('Input - RB')

        # fig.add_subplot(r, c, 4)
        # plt.imshow(op1)
        # plt.axis('off')
        # # plt.title('RB erosion')
        #
        # fig.add_subplot(r, c, 5)
        # # plt.imshow(cut_region)
        # plt.imshow(op2)
        # plt.axis('off')
        # # plt.title('Input - RB erosion')
        #
        # fig.add_subplot(r, c, 6)
        # # plt.imshow(cut_region)
        # plt.imshow(reg_f)
        # plt.axis('off')

        # plt.show()
        plt.savefig('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/fuzzy_sep/ATL_fuzzy_frame_%s' % f'{i}',
                    bbox_inches='tight')
        plt.close()

        # op = skimage.morphology.erosion(op)
        # cv2.imwrite('/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/RTN_mean_proj/R%s_mean_fuzz.png'%f'{i}', op)

        # print(np.unique(img))
        # print(np.unique(op))

        # nz_img = np.where(img != 0)
        # nz_op = np.where(op != 0)


cut_fuzzy_region()
exit()


def extract_fuzzy_region():
    atl = []
    for i in range(1, 27):
        img = imageio.imread(
            '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/ATL_mean_proj/A%s_mean.png' % f'{i}')
        # kernel = restoration.ellipsoid_kernel((2, 2), 2)
        # fuz = restoration.rolling_ball(img, kernel=kernel)
        fuz = restoration.rolling_ball(img, radius=1)
        op = skimage.morphology.erosion(fuz)
        # op = skimage.morphology.erosion(op)
        # cv2.imwrite('/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/RTN_mean_proj/R%s_mean_fuzz.png'%f'{i}', op)

        # print(np.unique(img))
        # print(np.unique(op))

        nz_img = np.where(img != 0)
        nz_op = np.where(op != 0)

        # print(len(nz_img[0]))
        # print(len(nz_op[0]))

        atl.append(len(nz_op[0]) / len(nz_img[0]))

    climp = []
    for i in range(1, 32):
        img = imageio.imread(
            '/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/Climp_mean_proj/C%s_mean.png' % f'{i}')
        fuz = restoration.rolling_ball(img, radius=1)
        op = skimage.morphology.erosion(fuz)
        # op = skimage.morphology.erosion(op)

        nz_img = np.where(img != 0)
        nz_op = np.where(op != 0)

        climp.append(len(nz_op[0]) / len(nz_img[0]))

    ctrl = []
    for i in range(1, 32):
        img = imageio.imread(
            '/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/Ctrl_mean_proj/Ct%s_mean.png' % f'{i}')
        fuz = restoration.rolling_ball(img, radius=1)
        op = skimage.morphology.erosion(fuz)
        # op = skimage.morphology.erosion(op)

        nz_img = np.where(img != 0)
        nz_op = np.where(op != 0)

        ctrl.append(len(nz_op[0]) / len(nz_img[0]))

    rtn = []
    for i in range(1, 30):
        img = imageio.imread(
            '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/RTN_mean_proj/R%s_mean.png' % f'{i}')
        fuz = restoration.rolling_ball(img, radius=1)
        op = skimage.morphology.erosion(fuz)
        # op = skimage.morphology.erosion(op)

        nz_img = np.where(img != 0)
        nz_op = np.where(op != 0)

        rtn.append(len(nz_op[0]) / len(nz_img[0]))

    atl = np.array(atl)
    climp = np.array(climp)
    ctrl = np.array(ctrl)
    rtn = np.array(rtn)

    # atl_norm = (atl - atl.min()) / (atl.max() - atl.min())
    # climp_norm = (climp - climp.min()) / (climp.max() - climp.min())
    # ctrl_norm = (ctrl - ctrl.min()) / (ctrl.max() - ctrl.min())
    # rtn_norm = (rtn - rtn.min()) / (rtn.max() - rtn.min())

    mx = max(atl.max(), climp.max(), ctrl.max(), rtn.max())
    mn = min(atl.min(), climp.min(), ctrl.min(), rtn.min())

    # sns.distplot(atl, hist=False, label='ATL')
    # sns.distplot(climp, hist=False, label='Climp')
    # sns.distplot(ctrl, hist=False, label='Control')
    # sns.distplot(rtn, hist=False, label='RTN')
    # plt.xlim(mn, mx)

    sns.boxplot(atl)
    sns.boxplot(climp)
    sns.boxplot(ctrl)
    sns.boxplot(rtn)

    # plt.scatter(atl, label='ATL')
    # plt.scatter(climp, label='Climp')
    # plt.scatter(ctrl, label='Control')
    # plt.scatter(rtn, label='RTN')
    plt.xlabel('Fuzzy vs ER ratio')
    plt.title('erosion ops: 1')
    plt.legend()
    plt.show()

    # fig = plt.figure(figsize=(8, 4))
    # # plt.title('R%s_fuzzy'%f'{i}')
    # plt.axis('off')
    # r, c = 1, 2
    #
    # fig.add_subplot(r, c, 1)
    # plt.imshow(img)
    # plt.axis('off')
    #
    # fig.add_subplot(r, c, 2)
    # plt.imshow(op)
    # plt.axis('off')
    #
    # plt.show()


extract_fuzzy_region()
exit()


def fuzzy_frames():
    for i in range(1, 30):
        img_path = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/RTN_mean_proj/R%s_mean.png' % f'{i}'
        fuzzy_path = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/fuzzy/R%s_mean_fuzz.png' % f'{i}'

        fig = plt.figure(figsize=(8, 4))
        plt.title('R%s_fuzzy' % f'{i}')
        plt.axis('off')
        r, c = 1, 2

        fig.add_subplot(r, c, 1)
        plt.imshow(cv2.imread(img_path))
        plt.axis('off')

        fig.add_subplot(r, c, 2)
        plt.imshow(cv2.imread(fuzzy_path))
        plt.axis('off')

        plt.savefig('R%s_fuzzy_frame_cv.png' % f'{i}', bbox_inches='tight')
        plt.close()


fuzzy_frames()
exit()


def preprocess_samples(group):
    path_pref = '/localhome/asa420/MIAL/data/confocal_movies/' + group + '/files/'
    new_pref = '/localhome/asa420/MIAL/data/confocal_movies/' + group + '/new_op_jul/preproc/'

    for i in range(27, 32):
        os.makedirs(new_pref + 'C%s' % f'{i}')
        for j in range(100):
            img = imageio.imread(path_pref + 'C%s_decon_t0%s_ch00.tif' % (f'{i}', f'{j:02d}'))
            std_img = ((img) / (img.max() - img.min())) * 255
            aop = skimage.morphology.area_opening(std_img, area_threshold=2)
            erod = skimage.morphology.erosion(aop)
            aop = skimage.morphology.area_opening(erod, area_threshold=2)
            cl = skimage.morphology.area_closing(aop, area_threshold=32)
            aop = skimage.morphology.area_opening(cl, area_threshold=2)
            loc = threshold_local(aop, 3)
            loc = threshold_local(loc, 3)
            cv2.imwrite(new_pref + 'C%s/C%s_decon_t0%s_ch00_proc.png' % (f'{i}', f'{i}', f'{j:02d}'), loc)


# preprocess_samples('Climp')
# exit()


def get_skel():
    pref = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/'
    for i in range(1, 30):
        os.makedirs(pref + 'skel/R%s' % f'{i}')
        for j in range(100):
            img = imageio.imread(
                pref + 'preproc/R%s/R%s_decon_t0%s_ch00_proc_enhance.png' % (f'{i}', f'{i}', f'{j:02d}'))
            sk = pcv.morphology.skeletonize(img)
            cv2.imwrite(pref + 'skel/R%s/R%s_decon_t0%s_ch00_skel.png' % (f'{i}', f'{i}', f'{j:02d}'), sk)


# get_skel()
# exit()


def Agg(path):
    for i in range(1, 32):
        files = glob.glob(path + 'C%s/*' % (f'{i}'))
        imgstack = []
        img_mean = np.zeros((128, 128))
        for each in files:
            img = imageio.imread(each)
            img_mean += img
            imgstack.append(img)
        cv2.imwrite('/localhome/asa420/MIAL/data/confocal_movies/Climp/new_op_jul/C%s_mean.png' % f'{i}',
                    img_mean / len(files))
        # pcv.print_image(img=img_mean/len(files), filename=home + '/Desktop/Climp/' + 'C%s-mean-brpts.png'%(f'{i}'))
        # # imageio.imwrite(home + '/Desktop/ATL/' + 'A%s-mean.png'%(f'{i}'), img_mean/len(files))
        # new = np.stack(imgstack, axis=2)
        # maximg = np.amax(new, axis=2)
        # pcv.print_image(img=maximg, filename=home + '/Desktop/Climp/' + 'C%s-max-brpts.png'%(f'{i}'))
        # imageio.imwrite(home + '/Desktop/ATL/' + 'A%s-max.png'%(f'{i}'), maximg)
