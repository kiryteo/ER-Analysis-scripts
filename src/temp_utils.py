import glob
import os
import shutil
from skan import draw
import numpy as np
import skimage.io as io
import skimage
from skimage import exposure
import imageio
import matplotlib.pyplot as plt

import cv2

confocal_data_path = '/localhome/asa420/MIAL/data/confocal_movies/'


skel_path = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/skel/'


def get_mean_img():
    for ser_num in range(1, 32):
        img_list = []
        for frame in range(100):            
            skel = imageio.imread(f'{skel_path}/R{ser_num}/R{ser_num}_decon_t0{frame:02d}_ch00_skel.png')
            img_list.append(skel)
            img_list = np.array(img_list)
            mean_img = np.mean(img_list, axis=0)
            imageio.imsave(f'/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/R{ser_num}_mean_skel.png', mean_img)

# get_mean_img()


def get_std_samples(group, num_series):
    for i in range(1, num_series+1):
        for j in range(100):
            img = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/files/{group[0]}{i}_decon_t0{j:02d}_ch01.tif')
            op = (img - img.min()) / (img.max() - img.min())
            cv2.imwrite(f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/std_mch/{group[0]}{i}_decon_t0{j:02d}_ch01_std.png', op*255)


# get_std_samples('ATL', 26)
# get_std_samples('RTN', 29)
# get_std_samples('Climp', 31)
# exit()




def preproc_individual_sample(img_path):
    """

    @param img_path: path to ER input sample
    @return: processed sample
    """
    img = imageio.imread(img_path)
    std_img = ((img) / (img.max() - img.min())) * 255
    img_aop = skimage.morphology.area_opening(std_img, area_threshold=2)
    img_erod = skimage.morphology.erosion(img_aop)
    img_aop = skimage.morphology.area_opening(img_erod, area_threshold=2)
    img_closing = skimage.morphology.area_closing(img_aop, area_threshold=32)
    img_aop = skimage.morphology.area_opening(img_closing, area_threshold=2)
    img_thr_loc = threshold_local(img_aop, 3)
    return threshold_local(img_thr_loc, 3)


def preproc_groups(path, group, num_series):
    """

    # @param path: path to all ER input files
    # @param group: group to process (ATL, Climp, Control, RTN)
    # @param num_series: number of movies in the group
    """
    # path_pref = '/localhome/asa420/MIAL/data/live-cell-movies/' + group + '/Decon/'
    # # new_pref = '{confocal_data_path}' + group + '/new_op_jul/preproc/'
    # new_pref = '/localhome/asa420/MIAL/data/live-cell-movies/' + group + '/Decon/'

    # for i in range(2, 3):
    for _ in range(num_series):
        os.makedirs(f'{new_pref}C{i}')
        for _ in range(100):
            img_path = ...
            # img = imageio.imread(path_pref + 'Series%s_decon_converted/files/Series%s_decon_converted_t%s_ch00.tif' % (f'{i:03d}', f'{i:03d}', f'{j:02d}'))
            processed_sample = preproc_individual_sample(img_path)

            cv2.imwrite(
                f'{new_pref}Series{i:03d}_decon_converted/new_op_sept/preproc/C{i}/C{i}_decon_t0{j:02d}_ch00_proc.png',
                processed_sample)



def sharpen_filter():
    img = imageio.imread('')

    low_vals = np.where(img < 10)
    img[low_vals] = 0

    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    op = cv2.filter2D(img, -1, kernel)

    cv2.imwrite('', op)

    img_and = cv2.bitwise_and(img, op)


def skel_over_input(img, skel):
    fig, ax = plt.subplots()
    op = draw.overlay_skeleton_2d(img, skel, dilate=0, axes=ax)
    plt.imshow(op)


home = os.path.expanduser('~')

files = glob.glob('/localhome/asa420/MIAL/data/confocal_movies/ATL/files/*')

for each in files:
    if 'A1_' in each and 'std_std_adj' in each:
        shutil.move(each, '/localhome/asa420/MIAL/data/confocal_movies/ATL/std_adj/')


exit()


dname = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series015_decon_converted/matching/'

files = glob.glob(f'{dname}*')
def match_enh_mover():
    ndir = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series015_decon_converted/match_enh/'
    os.makedirs(ndir)
    for each in files:
        if 'enhance' in each:
            shutil.move(each, ndir)

match_enh_mover()

exit()



# /localhome/asa420/MIAL/data/live-cell-movies/unet-exp/images/rtn/Series001_decon_converted_t00_ch00_std.png
#
# /localhome/asa420/MIAL/data/live-cell-movies/unet-exp/PGT/rtn/Series001_decon_converted_t00_ch00_std_erode_enhance_projection.png


dl = glob.glob('/localhome/asa420/MIAL/data/live-cell-movies/unet-exp/images/rtn/*')
print(dl[0])
#ids = [os.path.splitext(file)[0] for file in dl if not file.startswith('.')]

#print(ids[0])
exit()
# dirlist = glob.glob('/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/*')
# for each in dirlist:
#     if each.split('/')[-1].split('_')[-1] == 'converted':
#         files = glob.glob(each + '/std/*')
#         for file in files:
#             shutil.copy(file, '/localhome/asa420/MIAL/data/live-cell-movies/unet-exp/images/RTN/')


exit()

path = '/localhome/asa420/MIAL/data/mean-proj-annotations/RTN/'
dire = glob.glob(f'{path}*')

for each in dire:
    img = io.imread(each)
    op = (img==255).astype('int')
    fname = path + each.split('/')[-1].split('.')[0] + '_annot.png'
    io.imsave(fname, op)

exit()

def dirmaker():
    for i in range(1, 17):
        os.makedirs(f'{home}/MIAL/data/live-cell-movies/COSKDELRTN/R{i}')
        # os.makedirs(home + '/Desktop/RTN/skel/R' + str(i))
        # os.makedirs(home + '/Desktop/RTN/brpts/R' + str(i))
        # os.makedirs(home + '/Desktop/Climp/mcherry/C' + str(i))
        # os.makedirs(home + '/MIAL/data/live-cell-movies/COSKDELCLIMP/COSKDELCLIMP/Decon/Series0%s_decon_converted/erenh/'%(f'{i:02d}'))
        # os.makedirs(home + '/Desktop/RTN/frames_10/R' + str(i))
        # os.rename(home + '/Desktop/Climp/brpts/A' + str(i), home + '/Desktop/Climp/brpts/C' + str(i))
        # pref = home + '/Desktop/Climp/brpts/A%s'

# dirmaker()
# exit()

def mover():
    # files = glob.glob(home + '/MIAL/data/confocal_movies/Control/std/*')\
    dirs = glob.glob('/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/*')
    for each in dirs:
        dirend = each.split('/')[-1].split('_')[-1]
        sernum = each.split('/')[-1].split('_')[0][-3:]
        if dirend == 'converted':
            files = glob.glob(f'{each}/std/*')
            for file in files:
                shutil.copy(file, f'/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/R{int(sernum)}')

# mover()
# exit()

for each in files:
    fname = each.split('/')[-1]
    series = fname.split('_')[0]
        # series = fname.split('_')[1]
        # shutil.move(each, home + '/MIAL/data/confocal_movies/img' + series + '/')
    if fname.split('_')[-1] == 'std.png':
        shutil.move(each, f'{home}/MIAL/data/confocal_movies/{series}/')



# mover()
# exit()

def mover():
    for i in range(11, 17):
        origdir = f'{home}/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series0{i:02d}_decon_converted/erode/*'

        newdir = f'{home}/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series0{i:02d}_decon_converted/erenh/'

        files = glob.glob(origdir)
        for file in files:
            end = file.split('/')[-1].split('_')[-1]
            if end == 'enhance.png':
                shutil.move(file, newdir)


    # dr = '/localhome/asa420/Desktop/Climp/'
    # dr1 = '/localhome/asa420/Desktop/RTN/skel/'
    # files = glob.glob(dr + '*_skel_brpts.png')
    # for i in range(1, 32):
    #     for j in range(100):
    #         # fname_sk = dr1 + 'R%s_decon_t0%s_ch00_skel.png'%(f'{i}',f'{j:02d}')
    #         # fname_br = dr2 + 'R%s_decon_t0%s_ch00_skel_brpts.png'%(f'{i}',f'{j:02d}')
    #         fname = dr + 'files/C%s_decon_t0%s_ch01_std_adj.png'%(f'{i}',f'{j:02d}')
    #         # shutil.move(fname_sk, dr1 + 'R%s'%(f'{i}'))
    #         # shutil.move(fname_br, dr2 + 'R%s'%(f'{i}'))
    #         shutil.move(fname, dr + 'mcherry/C%s/'%(f'{i}'))

# mover()
# exit()

def file_rename():
    dir = f'{home}/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/rtn/'

    files = glob.glob(f'{dir}*')
    for each in files:
        name = each.split('/')[-1]
        ncomps = name.split('.')
        newname = dir + ncomps[0] + '_' + ncomps[1] + '.' + ncomps[2]
        os.rename(each, newname)

def file_move():
    dir = f'{home}/MIAL/live-cell-movies/COSKDELRTN/COSKDELRTN/Decon/'
    subdirs = glob.glob(f'{dir}*')
    # print(subdirs)
    for each in subdirs:
        newdir = f'{each}_MeanProjection_frames/'
        os.makedirs(newdir)
        filedir = f'{each}/files/*.tif_mean_skel.tif'
        files = glob.glob(filedir)
        for tfile in files:
            shutil.move(tfile, newdir)
        # print(files)


def normalize(img):
    return (img - np.min(img) / np.max(img) - np.min(img))


def apply_adapthist_movie():
    for i in range(100):
        img = imageio.imread(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/std_adj/A1_decon_t0{i:02d}_ch01_std_std_adj.png')
        op = exposure.equalize_adapthist(img, clip_limit=0.05)
        imageio.imwrite(f'/localhome/asa420/MIAL/data/confocal_movies/ATL/adapthist/A1_decon_t0{i:02d}_ch01_std_std_adj_adapthist.png', op)


def temp_fig():
    std_pref = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series005_decon_converted/std/'
    hist_pref = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series005_decon_converted/hist/'

    for i in range(100):
        fig = plt.figure(figsize=(12, 6))
        #plt.title('Confocal-RTN-Series%s-frame%s' % (f'{series}', f'{frame:02d}'), size=18)
        plt.axis('off')
        r, c = 1, 2

        # std = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series005_decon_converted/std/Series005_decon_converted_t00_ch00_std.png'
        # hist= '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series005_decon_converted/hist/Series005_decon_converted_t00_hist.png'

        fig.add_subplot(r, c, 1)
        plt.imshow(cv2.imread(std))
        plt.axis('off')
        plt.title('Input')

        fig.add_subplot(r, c, 2)
        plt.imshow(cv2.imread(hist))
        plt.axis('off')
        plt.title('Histogram')

        #plt.show()
        # plt.savefig('/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series005_decon_converted/img_hist' + , bbox_inches='tight')


exit()
img = imageio.imread('/localhome/asa420/ER-Analysis-scripts/Figure3-FuzIso/C12_t0_ch0_loc2_enhance_skel.png')
# aop = skimage.morphology.area_opening(img, area_threshold=2)
aop = skimage.morphology.remove_small_objects(img, 32)
plt.imshow(aop)
plt.show()


def preprocess_samples(group):
    path_pref = f'/localhome/asa420/MIAL/data/live-cell-movies/{group}/Decon/'
    # new_pref = '/localhome/asa420/MIAL/data/confocal_movies/' + group + '/new_op_jul/preproc/'
    new_pref = f'/localhome/asa420/MIAL/data/live-cell-movies/{group}/Decon/'

    for i in range(2, 3):
        os.makedirs(f'{new_pref}C{i}')
        for j in range(100):
            img = imageio.imread(f'{path_pref}Series{i:03d}_decon_converted/files/Series{i:03d}_decon_converted_t{j:02d}_ch00.tif')

            std_img = ((img) / (img.max() - img.min())) * 255
            aop = skimage.morphology.area_opening(std_img, area_threshold=2)
            erod = skimage.morphology.erosion(aop)
            aop = skimage.morphology.area_opening(erod, area_threshold=2)
            cl = skimage.morphology.area_closing(aop, area_threshold=32)
            aop = skimage.morphology.area_opening(cl, area_threshold=2)
            loc = threshold_local(aop, 3)
            loc = threshold_local(loc, 3)
            cv2.imwrite(f'{new_pref}Series{i:03d}_decon_converted/new_op_sept/preproc/C{i}/C{i}_decon_t0{j:02d}_ch00_proc.png', loc)



