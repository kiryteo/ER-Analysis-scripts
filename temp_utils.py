import glob
import os
import shutil
import numpy

home = os.path.expanduser('~')

def dirmaker():
    for i in range(1, 17):
        # os.makedirs(home + '/Desktop/RTN/skel/R' + str(i))
        # os.makedirs(home + '/Desktop/RTN/brpts/R' + str(i))
        # os.makedirs(home + '/Desktop/Climp/mcherry/C' + str(i))
        os.makedirs(home + '/MIAL/data/live-cell-movies/COSKDELCLIMP/COSKDELCLIMP/Decon/Series0%s_decon_converted/erenh/'%(f'{i:02d}'))
        # os.makedirs(home + '/Desktop/RTN/frames_10/R' + str(i))
        # os.rename(home + '/Desktop/Climp/brpts/A' + str(i), home + '/Desktop/Climp/brpts/C' + str(i))
        # pref = home + '/Desktop/Climp/brpts/A%s'

# dirmaker()
# exit()


def mover():
    for i in range(11, 17):
        origdir = home + '/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series0%s_decon_converted/erode/*'%(f'{i:02d}')
        newdir = home + '/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series0%s_decon_converted/erenh/'%(f'{i:02d}')
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

mover()

exit()

def file_rename():
    dir = home + '/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/rtn/'
    files = glob.glob(dir + '*')
    for each in files:
        name = each.split('/')[-1]
        ncomps = name.split('.')
        newname = dir + ncomps[0] + '_' + ncomps[1] + '.' + ncomps[2]
        os.rename(each, newname)

def file_move():
    dir = home + '/MIAL/live-cell-movies/COSKDELRTN/COSKDELRTN/Decon/'
    subdirs = glob.glob(dir + '*')
    # print(subdirs)
    for each in subdirs:
        newdir = each + '_MeanProjection_frames/'
        os.makedirs(newdir)
        filedir = each + '/files/*.tif_mean_skel.tif'
        files = glob.glob(filedir)
        for tfile in files:
            shutil.move(tfile, newdir)
        # print(files)


def normalize(img):
    return (img - np.min(img) / np.max(img) - np.min(img))
