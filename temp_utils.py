import glob
import os
import shutil
import numpy

home = os.path.expanduser('~')

def dirmaker():
    for i in range(1, 30):
        # os.makedirs(home + '/Desktop/RTN/skel/R' + str(i))
        # os.makedirs(home + '/Desktop/RTN/brpts/R' + str(i))
        os.makedirs(home + '/Desktop/RTN/frames/R' + str(i))
        # os.rename(home + '/Desktop/Climp/brpts/A' + str(i), home + '/Desktop/Climp/brpts/C' + str(i))
        # pref = home + '/Desktop/Climp/brpts/A%s'

# dirmaker()
#
# exit()

def mover():
    dr2 = '/localhome/asa420/Desktop/RTN/brpts/'
    dr1 = '/localhome/asa420/Desktop/RTN/skel/'
    # files = glob.glob(dr + '*_skel_brpts.png')
    for i in range(1, 30):
        for j in range(100):
            fname_sk = dr1 + 'R%s_decon_t0%s_ch00_skel.png'%(f'{i}',f'{j:02d}')
            fname_br = dr2 + 'R%s_decon_t0%s_ch00_skel_brpts.png'%(f'{i}',f'{j:02d}')
            # fname = dr + 'C%s_decon_t0%s_ch00_skel.png'%(f'{i}',f'{j:02d}')
            shutil.move(fname_sk, dr1 + 'R%s'%(f'{i}'))
            shutil.move(fname_br, dr2 + 'R%s'%(f'{i}'))

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
