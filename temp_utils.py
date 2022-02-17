import glob
import os
import shutil

home = os.path.expanduser('~')

def file_rename():
    dir = home + '/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/rtn/'
    files = glob.glob(dir + '*')
    for each in files:
        name = each.split('/')[-1]
        ncomps = name.split('.')
        newname = dir + ncomps[0] + '_' + ncomps[1] + '.' + ncomps[2]
        # print(newname)
        # break
        os.rename(each, newname)

def file_move():
    dir = home + '/MIAL/live-cell-movies/COSKDEL/COSKDEL/Decon/'
    subdirs = glob.glob(dir + '*')
    # print(subdirs)
    for each in subdirs:
        newdir = each + '_skel_frames/'
        os.makedirs(newdir)
        filedir = each + '/files/*.tif_mean_skel.tif'
        files = glob.glob(filedir)
        for tfile in files:
            shutil.move(tfile, newdir)
        # print(files)

file_move()
#
