import glob
import os

def file_rename():
    dir = '/localhome/asa420/Documents/ER-Full-data/FixedCell_for_Ashwin/FixedCell_for_Ashwin/numpys/rtn/'
    files = glob.glob(dir + '*')
    for each in files:
        name = each.split('/')[-1]
        ncomps = name.split('.')
        newname = dir + ncomps[0] + '_' + ncomps[1] + '.' + ncomps[2]
        # print(newname)
        # break
        os.rename(each, newname)
