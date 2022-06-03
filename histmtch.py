from skimage.exposure import match_histograms
import imageio
import shutil
import os

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



