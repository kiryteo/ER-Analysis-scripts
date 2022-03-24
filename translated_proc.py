from plantcv import plantcv as pcv
import numpy as np
import glob
import pickle
import skimage
import ks_multithresh
import skimage.io as io

def standardize_image(img):
    std_img = img - np.min(img) / (np.max(img) - np.min(img))
    return std_img

def skimage_erode_img(img):
    return skimage.morphology.erosion(img)

def rtog(img):
    return pcv.rgb2gray(rgb_img=img)

def thresh_image(img, threshold=12, max_value=255):
    return pcv.threshold.binary(gray_img=img, threshold=threshold, max_value=max_value)


def get_skel(img):
    return pcv.morphology.skeletonize(mask=img)


def get_brpts(img):
    return pcv.morphology.find_branch_pts(skel_img=img)


# dire = glob.glob('/localhome/asa420/MIAL/data/translated_confocal_livecell/live_cell_enhanced_128/*')

dire = glob.glob('/localhome/asa420/MIAL/data/live-cell-movies/dirs/*')

#
# print(dire[0])
#
# exit()

# gt, path, fname = pcv.readimage('/localhome/asa420/Downloads/AnalyzER_Simon_Fraser_University/gt1.tif')
# img, path, fname = pcv.readimage('/localhome/asa420/Downloads/AnalyzER_Simon_Fraser_University/img1.jpg')
# grimg = rtog(img)
# stdimg = standardize_image(grimg)
# thr = thresh_image(stdimg)
# skel = get_skel(thr)
# gt = pcv.invert(gt)
#
# pcv.print_image(img=grimg, filename='/localhome/asa420/Downloads/AnalyzER_Simon_Fraser_University/img1_gr.png')

# pcv.print_image(img=gt, filename='/localhome/asa420/Downloads/AnalyzER_Simon_Fraser_University/gt1_op.tif')
# pcv.print_image(img=skel, filename='/localhome/asa420/Downloads/AnalyzER_Simon_Fraser_University/img1_op.jpg')

# exit()

# def get_erenh():
#     for each in glob.glob('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/Decon/*'):
#         if each.split('/')[-1].split('_')[-1] == 'converted':
#             files = glob.glob(each + '/erode/*')


def runner():
    for each in dire:
        group = each.split('/')[-1][0]
        if group == 'C':
            erpath = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELCLIMP/Decon/'
            series = int(each.split('/')[-1][1:])
        elif group == 'i':
            erpath = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/Decon/'
            series = int(each.split('/')[-1][3:])
        else:
            erpath = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/'
            series = int(each.split('/')[-1][1:])

        # series = int(each[1:])
        if series < 10:
            dname = erpath + 'Series00' + str(series) + '_decon_converted/erenh/'
        else:
            dname = erpath + 'Series0' + str(series) + '_decon_converted/erenh/'
        files = glob.glob(each + '/*')
        dt = {}
        # skel_list = []
        # skel_avg = np.zeros((128, 128))
        # er_avg = np.zeros((128, 128))

        erenh_avg = np.zeros((128, 128))
        junc_list = []
        junc_avg = np.zeros((128, 128))

        for file in files:
            # print(file)
            # exit()
            dt[file] = []
            img, path, filename = pcv.readimage(file)
            # img = rtog(img)
            # img = standardize_image(img)

            beads = ks_multithresh.test_thresholds(img)

            # erimg = skimage_erode_img(img)

            # er_avg += erimg

            # print(file)
            erenh, pt, fn = pcv.readimage(dname + file.split('/')[-1].split('.')[0] + '_erode_enhance.png')
            # print(erenh_name)
            # erenh = io.imread(erenh_name)

            skel = get_skel(erenh)

            erenh_avg += erenh

            # thresh = thresh_image(erimg)

            # skel = get_skel(thresh)
            # skel_avg += skel
            # skel_list.append(skel)
            #
            brpts = get_brpts(skel)
            junc_avg += brpts
            junc_list.append(brpts)
            #
            dt[file].append(img)
            dt[file].append(skel)
            dt[file].append(brpts)
            dt[file].append(beads)

        # nm = each.split('/')[-1]
        # mean_er = er_avg / 100
        # pcv.print_image(img=mean_er, filename=nm+'_er.png')
        # mean_er_thresh = thresh_image(mean_er)
        # mean_er_skel = get_skel(mean_er_thresh)

        erenh_avg = erenh_avg / 100
        mean_er_skel = get_skel(erenh_avg)

        # pcv.print_image(img=mean_er_skel, filename=nm+'_erskel.png')
        # mean_skel = skel_avg / 100
        # # pcv.print_image(img=mean_skel, filename=nm+'.png')
        mean_junc = junc_avg / 100
        #
        # maxskel = np.stack(skel_list, axis=2)
        # maxskel = np.amax(maxskel, axis=2)
        # pcv.print_image(img=maxskel, filename=nm+'_max.png')

        # maxjunc = np.stack(junc_list, axis=2)
        # maxjunc = np.amax(maxjunc, axis=2)
        #
        dirname = each.split('/')[-1]
        dt[dirname] = []
        dt[dirname].append(mean_er_skel)
        # dt[dirname].append(maxskel)
        dt[dirname].append(mean_junc)
        # dt[dirname].append(maxjunc)
        #
        dtname = dirname + '.pickle'
        with open(dtname, 'wb') as hn:
            pickle.dump(dt, hn)

runner()

# file = '/localhome/asa420/MIAL/data/live-cell-movies/dirs/C1/Series001_decon_converted_t00_ch00_std.png'
# img, path, filename = pcv.readimage(file)
# img = standardize_image(img)
# beads = ks_multithresh.test_thresholds(img)
# import matplotlib.pyplot as plt
# plt.imshow(beads)
# plt.show()


