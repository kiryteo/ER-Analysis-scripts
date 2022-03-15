from plantcv import plantcv as pcv
import numpy as np
import glob
import pickle


def standardize_image(img):
    std_img = img - np.min(img) / (np.max(img) - np.min(img))
    return std_img


def rtog(img):
    return pcv.rgb2gray(rgb_img=img)

def thresh_image(img, threshold=12, max_value=255):
    return pcv.threshold.binary(gray_img=img, threshold=threshold, max_value=max_value)


def get_skel(img):
    return pcv.morphology.skeletonize(mask=img)


def get_brpts(img):
    return pcv.morphology.find_branch_pts(skel_img=img)


dire = glob.glob('/localhome/asa420/MIAL/data/translated_confocal_livecell/live_cell_enhanced_128/*')
#
# print(dire[0])
#
# exit()


def runner():
    for each in dire:
        files = glob.glob(each + '/*')
        dt = {}
        skel_list = []
        skel_avg = np.zeros((128, 128))
        junc_list = []
        junc_avg = np.zeros((128, 128))
        for file in files:
            dt[file] = []
            img, path, filename = pcv.readimage(file)
            img = rtog(img)
            img = standardize_image(img)
            thresh = thresh_image(img)

            skel = get_skel(thresh)
            skel_avg += skel
            skel_list.append(skel)

            brpts = get_brpts(skel)
            junc_avg += brpts
            junc_list.append(brpts)

            dt[file].append(skel)
            dt[file].append(brpts)

        mean_skel = skel_avg / 100
        mean_junc = junc_avg / 100

        maxskel = np.stack(skel_list, axis=2)
        maxskel = np.amax(maxskel, axis=2)

        maxjunc = np.stack(junc_list, axis=2)
        maxjunc = np.amax(maxjunc, axis=2)

        dirname = each.split('/')[-1]
        dt[dirname] = []
        dt[dirname].append(mean_skel)
        dt[dirname].append(maxskel)
        dt[dirname].append(mean_junc)
        dt[dirname].append(maxjunc)

        dtname = dirname + '.pickle'
        with open(dtname, 'wb') as hn:
            pickle.dump(dt, hn)


runner()
