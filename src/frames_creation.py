"""
Frames to captures changes over time,
later converted into a movie
"""


import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import imageio

home = os.path.expanduser('~')


input = '/localhome/asa420/MIAL/data/confocal_movies/RTN/files/'
preproc = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/preproc/'
skel = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/skel/'

meansk = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/'

svpath = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/frames/'

r, c = 1, 4

for i in range(1, 32):
    os.makedirs(f'{meansk}frames/R{i}')
    for j in range(100):
        img = f'{input}R{i}_decon_t0{j:02d}_ch00.tif'
        proc = f'{preproc}R{i}/R{i}_decon_t0{j:02d}_ch00_proc.png'
        sk = f'{skel}R{i}/R{i}_decon_t0{j:02d}_ch00_skel.png'
        meanfr = f'{meansk}R{i}_mean.png'

        img = imageio.imread(img)
        std_img = ((img) / (img.max() - img.min())) * 255

        fig = plt.figure(figsize=(8, 4))
        plt.title(f'R{i}_decon_t0{j:02d}_ch00_frame')
        plt.axis('off')
        fig.add_subplot(r, c, 1)
        plt.imshow(std_img)
        plt.axis('off')
        plt.title('Input')

        fig.add_subplot(r, c, 2)
        plt.imshow(cv2.imread(proc))
        plt.axis('off')
        plt.title('Preprocessing')

        fig.add_subplot(r, c, 3)
        plt.imshow(cv2.imread(sk))
        plt.axis('off')
        plt.title('Network')

        fig.add_subplot(r, c, 4)
        plt.imshow(cv2.imread(meanfr))
        plt.axis('off')
        plt.title('Mean Skel')

        plt.savefig(f'{svpath}R{i}/R{i}_decon_t0{j:02d}_ch00_frame.png', bbox_inches='tight')

        # pad_inches=0.1)
        plt.close()

exit()

er = f'{pre}samples/Series{ser:03d}_decon_ch02-{num}.tif'
lab_tub = f'{pre}labeled_tub/Series{ser:03d}_decon_ch02-{num}_lab.tif'
dil_tub = f'{pre}labeled_dil_tubules/Series{ser:03d}_decon_ch02-{num}_dil_tub.tif'

overlay = f'{pre}skel/Series{ser:03d}_decon_ch02-{num}_fuse.png'
skelbr = f'{pre}Series{ser:03d}_decon_ch02-{num}_skel_br.png'


fig = plt.figure(figsize=(15, 4))
plt.title(f'{grp}_S{ser:01d}_{num:01d}', size=18)
plt.axis('off')
r, c = 1, 5

fig.add_subplot(r, c, 1)
plt.imshow(cv2.imread(er))
plt.axis('off')
plt.title('Input')

fig.add_subplot(r, c, 2)
plt.imshow(cv2.imread(overlay))
plt.axis('off')
plt.title('Input+Skel')

fig.add_subplot(r, c, 3)
plt.imshow(cv2.imread(skelbr))
plt.axis('off')
plt.title('Skel+junc')

fig.add_subplot(r, c, 4)
plt.imshow(cv2.imread(lab_tub))
plt.axis('off')
plt.title('Tubule skeleton')

fig.add_subplot(r, c, 5)
plt.imshow(cv2.imread(dil_tub))
plt.axis('off')
plt.title('Dilated tubules')

plt.savefig(f'{grp}_S{ser:01d}_{num:01d}.png', bbox_inches='tight')
# pad_inches=0.1)
plt.close()

exit()

std = '/localhome/asa420/MIAL/data/selective_analysis/rtn/rtn_s10/std/Series010_decon_converted_t06_ch00_std.png'
skel = '/localhome/asa420/MIAL/data/selective_analysis/rtn/rtn_s10/skel/Series010_decon_converted_t62_ch00_std_enhance_skel.png'
brpts = '/localhome/asa420/ER-Analysis-scripts/rtn_t6_brpts.png'
dil_brpts = '/localhome/asa420/ER-Analysis-scripts/rtn_t6_dilbr.png'
tubules = '/localhome/asa420/ER-Analysis-scripts/rtn_t6_tub.png'
# labeled_tubules = '/localhome/asa420/ER-Analysis-scripts/lab_tub.png'
# overlay = '/localhome/asa420/ER-Analysis-scripts/er1-ov.png'


fig = plt.figure(figsize=(15, 3))
plt.title('Ctrl_S7_t00', size=18)
plt.axis('off')
r, c = 1, 6

fig.add_subplot(r, c, 1)
plt.imshow(cv2.imread(std))
plt.axis('off')
plt.title('Input')

fig.add_subplot(r, c, 2)
plt.imshow(cv2.imread(skel))
plt.axis('off')
plt.title('Skeleton')

fig.add_subplot(r, c, 3)
plt.imshow(cv2.imread(brpts))
plt.axis('off')
plt.title('Junctions')

fig.add_subplot(r, c, 4)
plt.imshow(cv2.imread(dil_brpts))
plt.axis('off')
plt.title('dilated junctions')

fig.add_subplot(r, c, 5)
plt.imshow(cv2.imread(tubules))
plt.axis('off')
plt.title('Tubules')

fig.add_subplot(r, c, 6)
plt.imshow(cv2.imread(overlay))
plt.axis('off')
plt.title('ER + tubules + junctions')


plt.savefig('Ctrl_s7_t00.png', bbox_inches='tight')
plt.close()

exit()

def frame_creator(total_series, num_frames):
    prefix = f'{home}/Desktop/RTN/'
    r, c = 2, 5

    # for i in range(1, 32):
    # 	for j in range(100):
    for series in range(1, total_series+1):
        for frame in range(num_frames):

            std = f'{prefix}std/R{series}_decon_t0{frame:02d}_ch00_std.png'
            skel = f'{prefix}skel/R{series}/R{series}_decon_t0{frame:02d}_ch00_skel.png'
            ipskel = f'{prefix}std/R{series}_decon_t0{frame:02d}_ch00_std_ip_skel_overlay.png'

            overlay = f'{prefix}overlay/R{series}/R{series}_decon_t0{frame:02d}_ch00_skel_br_overlay.png'

            # mcherry = prefix + 'mcherry/R%s/R%s_decon_t0%s_ch01_std_adj.png' % (f'{series}', f'{series}', f'{frame:02d}')
            brpts = f'{prefix}brpts/R{series}/R{series}_decon_t0{frame:02d}_ch00_skel_brpts.png'

            # mch_skel = prefix + 'mcherry/R%s/R%s_decon_t0%s_ch01_std_adj_dyn_skel_mcherry_overlay.png' % (f'{series}', f'{series}', f'{frame:02d}')
            # mch_junc = prefix + 'mch_junc_overlay/R%s/R%s_decon_t0%s_ch00_mch_junc_overlay.png' % (f'{series}', f'{series}', f'{frame:02d}')
            meanframe = f'{prefix}R{series}-mean.png'
            maxframe = f'{prefix}R{series}-max.png'
            # mean_br = prefix + 'Ct%s-mean-brpts.png' % f'{series}'
            # max_br = prefix + 'Ct%s-max-brpts.png' % f'{series}'
            # NW_mean_junc_max = prefix + 'Ct%s-NW_mean_junc_max.png' % f'{series}'
            # mch_sk_overlay = prefix + 'mcherry/R%s/R%s_decon_t0%s_ch01_std_adj_skel_mcherry_overlay.png' % (f'{series}', f'{series}', f'{frame:02d}')

            fig = plt.figure(figsize=(18, 9))
            plt.title(f'Confocal-RTN-Series{series}-frame{frame:02d}', size=18)
            plt.axis('off')
            fig.add_subplot(r, c, 1)
            plt.imshow(cv2.imread(std))
            plt.axis('off')
            plt.title('Input')

            # fig.add_subplot(r, c, 2)
            # plt.imshow(cv2.imread(mcherry))
            # plt.axis('off')
            # plt.title('MCherry')

            fig.add_subplot(r, c, 2)
            plt.imshow(cv2.imread(skel))
            plt.axis('off')
            plt.title('Skeleton')

            fig.add_subplot(r, c, 3)
            plt.imshow(cv2.imread(ipskel))
            plt.axis('off')
            plt.title('input + skel')

            fig.add_subplot(r, c, 4)
            plt.imshow(cv2.imread(overlay))
            plt.axis('off')
            plt.title('skel + junctions')

            fig.add_subplot(r, c, 5)
            plt.imshow(cv2.imread(mcherry))
            plt.axis('off')
            plt.title('MCherry')

            fig.add_subplot(r, c, 6)
            plt.imshow(cv2.imread(brpts))
            plt.axis('off')
            plt.title('Junctions')

            fig.add_subplot(r, c, 7)
            plt.imshow(cv2.imread(mch_skel))
            plt.axis('off')
            plt.title('MCherry + Skel')

            fig.add_subplot(r, c, 8)
            plt.imshow(cv2.imread(mch_junc))
            plt.axis('off')
            plt.title('MCherry + Junc')

            fig.add_subplot(r, c, 9)
            plt.imshow(cv2.imread(meanframe))
            plt.axis('off')
            plt.title('mean skel')

            fig.add_subplot(r, c, 10)
            plt.imshow(cv2.imread(maxframe))
            plt.axis('off')
            plt.title('max skel')

            plt.savefig(f'{prefix}frames_10/R{series}/R{series}_frame{frame:02d}.png', bbox_inches='tight', pad_inches=0.1)

            plt.close()

frame_creator(29, 100)

exit()


# img = [std, enh, overlay, avgframe, mxframe]
# images = [Image.open(x) for x in img]
# widths, heights = zip(*(i.size for i in images))
#
# total_width = sum(widths)
# max_height = max(heights)
#
# new_im = Image.new('RGB', (total_width, max_height))
#
# x_offset = 0
# for im in images:
#   new_im.paste(im, (x_offset,0))
#   x_offset += im.size[0]
#
# new_im.save('test2.png')


# list_im = [raw, enh, skel, brpts]
# imgs    = [ PIL.Image.open(i) for i in list_im ]
# # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
# # min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
# min_shape = imgs[0].size
# # imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
# imgs_comb = np.hstack( (np.asarray(i) for i in imgs ) )

# # save that beautiful picture
# imgs_comb = PIL.Image.fromarray( imgs_comb)

# plt.imshow(imgs_comb)
# plt.show()
# imgs_comb.save( 'Trifecta.jpg' )


# fig, ax = plt.subplots(1, 4)

# ax[0].plot(raw)
# ax[1].plot(enh)
# ax[2].plot(skel)
# ax[3].plot(brpts)

# plt.show()
