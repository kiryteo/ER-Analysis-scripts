import os
import matplotlib.pyplot as plt
import cv2

home = os.path.expanduser('~')


def frame_creator(prefix, total_series, num_frames):
    prefix = home + '/Desktop/Control/'
    # for i in range(1, 32):
    # 	for j in range(100):
    for series in range(1, total_series):
        for frame in range(num_frames):

            std = prefix + 'std/img_%s_decon_t0%s_std.png' % (f'{series}', f'{frame:02d}')
            skel = prefix + 'skel/Ct%s/img_%s_decon_t0%s_skel.png' % (f'{series}', f'{series}', f'{frame:02d}')
            brpts = prefix + 'brpts/Ct%s/img_%s_decon_t0%s_skel_brpts.png' % (f'{series}', f'{series}', f'{frame:02d}')
            overlay = prefix + 'overlay/Ct%s/img_%s_decon_t0%s_skel_br_overlay.png' % (f'{series}', f'{series}', f'{frame:02d}')
            meanframe = prefix + 'Ct%s-mean.png' % (f'{series}')
            maxframe = prefix + 'Ct%s-max.png' % (f'{series}')

            fig = plt.figure(figsize=(12, 9))
            plt.title('Confocal-Control-Series%s-frame%s' % (f'{series}', f'{frame:02d}'), size=18)
            plt.axis('off')
            r, c = 2, 3

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
            plt.imshow(cv2.imread(overlay))
            plt.axis('off')
            plt.title('Skeleton + Junctions')

            fig.add_subplot(r, c, 5)
            plt.imshow(cv2.imread(meanframe))
            plt.axis('off')
            plt.title('Mean Projection')

            fig.add_subplot(r, c, 6)
            plt.imshow(cv2.imread(maxframe))
            plt.axis('off')
            plt.title('Max Projection')

            plt.savefig(prefix + 'frames/Ct%s/img_%s_frame%s.png' %(f'{series}',\
			 			f'{series}', f'{frame:02d}'), bbox_inches='tight',
                        pad_inches=0.1)
            plt.close()

# frame_creator()

# exit()


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
