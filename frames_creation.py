import matplotlib.pyplot as plt
import imageio
import sys
from PIL import Image
# import PIL
import numpy as np

import cv2
import os

home = os.path.expanduser('~')
# import skvideo.io

def skvideo_creator():
	out_video = np.empty([100, 134, 795, 3], dtype=np.uint8)

	for i in range(100):
		img = cv2.imread('/home/ashwin/MIAL/live-cell-movies/COSKDEL/COSKDEL/Decon/Series003_decon_converted/frames/Series003-frame%s.png'%(f'{i:02d}'))
		# print(img.shape)
		# break
		out_video[i] = img

	skvideo.io.vwrite('Series003.mp4', out_video)


	writer = skvideo.io.FFmpegWriter("Series003.mp4")


def cv2_creator(num):
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	# video = cv2.VideoWriter('STED-RTN-Series%s.mp4'%(f'{num:03d}'), fourcc, 1, (1182, 740)) # 8 images in the frame
	# video = cv2.VideoWriter('RTN-Series%s.mp4'%(f'{num:03d}'), fourcc, 1, (1926, 642))
	video = cv2.VideoWriter('Climp-MP-Series%s.mp4'%(f'{num:03d}'), fourcc, 1, (1930, 386))

	for j in range(100):
		img = cv2.imread('/localhome/asa420/MIAL/live-cell-movies/COSKDELCLIMP/COSKDELCLIMP/Decon/Series%s_decon_converted_MeanProjection_frames/Series%s_decon_converted_t%s_ch00.tif_mean_skel.tif'%(f'{num:03d}',f'{num:03d}',f'{j:02d}'))
		video.write(img)

	cv2.destroyAllWindows()
	video.release()

# for i in range(3, 11):
# 	cv2_creator(i)

# exit()

# /home/ashwin/MIAL/aggregation-with-median/Control/avg/overlay/ControlSeries2-overlay_skel_brpts.png

def frame_creator():
	prefix = '/localhome/asa420/Desktop/ATL/'
	for i in range(1, 27):
		for j in range(100):
			std = prefix + 'std/A%s_decon_t0%s_ch00_std.png'%(f'{i}',f'{j:02d}')
			# enh = prefix + 'Series%s_decon_converted/enh/Series%s_decon_converted_t%s_ch00_std_enhance.png'%(f'{j:03d}',f'{j:03d}',f'{i:02d}')
			skel = prefix + 'skel/A%s_decon_t0%s_ch00_skel.png'%(f'{i}',f'{j:02d}')
			brpts = prefix + 'brpts/A%s_decon_t0%s_ch00_skel_brpts.png'%(f'{i}',f'{j:02d}')
			overlay = prefix + 'overlay/A%s_decon_t0%s_ch00_skel_br_overlay.png'%(f'{i}',f'{j:02d}')
			# avgframe = '/home/ashwin/MIAL/aggregation-with-median/Control/avg/series%s-avg.png'%(str(j))
			# mxframe = '/home/ashwin/MIAL/aggregation-with-median/Control/max/series%s-max.png'%(str(j))
			# medframe = '/home/ashwin/MIAL/aggregation-with-median/Control/median/series%s-median.png'%(str(j))


			fig = plt.figure(figsize=(12,9))
			plt.title('Confocal-ATL-Series%s-frame%s'%(f'{i}',f'{j:02d}'), size=18)
			plt.axis('off')
			r, c = 1, 4

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

			plt.savefig(prefix + 'frames/A%s_frame%s.png'%(f'{i}',f'{j:02d}'), bbox_inches='tight', pad_inches=0.1)
			plt.close()

frame_creator()

exit()

def frame_create():
	for i in range(2, 17):
		file = '/home/ashwin/MIAL/aggregation-with-median/Control/median/files/ClimpSeries%s-median.png'%f'{i}'
		skel = '/home/ashwin/MIAL/aggregation-with-median/Control/median/skel/ClimpSeries%s-median_skel.png'%f'{i}'
		brpts = '/home/ashwin/MIAL/aggregation-with-median/Control/median/brpts/ClimpSeries%s-median_brpts.png'%f'{i}'
		overlay = '/home/ashwin/MIAL/aggregation-with-median/Control/median/overlay/ClimpSeries%s-overlay_skel_brpts.png'%f'{i}'

		fig = plt.figure(figsize=(12,3.5))
		plt.title('STED-Control-Series%s'%i, size=18)
		plt.axis('off')
		r, c = 1, 4

		fig.add_subplot(r, c, 1)
		plt.imshow(cv2.imread(file))
		plt.axis('off')
		plt.title('median')

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

		plt.savefig('/home/ashwin/MIAL/aggregation-with-median/Control/median/frames/Series%s-frame.png'%i, bbox_inches='tight', pad_inches=0.1)
		plt.close()

# frame_create()
#
# exit()


def frame_creator():
	prefix = '/home/ashwin/MIAL/live-cell-movies/COSKDEL/COSKDEL/Decon/'
	for j in range(16, 17):
		for i in range(100):
			std = prefix + 'Series%s_decon_converted/std/Series%s_decon_converted_t%s_ch00_std.png'%(f'{j:03d}',f'{j:03d}',f'{i:02d}')
			enh = prefix + 'Series%s_decon_converted/enh/Series%s_decon_converted_t%s_ch00_std_enhance.png'%(f'{j:03d}',f'{j:03d}',f'{i:02d}')
			skel = prefix + 'Series%s_decon_converted/skel/Series%s_decon_converted_t%s_ch00_std_enhance_skel.png'%(f'{j:03d}',f'{j:03d}',f'{i:02d}')
			brpts = prefix + 'Series%s_decon_converted/brpts/Series%s_decon_converted_t%s_ch00_std_enhance_brpts.png'%(f'{j:03d}',f'{j:03d}',f'{i:02d}')
			overlay = prefix + 'Series%s_decon_converted/overlay/Series%s_decon_converted_t%s_ch00_skel_brpts.png'%(f'{j:03d}',f'{j:03d}',f'{i:02d}')
			avgframe = '/home/ashwin/MIAL/aggregation-with-median/Control/avg/series%s-avg.png'%(str(j))
			mxframe = '/home/ashwin/MIAL/aggregation-with-median/Control/max/series%s-max.png'%(str(j))
			medframe = '/home/ashwin/MIAL/aggregation-with-median/Control/median/series%s-median.png'%(str(j))


			fig = plt.figure(figsize=(15,9))
			plt.title('STED-Control-Series%s-frame%s'%(f'{j:03d}',f'{i:02d}'), size=18)
			plt.axis('off')
			r, c = 2, 4

			fig.add_subplot(r, c, 1)
			plt.imshow(cv2.imread(std))
			plt.axis('off')
			plt.title('STED')

			fig.add_subplot(r, c, 2)
			plt.imshow(cv2.imread(enh))
			plt.axis('off')
			plt.title('Tube-enhancement')

			fig.add_subplot(r, c, 3)
			plt.imshow(cv2.imread(skel))
			plt.axis('off')
			plt.title('Skeleton')

			fig.add_subplot(r, c, 4)
			plt.imshow(cv2.imread(brpts))
			plt.axis('off')
			plt.title('Junctions')

			fig.add_subplot(r, c, 5)
			plt.imshow(cv2.imread(overlay))
			plt.axis('off')
			plt.title('Skeleton + Junctions')

			fig.add_subplot(r, c, 6)
			plt.imshow(cv2.imread(avgframe))
			plt.axis('off')
			plt.title('Mean across time')

			fig.add_subplot(r, c, 7)
			plt.imshow(cv2.imread(medframe))
			plt.axis('off')
			plt.title('Median across time')

			fig.add_subplot(r, c, 8)
			plt.imshow(cv2.imread(mxframe))
			plt.axis('off')
			plt.title('Max across time')

			plt.savefig('/home/ashwin/MIAL/live-cell-movies/COSKDELRTN/COSKDELRTN/Decon/Series%s_decon_converted/frames/Series%s-frame%s.png'%(f'{j:03d}',f'{j:03d}',f'{i:02d}'), bbox_inches='tight', pad_inches=0.1)
			plt.close()

# frame_creator()
#
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
