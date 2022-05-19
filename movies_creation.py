import os
import numpy as np
import cv2


home = os.path.expanduser('~')


# import skvideo.io

def skvideo_creator(num_frames, height, width, channels):
    # out_video = np.empty([100, 134, 795, 3], dtype=np.uint8)
    out_video = np.empty([num_frames, height, width, channels], dtype=np.uint8)

    for i in range(num_frames):
        img = cv2.imread(home + '/MIAL/live-cell-movies/COSKDEL/COSKDEL/Decon/Series003_decon_converted/frames/Series003-frame%s.png'%(
                f'{i:02d}'))
        out_video[i] = img


# skvideo.io.vwrite('Series003.mp4', out_video)
# writer = skvideo.io.FFmpegWriter("Series003.mp4")

# /localhome/asa420/Desktop/ATL/std/A1_decon_t000_ch00_std_ip_mcherry_overlay.png


def series_movie_creator(series_num, num_frames, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter('STED-RTN-Series%s.mp4'%(f'{num:03d}'), fourcc, 1,\
	# 			 (1182, 740)) # 8 images in the frame
    # video = cv2.VideoWriter('Confocal-RTN-Series%s.mp4'%(f'{num}'), fourcc, 1,\
	#			 (950, 740))
    video = cv2.VideoWriter('Confocal-RTN-Series%s.mp4' % f'{series_num}', \
                            fourcc, 1, (width, height))

    for frame in range(num_frames):
        img = cv2.imread(
            home + '/Desktop/RTN/frames_10/R%s/R%s_frame%s.png' %(f'{series_num}', f'{series_num}', f'{frame:02d}'))
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

def group_movies_creator(total_series, num_frames, width, height):
    for series in range(1, total_series+1):
        series_movie_creator(series, num_frames, width, height)

# group_movies_creator(23, 100, 1415, 740)
# group_movies_creator(29, 100, 1926, 642)


def hist_movie_creator():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    prefix = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series005_decon_converted/hist/'
    video = cv2.VideoWriter('STED_RTN_Series5_hist.mp4', fourcc, 1.5, (584, 453))
    for frame in range(100):
        img = cv2.imread(prefix + 'Series005_decon_converted_t%s_hist.png'%f'{frame:02d}')
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

hist_movie_creator()