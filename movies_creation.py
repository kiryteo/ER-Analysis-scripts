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




def graph_movie_creator():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    prefix = '/localhome/asa420/MIAL/data/confocal_movies/Climp/graph_overlay_endpts/'
    video = cv2.VideoWriter('Climp_Confocal_S29_graphs_endpts.mp4', fourcc, 1.5, (480, 480))
    for frame in range(100):
        img = cv2.imread(prefix + 'C29_decon_t0%s_ch01_graph_overlay.png'%f'{frame:02d}')
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

# graph_movie_creator()
# exit()

def graph_overlay_movie_creator():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    prefix = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/graphs/A4/'
    video = cv2.VideoWriter('ATL_Confocal_graph_overlay_series4.mp4', fourcc, 1.5, (369, 369))
    for frame in range(100):
        img = cv2.imread(prefix + 'A4_decon_t0%s_ch00_graph.png'%f'{frame:02d}')
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


# graph_overlay_movie_creator()
# exit()

def hist_movie_creator():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    prefix = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series005_decon_converted/hist/'
    video = cv2.VideoWriter('STED_RTN_Series5_hist.mp4', fourcc, 1.5, (584, 453))
    for frame in range(100):
        img = cv2.imread(prefix + 'Series005_decon_converted_t%s_hist.png'%f'{frame:02d}')
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


def conf_movies_new():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    prefix = '/localhome/asa420/MIAL/data/confocal_movies/Control/new_op_jul/frames/'
    for i in range(1, 32):
        video = cv2.VideoWriter('Conf_Control_Series%s.mp4'%f'{i}', fourcc, 1.5, (640, 350))
        for frame in range(100):
            img = cv2.imread(prefix + 'Ct%s/Ct%s_decon_t0%s_ch00_frame.png'%(f'{i}', f'{i}', f'{frame:02d}'))
            video.write(img)

        cv2.destroyAllWindows()
        video.release()

import imageio



def std_input():
    prefix = '/localhome/asa420/MIAL/data/confocal_movies/ATL/files/'
    for frame in range(100):
        img = imageio.imread(prefix + 'A1_decon_t0%s_ch00.tif' % f'{frame:02d}')
        img = (img - img.min()) / (img.max() - img.min())
        op = img * 255.
        cv2.imwrite(prefix + 'A1_decon_t0%s_ch00.png' % f'{frame:02d}', op)

# std_input()
# exit()


def file_movie_creator():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    prefix = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/new_process_frames/'
    video = cv2.VideoWriter('ATL_er_skel_frame_two.mp4', fourcc, 1.5, (330, 189))
    for frame in range(100):
        img = cv2.imread(prefix + 'A1_decon_t0%s_ch00_new_frame_two.png' % f'{frame:02d}')
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

# file_movie_creator()
# exit()


# def gmovie_creator(group, num_series):
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video = cv2.VideoWriter('%s_s%s_junction_types.mp4'%(f'{group}', f'{num_series}'), fourcc, 1.5, (369, 369))
#     for frame in range(100):
#         img = cv2.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junc_types_movies/%s_decon_t0%s_ch00.png'%(f'{group}', f'{group[0]}{num_series}', f'{frame:02d}'))
#         video.write(img)
#
#     cv2.destroyAllWindows()
#     video.release()

# for i in range(2, 27):
#     gmovie_creator('ATL', i)


def gmovie_creator(group, num_series):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('%s_s%s_junction_types.mp4'%(f'{group}', f'{num_series}'), fourcc, 1.5, (369, 369))
    if group == 'Control':
        pref = 'Ct'
    else:
        pref = group[0]
    for frame in range(100):
        img = cv2.imread('/localhome/asa420/MIAL/data/confocal_movies/%s/new_op_jul/junc_types_movies/%s_decon_t0%s_ch00.png'%(f'{group}', f'{pref}{num_series}', f'{frame:02d}'))
        video.write(img)

    cv2.destroyAllWindows()
    video.release()