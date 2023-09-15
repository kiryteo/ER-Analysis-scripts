import glob
import imageio
from skimage.transform import resize

# pref = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/rtn_junc_viz/'


# pref = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junc_types_movies/'

# pref = '/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/junction_crops/S1_j15_29_74/'


# with imageio.get_writer('A1_junc_iso_fuz.gif', mode='I', duration=0.5) as writer:
# with imageio.get_writer('A1_iso_j15_egfp.gif', mode='I', duration=0.5) as writer:
#     # with imageio.get_writer('A1_junc.mp4', fps=2) as writer:
#     for frame in range(100):
#         filename = f'{pref}A1_decon_t0{frame:02d}_ch00.png'
#         image = imageio.imread(filename)
#         image = (image - image.min()) / (image.max() - image.min())
#         # image = resize(image, (64, 68))
#         writer.append_data(image)

import matplotlib.pyplot as plt

def create_frames():

    static = imageio.imread('/localhome/asa420/MIAL/data/sted-data/RTN4-sted.drawio.png')

    # draw a line for the given start and end points on static

    # start = (282, 282)
    # end = (589, 589)
    # plt.plot([start[0], end[0]], [start[1], end[1]], color='red', linewidth=2)
    # # static[start[0]:end[0], start[1]:end[1]] = (255, 0, 0)

    # start = (282, 0)
    # end = (589, 0)
    # # static[start[0]:end[0], start[1]:end[1]] = (255, 0, 0)
    # plt.plot([start[0], end[0]], [start[1], end[1]], color='red', linewidth=2)

    for i in range(100):
        crop = imageio.imread(f'/localhome/asa420/MIAL/data/sted-data/RTN/crops/R4_junc_repr_t{i:02d}.png')

        # start = (282, 282)
        # end = (589, 589)
        # plt.plot(, color='red', linewidth=2)

        plt.subplot(1, 2, 1)
        plt.imshow(static, interpolation='None')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(crop, interpolation='None')
        plt.axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(f'/localhome/asa420/MIAL/data/sted-data/RTN/frames/R4_junc_repr_t{i:02d}_draw.png', bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.show()
        plt.close()


# create_frames()
# exit()

def runner(writer, group, ser_num):#, group, ser_num, junc_id, channel):
    # pref = f'/localhome/asa420/MIAL/data/confocal_movies/{group}/new_op_jul/{group[0]}{ser_num}_j{junc_id}_t/'
    # ch = 0 if channel=='egfp' else 1
    # for frame in range(100):
    #     filename = f'{pref}{group[0]}{ser_num}_j{junc_id}_t0{frame:02d}_ch0{ch}.tif'
    #     image = imageio.imread(filename)
    #     image = (image - image.min()) / (image.max() - image.min())
    #     # image = resize(image, (64, 68))
    #     writer.append_data(image)
    pref = f'/localhome/asa420/MIAL/data/sted-data/{group}/frames/'

    for frame in range(100):
        filename = f'{pref}R{ser_num}_junc_repr_t{frame:02d}_draw.png'
        image = imageio.imread(filename)
        writer.append_data(image)


# def create_sequence(type, group, ser_num, junc_id, channel):
#     if type == 'gif':
#         with imageio.get_writer('A1_iso_j15_egfp.gif', mode='I', duration=0.5) as writer:
#             runner(writer, group, ser_num, junc_id, channel)
#     else:
#         with imageio.get_writer(f'{group[0]}{ser_num}_j{junc_id}_{channel}.mp4', fps=2) as writer:
#             runner(writer, group, ser_num, junc_id, channel)

def create_sequence(ser_num):
    with imageio.get_writer(f'R{ser_num}_junc_repr_draw.gif', mode='I', duration=0.5) as writer:
        runner(writer, 'RTN', ser_num)

create_sequence(4)
# create_sequence('mp4', 'RTN', 13, 82, 'egfp')
# create_sequence('mp4', 'RTN', 13, 82, 'mch')
