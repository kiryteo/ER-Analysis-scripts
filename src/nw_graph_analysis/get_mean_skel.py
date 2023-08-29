import imageio
import matplotlib.pyplot as plt
import numpy as np


skel_path = '/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/skel/'


def get_mean_img():
    for ser_num in range(1, 32):
        img_list = []
        for frame in range(100):            
            skel = imageio.imread(f'{skel_path}/R{ser_num}/R{ser_num}_decon_t0{frame:02d}_ch00_skel.png')
            img_list.append(skel)
            img_list = np.array(img_list)
            mean_img = np.mean(img_list, axis=0)
            imageio.imsave(f'/localhome/asa420/MIAL/data/confocal_movies/RTN/new_op_jul/R{ser_num}_mean_skel.png', mean_img)

get_mean_img()
