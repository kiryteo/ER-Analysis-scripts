import os
import matplotlib.pyplot as plt
import cv2

std_pref = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series005_decon_converted/std/'
hist_pref = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series005_decon_converted/hist/'

for i in range(100):
    fig = plt.figure(figsize=(12, 6))
    #plt.title('Confocal-RTN-Series%s-frame%s' % (f'{series}', f'{frame:02d}'), size=18)
    plt.axis('off')
    r, c = 1, 2

    # std = '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series005_decon_converted/std/Series005_decon_converted_t00_ch00_std.png'
    # hist= '/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series005_decon_converted/hist/Series005_decon_converted_t00_hist.png'

    fig.add_subplot(r, c, 1)
    plt.imshow(cv2.imread(std))
    plt.axis('off')
    plt.title('Input')

    fig.add_subplot(r, c, 2)
    plt.imshow(cv2.imread(hist))
    plt.axis('off')
    plt.title('Histogram')

    #plt.show()
    plt.savefig('/localhome/asa420/MIAL/data/live-cell-movies/COSKDELRTN/Decon/Series005_decon_converted/img_hist' + , bbox_inches='tight')