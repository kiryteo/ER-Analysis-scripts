import imageio
import numpy as np
import copy
import matplotlib.pyplot as plt
import cv2

import os

home = os.path.expanduser('~')

for i in range(3, 11):
    path = home + '/MIAL/aggregation-with-median/'
    avg_junc = imageio.imread(path + 'Climp/Climp-brpts{0}-avg.png'.format(str(i)))
    vals = np.unique(avg_junc)

    op = copy.deepcopy(avg_junc)

    first_th = np.where(avg_junc==vals[1])
    op[first_th] = 0

    ones = np.where(op!=0)
    op[ones] = 255

    cv2.imwrite(path + 'Climp/Climp-brpts{0}-avg-th-norm.png'.format(str(i)), op)

# plt.imshow(avg_junc)
# plt.imshow(op)
# #plt.imshow(op, ax=axes[1])
# plt.show()
