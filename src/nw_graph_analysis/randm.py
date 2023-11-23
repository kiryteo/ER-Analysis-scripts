import random

# generate 5 set of 21 random numbers between 1 and 117
# print rest of the numbers from 1 to 117 that are not in the set

# for i in range(5):
#     k = random.sample(range(0, 117), 21)
#     print(k)
#     print(set(range(0, 117)) - set(k))


# get mean and std for following list

# l = [0.9677618, 0.96121943, 0.97106934, 0.9666719, 0.972508]

# l = [0.911, 0.932, 0.954, 0.9]
# l = [0.805, 0.8127, 0.8355, 0.796]
l = [0.685, 0.777, 0.759, 0.698]

import numpy as np
# print(np.mean(l))
# print(np.std(l))


split1 = [6, 99, 88, 67, 108, 24, 76, 85, 61, 102, 69, 15, 34, 37, 110, 54, 39, 62, 82, 109, 78]

print(sorted(split1))

# gt = [1.59000000e+02, 1.60000000e+02, 1.69000000e+02, 8.80000000e+01, 1.16000000e+02]

# an = [1.58000000e+02, 2.08000000e+02, 1.72000000e+02, 1.18000000e+02, 1.41000000e+02]

# for v1, v2 in zip(gt, an):
#     print(abs(v1 - v2) / abs(v1))

# print(np.nanmean([0.00629, 0.3, 0.018, 0.340, 0.21551]))



# print(abs(-0.00034021705848178867 - 0.15849684436316602)) #/ abs(-0.00034021705848178867))
# 466.8697746387403