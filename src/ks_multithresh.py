import numpy as np
from itertools import combinations
from skimage.util.shape import view_as_windows

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy.misc import face
import argparse

from os.path import basename

# code adapted from https://github.com/manuelaguadomtz/pythreshold

def singh_threshold(img, w_size=15, k=0.85):    # 0.35
    """ Runs the Singh thresholding algorithm

    Reference:
    Singh, O. I., Sinam, T., James, O., & Singh, T. R. (2012). Local contrast
    and mean based thresholding technique in image binarization. International
    Journal of Computer Applications, 51, 5-10.

    Modifications: Using integral images to compute local mean
        and standard deviation

    @param img: The input image
    @type img: ndarray
    @param w_size: The size of the local window to compute
        each pixel threshold. Should be and odd value
    @type w_size: int
    @param k: Controls the value of the local threshold. It lies in the
        interval [0, 1]
    @type k: float

    @return: The estimated local threshold for each pixel
    @rtype: ndarray
    """
    img = img.astype(np.float64) / 255

    # Obtaining rows and cols
    rows, cols = img.shape
    i_rows, i_cols = rows + 1, cols + 1

    # Computing integral images
    # Leaving first row and column in zero for convenience
    integ = np.zeros((i_rows, i_cols), np.float64)

    integ[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.float64), axis=0), axis=1)

    # Defining grid
    x, y = np.meshgrid(np.arange(1, i_cols), np.arange(1, i_rows))

    # Obtaining local coordinates
    hw_size = w_size // 2
    x1 = (x - hw_size).clip(1, cols)
    x2 = (x + hw_size).clip(1, cols)
    y1 = (y - hw_size).clip(1, rows)
    y2 = (y + hw_size).clip(1, rows)

    # Obtaining local areas size
    l_size = (y2 - y1 + 1) * (x2 - x1 + 1)

    # Computing sums
    sums = (integ[y2, x2] - integ[y2, x1 - 1] -
            integ[y1 - 1, x2] + integ[y1 - 1, x1 - 1])

    # Computing local means
    means = sums / l_size

    # Obtaining windows
    padded_img = np.ones((rows + w_size - 1, cols + w_size - 1)) * np.nan
    padded_img[hw_size: -hw_size, hw_size: -hw_size] = img

    winds = view_as_windows(padded_img, (w_size, w_size))

    # Obtaining maximums and minimums
    mins = np.nanmin(winds, axis=(2, 3))
    maxs = np.nanmax(winds, axis=(2, 3))

    return k * (means + (maxs - mins) * (1 - img)) * 255


def kapur_threshold(image):
    """ Runs the Kapur's threshold algorithm.

    Reference:
    Kapur, J. N., P. K. Sahoo, and A. K. C.Wong. ‘‘A New Method for Gray-Level
    Picture Thresholding Using the Entropy of the Histogram,’’ Computer Vision,
    Graphics, and Image Processing 29, no. 3 (1985): 273–285.

    @param image: The input image
    @type image: ndarray

    @return: The estimated threshold
    @rtype: int
    """
    hist, _ = np.histogram(image, bins=range(256), density=True)
    c_hist = hist.cumsum()
    c_hist_i = 1.0 - c_hist

    # To avoid invalid operations regarding 0 and negative values.
    c_hist[c_hist <= 0] = 1
    c_hist_i[c_hist_i <= 0] = 1

    c_entropy = (hist * np.log(hist + (hist <= 0))).cumsum()
    b_entropy = -c_entropy / c_hist + np.log(c_hist)

    c_entropy_i = c_entropy[-1] - c_entropy
    f_entropy = -c_entropy_i / c_hist_i + np.log(c_hist_i)

    return np.argmax(b_entropy + f_entropy)


def _get_regions_entropy(hist, c_hist, thresholds):
    """Get the total entropy of regions for a given set of thresholds"""

    total_entropy = 0
    for i in range(len(thresholds) - 1):
        # Thresholds
        t1 = thresholds[i] + 1
        t2 = thresholds[i + 1]

        # print(thresholds, t1, t2)

        # Cumulative histogram
        hc_val = c_hist[t2] - c_hist[t1 - 1]

        # Normalized histogram
        h_val = hist[t1:t2 + 1] / hc_val if hc_val > 0 else 1

        # entropy
        entropy = -(h_val * np.log(h_val + (h_val <= 0))).sum()

        # Updating total entropy
        total_entropy += entropy

    return total_entropy


def _get_thresholds(hist, c_hist, nthrs):
    """Get the thresholds that maximize the entropy of the regions

    @param hist: The normalized histogram of the image
    @type hist: ndarray
    @param c_hist: The cummuative normalized histogram of the image
    @type c_hist: ndarray
    @param nthrs: The number of thresholds
    @type nthrs: int
    """
    # Thresholds combinations
    thr_combinations = combinations(range(255), nthrs)

    max_entropy = 0
    opt_thresholds = None

    # Extending histograms for convenience
    # hist = np.append([0], hist)
    c_hist = np.append(c_hist, [0])

    for thresholds in thr_combinations:
        # Extending thresholds for convenience
        e_thresholds = [-1]
        e_thresholds.extend(thresholds)
        e_thresholds.extend([len(hist) - 1])

        # Computing regions entropy for the current combination of thresholds
        regions_entropy = _get_regions_entropy(hist, c_hist, e_thresholds)

        if regions_entropy > max_entropy:
            max_entropy = regions_entropy
            opt_thresholds = thresholds

    return opt_thresholds


def kapur_multithreshold(image, nthrs):
    """ Runs the Kapur's multi-threshold algorithm.

    Reference:
    Kapur, J. N., P. K. Sahoo, and A. K. C.Wong. ‘‘A New Method for Gray-Level
    Picture Thresholding Using the Entropy of the Histogram,’’ Computer Vision,
    Graphics, and Image Processing 29, no. 3 (1985): 273–285.

    @param image: The input image
    @type image: ndarray
    @param nthrs: The number of thresholds
    @type nthrs: int

    @return: The estimated threshold
    @rtype: int
    """
    # Histogran
    hist, _ = np.histogram(image, bins=range(256), density=True)

    # Cumulative histogram
    c_hist = hist.cumsum()

    return _get_thresholds(hist, c_hist, nthrs)


def apply_threshold(img, threshold=128, wp_val=255):
    """Obtain a binary image based on a given global threshold or
    a set of local thresholds.

    @param img: The input image.
    @type img: ndarray
    @param threshold: The global or local thresholds corresponding
        to each pixel of the image.
    @type threshold: Union[int, ndarray]
    @param wp_val: The value assigned to foreground pixels (white pixels).
    @type wp_val: int

    @return: A binary image.
    @rtype: ndarray
    """
    return ((img >= threshold) * wp_val).astype(np.uint8)


def apply_multithreshold(img, thresholds):
    """Obtain a binary image based on a given global threshold or
    a set of local thresholds.

    @param img: The input image.
    @type img: ndarray
    @param thresholds: Global multi-thresholds.
    @type threshold: iterable

    @return: The thresholded image.
    @rtype: ndarray
    """
    # Extending entropy and thresholds for convenience
    e_thresholds = [-1]
    e_thresholds.extend(thresholds)

    # Threshold image
    t_image = np.zeros_like(img)

    for i in range(1, len(e_thresholds)):
        t_image[img >= e_thresholds[i]] = i

    wp_val = 255 // len(thresholds)

    return t_image * wp_val


# def test_thresholds_plt(img=None):
#     """Runs all the package thresholding algorithms on the input
#     image with default parameters and plot the results.
#
#     @param img: The input gray scale image
#     @type img: ndarray
#     """
#     # Loading image if needed
#     if img is None:
#         img = face(gray=True)
#
#     # Plotting test image histogram
#     plt.figure('Histogram')
#     plt.hist(img.ravel(), range=(0, 255), bins=255)
#
#     # Applying global entropy Kapur multi-trehshold method
#     # start = default_timer()
#     th = kapur_multithreshold(img, 2)
#
#     # Plotting results
#     plt.figure('Global entropy Kapur multi-threshold method')
#     plt.imshow(apply_multithreshold(img, th), cmap='gray')
#
#     # Applying local Singh method
#     th = singh_threshold(img)
#
#     # Plotting results
#     plt.figure('Local Singh method')
#     plt.imshow(apply_threshold(img, th), cmap='gray')
#
#     # Showing plots
#     plt.show()

def otsu_threshold(image=None, hist=None):
    """ Runs the Otsu threshold algorithm.

    Reference:
    Otsu, Nobuyuki. "A threshold selection method from gray-level
    histograms." IEEE transactions on systems, man, and cybernetics
    9.1 (1979): 62-66.

    @param image: The input image
    @type image: ndarray
    @param hist: The input image histogram
    @type hist: ndarray

    @return: The Otsu threshold
    @rtype int
    """
    if image is None and hist is None:
        raise ValueError('You must pass as a parameter either'
                         'the input image or its histogram')

    # Calculating histogram
    if not hist:
        hist = np.histogram(image, bins=range(256))[0].astype(np.float)

    cdf_backg = np.cumsum(np.arange(len(hist)) * hist)
    w_backg = np.cumsum(hist)  # The number of background pixels
    w_backg[w_backg == 0] = 1  # To avoid divisions by zero
    m_backg = cdf_backg / w_backg  # The means

    cdf_foreg = cdf_backg[-1] - cdf_backg
    w_foreg = w_backg[-1] - w_backg  # The number of foreground pixels
    w_foreg[w_foreg == 0] = 1  # To avoid divisions by zero
    m_foreg = cdf_foreg / w_foreg  # The means

    var_between_classes = w_backg * w_foreg * (m_backg - m_foreg) ** 2

    return np.argmax(var_between_classes)

def _get_variance(hist, c_hist, cdf, thresholds):
    """Get the total entropy of regions for a given set of thresholds"""

    variance = 0

    for i in range(len(thresholds) - 1):
        # Thresholds
        t1 = thresholds[i] + 1
        t2 = thresholds[i + 1]

        # Cumulative histogram
        weight = c_hist[t2] - c_hist[t1 - 1]

        # Region CDF
        r_cdf = cdf[t2] - cdf[t1 - 1]

        # Region mean
        r_mean = r_cdf / weight if weight != 0 else 0

        variance += weight * r_mean ** 2

    return variance


# def _get_thresholds(hist, c_hist, cdf, nthrs):
#     """Get the thresholds that maximize the variance between regions
#
#     @param hist: The normalized histogram of the image
#     @type hist: ndarray
#     @param c_hist: The normalized histogram of the image
#     @type c_hist: ndarray
#     @param cdf: The cummulative distribution function of the histogram
#     @type cdf: ndarray
#     @param nthrs: The number of thresholds
#     @type nthrs: int
#     """
#     # Thresholds combinations
#     thr_combinations = combinations(range(255), nthrs)
#
#     max_var = 0
#     opt_thresholds = None
#
#     # Extending histograms for convenience
#     c_hist = np.append(c_hist, [0])
#     cdf = np.append(cdf, [0])
#
#     for thresholds in thr_combinations:
#         # Extending thresholds for convenience
#         e_thresholds = [-1]
#         e_thresholds.extend(thresholds)
#         e_thresholds.extend([len(hist) - 1])
#
#         # Computing variance for the current combination of thresholds
#         regions_var = _get_variance(hist, c_hist, cdf, e_thresholds)
#
#         if regions_var > max_var:
#             max_var = regions_var
#             opt_thresholds = thresholds
#
#     return opt_thresholds


def otsu_multithreshold(image=None, hist=None, nthrs=2):
    """ Runs the Otsu's multi-threshold algorithm.

    Reference:
    Otsu, Nobuyuki. "A threshold selection method from gray-level
    histograms." IEEE transactions on systems, man, and cybernetics
    9.1 (1979): 62-66.

    Liao, Ping-Sung, Tse-Sheng Chen, and Pau-Choo Chung. "A fast algorithm
    for multilevel thresholding." J. Inf. Sci. Eng. 17.5 (2001): 713-727.

    @param image: The input image
    @type image: ndarray
    @param hist: The input image histogram
    @type hist: ndarray
    @param nthrs: The number of thresholds
    @type nthrs: int

    @return: The estimated thresholds
    @rtype: int
    """
    # Histogran
    if image is None and hist is None:
        raise ValueError('You must pass as a parameter either'
                         'the input image or its histogram')

    # Calculating histogram
    if not hist:
        hist = np.histogram(image, bins=range(256))[0].astype(np.float)

    # Cumulative histograms
    c_hist = np.cumsum(hist)
    cdf = np.cumsum(np.arange(len(hist)) * hist)

    return _get_thresholds(hist, c_hist, cdf, nthrs)

def apply_multithreshold(img, thresholds):
    """Obtain a binary image based on a given global threshold or
    a set of local thresholds.

    @param img: The input image.
    @type img: ndarray
    @param thresholds: Global multi-thresholds.
    @type threshold: iterable

    @return: The thresholded image.
    @rtype: ndarray
    """
    # Extending entropy and thresholds for convenience
    e_thresholds = [-1]
    e_thresholds.extend(thresholds)

    # Threshold image
    t_image = np.zeros_like(img)

    for i in range(1, len(e_thresholds)):
        t_image[img >= e_thresholds[i]] = i

    wp_val = 255 // len(thresholds)

    return t_image * wp_val


#def test_thresholds(img, odir, basename):
def test_thresholds(img):
    """Runs all the package thresholding algorithms on the input
    image with default parameters and plot the results.

    @param img: The input gray scale image
    @type img: ndarray
    """

    # Applying global entropy Kapur multi-trehshold method
    # start = default_timer()
    threshold_KM = kapur_multithreshold(img, 2)
    # fname_KM = join(odir, "%s_entropyKapurMultiTh.jpg" % basename)
    KM_output = apply_multithreshold(img, threshold_KM)
    # cv2.imwrite(fname_KM, KM_output)

    # Applying local Singh method
    # start = default_timer()
    threshold_SM = singh_threshold(img)
    # fname_SM = join(odir, "%s_singh.jpg" % basename)
    SM_output = apply_threshold(img, threshold_SM)
    # cv2.imwrite(fname_SM, SM_output)

    zeros = np.where(KM_output==0)
    SM_output[zeros] = 0

    # multithresh_op = join(odir, "%s_op.png" % basename)
    # cv2.imwrite(multithresh_op, SM_output)
    return SM_output

def test_thresholds_otml(img, odir, basename):
    th = otsu_multithreshold(img, nthrs=2)

    otsu_multi = apply_multithreshold(img, th)

    # signal_gr = np.where(otsu_multi==127)
    # otsu_multi[signal_gr] = 255

    otml_op = join(odir, f"{basename}_enh.png")
    cv2.imwrite(otml_op, otsu_multi)


# def test_thresholds_main():
# if __name__ == '__main__':
#     """Main entry point for the test thresholds script"""
#
#     # Parsing arguments
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-i", "--image", required=True, help="Input image")
#     ap.add_argument("-o", "--out_dir", required=True, help="Output directory")
#     args = ap.parse_args()
#
#     # Reading image
#     img = cv2.imread(args.image, 0)
#
#     if img is None:
#         print("Invalid input image")
#         exit()
#
#     img_name = basename(args.image).split(".")[0]
#
#     test_thresholds(img, args.out_dir, img_name)
    # test_thresholds_otml(img, args.out_dir, img_name)

import glob

def run_test_thresh(img):
    fname = basename(img).split(".")[0]
    img = cv2.imread(img, 0)
    out_dir = '/localhome/asa420/MIAL/data-Feb4-Guang/RTN2/'
    test_thresholds(img, out_dir, fname)

sted_files = glob.glob('/localhome/asa420/MIAL/data-Feb4-Guang/RTN2/STED-files/*')
for each in sted_files:
    run_test_thresh(each)
