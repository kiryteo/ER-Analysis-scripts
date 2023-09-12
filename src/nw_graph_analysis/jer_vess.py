# Description: Vesselness filter for 2D images
# Python implementation of Jerman Vesselness filter from https://github.com/timjerman/JermanEnhancementFilter

# The results are still not the same as in the original implementation, but they are close enough

import imageio
import numpy as np
from scipy.ndimage.filters import correlate
from scipy.ndimage import gaussian_filter


def eigvalOfHessian2D(Dxx, Dxy, Dyy):
    tmp = np.sqrt((Dxx - Dyy) ** 2 + 4 * Dxy ** 2)
    mu1 = 0.5 * (Dxx + Dyy + tmp)
    mu2 = 0.5 * (Dxx + Dyy - tmp)

    check = np.abs(mu1) > np.abs(mu2)

    Lambda1 = mu1.copy()
    Lambda1[check] = mu2[check]
    Lambda2 = mu2.copy()
    Lambda2[check] = mu1[check]

    return Lambda1, Lambda2


def imgaussian(I, sigma, spacing, siz=None):
    if siz is None:
        siz = sigma * 6

    if sigma > 0:
        x = np.arange(-np.ceil(siz / spacing[0] / 2), np.ceil(siz / spacing[0] / 2) + 1)
        H = np.exp(-(x ** 2 / (2 * (sigma / spacing[0]) ** 2)))
        H /= np.sum(H)
        Hx = H.reshape((len(H), 1))

        x = np.arange(-np.ceil(siz / spacing[1] / 2), np.ceil(siz / spacing[1] / 2) + 1)
        H = np.exp(-(x ** 2 / (2 * (sigma / spacing[1]) ** 2)))
        H /= np.sum(H)
        Hy = H.reshape((1, len(H)))

        I = correlate(correlate(I, Hx, mode='nearest'), Hy, mode='nearest')

    return I


def gradient2(F, option):
    k, l = F.shape
    D = np.zeros_like(F, dtype=F.dtype)

    if option.lower() == 'x':
        # Take forward differences on left and right edges
        D[0, :] = (F[1, :] - F[0, :])
        D[k - 1, :] = (F[k - 1, :] - F[k - 2, :])
        # Take centered differences on interior points
        D[1:k - 1, :] = (F[2:k, :] - F[0:k - 2, :]) / 2
    elif option.lower() == 'y':
        D[:, 0] = (F[:, 1] - F[:, 0])
        D[:, l - 1] = (F[:, l - 1] - F[:, l - 2])
        # Take centered differences on interior points
        D[:, 1:l - 1] = (F[:, 2:l] - F[:, 0:l - 2]) / 2
    else:
        print('Unknown option')

    return D


def Hessian2D(I, Sigma, spacing):
    if Sigma < 0:
        F = I
    else:
        F = imgaussian(I, Sigma, spacing)

    # Create first and second-order differentiations
    Dy = gradient2(F, 'y')
    Dyy = gradient2(Dy, 'y')
    Dx = gradient2(F, 'x')
    Dxx = gradient2(Dx, 'x')
    Dxy = gradient2(Dx, 'y')

    return Dxx, Dyy, Dxy


def imageEigenvalues(I, sigma, spacing, brightondark):
    Hxx, Hyy, Hxy = Hessian2D(I, sigma, spacing)

    # Correct for scaling
    c = sigma ** 2
    Hxx *= c
    Hxy *= c
    Hyy *= c

    # Reduce computation by computing vesselness only where needed
    B1 = -(Hxx + Hyy)
    B2 = Hxx * Hyy - Hxy ** 2

    T = np.ones_like(B1)

    if brightondark:
        T[B1 < 0] = 0
        T[(B2 == 0) & (B1 == 0)] = 0
    else:
        T[B1 > 0] = 0
        T[(B2 == 0) & (B1 == 0)] = 0

    indeces = np.where(T == 1)

    Hxx = Hxx[indeces]
    Hyy = Hyy[indeces]
    Hxy = Hxy[indeces]

    # Calculate eigen values
    Lambda1i, Lambda2i = eigvalOfHessian2D(Hxx, Hxy, Hyy)

    Lambda1 = np.zeros_like(T)
    Lambda2 = np.zeros_like(T)

    Lambda1[indeces] = Lambda1i
    Lambda2[indeces] = Lambda2i

    # Some noise removal
    Lambda1[~np.isfinite(Lambda1)] = 0
    Lambda2[~np.isfinite(Lambda2)] = 0

    Lambda1[np.abs(Lambda1) < 1e-4] = 0
    Lambda2[np.abs(Lambda2) < 1e-4] = 0

    return Lambda1, Lambda2


def vesselness2D(I, sigmas, spacing, tau, brightondark=False):
    verbose = 1

    if brightondark is None:
        brightondark = False  # Default mode for 2D is dark vessels compared to the background

    I = I.astype(np.float32)

    for j in range(len(sigmas)):

        if verbose:
            print(f'Current filter scale (sigma): {sigmas[j]}')

        Lambda1, Lambda2 = imageEigenvalues(I, sigmas[j], spacing, brightondark)
        if brightondark:
            Lambda2 = -Lambda2

        # Proposed filter at the current scale
        Lambda3 = Lambda2

        Lambda_rho = Lambda3
        Lambda_rho[(Lambda3 > 0) & (Lambda3 <= tau * np.max(Lambda3))] = tau * np.max(Lambda3)
        Lambda_rho[Lambda3 <= 0] = 0
        response = Lambda2 * Lambda2 * (Lambda_rho - Lambda2) * 27 / (Lambda2 + Lambda_rho) ** 3

        response[(Lambda2 >= Lambda_rho / 2) & (Lambda_rho > 0)] = 1
        response[(Lambda2 <= 0) | (Lambda_rho <= 0)] = 0
        response[~np.isfinite(response)] = 0

        # Max response over multiple scales
        if j == 0:
            vesselness = response
        else:
            vesselness = np.maximum(vesselness, response)

    vesselness = vesselness / np.max(vesselness)  # Should not be really needed
    vesselness[vesselness < 1e-2] = 0

    return vesselness



I = imageio.imread('/localhome/asa420/MIAL/data/live-cell-movies/Sep2023-sted-analysis/RTN/Series001_decon_converted/Series001_decon_converted_t37_ch00_proc.png')

Ip = I.astype(np.float32)
thr = np.percentile(Ip[Ip > 0], 1) * 0.9
Ip[Ip <= thr] = thr
Ip = Ip - np.min(Ip)
Ip = Ip / np.max(Ip)

v2 = vesselness2D(Ip, [0.5, 1.0, 1.5, 2.0, 2.5], [8, 8], 0.2, brightondark=True)

import matplotlib.pyplot as plt

plt.imshow(v2, cmap='gray')
plt.show()