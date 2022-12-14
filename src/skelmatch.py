from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment as lsa
import os
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
# from IPython import display


def skeletonTangentEstimate(edgeList, landMarkPoints=4):
    # edgeLength = [len(x) for x in edgeList]
    # mainSkeletonList = edgeList[np.argmax(edgeLength)]
    estimatedTangent = np.array([])
    skeletonPointsList = []
    for branchPoints in edgeList:
        numPoints = len(branchPoints)
        skeletonPointsList.extend(branchPoints)
        estimatedTangentBranch = np.zeros(numPoints)
        for i, point in enumerate(branchPoints):
            if i <= landMarkPoints:
                startPoint = np.array(branchPoints[0])
            else:
                startPoint = np.array(branchPoints[i - landMarkPoints])
            if i >= numPoints - landMarkPoints:
                endPoint = np.array(branchPoints[-1])
            else:
                endPoint = np.array(branchPoints[i + landMarkPoints])
            vector = endPoint - startPoint
            tan = vector[0] / vector[1] if vector[1] != 0 else np.inf
            estimatedTangentBranch[i] = np.arctan(tan)
        estimatedTangent = np.concatenate((estimatedTangent, estimatedTangentBranch))
    return estimatedTangent, skeletonPointsList


def mat2gray(img, minRange=0, maxRange=1):
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
        # Convert matrix to grayscale with the defined range
    minImg = np.min(img)
    maxImg = np.max(img)
    return (img - minImg) * (maxRange - minRange) / (maxImg - minImg) + minRange


def bdry_extract(skeleton, edgeList):
    t = np.zeros(len(edgeList))
    G2, G1 = np.gradient(skeleton)
    for i, point in enumerate(edgeList):
        t[i] = np.arctan2(G2[point[0], point[1]], G1[point[0], point[1]]) + np.pi / 2
    return t


def dist2(x, c):
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    if dimx != dimc:
        raise ValueError('Dimensions mismatch!')
    return (np.dot(np.ones((ncenters, 1)), np.sum(np.square(x).T, 0, keepdims=True))).T + np.dot(np.ones((ndata, 1)), np.sum(np.square(c).T, 0, keepdims=True)) - 2 * np.dot(x, c.T)


def get_samples(edgeList, t, tangent, nsamp, k=3):
    '''Using Jitendras sampling method'''
    edgeListi = np.array(edgeList)
    N = len(edgeList)
    sortInd = np.arange(N)
    Nstart = min(k * nsamp, N)

    ind0 = np.random.permutation(N)
    ind0 = ind0[:Nstart]

    edgeListi = edgeListi[ind0, :]
    ti = t[ind0]
    tangenti = tangent[ind0]
    sortIndi = sortInd[ind0]

    d2 = dist2(edgeListi, edgeListi)
    diag = np.zeros((Nstart, Nstart))
    np.fill_diagonal(diag, np.inf)
    d2 += diag

    s = 1

    while s:
        # Find Closest pair
        cp = np.argwhere(d2 == np.min(d2))
        cp = cp[0, :]
        # Remove one of the points
        edgeListi = np.delete(edgeListi, cp[1], 0)
        ti = np.delete(ti, cp[1], 0)
        tagenti = np.delete(tangenti, cp[1], 0)
        sortIndi = np.delete(sortIndi, cp[1], 0)
        d2 = np.delete(d2, cp[1], 0)
        d2 = np.delete(d2, cp[1], 1)
        if d2.shape[0] == nsamp:
            s = 0
    order = np.argsort(sortIndi)
    edgeListi = edgeListi[order]
    ti = ti[order]
    tangenti = tangenti[order]
    return edgeListi, ti, tangenti


def skeletonContext(Bsamp, Tsamp, nbins_theta, nbins_r, r_inner, r_outer, outVec, meanDistance=None):
    nsamp = Bsamp.shape[1]
    inVec = outVec == 0

    # Compute r and theta arrays
    rArray = np.sqrt(dist2(Bsamp.T, Bsamp.T))

    thetaArrayAbs = np.arctan2(
        np.tile(Bsamp[1, :][:, np.newaxis], nsamp) - np.tile(Bsamp[1, :][np.newaxis, :], (nsamp, 1)),
        np.tile(Bsamp[0, :][:, np.newaxis], nsamp) - np.tile(Bsamp[0, :][np.newaxis, :], (nsamp, 1)))
    thetaArray = thetaArrayAbs - np.tile(Tsamp[:, np.newaxis], nsamp)

    # Compute mean distance for normalization
    if not meanDistance:
        tmp = rArray[inVec, :]
        tmp = tmp[:, inVec]
        meanDistance = np.mean(tmp)
    rArrayNorm = rArray / meanDistance

    # Create LogSpace
    rBinEdges = np.logspace(np.log10(r_inner), np.log10(r_outer), nbins_r)
    rArrayBin = np.zeros((nsamp, nsamp))

    for rbin in rBinEdges:
        rArrayBin += (rArrayNorm < rbin).astype('int')
    # Indicate points inside outer boundry
    insdidePoints = rArrayBin > 0

    thetaArray = thetaArray % (2 * np.pi)
    thetaArrayBin = 1 + np.floor(thetaArray / (2 * np.pi / nbins_theta))

    nbins = nbins_r * nbins_theta

    pointHistogram = np.zeros((nsamp, nbins))
    for i in range(nsamp):
        insdidePointsi = insdidePoints[i, :] & inVec
        selectedPointsRBin = rArrayBin[i, insdidePointsi] - 1
        selectedPointsThetaBin = thetaArrayBin[i, insdidePointsi] - 1
        data = np.ones(selectedPointsRBin.shape)
        sparseMat = csr_matrix((data, (selectedPointsThetaBin, selectedPointsRBin)),
                               shape=(nbins_theta, nbins_r)).toarray()
        pointHistogram[i, :] = sparseMat.T.reshape(-1)
    return pointHistogram, meanDistance


def HistCost(SC1, SC2):
    nsamp1, nbins = SC1.shape
    nsamp2, _ = SC2.shape
    eps = np.finfo(float).eps
    SC1n = SC1 / (np.tile(np.sum(SC1, 1) + eps, (nbins, 1)).T)
    SC2n = SC2 / (np.tile(np.sum(SC2, 1) + eps, (nbins, 1)).T)

    SC1Temp = np.tile(SC1n.reshape(nsamp1, 1, nbins), [1, nsamp2, 1])
    SC2Temp = np.tile(SC2n.reshape(1, nsamp2, nbins), [nsamp1, 1, 1])

    return 0.5 * np.sum(pow((SC1Temp - SC2Temp), 2) / (SC1Temp + SC2Temp + eps), 2)


def hungarian(A):
    B = A.T
    rows, cols = lsa(B)
    return cols, sum(B[rows, cols])


def bookstien(X, Y, beta_k=None):
    '''Bookstien PAMI 89'''

    N = X.shape[0]

    if N != Y.shape[0]:
        raise ValueError(' Number of points must be equal')
    rX = dist2(X, X)

    # add identity matrix to rX to make zero on diagonal
    K = rX * np.log(rX + np.eye(N))
    P = np.array(np.bmat([np.ones((N, 1)), X]))
    L = np.array(np.bmat([[K, P], [P.T, np.zeros((3, 3))]]))
    V = np.array(np.bmat([Y.T, np.zeros((2, 3))]))

    # Check if regularization parameter provided
    if beta_k:
        L[0:N, 0:N] = L[0:N, 0:N] + beta_k * np.eye(N)

    invL = np.linalg.inv(L)

    c = np.dot(invL, V.T)
    cx = c[:, 0]
    cy = c[:, 1]

    Q = np.dot(np.dot(c[0:N, :].T, K), c[0:N, :])
    E = np.mean(np.diag(Q))
    return cx, cy, E, L


def SC_plot(SC, nbins_theta, nbins_r, r_inner, r_outer, N=20):
    '''Plotting polar histogram of skeleton context for each point'''
    import matplotlib.pyplot as plt
    if len(SC) != nbins_theta * nbins_r:
        raise ValueError('dimension mismatch, check the number of bins provided')
    SC_mat = reshape(nbins_theta, nbins_r)
    rbins = np.logspace(np.log10(r_inner), np.log10(r_outer), nbins_r)
    thetabins = np.linspace(0, 2 * np.pi, nbins_theta, endpoint=False)
    ranges2plot = np.argwhere(SC_mat)
    rplot = np.array([])
    thetaplot = np.array([])
    colors = np.array([])
    for rg in ranges2plot:
        rstart = rbins[rg[1] - 1] if rg[1] > 0 else 0
        rend = rbins[rg[1]]
        thetastart = thetabins[rg[0]]
        thetaend = thetabins[rg[0] + 1] if rg[0] < nbins_theta - 1 else 2 * np.pi
        r = np.linspace(rstart + 0.05, rend - 0.05, N)
        theta = np.linspace(thetastart + 0.05, thetaend - 0.05, N)
        rv, thetav = np.meshgrid(r, theta)
        rv = np.reshape(rv.T, -1)
        thetav = np.reshape(thetav.T, -1)
        c = SC_mat[rg[0], rg[1]] * np.ones(len(rv))
        rplot = np.concatenate((rplot, rv))
        thetaplot = np.concatenate((thetaplot, thetav))
        colors = np.concatenate((colors, c))

    area = 2
    ax = plt.subplot(111, projection='polar')
    ax.scatter(thetaplot, rplot, c=colors, cmap='hot_r')
    ax.set_yticks(rbins)
    ax.set_xticks(thetabins)
    plt.show()
    return

import numpy as np
from scipy import ndimage



'''Functions needed'''


def mat2gray(img, minRange=0, maxRange=1):
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
        # Convert matrix to grayscale with the defined range
    minImg = np.min(img)
    maxImg = np.max(img)
    return (img - minImg) * (maxRange - minRange) / (maxImg - minImg) + minRange


def removeEmpty(l):
    """Remove empty lists in a nested list"""
    return list(
        filter(lambda x: not isinstance(x, list) or x, (removeEmpty(x) if isinstance(x, list) else x for x in l)))


def gkern(kernLen, sigma=5):
    """Returns a 2D Gaussian kernel array."""

    xv, yv = np.meshgrid(range(-kernLen, kernLen + 1), range(-kernLen, kernLen + 1), sparse=False, indexing='xy')
    return np.exp(-(xv * xv + yv * yv) / (2 * pow(sigma, 2)))


def graphDrawing(skeleton, edgeList, eps):
    M, N = skeleton.shape
    edgeLen = len(edgeList)
    graphImg = np.ones((M, N, 3))
    colorMat = np.hstack((np.random.uniform(0, 1, size=(edgeLen, 1)), np.random.uniform(0, 1, size=(edgeLen, 1)),
                          np.random.uniform(0, 1, size=(edgeLen, 1))))
    for i, edge in enumerate(edgeList):
        if i > 0:
            while (np.linalg.norm(colorMat[i, :] - colorMat[i - 1, :]) < eps):
                colorMat[i, :] = np.array([np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)])
        c = colorMat[i,:]
        for point in edge:
            graphImg[point[0]-1:point[0]+2, point[1]-1:point[1]+2, :] = np.stack((c[0] * np.ones((3, 3)), c[1] * np.ones((3, 3)), c[2] * np.ones((3, 3)))).T
    return graphImg



def findBranchPoints(skeleton, return_image=False):
    pixelPoints = np.argwhere(skeleton)
    neighbFilter4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    neighbFilter8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    branchPoints = np.zeros((1, 2))
    branchImg = np.zeros(skeleton.shape)
    endPoints = np.zeros((1, 2))
    endImg = np.zeros(skeleton.shape)
    skeletonTemp = np.copy(skeleton)
    # print(pixelPoints)
    for selectedPoint in pixelPoints:
        pointMatrix = np.array(
            skeleton[selectedPoint[0] - 1:selectedPoint[0] + 2, selectedPoint[1] - 1:selectedPoint[1] + 2], copy=True)
        # print(pointMatrix.shape)
        # print(pointMatrix[1, 1])
        try:
            pointMatrix[1, 1] = 0
        except Exception:
            pass
        verticeNumber = np.count_nonzero(pointMatrix)
        edgeMap = pointMatrix * ndimage.convolve(pointMatrix, neighbFilter4, mode='constant', cval=0.0)
        edgeNumber = np.sum(edgeMap) / 2
        euilerNumber = verticeNumber - edgeNumber
        if (euilerNumber > 2):
            branchPoints = np.vstack((branchPoints, selectedPoint))
            branchImg[selectedPoint[0], selectedPoint[1]] = 1
            skeletonTemp[selectedPoint[0], selectedPoint[1]] = np.nan
        elif ((euilerNumber == 1) & (verticeNumber < 5)):
            endPoints = np.vstack((endPoints, selectedPoint))
            endImg[selectedPoint[0], selectedPoint[1]] = 1
            skeletonTemp[selectedPoint[0], selectedPoint[1]] = -np.inf
        elif ((euilerNumber == 2) & (verticeNumber >= 4)):
            connectedTrees4, connectedTrees4Num = ndimage.label(pointMatrix, neighbFilter4)
            connectedTrees8, connectedTrees8Num = ndimage.label(pointMatrix, neighbFilter8)
            label4, verticesNumberTrees = np.unique(connectedTrees4[connectedTrees4 > 0], return_counts=True)
            cornerCondition = (np.sum(pointMatrix[0:2, 0:2]) == 3) | (np.sum(pointMatrix[1:3, 1:3]) == 3) | (
            np.sum(pointMatrix[1:3, 0:2]) == 3) | (np.sum(pointMatrix[0:2, 1:3]) == 3)
            if ((abs(verticesNumberTrees[0] - verticesNumberTrees[1]) >= 2) & cornerCondition & (
                connectedTrees8Num > 1)):
                branchPoints = np.vstack((branchPoints, selectedPoint))
                branchImg[selectedPoint[0], selectedPoint[1]] = 1
                skeletonTemp[selectedPoint[0], selectedPoint[1]] = np.nan
    branchPoints = branchPoints[1:, :].astype('int64')
    endPoints = endPoints[1:, :].astype('int64')
    if not return_image:
        return branchPoints, endPoints, skeletonTemp
    skeletonGraphPointsImg = np.tile(skeleton, (3, 1, 1))
    skeletonGraphPointsImg = skeletonGraphPointsImg + np.stack(
        (np.zeros(skeleton.shape), -branchImg, -branchImg)) + np.stack((-endImg, np.zeros(skeleton.shape), -endImg))
    skeletonGraphPointsImg = np.moveaxis(skeletonGraphPointsImg, 0, -1)
    return branchPoints, endPoints, skeletonTemp, skeletonGraphPointsImg


def branchMask(searchMatrixPositive, branchsNearby):
    branchMask = ~ branchsNearby
    branchMat = np.argwhere(branchsNearby)
    # find special cases of branches
    branchMatRed = branchMat[np.where(
        np.all(branchMat == [1, 0], axis=1) | np.all(branchMat == [0, 1], axis=1) | np.all(branchMat == [2, 1],
                                                                                           axis=1) | np.all(
            branchMat == [1, 2], axis=1))]

    if (~ branchMatRed).all():
        for branch in branchMatRed:
            if branch[0] == 1:
                branchMask[[[0, 2], [branch[1], branch[1]]]] = 0
            else:
                branchMask[[[branch[0], branch[0]], [0, 2]]] = 0
    searchMatrixNew = searchMatrixPositive * branchMask
    return searchMatrixNew, branchMat

def branching(searchBranchPoint,skeletonTemp,edgeList,edgeNumber):
    branchNeighborMatrix = np.array(skeletonTemp[searchBranchPoint[0]-1:searchBranchPoint[0]+2,searchBranchPoint[1]-1:searchBranchPoint[1]+2],copy = True)
    branchNeighborMatrix[1,1] = 0
    branchNearbyBranch = np.isnan(branchNeighborMatrix)
    endNearbyBranch = np.isinf(branchNeighborMatrix)
    skeletonTemp[searchBranchPoint[0],searchBranchPoint[1]] = - edgeNumber
    with np.errstate( invalid='ignore'):
        branchNeighborMatrixPositive = branchNeighborMatrix > 0
    if (branchNearbyBranch.any()):
        branchNeighborMatrixNew, searchBranchMat2 = branchMask(branchNeighborMatrixPositive, branchNearbyBranch)
        searchBranchPoint2 = searchBranchMat2 + searchBranchPoint - [1,1]
        branchConnected4,branchConnected4Num = ndimage.label(branchNeighborMatrixNew, np.array([[0,1,0],[1,0,1],[0,1,0]]))
        if branchConnected4Num:
            labels = np.unique(branchConnected4[branchConnected4>0])
            for l in labels:
                newBranchPoints = np.argwhere(branchConnected4 == l) + searchBranchPoint - [1,1]
                edgeList.append(np.concatenate((np.array([searchBranchPoint]),newBranchPoints)).tolist())
                edgeNumberNew3 = len(edgeList)
                skeletonTemp[newBranchPoints.T.tolist()] = -edgeNumberNew3

        for branch2 in searchBranchPoint2:
            edgeList.append(np.stack((searchBranchPoint,branch2)).tolist())
            edgeNumberNew = len(edgeList)
            skeletonTemp, edgeList = branching(branch2, skeletonTemp, edgeList, edgeNumberNew)
    elif (endNearbyBranch.any()):
        endNearbyBranchPoints = np.argwhere(endNearbyBranch) + searchBranchPoint - [1,1]
        for branch2 in endNearbyBranchPoints:
            edgeList.append(np.stack((searchBranchPoint,branch2)).tolist())
            edgeNumberNew2 = len(edgeList)
            skeletonTemp[branch2.tolist()] = -edgeNumberNew2

        branchNeighborMatrixNew = np.array(branchNeighborMatrixPositive, copy=True)
        branchConnected4,branchConnected4Num = ndimage.label(branchNeighborMatrixNew, np.array([[0,1,0],[1,0,1],[0,1,0]]))
        if branchConnected4Num:
            labels = np.unique(branchConnected4[branchConnected4>0])
            for l in labels:
                newBranchPoints = np.argwhere(branchConnected4 == l) + searchBranchPoint - [1,1]
                edgeList.append(np.concatenate((np.array([searchBranchPoint]),newBranchPoints)).tolist())
                edgeNumberNew3 = len(edgeList)
                skeletonTemp[newBranchPoints.T.tolist()] = -edgeNumberNew3
    else:
        branchNeighborMatrixNew = np.array(branchNeighborMatrixPositive, copy=True)
        branchConnected4,branchConnected4Num = ndimage.label(branchNeighborMatrixNew, np.array([[0,1,0],[1,0,1],[0,1,0]]))
        if branchConnected4Num:
            labels = np.unique(branchConnected4[branchConnected4>0])
            for l in labels:
                newBranchPoints = np.argwhere(branchConnected4 == l) + searchBranchPoint - [1,1]
                edgeList.append(np.concatenate((np.array([searchBranchPoint]),newBranchPoints)).tolist())
                edgeNumberNew3 = len(edgeList)
                skeletonTemp[newBranchPoints.T.tolist()] = -edgeNumberNew3


    return skeletonTemp,edgeList


def mirrorBW(BW , t = 1):
    M,N = BW.shape
    mirrorImg = np.zeros([M+2*t,N+2*t])
    mirrorImg[t:M+t,t:N+t] = BW

    mirrorImg[0:t,t:N+t]           = np.flip(mirrorImg[t:2*t,t:N+t],0)
    mirrorImg[M+t:M+2*t,t:N+t]     = np.flip(mirrorImg[M:M+t,t:N+t],0)
    mirrorImg[t:M+t,0:t]           = np.flip(mirrorImg[t:M+t,t:2*t],1)
    mirrorImg[t:M+t,N+t:N+2*t]     = np.flip(mirrorImg[t:M+t,N:N+t],1)

    mirrorImg[0:t,0:t]             = np.flip(np.flip(mirrorImg[t:2*t,t:2*t],0),1)
    mirrorImg[M+t:M+2*t,N+t:N+2*t] = np.flip(np.flip(mirrorImg[M:M+t,N:N+t],0),1)
    mirrorImg[0:t,N+t:N+2*t]       = np.flip(np.flip(mirrorImg[t:2*t,N:N+t],0),1)
    mirrorImg[M+t:M+2*t,0:t]       = np.flip(np.flip(mirrorImg[M:M+t,t:2*t],0),1)
    return mirrorImg

def flux(delD_xn, delD_yn):
    Nx = -1/np.sqrt(2) * np.array([[-1, 0, 1],[-np.sqrt(2), 0, np.sqrt(2)],[-1, 0, 1]])
    Ny = -1/np.sqrt(2) * np.array([[-1, -np.sqrt(2), -1],[0, 0, 0],[1, np.sqrt(2), 1]])
    flux = np.zeros(delD_xn.shape)
    flux.fill(np.nan)
    nonNanPix = np.argwhere(np.invert(np.isnan(delD_xn) | np.isnan(delD_yn)))
    for pix in nonNanPix:
        flux_x = Nx * delD_xn[pix[0]-1:pix[0]+2,pix[1]-1:pix[1]+2]
        flux_y = Ny * delD_yn[pix[0]-1:pix[0]+2,pix[1]-1:pix[1]+2]
        flux_x[1,1] = np.nan
        flux_y[1,1] = np.nan
        flux_temp = flux_x + flux_y
        flux[pix[0]-1:pix[0]+2,pix[1]-1:pix[1]+2] = np.nansum(flux_temp)/np.count_nonzero(~np.isnan(flux_temp))
    return flux

'''Computing the graph of skeleton'''

def skeleton2Graph(skeleton, fluxMap, sigma = 5):
    branchPoints, endPoints, skeletonTemp = findBranchPoints(skeleton)
    skeletonTemp1 = np.copy(skeletonTemp)
    vertices = np.concatenate((endPoints, branchPoints))
    edgeList = [[]]

    # Initialization
    edgeList[0].append([endPoints[0, 0], endPoints[0, 1]])
    skeletonTemp[endPoints[0, 0], endPoints[0, 1]] = -1
    edgeNumber = 1
    pointNumber = 0
    adjacencyMatrix = np.zeros((len(vertices), len(vertices)))
    verticesProperties = [[] for _ in range(len(vertices))]
    verticesProperties2 = [[] for _ in range(len(vertices))]

    while (edgeNumber <= len(edgeList)):

        if (pointNumber > len(edgeList[edgeNumber - 1]) - 1):
            if ((not edgeList[edgeNumber - 1]) & (pointNumber == 1)):
                edgeNumber += 1
                continue
            searchPointValue = skeletonTemp[searchPoint[0], searchPoint[1]]
            newEdgeInd = np.argwhere((searchMatrix != -1) & (searchMatrix < 0) & (searchMatrix > -np.inf))
            if len(newEdgeInd) == 1:
                edgeNumber2 = -searchMatrix[newEdgeInd]
                edgePoints2 = list(np.flipud(edgeList[edgeNumber2 - 1]))
                edgeList[edgeNumber - 1].extend(edgePoints2)
                edgeList[edgeNumber2 - 1] = []
            edgeNumber += 1
            pointNumber = 1
            continue

        searchPoint = edgeList[edgeNumber - 1][pointNumber]
        if ((pointNumber == 1) & ((np.isnan(skeletonTemp1[searchPoint[0], searchPoint[1]])) | (
        np.isinf(skeletonTemp1[searchPoint[0], searchPoint[1]])))):
            edgeNumber += 1
            continue

        searchMatrix = np.array(
            skeletonTemp[searchPoint[0] - 1:searchPoint[0] + 2, searchPoint[1] - 1:searchPoint[1] + 2], copy=True)
        searchMatrix[1, 1] = 0
        vec2Branch = np.array(searchPoint) - np.array(edgeList[edgeNumber - 1][0])
        if (np.linalg.norm(vec2Branch) < 1.5):
            branchOldInd = [1, 1] - vec2Branch
            searchMatrix[branchOldInd[0], branchOldInd[1]] = 0

        with np.errstate(invalid='ignore'):
            searchMatrixPositive = searchMatrix > 0

        if (np.count_nonzero(searchMatrix)):
            branchsNearby = np.isnan(searchMatrix)
            endsNearby = np.isinf(searchMatrix)
            branchsEmpty = not np.count_nonzero(branchsNearby)
            endsEmpty = not np.count_nonzero(endsNearby)

            if (branchsEmpty & endsEmpty):
                edgePoints = np.argwhere(searchMatrixPositive) + searchPoint - [1, 1]
                edgeList[edgeNumber - 1].extend(edgePoints.tolist())
                skeletonTemp[edgePoints.T.tolist()] = - edgeNumber

                # New Assignment
                pointNumber += 1

            elif (not branchsEmpty):
                searchMatrixNew, branchMat = branchMask(searchMatrixPositive, branchsNearby)
                branchMat += np.array(searchPoint) - [1, 1]

                # Adding points to EdgeList while ommiting other branches points
                edgePoints = np.argwhere(searchMatrixNew) + searchPoint - [1, 1]
                edgeList[edgeNumber - 1].extend(edgePoints.tolist())
                skeletonTemp[edgePoints.T.tolist()] = - edgeNumber
                edgeList[edgeNumber - 1].append(list(branchMat[0, :]))

                for branch in branchMat:
                    if (not np.isnan(skeletonTemp[branch[0], branch[1]])):
                        continue
                    else:
                        skeletonTemp, edgeList = branching(branch, skeletonTemp, edgeList, edgeNumber)
                edgeNumber += 1
                pointNumber = 1

            elif (branchsEmpty & (not endsEmpty)):
                endPoint = np.argwhere(endsNearby) + searchPoint - [1, 1]
                edgePoints = np.argwhere(searchMatrixPositive) + searchPoint - [1, 1]
                edgeList[edgeNumber - 1].extend(edgePoints.tolist())
                edgeList[edgeNumber - 1].extend(endPoint.tolist())
                skeletonTemp[edgePoints.T.tolist()] = - edgeNumber

                edgeNumber += 1
                pointNumber = 1

        else:
            edgeNumber += 1
            pointNumber = 1

    edgeList = removeEmpty(edgeList)
    edgeLength = [len(edge) for edge in edgeList]
    maxEdgeLength = max(edgeLength)
    gaussianKernelMatrix = gkern(maxEdgeLength, sigma)
    edgeProperties = np.zeros((3, len(edgeList)))
    edgeProperties2 = np.zeros((3, len(edgeList)))

    for i, edgePoints in enumerate(edgeList):
        startInd = np.argwhere(np.all(vertices == edgePoints[0], axis=1))[0][0]
        endInd = np.argwhere(np.all(vertices == edgePoints[-1], axis=1))
        edgeProperties[2, i] = len(edgePoints)
        if endInd:
            endInd = endInd[0][0]
            adjacencyMatrix[startInd, endInd] = i + 1
            adjacencyMatrix[endInd, startInd] = -(i + 1)
            vector2EndGaussian = np.array(edgePoints) - edgePoints[-1] + [maxEdgeLength, maxEdgeLength]
            endGaussianValue = gaussianKernelMatrix[vector2EndGaussian.T.tolist()]
            endFluxValue = fluxMap[np.array(edgePoints).T.tolist()] * endGaussianValue
            edgeProperties[1, i] = np.sum(endFluxValue[:-1]) / (len(edgePoints) - 1)
            verticesProperties[endInd].append([edgeProperties[1, i], i])
        vector2StartGaussian = np.array(edgePoints) - edgePoints[0] + [maxEdgeLength, maxEdgeLength]
        startGaussianValue = gaussianKernelMatrix[vector2StartGaussian.T.tolist()]
        startFluxValue = fluxMap[np.array(edgePoints).T.tolist()] * startGaussianValue
        edgeProperties[0, i] = np.sum(startFluxValue[1:]) / (len(edgePoints) - 1)
        verticesProperties[startInd].append([edgeProperties[0, i], i])

    adjacencyMatrix = adjacencyMatrix.astype('int64')

    for v, vertex in enumerate(vertices):
        edgeLinkedNumber = adjacencyMatrix[v, np.where(adjacencyMatrix[v, :] != 0)[0]]
        if (~ edgeLinkedNumber.any()):
            continue
        # print(edgeLength)
        # print(type(edgeLength))
        # print(edgeLinkedNumber)
        # print(np.array(edgeLength))
        temp_index = list(map(int, list(np.abs(edgeLinkedNumber) - 1)))
        edgeLinkedLength = np.array(edgeLength)
        edgeLinkedLength = edgeLinkedLength[temp_index]
        #edgeLinkedLength = np.array(edgeLength)[map(int, list(np.abs(edgeLinkedNumber) - 1))].tolist()
        searchDepth = min(edgeLinkedLength)
        for el in edgeLinkedNumber:
            if el > 0:
                edgeLinkedPoints = edgeList[int(el) - 1][1:searchDepth]
            else:
                edgeLinkedPoints = edgeList[-int(el) - 1][-searchDepth:-1]
            vector2VertexGaussian = np.array(edgeLinkedPoints) - vertex + [maxEdgeLength, maxEdgeLength]
            vertexGaussianValue = gaussianKernelMatrix[vector2VertexGaussian.T.tolist()]
            vertexFluxValue = fluxMap[np.array(edgeLinkedPoints).T.tolist()] * vertexGaussianValue
            verticesProperties2[v].append([np.sum(vertexFluxValue) / (searchDepth - 1), np.abs(el) - 1])

    for edgeInd in range(len(edgeList)):
        verticesOnEdge = np.argwhere(adjacencyMatrix == edgeInd + 1)
        if (verticesOnEdge.any()):
            edgeProperties2[0, edgeInd] = verticesProperties[verticesOnEdge[0, 0]][
                np.argwhere(np.array(verticesProperties2[verticesOnEdge[0, 0]])[:, 1] == edgeInd)[0, 0]][0]
            edgeProperties2[1, edgeInd] = verticesProperties[verticesOnEdge[0, 1]][
                np.argwhere(np.array(verticesProperties2[verticesOnEdge[0, 1]])[:, 1] == edgeInd)[0, 0]][0]
        edgeProperties2[2, edgeInd] = edgeLength[edgeInd]

    return adjacencyMatrix, edgeList,edgeProperties,edgeProperties2, verticesProperties, verticesProperties2, endPoints, branchPoints




s1 = mat2gray(io.imread('/localhome/asa420/Desktop/Series002_decon_converted_t00_ch00_std_enhance_skel.png',as_gray=True)).astype('float64')
s2 = mat2gray(io.imread('/localhome/asa420/Desktop/Series002_decon_converted_t01_ch00_std_enhance_skel.png',as_gray=True)).astype('float64')

# s1 = mat2gray(io.imread('/localhome/asa420/Desktop/s1.png',as_gray=True)).astype('float64')
# s2 = mat2gray(io.imread('/localhome/asa420/Desktop/s2.png',as_gray=True)).astype('float64')
_, edgeList1, _, _, _, _, _, _ = skeleton2Graph(s1,s1)
_, edgeList2, _, _, _, _, _, _ = skeleton2Graph(s2,s2)
s1Tangent, s1PointsList = skeletonTangentEstimate(edgeList1)
s2Tangent, s2PointsList = skeletonTangentEstimate(edgeList2)


displayFlag=1
affineStartFlag=1
polarityFlag=0
nsamp= 100
costDum=0.15
numDumRate=0.6
thetaWeight=0.2
nbins_theta=12
nbins_r=5
r_inner=1/8
r_outer=2
tan_eps=1.0
numIter=20
betaInit=1
r=1
w=4
sf=2.5
neighborWeight = 0.6
skeletonMatchCost  = np.zeros(numIter)
affineCost         = np.zeros(numIter)
bendingEnergy      = np.zeros(numIter)
matchRatio         = np.zeros(numIter)
dimSize = np.max(np.array([s1.shape,s2.shape]),0)

t2 = bdry_extract(s2,s2PointsList)
if len(s2PointsList) >= nsamp:
    s2PointsList, t2,s2Tangent = get_samples(s2PointsList,t2,s2Tangent, nsamp)
else:
    raise ValueError('Skeleton 2 does not have enough samples')
t1 = bdry_extract(s1,s1PointsList)
if len(s1PointsList) >= nsamp:
    s1PointsList, t1,s1Tangent = get_samples(s1PointsList,t1,s1Tangent, nsamp)
else:
    raise ValueError('Skeleton 1 does not have enough samples')

# if displayFlag:
#     f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(6,8))
#     ax1.imshow(s1,cmap='gray_r')
#     ax1.set_title('Original Skeleton')
#     ax2.imshow(s2,cmap='gray_r')
#     ax2.set_title('Destination Skeleton')
#     ax3.quiver(s1PointsList[:,1],s1PointsList[:,0],np.sin(t1),np.cos(t1),color='b')
#     ax3.invert_yaxis()
#     ax3.set_title(str(len(s1PointsList)) + '  samples on original skeleton')
#     ax4.quiver(s2PointsList[:,1],s2PointsList[:,0],np.sin(t2),np.cos(t2),color='r')
#     ax4.invert_yaxis()
#     ax4.set_title(str(len(s2PointsList)) +'  samples on destination skeleton')
#     plt.tight_layout()
#     plt.show()

X = np.copy(s1PointsList)
Y = np.copy(s2PointsList)
Xk = X
tk = t1

numDumPoints = int(numDumRate * nsamp)
outVec1 = np.zeros(nsamp)
outVec2 = np.zeros(nsamp)
neighborMap = np.tile(np.arange(1,nsamp+1),(nsamp,1))
neighborCost = np.zeros((nsamp,nsamp))
if displayFlag:
    f, (ax1, ax2,ax3) = plt.subplots(1,3,figsize=(21,7))
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    x = np.linspace(0, dimSize[1], 36)
    y = np.linspace(0, dimSize[0],36)
    xv, yv = np.meshgrid(x, y)
    xv = np.reshape(xv.T,-1)
    yv = np.reshape(yv.T,-1)
    M = len(xv)
    printList = ['Bending Energy','Affine Cost', 'Matching Cost', 'Matching Ratio']
    row_format ="{:>20}" * (len(printList) + 1)
    print(row_format.format("Iteration ", *printList))
for k in range(numIter):
    pointHistogram1, meanDistance1 = skeletonContext(Xk.T,np.zeros(nsamp),nbins_theta,nbins_r,r_inner,r_outer,outVec1)
    pointHistogram2, meanDistance2 = skeletonContext(Y.T,np.zeros(nsamp),nbins_theta,nbins_r,r_inner,r_outer,outVec2)

    if affineStartFlag:
        lambda_o = 1000 if k == 0 else betaInit * pow(r,k-1)
    else:
        lambda_o = betaInit * pow(r,k)
    beta_k=(pow(meanDistance2,2)) * lambda_o

    costMatShape = HistCost(pointHistogram1,pointHistogram2)
    thetaDiff = np.tile(tk,[nsamp,1]).T - np.tile(t2,[nsamp,1])

    if polarityFlag:
        costMatTheta = 0.5 * (1 - np.cos(thetaDiff))
    else:
        costMatTheta = 0.5 * (1 - np.cos(2 * thetaDiff))

    costMat = (1 - thetaWeight) * costMatShape + thetaWeight * costMatTheta
    costMat += neighborCost

    # Calculate Skeleton Context cost
    costMatTemp = costMat - neighborCost
    a1 = np.min(costMatTemp,0)
    a2 = np.min(costMatTemp,1)
    skeletonMatchCost[k] = max(np.mean(a1),np.mean(a2))

    numMatchPoints = nsamp + numDumPoints
    costMatDum = costDum * np.ones((numMatchPoints,numMatchPoints))
    costMatDum[0:nsamp,0:nsamp] = costMat
    matchedVec, _ = hungarian(costMatDum)
    matchedVec2 = np.argsort(matchedVec)

    '''Neighboring Effect'''
    s1PointsMatchedInd = matchedVec2[:nsamp] <= nsamp
    s1PointsMatched = np.argwhere(s1PointsMatchedInd)
    matchDifference = s1PointsMatched - matchedVec2[s1PointsMatched]
    matchDifferenceMean = np.mean(matchDifference)
    matchDifferenceStd = np.std(matchDifference)
    outlierInd = (matchDifference >=  (matchDifferenceMean + 2.5 * matchDifferenceStd)) | (matchDifference <=  (matchDifferenceMean - 2.5 * matchDifferenceStd))
    matchRatio[k] = len(s1PointsMatched) / nsamp
    if matchRatio[k] <= 0.65:
        if (k != 0 & outlierInd.any() ):
            outlierMatchS2Points = s1PointsMatched[outlierInd]
            outlierMatchS1Points = matchedVec2[outlierMatchS2Points]
            matchedVec2[outlierMatchS2Points] = numMatchPoints
            matchedVec[outlierMatchS1Points] = numMatchPoints

            s1PointsMatchedInd = matchedVec2[:nsamp] <= nsamp
            s1PointsMatched = np.argwhere(s1PointsMatchedInd)
            matchDifference = s1PointsMatched - matchedVec2[s1PointsMatched]
            matchDifferenceMean = np.mean(matchDifference)
            matchDifferenceStd = np.std(matchDifference)

        neighborMean = neighborMap.T - matchDifferenceMean
        neighborCost = neighborWeight * (1 - np.exp(- (neighborMap - neighborMean)**2 / (2 * 10**2)))
    else:
        neighborCost = np.zeros((nsamp,nsamp))

    outVec1 = matchedVec2[:nsamp] > nsamp
    outVec2 = matchedVec[:nsamp] > nsamp

    X2 = np.nan * np.ones((numMatchPoints,2))
    X2[:nsamp,:] = np.copy(Xk)
    X2 = X2[matchedVec - 1,:]
    X2b = np.nan * np.ones((numMatchPoints,2))
    X2b[:nsamp,:] = X
    X2b = X2b[matchedVec - 1,:]
    Y2 = np.nan * np.ones((numMatchPoints,2))
    Y2[:nsamp,:] = np.copy(Y)

    indGood = np.where(~np.isnan(X2b[:nsamp,0]))[0]
    numGood = len(indGood)
    X3b = X2b[indGood,:]
    Y3  = Y2[indGood,:]

    if displayFlag:
        ax1.clear()
        ax1.scatter(X2[:,1],X2[:,0],c='b', marker='+',s=20)
        ax1.scatter(Y2[:,1],Y2[:,0],c='r',marker='o',s=20)
        ax1.plot(np.stack((X2[:,1],Y2[:,1]),axis=1).T,np.stack((X2[:,0],Y2[:,0]),axis=1).T,c='0.75')
        ax1.quiver(Xk[:,1],Xk[:,0],np.sin(tk),np.cos(tk),color='b')
        ax1.quiver(Y[:,1],Y[:,0],np.sin(t2),np.cos(t2),color='r')
        ax1.invert_yaxis()
        # display.clear_output(wait=True)
        # display.display(plt.gcf())
        ax2.clear()
        ax2.scatter(X[:,1],X[:,0],c='b', marker='+',s=20)
        ax2.scatter(Y[:,1],Y[:,0],c='r',marker='o',s=20)
        ax2.plot(np.stack((X2b[:,1],Y2[:,1]),axis=1).T,np.stack((X2b[:,0],Y2[:,0]),axis=1).T,c='0.75')
        ax2.invert_yaxis()
        # display.clear_output(wait=True)
        # display.display(plt.gcf())



    # Calculate bending energy
    cx,cy,E,_ = bookstien(X3b , Y3, beta_k)
    bendingEnergy[k] = E
    # Calculating affine cost
    A = np.vstack((cx[numGood + 1:numGood + 3], cy[numGood + 1:numGood + 3]))
    _,s,_ = np.linalg.svd(A)
    affineCost[k] = np.log(s[0]/s[1])

    # warp coordinates
    fx_aff = np.dot(cx[numGood:numGood + 3].T,np.vstack((np.ones((1,nsamp)),X.T)))
    d2 = dist2(X3b, X)
    U = d2 * np.log(d2 + np.finfo(float).eps)
    fx_wrp = np.dot(cx[:numGood].T,U)
    fx = fx_aff + fx_wrp
    fy_aff = np.dot(cy[numGood:numGood + 3].T,np.vstack((np.ones((1,nsamp)),X.T)))
    fy_wrp = np.dot(cy[:numGood].T,U)
    fy = fy_aff + fy_wrp

    Z=np.vstack((fx,fy)).T

    Xtan = X + tan_eps * np.vstack((np.cos(t1),np.sin(t1))).T
    fx_aff = np.dot(cx[numGood:numGood + 3].T,np.vstack((np.ones((1,nsamp)),Xtan.T)))
    d2 = dist2(X3b, Xtan)
    U = d2 * np.log(d2 + np.finfo(float).eps)
    fx_wrp = np.dot(cx[:numGood].T,U)
    fx = fx_aff + fx_wrp
    fy_aff = np.dot(cy[numGood:numGood + 3].T,np.vstack((np.ones((1,nsamp)),Xtan.T)))
    fy_wrp = np.dot(cy[:numGood].T,U)
    fy = fy_aff + fy_wrp

    Ztan = np.vstack((fx,fy)).T
    tk = np.arctan2(Ztan[:,1] - Z[:,1], Ztan[:,0] - Z[:,0])


    if displayFlag:
        ax3.clear()
        ax3.scatter(Z[:,1],Z[:,0],c='b', marker='+',s=20)
        ax3.scatter(Y[:,1],Y[:,0],c='r',marker='o',s=20)
        fx_aff = np.dot(cx[numGood:numGood + 3].T,np.vstack((np.ones((1,M)),np.stack((xv,yv)))))
        d2 = dist2(X3b,np.stack((xv,yv)).T)
        fx_wrp = np.dot(cx[:numGood].T, d2 * np.log(d2 + np.finfo(float).eps))
        fx = fx_aff + fx_wrp
        fy_aff = np.dot(cy[numGood:numGood + 3].T,np.vstack((np.ones((1,M)),np.stack((xv,yv)))))
        fy_wrp = np.dot(cy[:numGood].T, d2 * np.log(d2 + np.finfo(float).eps))
        fy = fy_aff + fy_wrp
        ax3.scatter(fy,fx,c='0.25',marker='.', s=1)
        ax3.invert_yaxis()
        # display.clear_output(wait=True)
        # display.display(plt.gcf())

    # Update Paramters
    Xk = Z
    matchingData = np.vstack((bendingEnergy[: k + 1], affineCost[: k + 1], skeletonMatchCost[: k + 1], matchRatio[: k + 1])).T

    print(row_format.format("Iteration ", *printList))
    for i, row in enumerate(matchingData):
        print(row_format.format(i+1, *row))
    if np.sum(matchingData[-1,0:3]) < 0.9:
        break
plt.close()
