"""
Collection of skeletonization algorithms
"""

import skimage
import skimage.io as io
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import numpy as np
import vigra
import cv2
import mahotas
import diplib as dip
from skimage.filters import threshold_niblack, threshold_local, threshold_sauvola


from sklearn.metrics import jaccard_score

# img0 = io.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/COSKDEL/Decon/Series002_decon_converted/std/Series002_decon_converted_t79_ch00_std.png')
#
# stdimg = io.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/COSKDEL/Decon/Series002_decon_converted/std/Series002_decon_converted_t65_ch00_std.png')
# img = io.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/COSKDEL/Decon/Series002_decon_converted/enh/Series002_decon_converted_t65_ch00_std_enhance.png')

# img0 =

def get_image(img):
    return io.imread(img)


def vigra_skel(img):
    cc = vigra.analysis.labelImageWithBackground(img.astype('uint32'))
    skel = vigra.filters.skeletonizeImage(cc, 'PruneLength', 0.5)
    featdict = vigra.analysis.extractSkeletonFeatures(cc)
    return skel


def pcv_skel(img):
    return pcv.morphology.skeletonize(mask=img)


def skimg_skel(img):
    if np.max(img) == 255:
        img = img / 255
    return skimage.morphology.skeletonize(img)


def skimg_thin(img):
    if np.max(img) == 255:
        img = img / 255
    return skimage.morphology.thin(img)


def skimg_medaxis(img):
    if np.max(img) == 255:
        img = img / 255
    return skimage.morphology.medial_axis(img)


def cv_skel(img, type):
    if type == 'gh':
        return cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    else:
        return cv2.ximgproc.thinning(img)


def mahotas_thin(img):
    return mahotas.thin(img)


def dip_skel(img):
    dipimg = (img>0)
    skel_dip_three = dip.EuclideanSkeleton(dipimg, endPixelCondition='three neighbors')
    skel_dip_two = dip.EuclideanSkeleton(dipimg, endPixelCondition='two neighbors')
    skel_dip_one = dip.EuclideanSkeleton(dipimg, endPixelCondition='one neighbor')
    skel_dip_nat = dip.EuclideanSkeleton(dipimg, endPixelCondition='natural')
    return skel_dip_nat, skel_dip_one, skel_dip_two, skel_dip_three

def thresh_image(img, threshold=12, max_value=255):
    return pcv.threshold.binary(gray_img=img, threshold=threshold, max_value=max_value)

def thresh_niblack(img):
    """Grayscale image"""
    return threshold_niblack(img, window_size=5)


def thresh_loc(img, block=5, method='mean', offset=0, mode='nearest'):
    return threshold_local(img, block_size=block, method=method, offset=offset, mode=mode)


def skimage_dilate_img(img):
    return skimage.morphology.dilation(img)


def skimage_erode_img(img):
    return skimage.morphology.erosion(img.astype('int'))


def plot_samples():
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(dilsk)
    ax[1].imshow(dilavg)

    plt.show()

    # fig, ax = plt.subplots(2, 6)
    #
    # ax[0][0].imshow(stdimg)
    # # ax[0][1].imshow(er1)
    # # ax[0][2].imshow(loc)
    # # ax[0][3].imshow(er2)
    # ax[0][4].imshow(er3)
    # ax[0][5].imshow(mh_thin)
    # ax[1][0].imshow(skcv)
    # ax[1][1].imshow(skghcv)
    # ax[1][2].imshow(skpcv)
    # ax[1][3].imshow(sksk)
    # ax[1][4].imshow(skth)
    # ax[1][5].imshow(skmed)
    # plt.show()

    # ax[0].imshow(dipimg)
    # ax[1].imshow(thrlocal3)
    # ax[2].imshow(thrlocal5)
    # ax[3].imshow(thrlocal7)
    #
    # # ax[0].imshow(skel_dip_three)
    # # # ax[1].imshow(thin)
    # # # ax[2].imshow(medax)
    # # # ax[3].imshow(skel_pcv)
    # # # ax[4].imshow(skel_cv)
    # # # ax[5].imshow(skel_cv_gh)
    # # # ax[6].imshow(skel_maho)
    # # ax[1].imshow(skel_dip_two)
    # # ax[2].imshow(skel_dip_one)
    # # ax[3].imshow(skel_dip_nat)
    #
    # plt.show()

# stdimg = io.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDEL/COSKDEL/Decon/Series002_decon_converted/std/Series002_decon_converted_t00_ch00_std.png')

def exp():
    er1 = skimage_erode_img(stdimg)
    loc = thresh_loc(er1)
    er2 = skimage_erode_img(loc)
    er3 = skimage_erode_img(er2)

    mh_thin = mahotas_thin(er3)
    skcv = cv_skel(er3.astype('uint8'), 'e')

    # fig, ax = plt.subplots(1, 4)
    # ax[0].imshow(er1)
    # ax[1].imshow(loc)
    # ax[2].imshow(er2)
    # ax[3].imshow(er3)

    # biner3 = threshold_niblack(er3)
    # skghcv = cv_skel(biner3.astype('uint8'), 'gh')
    # skpcv = pcv_skel(biner3)
    # sksk = skimg_skel(biner3.astype('int')/255)
    # skth = skimg_thin(er3.astype('int')/255)
    # skmed = skimg_medaxis(er3.astype('int')/255)

    # fsk = io.imread('/localhome/asa420/MIAL/data/live-cell-movies/COSKDELCLIMP/COSKDELCLIMP/Decon/Series007_decon_converted/skel/Series007_decon_converted_t00_ch00_std_enhance_skel.png')
    # favg = io.imread('/localhome/asa420/MIAL/data/aggregation-with-median/Climp/avg/skel/ClimpSeries7-avg_skel.png')
    #
    # dilsk = skimage_dilate_img(fsk)
    # dilavg = skimage_dilate_img(favg)




    # timg = io.imread('/localhome/asa420/ER-Analysis-scripts/ero5.png')
    #
    op = skimage.filters.butterworth(er3, cutoff_frequency_ratio=0.02, order=3.0)#, high_pass=False)
    # k = np.where(op < 0)
    # op[k]=0
    plt.imshow(op)
    plt.colorbar()
    plt.show()

# exp()

# skel_dip_away = dip.EuclideanSkeleton(dipimg, endPixelCondition='loose ends away') ####### not useful


import numpy as np
import skfmm


class skeletonize():
    def __init__(self, speed_power=1.2, Euler_step_size=0.5, depth_th=2, length_th=None, simple_path=False, verbose=False):
        super().__init__()
        self.speed_power = speed_power
        self.Euler_step_size = Euler_step_size
        self.simple_path = simple_path
        self.length_th = length_th
        self.depth_th = depth_th
        self.verbose = verbose


    def _get_line_length(self, L):
        length = np.sum( np.sum( (L[1:]-L[:-1])**2, axis=1 )**0.5 )
        return length


    def _point_min(self, dist, im_2d):
        sz = dist.shape
        max_dist_ = np.max(dist)
        pd_dist = max_dist_ * np.ones(np.array(sz)+2)

        if im_2d:
            pd_dist[1:-1, 1:-1] = dist

            Fx = np.zeros(sz, dtype=np.float64)
            Fy = np.zeros(sz, dtype=np.float64)

            x = [1,-1, 0, 0, 1, 1,-1,-1]
            y = [0, 0, 1,-1, 1,-1, 1,-1]

            for i in range(len(x)):
                in_ = pd_dist[1+x[i]:1+sz[0]+x[i], 1+y[i]:1+sz[1]+y[i]]
                check = in_<dist
                dist[check] = in_[check]

                den = (x[i]**2 + y[i]**2)**0.5
                Fx[check] = x[i]/den
                Fy[check] = y[i]/den
            return [Fx, Fy]

        else:
            pd_dist[1:-1, 1:-1, 1:-1] = dist

            Fx = np.zeros(sz, dtype=np.float64)
            Fy = np.zeros(sz, dtype=np.float64)
            Fz = np.zeros(sz, dtype=np.float64)

            x = [0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 1,-1, 0, 0, 1, 1,-1,-1, 1,-1, 0, 0, 1, 1,-1,-1]
            y = [0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1]
            z = [1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0]

            for i in range(len(x)):
                in_ = pd_dist[1+x[i]:1+sz[0]+x[i], 1+y[i]:1+sz[1]+y[i], 1+z[i]:1+sz[2]+z[i]]
                check = in_<dist
                dist[check] = in_[check]

                den = (x[i]**2 + y[i]**2 + z[i]**2)**0.5
                Fx[check] = x[i]/den
                Fy[check] = y[i]/den
                Fz[check] = z[i]/den
            return [Fx, Fy, Fz]



    def _Euler_path_3d(self, Fx, Fy, Fz, start_point, step_size):
        f_start_point = np.floor(start_point).astype(int)
        sz = Fx.shape

        x = [0, 0, 0, 0, 1, 1, 1, 1]
        y = [0, 0, 1, 1, 0, 0, 1, 1]
        z = [0, 1, 0, 1, 0, 1, 0, 1]

        neighbor_inx = np.array((x,y,z)).T

        base = f_start_point + neighbor_inx
        base[base<0] = 0
        xbase = base[:,0]; xbase[xbase>=sz[0]] = sz[0]-1
        ybase = base[:,1]; ybase[ybase>=sz[1]] = sz[1]-1
        zbase = base[:,2]; zbase[zbase>=sz[2]] = sz[2]-1
        base  = np.array((xbase,ybase,zbase)).T

        dist2f = np.squeeze(start_point-f_start_point)
        dist2c = 1-dist2f

        perc = np.array((   dist2c[0]*dist2c[1]*dist2c[2],
                            dist2c[0]*dist2c[1]*dist2f[2],
                            dist2c[0]*dist2f[1]*dist2c[2],
                            dist2c[0]*dist2f[1]*dist2f[2],
                            dist2f[0]*dist2c[1]*dist2c[2],
                            dist2f[0]*dist2c[1]*dist2f[2],
                            dist2f[0]*dist2f[1]*dist2c[2],
                            dist2f[0]*dist2f[1]*dist2f[2]  ))

        gradient_valueX = [Fx[tuple(i)] for i in base]*perc
        gradient_valueY = [Fy[tuple(i)] for i in base]*perc
        gradient_valueZ = [Fz[tuple(i)] for i in base]*perc

        gradient_value = np.array((gradient_valueX, gradient_valueY, gradient_valueZ))
        sum_g = np.sum(gradient_value, axis=1)
        gradient = sum_g / ((np.sum(sum_g**2)+0.000001)**0.5)
        end_point = start_point - step_size*gradient

        if (np.any(end_point<0) or end_point[0,0]>sz[0] or end_point[0,1]>sz[1] or end_point[0,2]>sz[2]):
            end_point = np.zeros((1,3))
        return end_point


    def _Euler_path_2d(self, Fx, Fy, start_point, step_size):
        f_start_point = np.floor(start_point).astype(int)
        sz = Fx.shape

        x = [0, 0, 1, 1]
        y = [0, 1, 0, 1]

        neighbor_inx = np.array((x,y)).T

        base = f_start_point + neighbor_inx
        base[base<0] = 0
        xbase = base[:,0]; xbase[xbase>=sz[0]] = sz[0]-1
        ybase = base[:,1]; ybase[ybase>=sz[1]] = sz[1]-1
        base  = np.array((xbase,ybase)).T

        dist2f = np.squeeze(start_point-f_start_point)
        dist2c = 1-dist2f

        perc = np.array((   dist2c[0]*dist2c[1],
                            dist2c[0]*dist2f[1],
                            dist2f[0]*dist2c[1],
                            dist2f[0]*dist2f[1]  ))

        gradient_valueX = [Fx[tuple(i)] for i in base]*perc
        gradient_valueY = [Fy[tuple(i)] for i in base]*perc

        gradient_value = np.array((gradient_valueX, gradient_valueY))
        sum_g = np.sum(gradient_value, axis=1)
        gradient = sum_g / ((np.sum(sum_g**2)+0.000001)**0.5)
        end_point = start_point - step_size*gradient

        if np.any(end_point<0) or np.any(end_point>sz):
            end_point = np.zeros_like(end_point)

        return end_point


    def _Euler_shortest_path(self, dist, start_point, source_point, step_size, im_2d):
        F = self._point_min(dist, im_2d)

        if im_2d:
            Fx, Fy = -F[0], -F[1]
        else:
            Fx, Fy, Fz = -F[0], -F[1], -F[2]

        itr = 0
        path = start_point
        while True:
            if im_2d:
                end_point = self._Euler_path_2d(Fx, Fy, start_point, step_size)
            else:
                end_point = self._Euler_path_3d(Fx, Fy, Fz, start_point, step_size)

            endpoint_dist_to_all = np.sum((source_point-end_point)**2, axis=1)**0.5
            distance_to_endpoint = np.min(endpoint_dist_to_all)

            if itr>=10:
                movement = np.sum((end_point-path[itr-10])**2)**0.5
            else:
                movement = step_size+1

            if np.all(end_point==0) or movement<step_size: break

            itr += 1
            path = np.append(path, end_point, axis=0)
            if distance_to_endpoint<4*step_size:
                source_inx = source_point[np.argmin(endpoint_dist_to_all)]
                path = np.append(path, np.expand_dims(source_inx, axis=0), axis=0)
                break

            start_point = end_point
        return path


    def _discrete_shortest_path(self, dist, start_point, im_2d):
        sz = dist.shape
        if im_2d:
            x = [0, 1,-1, 0, 0, 1, 1,-1,-1]
            y = [0, 0, 0, 1,-1, 1,-1, 1,-1]
            neighbor_inx = np.array((x,y)).T
        else:
            x = [0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 1,-1, 0, 0, 1, 1,-1,-1, 1,-1, 0, 0, 1, 1,-1,-1]
            y = [0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 0, 1,-1, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 1,-1]
            z = [1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0]
            neighbor_inx = np.array((x,y,z)).T

        path = start_point.copy()
        min_v = np.inf
        while min_v!=0:
            ngb = start_point + neighbor_inx
            valid_ngb = np.all((np.all(ngb>=0, axis=1), np.all(ngb<sz, axis=1)), axis=0)
            ngb = ngb[valid_ngb]
            ngb_value = dist[tuple(ngb.T)]
            min_ind = np.argmin(ngb_value)
            min_v = ngb_value[min_ind]
            start_point = ngb[min_ind]
            path = np.append(path, np.expand_dims(start_point, axis=0), axis=0)
        return path


    def _organize_skeleton(self, skel_seg, length_th, im_2d):
        final_skeleton = []
        n = len(skel_seg)
        if im_2d:
            end_points = np.zeros((n*2, 2))
        else:
            end_points = np.zeros((n*2, 3))

        l = 0
        for i in range(n):
            ss = skel_seg[i]
            l = max(l, len(ss))
            end_points[i*2] = ss[0]
            end_points[i*2+1] = ss[-1]

        connecting_distance = 2
        for i in range(n):
            ss = np.asarray(skel_seg[i])

            ex = np.reshape(end_points[:,0], (-1,1)); ex = np.repeat(ex, len(ss), axis=1)
            sx = np.reshape(ss[:,0], (1,-1)); sx = np.repeat(sx, len(end_points), axis=0)

            ey = np.reshape(end_points[:,1], (-1,1)); ey = np.repeat(ey, len(ss), axis=1)
            sy = np.reshape(ss[:,1], (1,-1)); sy = np.repeat(sy, len(end_points), axis=0)

            if im_2d:
                dist_ = (ex-sx)**2 + (ey-sy)**2
            else:
                ez = np.reshape(end_points[:,2], (-1,1)); ez = np.repeat(ez,len(ss), axis=1)
                sz = np.reshape(ss[:,2], (1,-1)); sz = np.repeat(sz,len(end_points), axis=0)
                dist_ = (ex-sx)**2 + (ey-sy)**2 + (ez-sz)**2

            check = np.amin(dist_, axis=1)<connecting_distance
            check[i*2] = False
            check[i*2+1] = False
            cut_skel = [0, len(ss)]
            if(any(check)):
                for ii in range(len(check)):
                    if(check[ii]):
                        line = dist_[ii]
                        min_ind = np.ma.argmin(line)
                        if (min_ind>2) and (min_ind<(len(line)-2)):
                            cut_skel.append(min_ind)

            cut_skel = sorted(cut_skel)
            for j in range(len(cut_skel)-1):
                skel_breaked_seg = ss[cut_skel[j]:cut_skel[j+1]]
                length_skel_seg = self._get_line_length(skel_breaked_seg)
                if length_skel_seg>=length_th:
                    final_skeleton.append(skel_breaked_seg)
        return final_skeleton


    def skeleton(self, obj):
        obj = np.array(obj, dtype=np.bool)
        im_2d = True if obj.ndim==2 else False

        boundary_dist = skfmm.distance(obj)
        source_point = np.unravel_index(np.argmax(boundary_dist), boundary_dist.shape)
        max_dist_ = boundary_dist[source_point]
        speed_im = (boundary_dist / max_dist_) ** self.speed_power
        del boundary_dist

        flag = True
        length_threshold = 0.0
        obj = np.ones(obj.shape, dtype=np.float64)
        obj[source_point] = 0.0
        skeleton_segments = []
        source_point = np.expand_dims(source_point, axis=0)
        while True:
            dist = skfmm.travel_time(obj, speed_im)
            end_point = np.unravel_index(np.ma.argmax(dist), dist.shape)
            max_dist = dist[end_point]
            dist = np.ma.filled(dist, max_dist)
            end_point = np.expand_dims(end_point, axis=0)

            if self.simple_path:
                shortest_path = self._discrete_shortest_path(dist, end_point, im_2d)
            else:
                shortest_path = self._Euler_shortest_path(dist, end_point, source_point, self.Euler_step_size, im_2d)

            path_length = self._get_line_length(shortest_path)
            if self.verbose:
                print(path_length)

            if flag:
                depth_threshold  = self.depth_th * max_dist_

                longest_line_threshold = np.inf
                if self.length_th:
                    longest_line_threshold = self.length_th * path_length

                length_threshold = min(depth_threshold, longest_line_threshold)
                flag = False

            if path_length<=length_threshold: break

            source_point = np.append(source_point, shortest_path, axis=0)
            skeleton_segments.append(shortest_path)

            shortest_path = np.floor(shortest_path).astype(int)
            obj[tuple(shortest_path.T)] = 0

        final_skeleton = None
        if len(skeleton_segments) != 0:
            final_skeleton = self._organize_skeleton(skeleton_segments, length_threshold, im_2d)

        return final_skeleton



# img = io.imread('/localhome/asa420/ER-Analysis-scripts/er1_enhance.png')
# sk = skeletonize(speed_power=1.2, Euler_step_size=0.5, depth_th=2, length_th=None, simple_path=False, verbose=False)
# skel = sk.skeleton(img)
# print(len(skel[6]))

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(img)
# ax[1].imshow(skel)
# plt.show()
