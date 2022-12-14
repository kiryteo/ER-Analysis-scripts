from __future__ import print_function
import cv2 as cv
import numpy as np

# # The video feed is read in as a VideoCapture object
# cap = cv.VideoCapture("/localhome/asa420/ER-Analysis-scripts/A2_skel.mp4")
# # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
# ret, first_frame = cap.read()
# # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
# prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# # Creates an image filled with zero intensities with the same dimensions as the frame
# mask = np.zeros_like(first_frame)
# # Sets image saturation to maximum
# mask[..., 1] = 255
#
# while(cap.isOpened()):
#     # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
#     ret, frame = cap.read()
#     # Opens a new window and displays the input frame
#     cv.imshow("input", frame)
#     # Converts each frame to grayscale - we previously only converted the first frame to grayscale
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # Calculates dense optical flow by Farneback method
#     # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
#     flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     # Computes the magnitude and angle of the 2D vectors
#     magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
#     # Sets image hue according to the optical flow direction
#     mask[..., 0] = angle * 180 / np.pi / 2
#     # Sets image value according to the optical flow magnitude (normalized)
#     mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
#     # Converts HSV to RGB (BGR) color representation
#     rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
#     # Opens a new window and displays the output frame
#     cv.imshow("dense optical flow", rgb)
#     # Updates previous frame
#     prev_gray = gray
#     # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
#     if cv.waitKey(10) & 0xFF == ord('q'):
#         break
# # The following frees up resources and closes all windows
# cap.release()
# cv.destroyAllWindows()




# import numpy as np
# import cv2
# import time
#
# help_message = \
#     '''
# USAGE: optical_flow.py [<video_source>]
# Keys:
#  1 - toggle HSV flow visualization
#  2 - toggle glitch
# '''
# count = 0
#
#
# def draw_flow(img, flow, step=16):
#     # from the beginning to position 2 (excluded channel info at position 3)
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
#     fx, fy = flow[y, x].T
#     lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines + 0.5)
#     vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     cv2.polylines(vis, lines, 0, (0, 255, 0))
#     for (x1, y1), (x2, y2) in lines:
#         cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
#     return vis
#
#
# def draw_hsv(flow):
#     (h, w) = flow.shape[:2]
#     (fx, fy) = (flow[:, :, 0], flow[:, :, 1])
#     ang = np.arctan2(fy, fx) + np.pi
#     v = np.sqrt(fx * fx + fy * fy)
#     hsv = np.zeros((h, w, 3), np.uint8)
#     hsv[..., 0] = ang * (180 / np.pi / 2)
#     hsv[..., 1] = 0xFF
#     hsv[..., 2] = np.minimum(v * 4, 0xFF)
#     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     cv2.imshow('hsv', bgr)
#     return bgr
#
#
# def warp_flow(img, flow):
#     (h, w) = flow.shape[:2]
#     flow = -flow
#     flow[:, :, 0] += np.arange(w)
#     flow[:, :, 1] += np.arange(h)[:, np.newaxis]
#     res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
#     return res
#
#
# # if __name__ == '__main__':
# #     import sys
# #
# #     print(help_message)
# #     try:
# #         fn = sys.argv[1]
# #     except:
# #         fn = 0
#
# fn = '/localhome/asa420/ER-Analysis-scripts/A2_skel.mp4'
#
# cam = cv2.VideoCapture(fn)
# (ret, prev) = cam.read()
#
# prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
# show_hsv = True
# show_glitch = False
# cur_glitch = prev.copy()
#
# while True:
#     (ret, img) = cam.read()
#     vis = img.copy()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
#     prevgray = gray
#     cv2.imshow('flow', draw_flow(gray, flow))
#     if show_hsv:
#         gray1 = cv2.cvtColor(draw_hsv(flow), cv2.COLOR_BGR2GRAY)
#         thresh = cv2.threshold(gray1, 25, 0xFF,
#                                cv2.THRESH_BINARY)[1]
#         thresh = cv2.dilate(thresh, None, iterations=2)
#         cv2.imshow('thresh', thresh)
#         cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         # loop over the contours
#         for c in cnts:
#
#             # if the contour is too small, ignore it
#             (x, y, w, h) = cv2.boundingRect(c)
#             if w > 100 and h > 100 and w < 900 and h < 680:
#                 cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0xFF, 0), 4)
#                 cv2.putText(vis, str(time.time()), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0xFF), 1)
#         cv2.imshow('Image', vis)
#     if show_glitch:
#         cur_glitch = warp_flow(cur_glitch, flow)
#         cv2.imshow('glitch', cur_glitch)
#     ch = 0xFF & cv2.waitKey(5)
#     if ch == 27:
#         break
#     if ch == ord('1'):
#         show_hsv = not show_hsv
#         print('HSV flow visualization is', ['off', 'on'][show_hsv])
#     if ch == ord('2'):
#         show_glitch = not show_glitch
#         if show_glitch:
#             cur_glitch = img.copy()
#         print('glitch is', ['off', 'on'][show_glitch])
# cv2.destroyAllWindows()

import cv2 as cv
import numpy as np

# The video feed is read in as a VideoCapture object
cap = cv.VideoCapture("/localhome/asa420/ER-Analysis-scripts/A2_skel.mp4")
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = cap.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Creates an image filled with zero intensities with the same dimensions as the frame
mask = np.zeros_like(first_frame)
# Sets image saturation to maximum
mask[..., 1] = 255

while(cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
    ret, frame = cap.read()
    # Opens a new window and displays the input frame
    cv.imshow("input", frame)
    # Converts each frame to grayscale - we previously only converted the first frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    # Opens a new window and displays the output frame
    cv.imshow("dense optical flow", rgb)
    # Updates previous frame
    prev_gray = gray
    # Frames are read by intervals of 1 millisecond. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(100) & 0xFF == ord('q'):
        break
# The following frees up resources and closes all windows
cap.release()
cv.destroyAllWindows()


#
# #imports
# import cv2
# import time
# import datetime
# import imutils
#
#
# def motion_detection():
#     video_capture = cv2.VideoCapture('/localhome/asa420/ER-Analysis-scripts/A2_skel.mp4') # value (0) selects the devices default camera
#     time.sleep(2)
#
#     first_frame = None # instinate the first fame
#
#     while True:
#         frame = video_capture.read()[1] # gives 2 outputs retval,frame - [1] selects frame
#         text = 'Unoccupied'
#
#         greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # make each frame greyscale wich is needed for threshold
#
#         gaussian_frame = cv2.GaussianBlur(greyscale_frame, (21,21),0)
#         # uses a kernal of size(21,21) // has to be odd number to to ensure there is a valid integer in the centre
#         # and we need to specify the standerd devation in x and y direction wich is the (0) if only x(sigma x) is specified
#         # then y(sigma y) is taken as same as x. sigma = standerd deveation(mathmetics term)
#
#         blur_frame = cv2.blur(gaussian_frame, (5,5)) # uses a kernal of size(5,5)(width,height) wich goes over 5x5 pixel area left to right
#         # does a calculation and the pixel located in the centre of the kernal will become
#         # a new value(the sum of the kernal after the calculations) and then it moves to the right one and has a new centre pixel
#         # and does it all over again..untill the image is done, obv this can cause the edges to not be changed, but is very minute
#
#         greyscale_image = blur_frame
#         # greyscale image with blur etc wich is the final image ready to be used for threshold and motion detecion
#
#         if first_frame is None:
#             first_frame = greyscale_image
#             # first frame is set for background subtraction(BS) using absdiff and then using threshold to get the foreground mask
#             # foreground mask (black background anything that wasnt in image in first frame but is in newframe over the threshold will
#             # be a white pixel(white) foreground image is black with new object being white...there is your motion detection
#         else:
#             pass
#
#
#         frame = imutils.resize(frame, width=500)
#         frame_delta = cv2.absdiff(first_frame, greyscale_image)
#         # calculates the absolute diffrence between each element/pixel between the two images, first_frame - greyscale (on each element)
#
#         # edit the ** thresh ** depending on the light/dark in room, change the 100(anything pixel value over 100 will become 255(white))
#         thresh = cv2.threshold(frame_delta, 100, 255, cv2.THRESH_BINARY)[1]
#         # threshold gives two outputs retval,threshold image. using [1] on the end i am selecting the threshold image that is produced
#
#         dilate_image = cv2.dilate(thresh, None, iterations=2)
#         # dilate = dilate,grow,expand - the effect on a binary image(black background and white foregorund) is to enlarge/expand the white
#         # pixels in the foreground wich are white(255), element=Mat() = default 3x3 kernal matrix and iterartions=2 means it
#         # will do it twice
#
#         cnt = cv2.findContours(dilate_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
#         # contours gives 3 diffrent ouputs image, contours and hierarchy, so using [1] on end means contours = [1](cnt)
#         # cv2.CHAIN_APPROX_SIMPLE saves memory by removing all redundent points and comppressing the contour, if you have a rectangle
#         # with 4 straight lines you dont need to plot each point along the line, you only need to plot the corners of the rectangle
#         # and then join the lines, eg instead of having say 750 points, you have 4 points.... look at the memory you save!
#
#         print(cnt)
#
#         for c in cnt:
#             if cv2.contourArea(c) > 800: # if contour area is less then 800 non-zero(not-black) pixels(white)
#                 (x, y, w, h) = cv2.boundingRect(c) # x,y are the top left of the contour and w,h are the width and hieght
#
#                 cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2) # (0, 255, 0) = color R,G,B = lime / 2 = thickness(i think?)(YES IM RITE!)
#                 # image used for rectangle is frame so that its on the secruity feed image and not the binary/threshold/foreground image
#                 # as its already used the threshold/(binary image) to find the contours this image/frame is what image it will be drawed on
#
#                 text = 'Occupied'
#                 # text that appears when there is motion in video feed
#             else:
#                 pass
#
#
#         ''' now draw text and timestamp on security feed '''
#         font = cv2.FONT_HERSHEY_SIMPLEX
#
#         cv2.putText(frame, '{+} Room Status: %s' % (text),
#                     (10,20), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 0, 255), 2)
#         # frame is the image on wich the text will go. 0.5 is size of font, (0,0,255) is R,G,B color of font, 2 on end is LINE THICKNESS! OK :)
#
#
#         cv2.putText(frame, datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p'),
#                     (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX , 0.35, (0, 0, 255),1) # frame.shape[0] = hieght, frame.shape[1] = width,ssssssssssssss
#         # using datetime to get date/time stamp, for font positions using frame.shape() wich returns a tuple of (rows,columns,channels)
#         # going 10 accross in rows/width so need columns with frame.shape()[0] we are selecting columns so how ever many pixel height
#         # the image is - 10 so oppisite end at bottom instead of being at the top like the other text
#
#         cv2.imshow('Security Feed', frame)
#         cv2.imshow('Threshold(foreground mask)', dilate_image)
#         cv2.imshow('Frame_delta', frame_delta)
#
#         key = cv2.waitKey(1) & 0xFF # (1) = time delay in seconds before execution, and 0xFF takes the last 8 bit to check value or sumin
#         if key == ord('q'):
#             cv2.destroyAllWindows()
#             break
#
#
#
# if __name__=='__main__':
#     motion_detection()


def motion_det():

    # img_brg = np.array(ImageGrab.grab())
    img_brg = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A2/A2_decon_t000_ch00_skel.png')
    img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)

    prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5,5), sigmaX=0)


    sr2_frame = imageio.imread('/localhome/asa420/MIAL/data/confocal_movies/ATL/new_op_jul/skel/A2/A2_decon_t001_ch00_skel.png')
    sr2_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)

    frame_2 = cv2.cvtColor(sr2_rgb, cv2.COLOR_BGR2GRAY)
    frame_2 = cv2.GaussianBlur(src=frame_2, ksize=(5,5), sigmaX=0)


    diff_frame = cv2.absdiff(src1=prepared_frame, src2=frame_2)

    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)

    thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(image=thresh_frame,mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    plt.imshow(img_rgb)
    plt.show()

motion_det()
exit()