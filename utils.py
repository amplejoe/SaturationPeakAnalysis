#!/usr/bin/python
#title			:histogram_utils.py
#description	:Various utility functions.
#author			:Andreas Leibetseder (aleibets@itec.aau.at)
#author         :theodore from answers.opencv.org (see code)
#date			:20180404
#version		:1.0
#notes			:Requirements: OpenCV and various python packages (see Readme.txt).
#python_version	:2.7.6
#==============================================================================

## imports
import cv2
import numpy as np
from collections import namedtuple
import os
import sys
import errno

## general constants
H_NUM_BINS = 180
S_NUM_BINS = 256
V_NUM_BINS = 256

## histogram utilities

# classes
# PeakInfo = namedtuple('PeakInfo', ['pos', 'left_size', 'right_size', 'value'])
class PeakInfo:
    pos = left_size = right_size = value = -1

    def __init__(self, pos, left_size, right_size, value):
        self.pos = pos
        self.left_size = left_size
        self.right_size = right_size
        self.value = value

# Length = namedtuple('Length', ['pos1', 'pos2'])
class Length:
    pos1 = pos2 = 0

    def __init__(self, pos1, pos2):
        self.pos1 = pos1
        self.pos2 = pos2

    def getSize(self):
        return self.pos2 - self.pos1 + 1

# find local maxima in 1D matrix using derivatives
# original article: http://answers.opencv.org/question/54672/count-number-of-peaks-in-histogram/
def findPeaks(_src, window_size):

    slope_mat = _src.copy()

    # Transform initial matrix into 1channel, and 1 row matrix
    height, width = _src.shape
    src2 = _src.flatten()

    size = window_size / 2

    up_hill = Length(0,0)
    down_hill = Length(0,0)
    output = []

    pre_state = 0
    i = size

    # R,L ... Right and Left neighbors
    # state vars (cur_state, pre_state)
    #   2: L < R
    #   1: L > R
    #   0: R = L
    # hill vars (up_hill, down_hill)
    #   0 -> 2 up
    #   2 -> 1 down
    #   1 -> 2 | 1 -> 0 down slope turning up/finished: peak found write peak_info
    while (i < len(src2) - size):
        cur_state = src2[i + size] - src2[i - size]
        if (cur_state > 0):
            cur_state = 2 #
        elif (cur_state < 0):
            cur_state = 1
        else:
            cur_state = 0
        # In case you want to check how the slope looks like
        slope_mat[i, 0] = cur_state
        if (pre_state == 0 and cur_state == 2):
            up_hill.pos1 = i
        elif (pre_state == 2 and cur_state == 1):
            up_hill.pos2 = i - 1
            down_hill.pos1 = i
        if ((pre_state == 1 and cur_state == 2) or (pre_state == 1 and cur_state == 0)):
            down_hill.pos2 = i - 1
            max_pos = up_hill.pos2
            if (src2[up_hill.pos2] < src2[down_hill.pos1]):
                max_pos = down_hill.pos1
            peak_info = PeakInfo(max_pos, up_hill.getSize(), down_hill.getSize(), src2[max_pos])
            output.append(peak_info)
        i += 1
        pre_state = cur_state
    return output

def getMinPeakValue(histogram, peak_thresh):
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(histogram)
    return (max_val * peak_thresh)

# if you play with the thresh_peak attribute value, you can increase/decrease the number of peaks found
def getLocalMaxima(_src, useGaussian, smooth_size, neighbor_size, thresh_peak, peak_width):

    output = []
    src = _src.copy()

    if useGaussian == True:
        src = cv2.GaussianBlur(src, (smooth_size, smooth_size), 0)

    peaks = findPeaks(src, neighbor_size)

    min_peak_value = getMinPeakValue(src, thresh_peak)

    for i in range(0,len(peaks)):
        if ((peaks[i].value > min_peak_value) and (peaks[i].left_size >= int(peak_width)) and (peaks[i].right_size >= int(peak_width))):
            output.append(peaks[i].pos)

    # free mem - handled automatically?
    # gc.collect()

    return output

def predictSPA(peaks, threshold):
    num_peaks = len(peaks)
    thresh_bin = int((S_NUM_BINS-1) * threshold)
    total = 0
    for i in range(0, num_peaks):
        # check if peak_bin below thresh_bin
        if peaks[i] <= thresh_bin:
            total += 1
    # smoke, non_smoke predictions
    smoke = total / float(num_peaks)
    # prediction[NO_SMOKE, SMOKE]
    prediction = [1 - smoke, smoke]
    return prediction

def predictSAN(s_histogram, threshold):
    s_bins = len(s_histogram)
    print "his len" + str(s_bins)
    thresh_bin = int((s_bins-1) * threshold)
    part1 = s_histogram[:thresh_bin]
    part2 = s_histogram[(thresh_bin+1):]
    sumOverall = np.sum(s_histogram)
    # print "Overall sum: ", sumOverall
    if (sumOverall <= 0):
        return [0,0]
    smoke = np.sum(part1) / float(sumOverall)
    # prediction[NO_SMOKE, SMOKE]
    prediction = [1 - smoke, smoke]
    return prediction

def getSatHisto(image, hist_height):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv) # sat channel is at position 1
    histSize = S_NUM_BINS # s_bins
    s_ranges = [0,histSize]
    uniform = True
    accumulate = False
    s_hist = cv2.calcHist([hsv_planes[1]], [0], None, [histSize], s_ranges)
    cv2.normalize(s_hist, s_hist, 0, hist_height, cv2.NORM_MINMAX)
    return s_hist

## output and display helper functions

def getFilledHistoImage(histo, width, height):
    bin_w = width / float(len(histo)-1)
    # create 3 channel color image
    histImage = np.zeros((height, width, 3), np.uint8)
    # normalize result to [0, histImage.rows]
    # cv2.normalize(histo, histo, 0, HIST_H, cv2.NORM_MINMAX) # HISTO IS ALREADY NORMALIZED!!
    for i in range(1,len(histo)):
        pt1 = (bin_w * (i-1), height)
        pt2 = (bin_w * i, height)
        pt3 = (bin_w * i, np.int32(height - np.round(histo[i][0])))
        pt4 = (bin_w * (i-1), np.int32(height - np.round(histo[i-1][0])))
        pts = np.array([pt1, pt2, pt3, pt4], dtype=(int,2))
        # print "bin_w", bin_w
        # np.set_printoptions(precision=3)
        # print(pts)
        # print "-------------------------"
        cv2.fillConvexPoly(histImage, pts, (255,255,255))
    return histImage

def drawline(img,pt1,pt2,color,thickness=1,style='solid',gap=10):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    elif style=='dashed':
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1
    else:
        # solid
        cv2.line(img, pt1, pt2, color, thickness)

def addVerticalLine(image, height, x_coordinate, color, thickness = 1, style='dashed'):
    drawline(image, (x_coordinate, height), (x_coordinate, 0), color, thickness, style)

def addHorizontalLine(image, width, y_coordinate, color, thickness = 1, style='dashed'):
    drawline(image, (0, y_coordinate), (width, y_coordinate), color, thickness, style)

def addPeaks(histo_image, width, peaks, color = (0, 0, 255)):
    bin_w = width / float(S_NUM_BINS - 1) # histo image starts from 0 - 255, peaks can take values from 0 - 255
    for i in range(0, len(peaks)):
        addVerticalLine(histo_image, width, int(np.round(bin_w * (peaks[i]))), color, 1, 'solid')
    return True

def addText(image, text, ypos = 28, xpos = 0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    cv2.putText(image,text,(xpos,ypos), font, scale,(183, 192, 206), thickness,cv2.LINE_AA)

def getHistoImage(histo, width, height, peaks, peak_thresh = None, class_threshs = []):
    histImg = getFilledHistoImage(histo, width, height)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(histo)

    if peak_thresh is not None:
        addPeaks(histImg, width, peaks)
    # cv2.imwrite('%s/%s' % (file_root, file_name), histImg)

    thickness = 2
    style = 'dashed'
    if peak_thresh is not None:
        # histogram is normalized to HIST_H, but max_val can still sometimes be slightly lower than HIST_H
        # (coordinate system starts from upper left, therefore the inverse percentage of peak_thresh is used)
        addHorizontalLine(histImg, width, np.int32(np.round(max_val * (1.0 - peak_thresh))), (178,223,138), thickness, style)

    if len(class_threshs) > 0:
        for th in class_threshs:
            addVerticalLine(histImg, height, np.int32(np.round(width * th)), (237, 149, 100), thickness, style)
    return histImg

def saveImage(image, file_path):
    cv2.imwrite(file_path, image)
    print "Created file: ", file_path
    sys.stdout.flush()
    pass

def showImage(image, name, text = "SATHIST"):
    # add text to image
    lines = text.splitlines()
    y0, dy = 28, 40
    for i, line in enumerate(lines):
        y = y0 + i*dy
        addText(image, line, y)
    # show image using opencv
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass

## other utilities

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print "Unexpected error: ", str(e.errno)
            raise  # This was not a "directory exist" error..
        # print "Directory exists: ", path
        return False
    return True
