#!/usr/bin/python
#title			:extractSPA.py
#description	:Uses SPA for classifying an input image.
#author			:Andreas Leibetseder (aleibets@itec.aau.at)
#date			:20180404
#version		:1.0
#usage			:python extractSPA.py [path_to_image]
#notes			:Requirements: OpenCV and various python packages (see Readme.txt).
#python_version	:2.7.6
#==============================================================================

## imports
import sys
import cv2
import utils

## settings from ACMMM'17 paper (change these if you desire)
thresh_class        =       float(0.35)    # classification threshold (t_c)
class_conf          =       float(0.50)    # classification confidence (c_c)
# local maxima calculation
use_gaussian_blur   =       False          # smooth histogram with gaussian blur
gauss_smooth_size   =       9              # gaussian smooth size
thresh_peak         =       float(0.35)    # peak threshold (t_p) - peaks must be larger than max_bin_val * peak_thresh
neighbor_size       =       3              # size of peak finding window
peak_width          =       2              # any peak found must be 2*peak_width wide
# display and output
display_histogram   =       True
save_histogram      =       True
out_folder          =       "out"
out_prefix          =       "spa_histogram"
# output image dims
HIST_W = 720
HIST_H = 480

## other constants
LABEL_NO_SMOKE = 0
LABEL_SMOKE = 1


def classifySPA(image):
    # read image
    input_image = cv2.imread(image)
    # get saturation histogram
    sat_histogram = utils.getSatHisto(input_image, HIST_H)
    # calculate local maxima
    peaks = utils.getLocalMaxima(sat_histogram, use_gaussian_blur, gauss_smooth_size, neighbor_size, thresh_peak, peak_width)
    # make predictions (use SAN as fallback)
    prediction = utils.predictSPA(peaks, thresh_class) if (len(peaks) > 0) else utils.predictSAN(sat_histogram, thresh_class)
    return prediction

def saveHistogram(image):
    # read image
    input_image = cv2.imread(image)
    # get saturation histogram
    sat_histogram = utils.getSatHisto(input_image, HIST_H)
    # calculate local maxima
    peaks = utils.getLocalMaxima(sat_histogram, use_gaussian_blur, gauss_smooth_size, neighbor_size, thresh_peak, peak_width)
    # paint histogram (with thresholds)
    histoImage = utils.getHistoImage(sat_histogram, HIST_W, HIST_H, peaks, thresh_peak, [thresh_class])
    # save image
    utils.make_dir(out_folder)
    file_suffix = "_tc" + str(thresh_class) + "_tp"+str(thresh_peak)
    utils.saveImage(histoImage, out_folder + "/" + out_prefix + file_suffix + ".jpg")

def showHistogram(image):
    # read image
    input_image = cv2.imread(image)
    # get saturation histogram
    sat_histogram = utils.getSatHisto(input_image, HIST_H)
    # calculate local maxima
    peaks = utils.getLocalMaxima(sat_histogram, use_gaussian_blur, gauss_smooth_size, neighbor_size, thresh_peak, peak_width)
    # paint histogram (with thresholds)
    histoImage = utils.getHistoImage(sat_histogram, HIST_W, HIST_H, peaks, thresh_peak, [thresh_class])
    # show image
    desc_text = "#peaks: " + str(len(peaks))
    desc_text += "\nt_c: " + str(thresh_class)
    desc_text += "\nt_p: " + str(thresh_peak)
    utils.showImage(histoImage, "Saturation Histogram", desc_text)

def printPrediction(image_path, confidences):
    # prediction based on classification confidence
    predictionIdx = LABEL_SMOKE if (confidences[LABEL_SMOKE] >= class_conf) else LABEL_NO_SMOKE
    predictionText = "Smoke" if (confidences[LABEL_SMOKE] >= class_conf) else "No Smoke"
    print("Classification (cc = "+ str(class_conf)+"): " + image_path)
    print("Prediction: " + predictionText)
    print("Confidence: " + str(confidences[predictionIdx]))

## main
if __name__ == '__main__':

    # arguments
    if len(sys.argv) < 2:
        print "Usage: python extractSPA.py [path_to_image]"
        sys.exit()
    image_path = sys.argv[1]

    # classification
    confidences = classifySPA(image_path)
    printPrediction(image_path, confidences)

    # additional
    if (save_histogram):
        saveHistogram(image_path)
    if (display_histogram):
        showHistogram(image_path)
