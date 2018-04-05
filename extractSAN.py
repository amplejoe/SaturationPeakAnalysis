#!/usr/bin/python
#title			:extractSAN.py
#description	:Calculates SAN for input image.
#author			:Andreas Leibetseder (aleibets@itec.aau.at)
#date			:20180404
#version		:1.0
#usage			:python extractSAN.py
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
# display and output
display_histogram   =       True
save_histogram      =       True
out_folder          =       "out"
out_prefix          =       "san_histogram"
# output image dims
HIST_W = 720
HIST_H = 480

## other constants
LABEL_NO_SMOKE = 0
LABEL_SMOKE = 1


def classifySAN(image):
    # read image
    input_image = cv2.imread(image)
    # get saturation histogram
    sat_histogram = utils.getSatHisto(input_image, HIST_H)
    # make predictions
    prediction = utils.predictSAN(sat_histogram, thresh_class)
    return prediction

def saveHistogram(image):
    # read image
    input_image = cv2.imread(image)
    # get saturation histogram
    sat_histogram = utils.getSatHisto(input_image, HIST_H)
    # paint histogram (with thresholds)
    histoImage = utils.getHistoImage(sat_histogram, HIST_W, HIST_H, None, None, [thresh_class])
    # save image
    utils.make_dir(out_folder)
    file_suffix = "_tc" + str(thresh_class)
    utils.saveImage(histoImage, out_folder + "/" + out_prefix + file_suffix + ".jpg")

def showHistogram(image):
    # read image
    input_image = cv2.imread(image)
    # get saturation histogram
    sat_histogram = utils.getSatHisto(input_image, HIST_H)
    # paint histogram (with thresholds)
    histoImage = utils.getHistoImage(sat_histogram, HIST_W, HIST_H, None, None, [thresh_class])
    # show image
    desc_text = "t_c: " + str(thresh_class)
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
        print "Usage: python extractSAN.py [path_to_image]"
        sys.exit()
    image_path = sys.argv[1]

    # classification
    confidences = classifySAN(image_path)
    printPrediction(image_path, confidences)

    # additional
    if (save_histogram):
        saveHistogram(image_path)
    if (display_histogram):
        showHistogram(image_path)
