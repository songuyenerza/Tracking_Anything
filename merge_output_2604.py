import os
import cv2
import glob
import argparse
import numpy as np
import re
import math
import shutil
from resnet_extract_ft import ResidualNet, predict_resnet
# from GroundingDINO.test_end2end import GroundDino_clas

from scipy.optimize import linear_sum_assignment
from detect_carton_output import *

def get_mean(img):
    mean_red = img[:,:,0].mean()
    mean_blue = img[:,:,1].mean()
    mean_green = img[:,:,2].mean()
    return (mean_red, mean_blue, mean_green)

def get_moment(img):
    moment_red = cv2.moments(img[:,:,0])['m00']
    moment_blue = cv2.moments(img[:,:,1])['m00']
    moment_green = cv2.moments(img[:,:,2])['m00']
    return (moment_red, moment_blue, moment_green)


def create_histogram(image_matrix, max_val):
    """
    image_matrix : 2-d array
        a matrix of an image consisting of only 1 channel, [[ r ]]
    max_val : int
        maximum value that the image_matrix could have
    returns : array
        a normalized histogram array
    """
    # create an array with length max_val initialized to 0
    histogram = [0] * max_val
    height, width = image_matrix.shape
    count = height * width

    for row in image_matrix:
        for val in row:
            if val < 0 or val > max_val - 1:
                raise Exception(f"Histogram value must be between 0 and {max_val}. Received {val}.")
            # increment the histogram value of the color value
            histogram[int(val)] += 1

    # normalize the histogram by dividing the values by the amount of pixels and return as a list
    return list(map(lambda x: x / count, histogram))

def rgb_to_hsv(rgb):
    """ 
    rgb : array
        array consisting of RGB values, [ r, g, b ]
    returns : array
        array consisting of HSV values, [ h, s, v ]
    """
    r = rgb[0] / 255.
    g = rgb[1] / 255.
    b = rgb[2] / 255.
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    h = 60

    if delta == 0:
        h = 0
    elif cmax == r:
        h *= ((g - b) / delta) % 6
    elif cmax == g:
        h *= (b - r) / delta + 2
    elif cmax == b:
        h *= (r - g) / delta + 4

    return [round(h), delta / cmax if cmax != 0 else 0, cmax]

def rgb_to_hsv_image(image):
    """
    image : 3-d array
        a matrix of an image consisting of RGB channels, [[ [r, g, b] ]]
    returns : 3-d array
        a matrix of an image consisting of HSV channels, [[ [h, s, v] ]]
    """
    return np.array([[rgb_to_hsv(rgb) for rgb in row] for row in image])


def create_histogram(image_matrix, max_val):
	"""
	image_matrix : 2-d array
		a matrix of an image consisting of only 1 channel, [[ r ]]
	max_val : int
		maximum value that the image_matrix could have
	returns : array
		a normalized histogram array
	"""
	# create an array with length max_val initialized to 0
	histogram = [0] * max_val
	height, width = image_matrix.shape
	count = height * width

	for row in image_matrix:
		for val in row:
			if val < 0 or val > max_val - 1:
				raise Exception(f"Histogram value must be between 0 and {max_val}. Received {val}.")
			# increment the histogram value of the color value
			histogram[int(val)] += 1

	# normalize the histogram by dividing the values by the amount of pixels and return as a list
	return list(map(lambda x: x / count, histogram))




def calcal_img(folder, name_img, res_model):
    match = re.search(r'size([\d\.]+)x([\d\.]+)\.', name_img)
    # size1 = float(match.group(1))
    # size2 = float(match.group(2))

    # size_box = str(box.split('jpg')[0]).split('size')[-1]
    # size_box = [size1, size2]
    
    path_box = os.path.join(folder, name_img)
    img_box = cv2.imread(path_box)
    img_pill = Image.fromarray(img_box).convert("RGB")

    h, w = img_box.shape[0], img_box.shape[1]
    size_box = [h, w]

    ratio = math.sqrt(min(size_box) / max(size_box))
    size_box = [ratio, ratio]

    # box_carton_list =  model_dino.detect_carton(img_pill, BOX_TRESHOLD = 0.7)   
    # if len(box_carton_list) == 0:
    #     size_box = [1.1, 1.1]

    path_img = path_box

    center = img_box.shape

    w =  center[1] * 0.8
    h =  center[0] * 0.8
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2

    # d_hist = predict_resnet(res_model, cv2.resize(img_box, (256, 256)))

    img_box = img_box[int(y):int(y+h), int(x):int(x+w)]
    img_box = cv2.resize(img_box, (256, 256))

    d_hist = predict_resnet(res_model, img_box)


    #get feature hsv

    image_rgb = cv2.cvtColor(img_box.copy(), cv2.COLOR_BGR2RGB)

    histogram_h = create_histogram(rgb_to_hsv_image(image_rgb)[...,0], 361)
    # print("histogram_h", len(histogram_h))
    # exit()
    # mean_img = get_mean(img_box)
    # moment_img = get_moment(img_box)


    return size_box, histogram_h, d_hist, path_img


def euclidean(hist1, hist2):
    """
    find end return the euclidean distance between 2 histograms
    hist1, hist2 : array
        1-d array of an image histogram
    returns : int
        euclidean distance of the given histograms
    """
    len1, len2 = len(hist1), len(hist2)
    if len1 != len2:
        raise Exception(f"Histogram lengths must be equal. Received {len1} and {len2}.")

    result = 0
    for val1, val2 in zip(hist1, hist2):
        result += (val1 - val2) ** 2

    return math.sqrt(result)

def euclidean_distance(x=(0,0,0), y=(0,0,0)):
    return np.sqrt(np.square(x[0] - y[0]) 
                   + np.square(x[1] - y[1])
                   + np.square(x[2] - y[2]))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Merge results box")
    parser.add_argument("--input_video", default="/media/anlabadmin/Data/SonG/Moving_recog_V2/video_output_1204/output_tach_box_2004008_01_PXL_20230125_223312522")
    parser.add_argument("--save_result", default="/media/anlabadmin/Data/SonG/moving-recog/results_carton/result_merge")

    args = parser.parse_args()
    folder_video1 = args.input_video
    folder_save_output = args.save_result

    #load model resnet extract feature
    RES_model = "RES_model"
    res_model = ResidualNet(model=RES_model, pretrained= True)
    res_model.eval()

    if not os.path.exists(folder_save_output):
        os.makedirs(folder_save_output)

    list_results_video = glob.glob1(folder_video1,"*.jpg")
    # print(len(list_results_video))
    for i in range(0, len(list_results_video), 3):
        image_group = list_results_video[i : i + 3]
        for j in range(len(image_group)):
            img_path = image_group[j]
            print(img_path)