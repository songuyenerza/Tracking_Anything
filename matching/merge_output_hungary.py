import sys
sys.path.append('../')

import os
import cv2
import glob
import argparse
import numpy as np
import re
import math
import shutil
from resnet_extract_ft import ResidualNet, predict_resnet
from GroundingDINO.test_end2end import GroundDino_clas

from scipy.optimize import linear_sum_assignment
from detect_carton_output import *
import time
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




def calcal_img(folder, name_img, res_model, model_dino):
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
    size_box = [ratio]

    # box_carton_list =  model_dino.detect_carton(img_pill, BOX_TRESHOLD = 0.7)   
    # if len(box_carton_list) == 0:
    #     size_box = [1.1, 1.1]

    path_img = path_box

    center = img_box.shape

    w =  center[1] * 0.9
    h =  center[0] * 0.9
    x = center[1]/2 - w/2
    y = center[0]/2 - h/2

    # d_hist = predict_resnet(res_model, cv2.resize(img_box, (256, 256)))
    d_hist = predict_resnet(res_model, img_box)

    img_box = img_box[int(y):int(y+h), int(x):int(x+w)]
    img_box = cv2.resize(img_box, (256,256))

    #get feature hsv

    # image_rgb = cv2.cvtColor(img_box.copy(), cv2.COLOR_BGR2RGB)

    # histogram_h = create_histogram(rgb_to_hsv_image(image_rgb)[...,0], 361)

    # # Convert the image to HSV color space
    img_hsv = cv2.cvtColor(img_box, cv2.COLOR_BGR2HSV)

    # Calculate the color histograms of each channel
    hist_hue = cv2.calcHist([img_hsv],[0],None,[180],[0,180])
    hist_sat = cv2.calcHist([img_hsv],[1],None,[256],[0,256])
    hist_val = cv2.calcHist([img_hsv],[2],None,[256],[0,256])

    # Concatenate the histograms into a single vector
    color_vector = np.concatenate((hist_hue, hist_sat, hist_val), axis=0)

    # Normalize the vector to have unit length
    histogram_h = cv2.normalize(color_vector, None)

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


def get_object_id(filename):
    # print( "filename", filename)
    return int(filename.split("_")[1][:-4])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Merge results box")
    parser.add_argument("--input_video_2", default="/media/anlabadmin/Data/SonG/Moving_recog_V2/video_output_1204/0505/output_tach_box_2004008_01_PXL_20230125_223312522")
    parser.add_argument("--input_video_1", default="/media/anlabadmin/Data/SonG/Moving_recog_V2/video_output_1204/0505/output_tach_box_2004008_02_IMG_0837")
    parser.add_argument("--save_result", default="/media/anlabadmin/Data/SonG/moving-recog/results_carton/result_merge")

    args = parser.parse_args()

    folder_video1 = args.input_video_1
    folder_video2 = args.input_video_2
    folder_save_output = args.save_result

    #load model resnet extract feature
    RES_model = "RES_model"
    res_model = ResidualNet(model=RES_model, pretrained= True)
    res_model.eval()

    #load model DINO
    TEXT_PROMPT = "carton box"
    BOX_TRESHOLD = 0.25
    TEXT_TRESHOLD = 0.2
    config_file = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" # change the path of the model config file
    checkpoint_path = "../GroundingDINO/CP/groundingdino_swint_ogc.pth"  # change the path of the model
    # model_dino = GroundDino_clas( config_file, checkpoint_path)
    model_dino = None
    print("==============load done model dino==========")

    time_start = time.time()
    if not os.path.exists(folder_save_output):
        os.makedirs(folder_save_output)

    #mkdir folder no match
    folder_save_nomatch = os.path.join(folder_save_output, "nomatch")
    if not os.path.exists(folder_save_nomatch):
        os.makedirs(folder_save_nomatch)

    len_results_video_1 = glob.glob1(folder_video1,"*.jpg")
    len_results_video_1 = sorted(len_results_video_1, key=get_object_id)
    
    len_results_video_2 = glob.glob1(folder_video2,"*.jpg")
    len_results_video_2 = sorted(len_results_video_2, key=get_object_id)


    index = np.argmax([len(len_results_video_1), len(len_results_video_2)])
    results_video_min = [len(len_results_video_1), len(len_results_video_2)][index]

    # print("video have len boxes nho hon :",  "len_results_video_%d" % index, abs(len(len_results_video_1) - len(len_results_video_2)))
    
    results_video_2 = {}
    for box in len_results_video_2:
        box_dic = {}
        # results_video_2["name"] = box
        size, mean_img, img1_moments, img_path = calcal_img(folder_video2, box, res_model, model_dino)
        box_dic["size"] = size
        box_dic["histogram_key"] = mean_img
        box_dic["img1_moments"] = img1_moments
        box_dic["img_path"] = img_path

        results_video_2[box] = box_dic

    count = 1

    cost_matrix = np.zeros((len(len_results_video_1), len(len_results_video_2)))
    for i, box in enumerate(len_results_video_1):

        folder_save_box = os.path.join(folder_save_output, str(count))
        if not os.path.exists(folder_save_box):
            os.makedirs(folder_save_box)

        count += 1
        size, mean_img, img1_moments, img_path = calcal_img(folder_video1, box, res_model, model_dino)

        distan_list = []
        list_search = []
        distan_list_mean = []
        distan_list_moment = []

        distan_list_size = []

        for j, box_search in enumerate(len_results_video_2):
            size_search, mean_img_search, img1_moments_search = results_video_2[box_search]["size"], results_video_2[box_search]["histogram_key"], results_video_2[box_search]["img1_moments"]
                        
            dis_size = euclidean(size, size_search)
            dis_mean = euclidean(mean_img, mean_img_search)
            dis_moments = euclidean(img1_moments, img1_moments_search)

            distan_list_mean.append(dis_mean)
            distan_list_size.append(dis_size)    
            distan_list_moment.append(dis_moments)
            
            list_search.append(box_search)    #name
            # print(box_search, box)
            # print(size_search, histogram_key_search, img1_moments_search)
        distan_list_mean = [float(i)/sum(distan_list_mean) for i in distan_list_mean]
        distan_list_size = [float(i)/sum(distan_list_size) for i in distan_list_size]
        distan_list_moment = [float(i)/sum(distan_list_moment) for i in distan_list_moment]
        
        for j in range(len(distan_list_size)):

            distan_final = distan_list_mean[j] * 1.6/6 + distan_list_moment[j] * 4/6 + distan_list_size[j] *0.4/6
            # print(distan_final)
            cost_matrix[i, j] = distan_final
    print("time per video extract feature:", time.time() - time_start)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Print the matched indices
    print("time per video:", time.time() - time_start)

    thresh_hold = 0.1
    
    for i, box in enumerate(len_results_video_1):
        if cost_matrix[i,col_ind[i] ] > thresh_hold:
            result_box_merge = len_results_video_2[col_ind[i]]
            folder_save_box = os.path.join(folder_save_output, str(i+1))
            
            #save output 2 img matching
            shutil.copy( os.path.join(folder_video1, box), os.path.join(folder_save_box, "2004008_02_IMG_0837" + box ) ) 
            shutil.copy( os.path.join(folder_video2, result_box_merge),  os.path.join(folder_save_box, "2004008_01_PXL_20230125_223312522" + result_box_merge ))
            
            print("stt = " ,i, "input query = ", box, "img_result merge = ", result_box_merge, "== distance ===", cost_matrix[i,col_ind[i] ])

