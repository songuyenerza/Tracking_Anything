
import argparse
import sys

sys_path = '../'
sys.path.insert(0, sys_path)
sys.path.append(sys_path)
from logger import AppLogger

from csv import writer, DictWriter
import math
from tabnanny import check
import cv2
import os
import time
import torch

from tracker.byte_tracker import BYTETracker
from Grounding_Dino import GroundDino_clas
from detect_person_yolov5 import yolo_detect_person

from PIL import Image
import json
import numpy as np
from pathlib import Path
# import torch.backends.cudnn as cudnn
from yolov5.utils.plots import Annotator, colors
import logging

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / '../yolov5') not in sys.path:
    sys.path.append(str(ROOT / '../yolov5'))  # add yolov5 ROOT to PATH


def save_dict_box(results_tracker, result_person_box,  out_path, name_video):
    # save dict box id object of video
    path_save_track = os.path.join(out_path, str(name_video))
    if not os.path.exists(path_save_track):
        os.makedirs(path_save_track)

    if os.path.exists(os.path.join(path_save_track, name_video + ".json")):
        os.remove(os.path.join(path_save_track, name_video + ".json"))
    
    with open(os.path.join(path_save_track, name_video + ".json"), "w") as outfile:
        json.dump(results_tracker, outfile)
    
    # save dict box person
    if os.path.exists(os.path.join(path_save_track, "box_person_" + name_video + ".json") ):
        os.remove(os.path.join(path_save_track, "box_person_" + name_video + ".json") )
    
    with open(os.path.join(path_save_track, "box_person_" + name_video + ".json") , "w") as outfile_person:
        json.dump(result_person_box, outfile_person)
    print("=======================================save dict done=====================================================")
    return "save dict done"


def mer_box(box_list):
    box1 = []
    box2 = []
    box3 = []
    box4 = []

    for box in box_list:
        box1.append(box[0])
        box2.append(box[1])
        box3.append(box[2])
        box4.append(box[3])
    return [min(box1), min(box2), max(box3), max(box4)]

def xywh2xy(box):
    # convert xywh 2 xmin ymin xmax ymax
    x0 = int((box[0] - ((box[2]) / 2)))
    y0 = int((box[1] - ((box[3]) / 2)))
    x1 = int((box[0] + ((box[2]) / 2)))
    y1 = int((box[1] + ((box[3]) / 2)))
    return [x0, y0, x1, y1]

def check_over_box(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    box_over = [xA, yA, xB, yB]
    # print('box_over=======', boxA, boxB, box_over)
    if box_over == boxB:
        return False
    else:
        return True

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def check_iou_person(box_, box_person, iou_thresh):
    check = 0
    box_ps_list = []
    for box_ps in box_person:
        iou = bb_intersection_over_union(box_, box_ps)
        # print("check iou with box person = ", iou)
        if iou >= iou_thresh:
            box_ps_list.append(box_ps)

            check += 1
    if check > 0:
        return True, box_ps_list[0]
    else:
        return False, []

def pading_box(box_person, image_width, image_height, num_person):
    # Calculate padding widths
    w_box, h_box = box_person[2] - box_person[0], box_person[3] - box_person[1]
    if num_person > 1:
        w_padding = int(w_box * 0.2)
        h_padding = int(h_box * 0.2)
    else:
        w_padding = int(w_box * 2)
        h_padding = int(h_box * 1)

    # Adjust box coordinates to account for padding
    x0 = max(0, box_person[0] - w_padding)
    x1 = max(0, box_person[1] - h_padding)
    x2 = min(image_width, box_person[2] + w_padding)
    x3 = min(image_height, box_person[3] + h_padding)

    return [x0, x1, x2, x3]


def get_box_final(box_detect, box_crop):
    h_boxcrop = box_crop[3] - box_crop[1]
    w_boxcrop = box_crop[2] - box_crop[0]

    return [box_crop[0] + box_detect[0] , box_crop[1] + box_detect[1], box_crop[2] - (w_boxcrop - box_detect[2]) , box_crop[3] - ( h_boxcrop - box_detect[3])]

def get_list_box_final(list_box_detect, box_crop):
    #   Convert a list of object bounding boxes detected in an image crop to their final bounding box coordinates in the original image.
    list_box_final = []
    for box_detect in list_box_detect:
        h_boxcrop = box_crop[3] - box_crop[1]
        w_boxcrop = box_crop[2] - box_crop[0]
        box_final = [box_crop[0] + box_detect[0] , box_crop[1] + box_detect[1], box_crop[2] - (w_boxcrop - box_detect[2]) , box_crop[3] - ( h_boxcrop - box_detect[3])]
        list_box_final.append(box_final)
    return list_box_final


def iou(box1, box2):
    # box1 and box2 are lists of [x1, y1, x2, y2] coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = intersection / float(area1 + area2 - intersection)
    return iou

def area_box(box):
    # input = box [xmin ymin xmax, ymax]
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    return area1

def iou_person(box1, box2):
    # box1 and box2 are lists of [x1, y1, x2, y2] coordinates ( xmin ymin xmax ymax)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = intersection / float(area1 + area2 - intersection)
    return iou

def check_merge_with_person(box_obj, box_person_list):
    """
    Check if a given object bounding box can be merged with a person bounding box in a given list.
    Parameters:
        box_obj (tuple): format (x_min, y_min, x_max, y_max).
        box_person_list (list): A list of bounding box coordinates of person objects in the format [(x_min, y_min, x_max, y_max), ...].
    Returns:
        result (bool):  Returns True if the object can be merged, False otherwise.
    """
    check = 0
    check_iou = 0
    result = None
    area_box_person = []

    if len(box_person_list) > 0:
        for box_person in box_person_list:
            iou = iou_person(box_obj, box_person)
            area_box_person.append(area_box(box_person)) 
            if iou > 0.85 and 0.87 < area_box(box_obj) / area_box(box_person) < 1.15 :
                check += 1
            if iou <= 0:
                check_iou += 1
        ratio = area_box(box_obj) / max(area_box_person)    # area box obj / area box person

        if check == 0  and  0.2 < ratio < 2 and check_iou != len(box_person_list):
            result = True
        else:
            result = False
    else:
        result = False
    return result

def check_merge_with_person_ver2(box_obj, box_person_list):
    """
    Check if a given object bounding box can be merged with a person bounding box in a given list.
    Parameters:
        box_obj (tuple): format (x_min, y_min, x_max, y_max).
        box_person_list (list): A list of bounding box coordinates of person objects in the format [(x_min, y_min, x_max, y_max), ...].
    Returns:
        result (bool):  Returns True if the object can be merged, False otherwise.
    """
    check = 0
    check_iou = 0
    result = None
    area_box_person = []

    if len(box_person_list) > 0:
        for box_person in box_person_list:
            iou = iou_person(box_obj, box_person)

            box1_xywh = xy2xywh(box_obj)
            box2_xywh = xy2xywh(box_person)

            distan_box12_w = abs(box1_xywh[0] - box2_xywh[0]) # distance Ox between box1 and box2 

            area_box_person.append(area_box(box_person)) 
            if iou > 0.8 and 0.8 < area_box(box_obj) / area_box(box_person) < 1.3 and distan_box12_w < 0.4 * min(box1_xywh[2], box2_xywh[2]) :
                check += 1
            if iou <= 0:
                check_iou += 1
        ratio = area_box(box_obj) / max(area_box_person)    # area box obj / area box person

        if check == 0  and  0.2 < ratio < 2 :
            result = True
        else:
            result = False
    else:
        result = False
    return result

def xy2xywh(box):
    w_ = box[2] - box[0]
    h_ = box[3] - box[1]
    return [int(box[0] + w_/2), int(box[1] + h_/2), w_, h_]

def merge_box_object(boxes):   #box = [xmin ymin xmax ymax]
    """
    Merge overlapping object bounding boxes in a given list.

    Parameters:
        boxes (list): A list of tuples representing the bounding box coordinates of objects in the format [(x_min, y_min, x_max, y_max), ...].

    Returns:
        merged_boxes (list): A list of tuples representing the merged bounding box, format [(x_min, y_min, x_max, y_max), ...].
    """
    merged_boxes = []
    boxes = sorted(boxes, key=lambda x: x[0])
    while len(boxes) > 0:
        current_box = boxes[0]
        del boxes[0]
        merged_box = current_box
        i = 0
        while i < len(boxes):
            box1_xywh = xy2xywh(merged_box)
            box2_xywh = xy2xywh(boxes[i])

            distan_box12_w = abs(box1_xywh[0] - box2_xywh[0]) # distance Ox between box1 and box2 
            if iou(merged_box, boxes[i]) > 0.4 and distan_box12_w < 0.5 * min(box1_xywh[2], box2_xywh[2]):
                x1 = min(merged_box[0], boxes[i][0])
                y1 = min(merged_box[1], boxes[i][1])
                x2 = max(merged_box[2], boxes[i][2])
                y2 = max(merged_box[3], boxes[i][3])
                merged_box = [x1, y1, x2, y2]
                del boxes[i]
            else:
                i += 1
        merged_boxes.append(merged_box)

    return merged_boxes


def check_height_box(box, box_person_list):     # add clear box upper and top box person
    """
    Check if a given object bounding box is located at an appropriate height relative to a given list of person bounding boxes.
    Parameters:
        box (tuple): format (x_min, y_min, x_max, y_max).
        box_person_list (list):format [(x_min, y_min, x_max, y_max), ...].
    Returns:
        result (bool):Returns True if the object is at an appropriate height, False otherwise.
    """
    y_center_list = []
    for box_person in box_person_list:
        y_center = box_person[1] + (box_person[3] - box_person[1]) /2
        y_center_list.append(y_center)
    if box[1] > max(y_center_list) or box[3] < min(y_center_list) * 0.85:
        return False
    else:
        return True

def get_obj(image, model_dino, detect_person, box_list_check, TEXT_PROMPT, Debug):          # input image; output = box
    im0 = image.copy()
    h_image = image.shape[0]
    w_image = image.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))

    box_person_list = [] #list box of person
    # detect person use yolov5m
    box_person_list = detect_person.detect_person(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # visual box person
    if Debug == True:
        for box_per in box_person_list:
            cv2.rectangle(im0, (box_per[0], box_per[1]), (box_per[2], box_per[3]), (255,255,255), 2)   

    xywhs = []
    box_carton_list = []

    if len(box_person_list) > 0:
        # crop video base on mask person
        box_person_all_merge = mer_box(box_person_list)        #box_person_all_merge: merge all box person in image  
        num_person = len(box_person_list)
        box_person_all_merge = pading_box(box_person_all_merge, w_image, h_image, num_person)     #pading 1,2 height, 1.2 width

        if Debug == True:
            #   visual box merge person 
            cv2.rectangle(im0, (box_person_all_merge[0], box_person_all_merge[1]), (box_person_all_merge[2], box_person_all_merge[3]), (255,255,255), 2)     #visual all box carton 

        img_crop_fr_person = image[box_person_all_merge[1]: box_person_all_merge[3], box_person_all_merge[0]: box_person_all_merge[2]]
        
        # finish crop image from box merge person
        # Predict moving objiect with DINO
        box_predict_ = model_dino.predict_box(Image.fromarray(img_crop_fr_person), TEXT_PROMPT = TEXT_PROMPT, BOX_TRESHOLD = 0.25, TEXT_TRESHOLD = 0.35  )   #predict in image crop by box merge person
        box_predict = get_list_box_final(box_predict_, box_person_all_merge)

    else:
        box_predict = []

    if len(box_predict) > 0:
        for box_final_ in box_predict:
            
            # visual all box carton 
            if Debug == True:
                cv2.rectangle(im0, (box_final_[0], box_final_[1]), (box_final_[2], box_final_[3]), (0,0,0), 7)     

            # add ymin box_onject > y center box person
            if  check_merge_with_person(box_final_, box_person_list) == True and check_height_box(box_final_, box_person_list ) == True :
                # visual box carton after checker
                if Debug == True:
                    cv2.rectangle(im0, (box_final_[0], box_final_[1]), (box_final_[2], box_final_[3]), (255,255,0), 2)
                box_carton_list.append(box_final_)

            else:
                if box_list_check != None:
                    if check_iou_person(box_final_, box_list_check, 0.7)[0] == True and check_merge_with_person_ver2(box_final_, box_person_list) == True and check_height_box(box_final_, box_person_list ) == True:
                        box_carton_list.append(box_final_)

                    if len(box_list_check) > 0:
                        if check_iou_person(box_final_, box_list_check, 0.7)[0] == True :
                            box_carton_list.append(box_final_)
                else:
                    if check_merge_with_person_ver2(box_final_, box_person_list) == True and check_height_box(box_final_, box_person_list ) == True:
                        box_carton_list.append(box_final_)

    for i , box_carton in enumerate(box_carton_list):
        check = 0
        for box_person in box_person_list:

            if iou(box_carton, box_person) > 0.75 :
                check += 1
        if check > 0:
            del box_carton_list[i]

    box_carton_list_merge = merge_box_object(box_carton_list)

    if Debug == True:
        print("check box carton merge = ", box_carton_list_merge, "box check ",box_list_check , box_person_list )
    if box_carton_list_merge != []:
        box_list_check = box_carton_list_merge.copy()

    for box_final_ in box_carton_list_merge:

        if box_final_ not in xywhs:
            if len(xywhs) > 0:
                for a in range(len(xywhs)):
                    box = xywhs[a]
                    if bb_intersection_over_union(box, box_final_) > 0.2:
                        box_mer = mer_box([box, box_final_])
                        xywhs[a] = box_mer
                    else:
                        xywhs.append(box_final_)
          
            else:
                xywhs.append(box_final_)

    # merge list box putput
    xywhs = merge_box_object(xywhs)

    return xywhs, im0, box_person_list, box_list_check

@torch.no_grad()
def run(
        in_path ='/media/anlabadmin/Data/SonG/DAVIS-data/DAVIS/JPEGImages/480p/video1/',
        out_path ='/media/anlabadmin/Data/SonG/moving-recog/results_carton',
        device = 'cuda',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid= False,  # show results
        save_vid= False,  # show results
        frame_skip = 3

):
    # Check loger
    logger = AppLogger('update')
    Debug = False    # visual boxes

    print("==================load model Ground Dino=======================")
    t_start = time.time()
    ## limited GPU
    # gpu_fraction = 0.9
    # torch.cuda.set_per_process_memory_fraction(gpu_fraction)

    # load model
    TEXT_PROMPT = "carrying a object, bring object, bringing object"
    BOX_TRESHOLD = 0.25
    TEXT_TRESHOLD = 0.25

    config_file = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" # change the path of the model config file
    checkpoint_path = "../GroundingDINO/CP/groundingdino_swint_ogc.pth"  # change the path of the model
    model_dino = GroundDino_clas( config_file, checkpoint_path)

    print("==================Finish Load model Ground Dino=======================", time.time() - t_start)
    
    print("==================Start Load model Yolo_V5=======================")

    detect_person = yolo_detect_person(weight='./weights/yolov5l.pt')
    
    print("==================finish load model yolo v5==========")

    print('start processing...')

    # initialize Byte Tracker
    track_thresh  = 0.5
    track_buffer = 32
    match_thresh = 0.8
    Tracker_bytetrack = BYTETracker(track_thresh, track_buffer , match_thresh, frame_rate=30)

    outputs = [None]

    # Stat data structure init
    raw_stats = {}
    results_tracker = {}
    result_person_box = {}

    path_video = in_path
    name_video = in_path.split("/")[-1][:-4]

    logger.info(f'path_video = {path_video}')
    print("path_video", path_video)
    vidcap = cv2.VideoCapture(path_video)
    W_vid = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_vid = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = (H_vid, W_vid) 
    
    if save_vid:
        fps = vidcap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
        fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
        path_check_apply_mask = os.path.join(out_path, name_video + ".avi")
        vid_writer = cv2.VideoWriter(path_check_apply_mask, cv2.VideoWriter_fourcc(*'MJPG'), fps, resolution)

    # Run tracking
    success , image = vidcap.read()
    count = 0
    #start time per video
    time_start = time.time()
    box_list_check = None

    while success:
        
        if count % 30 == 0:
            t1 = time.time()
        # skip frame
        if count % int(frame_skip) == 0:
            xywhs, im0, box_person_list, box_list_check = get_obj(image, model_dino,detect_person, box_list_check , TEXT_PROMPT, Debug)
        else:
            xywhs = []
        if count % 30 == 0:
            print("===========time per image ==============", time.time() - t1)
        result_person_box[str(count)] = box_person_list

        annotator = Annotator(im0, line_width=2, pil=not ascii)
        confs = np.array([1] * len(xywhs))
        xywhs = np.array(xywhs)

        if xywhs is not None and len(xywhs):
            print('xywhs == ', xywhs, count)
            outputs = Tracker_bytetrack.update(xywhs)
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):
                    bboxes = output.tlwh
                    bboxes = [bboxes[0], bboxes[1] , bboxes[0] + bboxes[2], bboxes[1] + bboxes[3]]
                    id = output.track_id
                    res = {}
                    res["box"] = [int(bboxes[0] ) , int(bboxes[1] ),int(bboxes[2] ) ,int(bboxes[3] )]
                    frame = count
                    res["fr"] = frame
                    list_data = []

                    if id in results_tracker.keys():
                        list_data = results_tracker[id]

                    list_data.append(res)
                    results_tracker[id] = list_data

                    print('id+check=========================', id)
                    cls = 1

                    stats = bboxes
                    if id in raw_stats:
                        raw_stats[id].append(stats)
                    else:
                        raw_stats[id] = [stats]

                    label = f'{id}'
                    if show_vid or save_vid:
                        annotator.box_label(bboxes, label, color=colors(cls, True))

        success,image = vidcap.read()
        count +=1

        # results
        if show_vid or save_vid:
            im0 = annotator.result()
        if save_vid:
            im0 = cv2.resize(im0, resolution)
            vid_writer.write(im0)
        
        #if imshow frame img
        if show_vid:
            cv2.imshow('video output', cv2.resize(im0, (360, int(resolution[1] *360 / resolution[0]))))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if save_vid:
        vid_writer.release()

    logger.info(f'Videoooo {name_video}, Time tracking: {time.time() - time_start}(s), num of frames: { count }, FPS: {1/((time.time() - time_start)/count)}')

    print("results_tracker " , results_tracker)

    save_dict_box(results_tracker, result_person_box,  out_path, name_video)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in-path', type=str, default='./video_test/2004008_01_PXL_20230125_223312522.mp4', help='input video file path')
    parser.add_argument('-o', '--out-path', type=str, default='./results_carton/', help=' output file path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save tracking video results')
    parser.add_argument('--frame_skip', type = int, default= 3, help='skip n frames when tracking')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)