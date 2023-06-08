import json
import math

import json 
# import _init_paths
import os
import sys
import numpy as np
import pprint
import pdb
import time
import cv2
from PIL import Image
import argparse
import statistics
def area_box(box):
    #input = box [xmin ymin xmax, ymax]
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    return area1

def caculator_person_per_img(mask_person):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    mask_person = cv2.erode(mask_person, kernel, iterations = 2) 
    contours, _  = cv2.findContours(mask_person.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height_person = [0]
    width_person = [0]
    for i in contours:
        x, y, width, height = cv2.boundingRect(i)
        # box_person.append([x, y, x+ width, y + height ])
        height_person.append(height)

        width_person.append(width)
    
    return [max(height_person)*2, max(width_person)*2]

def caculator_box_person(box_person_list):
    area_box_person_list = []
    for box_person in box_person_list:

        area_box_person_list.append(area_box(box_person))
    index = area_box_person_list.index(max(area_box_person_list))
    # print(index)
    box_person_max = box_person_list[index]
    return [int(box_person_max[2]) - int(box_person_max[0]) , int(box_person_max[3]) - int(box_person_max[1]) ]
        

def interpolateObjects(box1, box2, weight):
    box = [int(r1 *weight + r2*(1-weight )) for r1, r2 in zip(box1, box2) ]
    return box

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
def get_box_final(box_detect, box_crop):
    h_boxcrop = box_crop[3] - box_crop[1]
    w_boxcrop = box_crop[2] - box_crop[0]

    return [box_crop[0] + box_detect[0] , box_crop[1] + box_detect[1], box_crop[2] - (w_boxcrop - box_detect[2]) , box_crop[3] - ( h_boxcrop - box_detect[3])]


def dist_box(ob_start, ob_end):
    x1 = (ob_start[0] + ob_start[2])/2
    y1 = (ob_start[1] + ob_start[3])/2

    x2 = (ob_end[0] + ob_end[2])/2
    y2 = (ob_end[1] + ob_end[3])/2
    dist = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
    return dist
def detect_frame_best(box_obj_list, w_vid):

    # box_list = []
    frame_list = []
    area_box_list = []
    H_box_list = []
    W_box_list = []
    W_list = []
    for box_dict in box_obj_list:
        # box_list.append(box_dict["box"])
        box = box_dict["box"]

        w_box = box[2] - box[0]
        h_box = box[3] - box[1]
        H_box_list.append(h_box)
        W_box_list.append(w_box)
        w_center = (box[2] + box[0]) / 2
        W_list.append(w_center)

        area_box = w_box * h_box
        area_box_list.append(area_box)
        frame_list.append(box_dict["fr"])
    area_avr = statistics.mean(area_box_list)
    W_avr = statistics.mean(W_box_list)

    H_avr = statistics.mean(H_box_list)
    frame_best_list = []
    index_best_list = []
    
    for i in range(len(area_box_list)):
        if area_box_list[i] > area_avr * 0.8 and W_box_list[i] < W_avr * 1.2  and W_list[i] > 0.25 * w_vid  and 0.1 * len(area_box_list) < i < 0.85* len(area_box_list)  and H_box_list[i] < 1.1 * H_avr :
        
            index = i
            index_best_list.append(index)
            # frame_best_list.append(frame_list[index])
    if len(index_best_list) > 5:
        step = len(index_best_list) // 5
        index_best_list = [index_best_list[i] for i in range(0, len(index_best_list), step)]
    for index in index_best_list:
        frame_best_list.append(frame_list[index])
    return frame_best_list
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract frames")
    parser.add_argument("-i", "--input_video", default="/media/anlabadmin/setup/dungtd/carton/video_test/2004008_01_PXL_20230125_223312522.mp4")
    parser.add_argument("-o", "--output_save", default="/media/anlabadmin/setup/dungtd/carton/mask_video/")
    
    args = parser.parse_args()

    data_folder = args.output_save

    path_video = args.input_video
    name_video = path_video.split("/")[-1][:-4]

    path_save_track = os.path.join("./results_track", str(name_video))

    # Opening JSON file
    folder_output = os.path.join(data_folder, name_video)
    folder_images = os.path.join(folder_output, 'images')

    folder_mask_person = os.path.join(folder_output, 'mask_person')

    #load box person per frame
    file_box_person = os.path.join(path_save_track, "box_person_" + name_video + ".json")
    dict_box_person = {}
    with open(file_box_person, 'r', encoding='utf-8') as p:
        dict_box_person = json.load(p)
    
    #load box merge dic:
    file_box_merge = os.path.join('0505merge_video_2004008_02_IMG_0837.json')
    dict_box_merge = {}
    with open(file_box_merge, 'r', encoding='utf-8') as p:
        dict_box_merge = json.load(p)

    #path save video output
    file_name = os.path.join(path_save_track, name_video + ".json")
    save_path =  os.path.join(path_save_track, name_video + ".avi")  # force *.mp4 suffix on results videos
    
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (1280, 720))
    print('path_video', path_video)
    cap = cv2.VideoCapture(path_video)
    frame_ids = {}
    w_vid  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # returns JSON object as 
    # a dictionary
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    count_object = 0 
    dict_ids = {}
    size_ids = {}
    frame_results = {}
    dict_label_frame = {}

    end_frame = 0
    box_check = None

    direction_count = 0
    count = 0
    for id in data.keys(): #id = id box
        count += 1
        objs = data[id]
        num_data = len(objs)
        # print('num_data', num_data)
        if num_data >= 3:
            # print('num_data', num_data, objs)
            ob_start = objs[0]["box"]
            ob_end = objs[num_data -1 ]["box"]

            x1 = (ob_start[0] + ob_start[2])/2
            x2 = (ob_end[0] + ob_end[2])/2

            if x2 - x1 > 0:
                direction_count += 1
    if direction_count / count > 0.5:
        direction = 1
    else:
        direction = -1
    
    print('direction', direction) #direction = direction of video

    for id in data.keys(): #id = id box
        objs = data[id]
        num_data = len(objs)
        # print('num_data', num_data)
        if num_data >= 5:
            # print('num_data', num_data, objs)
            # print("objs ", id , objs)
            ob_start = objs[0]["box"]
            ob_end = objs[num_data -1 ]["box"]

            x1 = (ob_start[0] + ob_start[2])/2
            y1 = (ob_start[1] + ob_start[3])/2

            x2 = (ob_end[0] + ob_end[2])/2
            y2 = (ob_end[1] + ob_end[3])/2
            dist = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
            direction_check = x2 - x1
            # print("objs ", id  ,num_data, dist)
            # print('dist',id, dist, dist/num_data, direction_check, direction)
            # if dist > 30 and direction * direction_check >= 0 and -200 < y2 - y1 < 200 :
            # exit()
            if dist > 40 and direction * direction_check >= 0:
                size_data = [0 , 0]

                for ob in objs:
                    frame_idx_str = ob["fr"]
                    frame_idx = int(frame_idx_str)
                    box = ob["box"]
                    size_data = [size_data[0] + box[2] - box[0] ,size_data[1] + box[3] - box[1] ]
                size_data= [int(size_data[0]/num_data) ,int(size_data[1]/num_data) ]
                frame_best = objs[0]["fr"]
                area_max = 10000
                for ob in objs:
                    frame_idx_str = ob["fr"]
                    frame_idx = int(frame_idx_str)
                    box = ob["box"]
                    dist_est = abs((box[2] - box[0])*(box[3] - box[1]) - (size_data[0]*size_data[1]))
                    if dist_est < area_max:
                        area_max = dist_est
                        frame_best = int(frame_idx)

                print('check', id, frame_best, dist)
         
                count_object += 1
                index_start = int(objs[0]["fr"])

                print("index start ",objs[0]["fr"] , objs[num_data-1]["fr"], end_frame)

                ob_start = objs[0]["box"]
                ob_end = objs[-1]["box"]
                # print("ob_start - end:", ob_start, ob_end, box_check)
                #add rule logic loc box
                # if objs[0]["fr"] > end_frame + 35 and ob_start[2] > 640 and ob_end[3] < 700:
                if objs[0]["fr"] > end_frame + 50 :
                    # count_object -= 1
                    dist_frame = int(objs[0]["fr"] - end_frame)
                    end_frame = objs[num_data-1]["fr"]
                    box_check = ob_end

                else:
                    dist_frame = int(objs[0]["fr"] - end_frame)
                    end_frame = objs[num_data-1]["fr"]

                    if dist_frame < 0:
                        count_object -= 1
                    else:
                        if box_check != None:
                            print("check ob_start=====================================", ob_start, box_check)
                            print('dist_frame',count_object, dist_frame, dist_box(ob_start, box_check),  bb_intersection_over_union(ob_start, box_check))
                            if (dist_box(ob_start, box_check) < 200 and  (direction * 640* (direction-1) + ob_start[0]) >  direction * 200) or bb_intersection_over_union(ob_start, box_check) > 0 :
                                box_check = ob_end
                                count_object -= 1
                            else:
                                box_check = ob_end

                        else:
                            box_check = ob_end
                            
                        # count_object -= 1
                print('check_object_box:', count_object)
                if str(count_object) in dict_label_frame.keys():
                    dict_label_frame[str(count_object)].extend(objs)
                else:
                    dict_label_frame[str(count_object)] = objs
                for idx , ob in enumerate(objs):
                    # print('idx', idx, ob)
                    frame_idx_str = ob["fr"]
                    frame_idx = int(frame_idx_str)
                    box = ob["box"]
                    if frame_idx > index_start + 1:
                        box_current = objs[idx-1]["box"]
                        # print('count_object', count_object , frame_idx , index_start + 1, box_current)
                        for k in range(index_start +1 , frame_idx): 
                            w = 1 - (k - index_start)/(frame_idx -index_start )
                            box_inter = interpolateObjects(box_current, box, w)
                            ob_frame = {}
                            if k in frame_ids.keys():
                                ob_frame = frame_ids[k]
                            ob_frame[count_object] = box_inter
                            frame_ids[k] = ob_frame
                        
                    ob_frame = {}
                    if frame_idx in frame_ids.keys():
                        ob_frame = frame_ids[frame_idx]
                    ob_frame[count_object] = box
                    frame_ids[frame_idx] = ob_frame
                    index_start = frame_idx

                ifo_data = {}
                ifo_data["size"] = size_data
                ifo_data["frame"] = frame_best
                size_ids[count_object] = ifo_data

    f.close()
    cout_data = 0
    num_obj = 0
    list_putext = {}
    fontScale = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    folder_results = "results_carton/" +  name_video

    if not os.path.exists(folder_results):
        os.makedirs(folder_results)

    carton_size = 1/1
    while (1):
        cout_data += 1

        ret, frame = cap.read()
        img_path = "%d.png" % cout_data

        try:
            h_image = frame.shape[0]
            w_image = frame.shape[1]
            # print( "check height, weight ==" , h_image, w_image)
            frame_ori = frame.copy()
            
        except:
            print("frame_error")
        if frame is None:
            break
        if cout_data in frame_ids.keys():

            objs_frame = frame_ids[cout_data]
            box_list = []
            id_list = []
            for id in objs_frame.keys():

                box_imgmerge = dict_box_merge[str(id)]
                # print("check dict_box_imgmerge======", box_imgmerge)
                list_box_merge = box_imgmerge['box']
                list_img_merge = box_imgmerge['img'] 

                box = objs_frame[id]

                size_ob = size_ids[id]["size"]
                fr_best = size_ids[id]["frame"]
                ids_obj_list = box_imgmerge['obj_id']
                try:
                    fr_best = detect_frame_best(dict_label_frame[str(id)], w_vid)[-2]
                except:
                    fr_best = size_ids[id]["frame"]

                w = (box[2] - box[0])
                h = (box[3] - box[1])
             
                x_center = (box[0] + w/2)/frame.shape[1]
                y_center = (box[1] + h/2)/frame.shape[0]
                w = w/frame.shape[1]
                h = h/frame.shape[0]
               
                frame = cv2.rectangle(frame, (int(box[0]) , int(box[1])), (int(box[2]) , int(box[3])), (255 , 255 , 255 ), 7)

                #draw merge box in video
                frame = cv2.putText(frame, f'{id}', (int(box[0]) , int(box[1])), font, fontScale, (0 , 0 , 174), 2, cv2.LINE_AA)
                num_obj = max(ids_obj_list)
                
                if id not in list_putext.keys():
                    d_id = {}
                    d_id["count"] = 1
                    d_id["size"] = size_ob
                    list_putext[id] = d_id
                if fr_best == int(cout_data):
                    im_object = frame_ori[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    box_person = caculator_box_person(dict_box_person[str(id)])
                    path_save_obj = os.path.join(folder_results, f"objectId_{id}_size{round(float(size_ob[0]/box_person[0]), 2)}x{round(float(size_ob[1]/box_person[1]), 2)}.jpg")
                    print("check id img",path_save_obj, im_object.shape)
                    
                    cv2.imwrite( path_save_obj , im_object)
        # frame = cv2.putText(frame, f'frame id : {cout_data}', (50, 50), font, fontScale, (0 , 200 , 200),3 , cv2.LINE_AA)
        frame = cv2.putText(frame, f'#objects : {len(list_putext.keys())}', (50, 50), font, fontScale, (0 , 200 , 200),2 , cv2.LINE_AA)
        
        y_size = 50
        for key_text in list_putext.keys():
            if list_putext[key_text]["count"] < 120:
                list_putext[key_text]["count"] += len(ids_obj_list)
                size_ob = list_putext[key_text]["size"]
                y_size += 70
        frame_rsize =  cv2.resize(frame, (1280, 720))

        vid_writer.write(frame_rsize)
    vid_writer.release()