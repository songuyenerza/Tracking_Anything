import os
import cv2
from PIL import Image
from GroundingDINO.test_end2end import GroundDino_clas
import torch
import math
import re
import json
import glob

def crop_center(image_pill, scale=0.8):

    with image_pill as img:
        width, height = img.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height

        cropped_img = img.crop((left, top, right, bottom))

    return cropped_img

def is_image_close_to_white(img_pill, threshold = 150):   #150

    with img_pill as img:
        img = img.convert('RGB')
        # Get size image
        width, height = img.size
        total_r, total_g, total_b = 0, 0, 0
        for x in range(width):
            for y in range(height):
                r, g, b = img.getpixel((x, y))
                total_r += r
                total_g += g
                total_b += b
        avg_r = total_r / (width * height)
        avg_g = total_g / (width * height)
        avg_b = total_b / (width * height)
    
    white = (255, 255, 255)
    diff_r = abs(avg_r - white[0])
    diff_g = abs(avg_g - white[1])
    diff_b = abs(avg_b - white[2])
    # print("check color ======",diff_r, diff_g, diff_b)
    if diff_r <= threshold and diff_g <= threshold and diff_b <= threshold:
        return True
    else:
        return False

def xy2xywh(box):
    w_ = box[2] - box[0]
    h_ = box[3] - box[1]
    return [int(box[0] + w_/2), int(box[1] + h_/2), w_, h_]

def xy2xywh_nom(box, w_img, h_img):  #add nomalize
    w_ = box[2] - box[0]
    h_ = box[3] - box[1]
    return [(box[0] + w_/2)/w_img, (box[1] + h_/2)/h_img, w_/w_img, h_/h_img]

def iou(box1, box2):
    # box1 and box2 are lists of [x1, y1, x2, y2] coordinates
    intersection = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * \
                   max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # union = area1 + area2 - intersection
    union = min(area1, area2)
    iou = intersection / union if union > 0 else 0
    return iou

def get_object_id(filename):
    return int(filename.split("_")[1])

if __name__ == "__main__":

    #load model
    #limited gpu
    # gpu_fraction = 0.5
    # torch.cuda.set_per_process_memory_fraction(gpu_fraction)
    #load model
    TEXT_PROMPT = "carton box"
    BOX_TRESHOLD = 0.25
    TEXT_TRESHOLD = 0.2
    config_file = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" # change the path of the model config file
    checkpoint_path = "./GroundingDINO/CP/groundingdino_swint_ogc.pth"  # change the path of the model
    model_dino = GroundDino_clas( config_file, checkpoint_path)
    print("==============load done model==========")
    #finish load model

    data_folder = "./video_output_1204/2004008_01_PXL_20230125_223312522/"
    folder_check = "./video_output_1204/output_tach_box_2004008_01_PXL_20230125_223312522/"

    #create path folder_check
    if not os.path.exists(folder_check):
        os.mkdir(folder_check)
    count = len(os.listdir(data_folder))
    print(count)
    obj_ids = 0
    # exit()
    dict_box = {}
    list_img =  os.listdir(data_folder)

    for path in sorted(list_img, key=get_object_id):
        ids = int(path.split("_")[1])
        obj_id_list = []
        img_path = os.path.join(data_folder, path)
        image = cv2.imread(img_path)
        print("shape image: ", path,  image.shape)
        H, W = image.shape[0], image.shape[1]
        ratio_img_ori = round(H/W, 3)
        area_image = H*W
        image_pill = Image.fromarray(image).convert("RGB")
        list_box_carton_nom = []
        dict_box_img = {}
        if is_image_close_to_white(crop_center(image_pill)) == True:

            box_carton_list =  model_dino.detect_carton(image_pill, BOX_TRESHOLD = 0.22, TEXT_TRESHOLD = 0.3)   
            H_box_list = []
            box_carton_final = []
            for box_carton in box_carton_list:
                area_box = (box_carton[2] - box_carton[0]) * (box_carton[3] - box_carton[1])
                
                if  0.2 * area_image < area_box < 0.9 * area_image and 0.2 * H <  (box_carton[3] - box_carton[1]) < 0.8 * H and box_carton[2] - box_carton[0] > 0.7* W:
                    # print(box_carton)
                    H_box_list.append(box_carton[3] - box_carton[1])
                    box_carton_final.append(box_carton)

            if len(H_box_list) > 0:
                ratio = round(H / min(H_box_list))
                print("ratio:",round(ratio))
                # H_box_list_new = []
                box_carton_final_new = []
                for i in range(len(H_box_list)):
                    h_box = H_box_list[i]
                    if h_box / min(H_box_list) < 1.4:
                        box_carton_final_new.append(box_carton_final[i])
                print(box_carton_final_new)
                area_box_list = []
                for box in box_carton_final_new:
                    area_box = (box[2] - box[0]) * (box[3] - box[1])
                    area_box_list.append(box)
                
                box_max = box_carton_final_new[area_box_list.index(max(area_box_list))]

                if ratio == len(box_carton_final_new):
                    for i in range(len(H_box_list)):
                        count += 1
                        box_car = box_carton_final[i]
                        img_crop = image[box_car[1]: box_car[3], box_car[0] : box_car[2]]
                        # cv2.rectangle(image, (box_car[0], box_car[1]), (box_car[2], box_car[3]), (255,0,0), 2)     #visual all box carton 
                        # print("box=====", box_car)
                        box_nom = xy2xywh_nom(box_car, W, H)
                        list_box_carton_nom.append(box_nom)

                        path_save = re.sub(r'_\d+_', '_{}_'.format(count), path)
                        obj_ids += 1
                        path_save = f"objectId_{obj_ids}.jpg"
                        obj_id_list.append(obj_ids)
                        cv2.imwrite(os.path.join(folder_check, path_save), img_crop)
                
                else:
                    # img_crop = image[box_max[1]: box_max[3], box_max[0] : box_max[2]]
                    # cv2.rectangle(image, (box_max[0], box_max[1]), (box_max[2], box_max[3]), (255,0,0), 2)     #visual all box carton 
                    
                    box_max_xywh = xy2xywh(box_max)
                    h_box = box_max_xywh[3]
                    coords = []
                    print( "check box_max ======" , box_max, box_max_xywh)
                    for i in range(ratio):
                        h_b = int(H / ratio)
                        x1 = h_b * i
                        y1 = 0
                        x2 = h_b * (i + 1)
                        y2 = W
                        box = [y1, x1, y2, x2]
                        coords.append([y1, x1, y2, x2])
                        iou_score = iou(box_max, box)
                        if iou_score > 0.7:
                            coords[i] = box_max
                    
                    # print(coords)
                    for i, box_ in enumerate(coords):
                        count += 1
                        # cv2.rectangle(image, (box_[0], box_[1]), (box_[2], box_[3]), (255,0,0), 2)     #visual all box carton 
                        img_crop = image[box_[1]: box_[3], box_[0] : box_[2]]
                        list_box_carton_nom.append(xy2xywh_nom(box_, W, H))

                        
                        path_save = re.sub(r'_\d+_', '_{}_'.format(count), path)
                        obj_ids += 1
                        path_save = f"objectId_{obj_ids}.jpg"
                        obj_id_list.append(obj_ids)

                        cv2.imwrite(os.path.join(folder_check, path_save), img_crop)
                        # print("save ok=========")
            
            else:
                box = [0, 0, W, H]
                list_box_carton_nom.append(xy2xywh_nom(box, W, H))
                obj_ids += 1
                path = f"objectId_{obj_ids}.jpg"
                obj_id_list.append(obj_ids)

                cv2.imwrite(os.path.join(folder_check, path), image)

        else:
            box = [0, 0, W, H]
            list_box_carton_nom.append(xy2xywh_nom(box, W, H))
            image = image[0: H, 0:W]

            obj_ids += 1
            path = f"objectId_{obj_ids}.jpg"
            obj_id_list.append(obj_ids)

            cv2.imwrite(os.path.join(folder_check, path), image)
        
        folder_match = "/media/anlabadmin/Data/SonG/moving-recog/results_carton/result_merge/0505/"
        list_img_matching = []
        for id_match in obj_id_list:
            img_match = os.path.join(folder_match, str(id_match) + '_')
            list_img = glob.glob1(img_match,"*.jpg") 
            if len(list_img) == 2:
                img_match = list_img[1]
                print(ids, "img_match",img_match)
                img_match = img_match.split('2004008_02_IMG_0837')[-1]
                list_img_matching.append(img_match)

        dict_box_img['box'] = list_box_carton_nom
        dict_box_img['ratio'] = ratio_img_ori
        dict_box_img['img'] = list_img_matching
        dict_box_img['obj_id'] = obj_id_list
        dict_box[ids] = dict_box_img

    print("check =======",dict_box )
    name_save_dict = "0505merge_video_2004008_01_PXL_20230125_223312522"
    with open(name_save_dict + ".json", "w") as outfile:
        json.dump(dict_box, outfile)
