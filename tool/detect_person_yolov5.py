yolo_path = "../yolov5"
from distutils.log import error
import sys
from tkinter import W, Image
from turtle import window_height
import cv2
import os
import numpy as np
sys.path.append(yolo_path)
from utils.augmentations import letterbox
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, time_sync
from models.common import DetectMultiBackend
import torch
import torch.backends.cudnn as cudnn

import timeit
from PIL import Image
import time
def maxx(box):
    conf_list = []
    if len(box) > 1:
        for b in box:
            conf_list.append(b[5])
        index = np.argmax(conf_list)
        return [box[index]]
    else:
        return box

def convert_box(box, img_width, img_height, cls):    #output  = [xmin, ymin, xmax, ymax]
    x0 = int((box[0] - ((box[2]) / 2)) * img_width)
    y0 = int((box[1] - ((box[3]) / 2)) * img_height)
    x1 = int((box[0] + ((box[2]) / 2)) * img_width)
    y1 = int((box[1] + ((box[3]) / 2)) * img_height)
    if x0<0:
        x0 = 0
    if y0<0:
        y0 = 0
    return [x0, y0, x1, y1]
def convert_box_no(box, img_width, img_height, cls, conf):
    x0 = int(box[0] * img_width)
    y1 = int(box[1] * img_height)
    w = int(box[2] * img_width)
    h = int(box[3] * img_height)
    conf = conf.cpu().data.numpy()
    return [x0, y1, w, h, cls, float(conf)]

@torch.no_grad()
def load_model_detect_person(weights="",  # model.pt path(s)
        data='../yolov5/data/coco.yaml',  # dataset.yaml path
        imgsz=[640, 640],  # inference size (height, width)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    model.warmup(imgsz=(1 , 3, *imgsz))  # warmup
    # print("device",device)
    return model,device
@torch.no_grad()
def detect_person(model,
        device,
        source,  # file/dir/URL/glob, 0 for webcam
        imgsz=[640,640],  # inference size (height, width)
        conf_thres=0.2,  # confidence threshold
        iou_thres=0.7,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        ):
    
    # Load model
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    im0s = source
    img = letterbox(im0s, imgsz, stride=stride, auto=pt)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(img)

    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im)
    im0s = source
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    result=[]
    # Process predictions
    for i, det in enumerate(pred):  # per image
        im0= im0s.copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        box_image=[]
        box_image_no = []
        # print(det[:, :4])
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            box_image=[]
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                line=(('%g ' * len(line)).rstrip() % line)
                line=line.split(" ")

                line= [float(value) if i!=0 else int(value) for i,value in enumerate(line)]
                cls=line[0]
                box=convert_box(line[1:],im0.shape[1],im0.shape[0], cls)
                # box_no = convert_box_no(line[1:],im0.shape[1],im0.shape[0], cls, conf)
                # if box[0] > int(im0.shape[1]*0.02):
                    # if int(im0.shape[1]*0.1) < box[2] < int(im0.shape[1]*0.9):
                if cls == 0:
                    box_image.append(box)
                    # box_image_no.append(box_no)
       

    return box_image


class yolo_detect_person():
    def __init__(self, weight = './weights/yolov5l.pt'):
        self.model, self.device =  load_model_detect_person( weights = weight)
        
    def detect_person(self,
        source,  # file/dir/URL/glob, 0 for webcam
        imgsz=[736,736],  # inference size (height, width)
        conf_thres=0.05,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        ):
        model = self.model
        device = self.device
        # Load model
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        im0s = source
        img = letterbox(im0s, imgsz, stride=stride, auto=pt)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)

        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = model(im)
        im0s = source
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        result=[]
        # Process predictions
        for i, det in enumerate(pred):  # per image
            im0= im0s.copy()
            H = im0.shape[0]
            W = im0.shape[1]

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            box_image=[]
            box_image_no = []
            # print(det[:, :4])
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                box_image=[]
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh)  # label format
                    line=(('%g ' * len(line)).rstrip() % line)
                    line=line.split(" ")

                    line= [float(value) if i!=0 else int(value) for i,value in enumerate(line)]
                    cls=line[0]
                    box=convert_box(line[1:],im0.shape[1],im0.shape[0], cls)

                    if cls == 0:    # 0: id of class person
                        if (box[2] - box[0]) * (box[3] - box[1]) > 0.02 * H * W :
                            box_image.append(box)
        
        return box_image

if __name__ == "__main__":

    weight = "./weights/yolov5m.pt"
    #link model: https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l.pt

    torch.cuda.set_per_process_memory_fraction(0.2, 0)

    # ''' test model in one image'''
    img_path = "/media/anlabadmin/Data/SonG/Moving_recog_V2/GroundingDINO/pred.jpg"
    img_ori = cv2.imread(img_path)
    detect_person = yolo_detect_person(weight='./weights/yolov5m.pt')
    box_list = detect_person.detect_person(img_ori)
    print(box_list)
