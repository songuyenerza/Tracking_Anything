GrDino_path = "./GroundingDINO"
import sys
sys.path.append(GrDino_path)
import argparse
import os

import cv2
import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def load_image(image_pil):
    # load image

    transform = T.Compose(
        [
            T.RandomResize([600], max_size=1333), #600  #700 #800
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_image_person(image_pil):
    # load images

    transform = T.Compose(
        [
            T.RandomResize([700], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location= args.device)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, cpu_only=False):

    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    # print("check image[None]", image[None].shape, image.shape)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # print("caption", caption)
    # get phrase
    tokenlizer = model.tokenizer
    t0 = time.time()
    tokenized = tokenlizer(caption)

    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def get_grounding_multi_output(model, images, caption, box_threshold, text_threshold, with_logits=True, cpu_only=False):
    # print(images)
    images = torch.FloatTensor(images)
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    images = images.to(device)

    print("check image[None]", images.shape)

    with torch.no_grad():
        outputs = model(images, captions=[caption])

    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # print("caption", caption)
    # get phrase
    # tokenlizer = model.tokenizer
    # t0 = time.time()
    # tokenized = tokenlizer(caption)

    # build pred
    pred_phrases = []
    # for logit, box in zip(logits_filt, boxes_filt):
    #     pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
    #     if with_logits:
    #         pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
    #     else:
    #         pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

class GroundDino_clas():
    def __init__(self, config_file, checkpoint_path):

        #load model
        self.model = load_model(config_file, checkpoint_path, cpu_only= False)

    def predict(self, image_pil, TEXT_PROMPT = "carrying a object, bring object, bringing object", BOX_TRESHOLD = 0.25, TEXT_TRESHOLD = 0.25 ):
        image_pil, image = load_image(image_pil)
        boxes_filt, pred_phrases = get_grounding_output(
        self.model, image, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD, cpu_only= False
        )

        return boxes_filt, pred_phrases
    

    def perdict_batch_img(self, list_image_pil, TEXT_PROMPT = "carrying a object, bringing object", BOX_TRESHOLD = 0.25, TEXT_TRESHOLD = 0.4):
        #input = list img_pil rgb
        list_imgpill_input = []
        list_image_input = []
        for image_pill in list_image_pil:
            image_pil, image = load_image(image_pill)
            list_imgpill_input.append(image_pil)

            # print("image=", image.shape)
            list_image_input.append(image.numpy())
        boxes_filt, pred_phrases = get_grounding_multi_output(
                self.model, list_image_input, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD, cpu_only= False
                )
        
        return boxes_filt, None

    def predict_box(self, image_pil, TEXT_PROMPT = "carrying a object, bring object", BOX_TRESHOLD = 0.25, TEXT_TRESHOLD = 0.35):
        #check out thresh = 0.22 - 0.23 - 0.25
        #output = list boxes [xmin ymin xmax ymax]
        image_pil, image = load_image(image_pil)
        boxes_filt, pred_phrases = get_grounding_output(
        self.model, image, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD, cpu_only=False
        )
        size = image_pil.size
        # size_HW = [size[1], size[0]]
        H = size[1]
        W = size[0]
        boxes = boxes_filt
        box_final_list = []
        for box in boxes:
            box = box * torch.Tensor([W, H, W, H])
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            box_final = [x0, y0, x1, y1]
            if (x1 - x0)* (y1 - y0) > 0.01 * H * W:

                box_final_list.append(box_final)

        return box_final_list 
    
    def detect_carton(self, image_pil, TEXT_PROMPT = "carton box", BOX_TRESHOLD = 0.25, TEXT_TRESHOLD = 0.3):  #0.23
        
        #output = list boxes [xmin ymin xmax ymax]
        image_pil, image = load_image(image_pil)
        boxes_filt, pred_phrases = get_grounding_output(
        self.model, image, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD, cpu_only= True
        )
        size = image_pil.size
        # size_HW = [size[1], size[0]]
        H = size[1]
        W = size[0]
        boxes = boxes_filt
        box_final_list = []
        for box in boxes:
            box = box * torch.Tensor([W, H, W, H])
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            box_final = [x0, y0, x1, y1]
            if (x1 - x0)* (y1 - y0) > 0.01 * H * W:
                box_final_list.append(box_final)
        return box_final_list

    def predict_person(self, image_pil, TEXT_PROMPT = "person, human", BOX_TRESHOLD = 0.1, TEXT_TRESHOLD = 0.2):
        #detect person
        #output = list boxes [xmin ymin xmax ymax]
        image_pil, image = load_image_person(image_pil)
        boxes_filt, pred_phrases = get_grounding_output(
        self.model, image, TEXT_PROMPT, BOX_TRESHOLD, TEXT_TRESHOLD, cpu_only=False
        )
        size = image_pil.size
        # size_HW = [size[1], size[0]]
        H = size[1]
        W = size[0]
        boxes = boxes_filt
        box_final_list = []
        for box in boxes:
            box = box * torch.Tensor([W, H, W, H])
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            box_final = [x0, y0, x1, y1]
            if (x1 - x0) * (y1 - y0) > 0.02 * H * W:
                box_final_list.append(box_final)
        return box_final_list


if __name__ == "__main__":
    gpu_fraction = 0.5
    torch.cuda.set_per_process_memory_fraction(gpu_fraction)
    #load model
    TEXT_PROMPT = "bring object"
    BOX_TRESHOLD = 0.25
    TEXT_TRESHOLD = 0.2

    config_file = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py" # change the path of the model config file
    checkpoint_path = "./GroundingDINO/CP/groundingdino_swint_ogc.pth"  
    # model = load_model(config_file, checkpoint_path, cpu_only= False)
    print("load model finish")
    model = GroundDino_clas( config_file, checkpoint_path)
    #Detect
    data_folder = "/media/anlabadmin/Data/SonG/moving-recog/results_carton/frame_debug/"
    folder_check = "/media/anlabadmin/Data/SonG/moving-recog/results_carton/check_frame_debug/"
    for path in os.listdir(data_folder):
        print("path", path)
        img_path = os.path.join(data_folder, path)

        image_pil = Image.open(img_path).convert("RGB")

        # image_pil, image = load_image(image_pil)
        t0 = time.time()

        # run model

        boxes_filt, pred_phrases = model.perdict_batch_img([image_pil ])

        print("time per img = ", time.time() - t0)
        # # visualize pred
        # size = image_pil.size
        # pred_dict = {
        #     "boxes": boxes_filt,
        #     "size": [size[1], size[0]],  # H,W
        #     "labels": pred_phrases,
        # }

        # image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        # path_save = os.path.join(folder_check, path)

        #if want save to check output
        # image_with_box.save(path_save)

        break

