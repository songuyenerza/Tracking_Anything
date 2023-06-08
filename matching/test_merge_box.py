def iou(box1, box2):
    """
    Calculates the intersection over union (IoU) of two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    iou = intersection / float(area1 + area2 - intersection)
    return iou

def merge_boxes(boxes, iou_threshold=0.7):
    """
    Merges bounding boxes in a list that have an IoU greater than a specified threshold.
    """
    merged_boxes = []
    boxes = sorted(boxes, key=lambda x: x[0])
    while len(boxes) > 0:
        current_box = boxes[0]
        del boxes[0]
        merged_box = current_box
        i = 0
        while i < len(boxes):
            if iou(merged_box, boxes[i]) > iou_threshold:
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

# Example usage:
boxes = [[366, 231, 523, 610], [580, 478, 860, 864], [366, 231, 522, 610]]
merged_boxes = merge_boxes(boxes, iou_threshold=0.7)
print(merged_boxes)


box1 = [553, 266, 776, 537]
box2 = [566, 365, 776, 534]
print(iou(box1, box2))
exit()

box_carton_list =  [[594, 374, 830, 568]]
box_person_list =   [[1572, 216, 1852, 451], [1571, 269, 1733, 449], [534, 5, 673, 390], [685, 267, 929, 645]]

for i , box_carton in enumerate(box_carton_list):
    check = 0
    for box_person in box_person_list:
        print(iou(box_carton, box_person), box_person)
        if iou(box_carton, box_person) > 0.9:
            check += 1
    print(i, box_carton, check)
    if check > 0:
        del box_carton_list[i]
print(box_carton_list)