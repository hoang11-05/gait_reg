#! /usr/bin/python3
# coding=utf-8

import cv2
from .hubconf import custom

model = custom("model/person_det/yolov5/yolov5n.pt")
model.conf = 0.80
model.iou = 0.45
model.classes = [0]

_frame_counter = 0
_last_boxes = []
_last_label = ""
_last_has_person = False


def yolov5_detect_person(img, label='', skip_frame=2):
    global _frame_counter, _last_boxes, _last_label, _last_has_person

    _frame_counter += 1
    is_detection_frame = (_frame_counter % skip_frame) == 0

    if is_detection_frame:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb, size=640)

        _last_boxes = []
        _last_has_person = False
        _last_label = label

        df = results.pandas().xyxy[0]
        persons = df[df['name'] == 'person']

        if not persons.empty:
            _last_has_person = True
            for _, row in persons.iterrows():
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                _last_boxes.append((x1, y1, x2, y2))

            if len(_last_boxes) > 1:
                _last_label = ''

    boxes_to_draw = _last_boxes if _last_has_person else []
    if is_detection_frame and not boxes_to_draw:
        return img

    for (x1, y1, x2, y2) in boxes_to_draw:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if _last_label:
            cv2.putText(img, _last_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return img


if __name__ == '__main__':
    img = cv2.imread('./test.jpg', cv2.IMREAD_COLOR)
    out = yolov5_detect_person(img, label='UserA', skip_frame=2)
    cv2.imwrite("test_out.jpg", out)
