#!/usr/bin/env python3
# coding=utf-8
import warnings
warnings.filterwarnings("ignore")

import os
from config import conf
os.environ["CUDA_VISIBLE_DEVICES"] = conf["CUDA_VISIBLE_DEVICES"]
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_AGGREGATE"] = "0"

from ultralytics import YOLO
import cv2

# === Load YOLO model ===
device = conf["CUDA_VISIBLE_DEVICES"]
model_path = os.path.join(os.path.dirname(__file__), "yolov8n.onnx")
model = YOLO(model_path)

# Frame skip state (giống SSD)
_frame_counter = 0
_last_boxes = []
_last_label = ""
_last_has_person = False


def yolov8_detect_person(img, label="", skip_frame=2, conf_thres=0.8, iou_thres=0.45):
    """
    Detect person in image using YOLOv8 with frame skipping (giống SSD)
    """
    global _frame_counter, _last_boxes, _last_label, _last_has_person

    _frame_counter += 1
    is_detection_frame = (_frame_counter % skip_frame) == 0

    if is_detection_frame:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model.predict(
            img_rgb,
            conf=conf_thres,
            iou=iou_thres,
            classes=[0],   # chỉ detect person
            device=device,
            verbose=False
        )

        _last_boxes = []
        _last_has_person = False
        _last_label = label

        if results and results[0].boxes:
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                _last_boxes.append((x1, y1, x2, y2))
            _last_has_person = len(_last_boxes) > 0

    # Dùng lại kết quả gần nhất nếu không detect mới
    boxes_to_draw = _last_boxes if _last_has_person else []

    if is_detection_frame and not boxes_to_draw:
        return img

    multiple_person = len(boxes_to_draw) > 1

    for (x1, y1, x2, y2) in boxes_to_draw:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if _last_label and not multiple_person:
            cv2.putText(
                img, _last_label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

    return img


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--source', type=str, default='0', help='0 = webcam hoặc đường dẫn video')
#     parser.add_argument('--output', type=str, default='output.mp4')
#     parser.add_argument('--skip', type=int, default=5, help='Skip frame interval')
#     args = parser.parse_args()

#     cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
#     width, height = int(cap.get(3)), int(cap.get(4))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(args.output, fourcc, 20.0, (width, height))

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         result_frame = yolov8_detect_person(frame, label="person", skip_frame=args.skip)
#         out.write(result_frame)
#         cv2.imshow('YOLOv8 Person Detection', result_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
