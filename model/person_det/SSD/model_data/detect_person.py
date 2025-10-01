#!/usr/bin/env python3
# coding=utf-8
import warnings
warnings.filterwarnings("ignore")

import os
import cv2
from config import conf

# === Thiết lập GPU (nếu có) ===
os.environ["CUDA_VISIBLE_DEVICES"] = conf["CUDA_VISIBLE_DEVICES"]

# === Load SSD model ===
base_dir = os.path.dirname(__file__)
configPath = os.path.join(base_dir, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
modelPath = os.path.join(base_dir, "frozen_inference_graph.pb")
classesPath = os.path.join(base_dir, "coco.names")

with open(classesPath, "r") as f:
    CLASSES = f.read().strip().split("\n")
CLASSES.insert(0, "__Background__")

net = cv2.dnn.DetectionModel(modelPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# === Skip frame control ===
SSD_PROCESS_INTERVAL = 2  # detect mỗi 2 frame
_SSD_FRAME_COUNTER = 0
_SSD_LAST_ANNOTATED = None  # cache ảnh kết quả đã vẽ


def reset_ssd_skip_state():
    """Reset bộ đếm khi đổi video."""
    global _SSD_FRAME_COUNTER, _SSD_LAST_ANNOTATED
    _SSD_FRAME_COUNTER = 0
    _SSD_LAST_ANNOTATED = None


def ssd_detect_person(img, label="", conf_threshold: float = 0.6):
    """
    Detect người trong ảnh bằng SSD + skip frame.
    - img: ảnh BGR
    - label: text dán khi có đúng 1 người
    - return: ảnh đã annotate
    """
    global _SSD_FRAME_COUNTER, _SSD_LAST_ANNOTATED

    _SSD_FRAME_COUNTER += 1
    if _SSD_FRAME_COUNTER > 1e9:  # tránh tràn số
        _SSD_FRAME_COUNTER = 0

    # Chỉ detect ở frame chia hết cho interval
    is_detection_frame = (_SSD_FRAME_COUNTER % SSD_PROCESS_INTERVAL) == 0

    if is_detection_frame:
        class_ids, confidences, boxes = net.detect(img, confThreshold=conf_threshold, nmsThreshold=0.1)

        annotated_img = img.copy()
        person_boxes = []

        if class_ids is not None and len(class_ids) > 0:
            for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
                if CLASSES[class_id].lower() == "person":
                    x, y, w, h = box
                    person_boxes.append((x, y, w, h))

        # Vẽ box
        for (x, y, w, h) in person_boxes:
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if len(person_boxes) == 1 and label:
            x, y, w, h = person_boxes[0]
            cv2.putText(
                annotated_img, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        _SSD_LAST_ANNOTATED = annotated_img
        return annotated_img

    # Nếu frame skip thì trả về ảnh cache gần nhất
    return _SSD_LAST_ANNOTATED if _SSD_LAST_ANNOTATED is not None else img


# === Demo chạy thử với video/webcam ===
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--source', type=str, default='0', help='0 = webcam hoặc đường dẫn video')
#     parser.add_argument('--output', type=str, default='output_ssd.mp4')
#     args = parser.parse_args()

#     cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if out is None:
#             h, w = frame.shape[:2]
#             out = cv2.VideoWriter(args.output, fourcc, 20.0, (w, h))

#         result_frame = ssd_detect_person(frame, label="User A")
#         out.write(result_frame)
#         cv2.imshow('SSD Person Detection', result_frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     if out:
#         out.release()
#     cv2.destroyAllWindows()
