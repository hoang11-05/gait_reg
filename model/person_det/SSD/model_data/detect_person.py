import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import numpy as np
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

# Tối ưu: Nếu có CUDA/OpenVINO thì bật backend (tùy môi trường)
try:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
except Exception:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# === Detection skip control (process every N frames) ===
SSD_PROCESS_INTERVAL = 2   # 2 frame bỏ qua, 1 frame detect
_SSD_FRAME_COUNTER = 0
_SSD_LAST_BOX = None
_SSD_LAST_CONFIDENCE = 0.0
_SSD_LAST_HAS_PERSON = False
_SSD_LAST_LABEL = ""


def reset_ssd_skip_state():
    """Reset trạng thái bộ nhớ detection"""
    global _SSD_FRAME_COUNTER, _SSD_LAST_BOX, _SSD_LAST_CONFIDENCE, _SSD_LAST_HAS_PERSON, _SSD_LAST_LABEL
    _SSD_FRAME_COUNTER = 0
    _SSD_LAST_BOX = None
    _SSD_LAST_CONFIDENCE = 0.0
    _SSD_LAST_HAS_PERSON = False
    _SSD_LAST_LABEL = ""


def ssd_detect_person(img, label="", conf_threshold: float = 0.6):
    """
    Phát hiện 1 người trong ảnh bằng SSD, lấy bbox có confidence cao nhất.
    Args:
        img: ảnh BGR đầu vào
        label: text hiển thị trên bbox
        conf_threshold: ngưỡng confidence tối thiểu
    Returns:
        img đã annotate
    """
    global _SSD_FRAME_COUNTER, _SSD_LAST_BOX, _SSD_LAST_CONFIDENCE, _SSD_LAST_HAS_PERSON, _SSD_LAST_LABEL

    _SSD_FRAME_COUNTER += 1
    is_detection_frame = (_SSD_FRAME_COUNTER % SSD_PROCESS_INTERVAL) == 1

    if is_detection_frame:
        class_ids, confidences, boxes = net.detect(
            img, confThreshold=conf_threshold, nmsThreshold=0.1
        )

        has_detection = not (isinstance(class_ids, tuple) or len(class_ids) == 0)
        _SSD_LAST_HAS_PERSON = False
        _SSD_LAST_BOX = None
        _SSD_LAST_CONFIDENCE = 0.0
        _SSD_LAST_LABEL = label

        if has_detection:
            # Tìm bbox person có confidence cao nhất
            best_conf = -1
            best_box = None
            for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
                if CLASSES[class_id].lower() == "person" and confidence > best_conf:
                    best_conf = confidence
                    best_box = box

            if best_box is not None:
                _SSD_LAST_HAS_PERSON = True
                _SSD_LAST_BOX = tuple(best_box)
                _SSD_LAST_CONFIDENCE = float(best_conf)

    # Vẽ bbox nếu có
    if _SSD_LAST_HAS_PERSON and _SSD_LAST_BOX:
        x, y, w, h = _SSD_LAST_BOX
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if _SSD_LAST_LABEL:
            text = f"{_SSD_LAST_LABEL} ({_SSD_LAST_CONFIDENCE:.2f})"
            cv2.putText(
                img,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    return img



# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--source', type=str, default='0', help='0 = webcam hoặc đường dẫn video')
#     parser.add_argument('--output', type=str, default='output_ssd.mp4')
#     args = parser.parse_args()

#     cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(args.output, fourcc, 20.0, (640, 480))

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         result_frame = ssd_detect_person(frame, label="User A")
#         out.write(result_frame)
#         cv2.imshow('SSD Person Detection', result_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
