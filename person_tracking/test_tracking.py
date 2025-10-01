import cv2
import importlib.util
import os
from .bytetrack_tracker import SORTTracker
from .tracking_utils import crop_person


def main():
    # Import detect_person.py as module
    DETECT_PERSON_PATH = r'D:\gait_reg\model\person_det\SSD\model_data\detect_person.py'
    spec = importlib.util.spec_from_file_location('detect_person', DETECT_PERSON_PATH)
    detect_person = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detect_person)

    tracker = SORTTracker()
    cap = cv2.VideoCapture(r'D:\gait_reg\vlc-record-2025-09-26-10h45m32s-rtsp___117.2.126.196_554_stream1-.avi')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # SSD detection (luôn annotate vào frame nếu có bbox)
        detect_person.ssd_detect_person(frame)

        detections = []
        if detect_person._SSD_LAST_HAS_PERSON and detect_person._SSD_LAST_BOX is not None:
            x, y, w, h = detect_person._SSD_LAST_BOX
            detections.append({'bbox': (x, y, x + w, y + h),
                               'conf': detect_person._SSD_LAST_CONFIDENCE})

        # Update tracker
        tracked_objects = tracker.update(detections, frame)

        # Vẽ kết quả tracking
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {obj.id}',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            # Crop ảnh người
            person_img = crop_person(frame, obj.bbox)
            print(f"Track ID: {obj.id}, BBox: {obj.bbox}, Crop shape: {person_img.shape}")

        cv2.imshow('Tracking Visualization', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
