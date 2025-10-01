import cv2
from yolo_detector import YOLODetector
from bytetrack_tracker import SORTTracker
from tracking_utils import crop_person

def main():
    detector = YOLODetector()
    tracker = SORTTracker()
    cap = cv2.VideoCapture(r'D:\Gait_RECOGNITION\data\upload\tran-hoang\video\ee3621ca.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections, frame)
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            # Vẽ bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Vẽ track ID
            cv2.putText(frame, f'ID: {obj.id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            person_img = crop_person(frame, obj.bbox)
            print(f"Track ID: {obj.id}, BBox: {obj.bbox}, Crop shape: {person_img.shape}")
        cv2.imshow('Tracking Visualization', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()