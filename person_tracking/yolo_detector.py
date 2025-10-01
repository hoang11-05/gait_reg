from ultralytics import YOLO
import cv2

class YOLODetector:
	def __init__(self, model_path='D:/Gait_RECOGNITION/model/person_det/yolov8/yolov8n.onnx', device='cpu'):
		self.model = YOLO(model_path)
		self.model.to(device)
		self.device = device

	def detect(self, frame, conf_thres=0.8):
		results = self.model(frame, conf=conf_thres, classes=[0])  # class 0: person
		detections = []
		for result in results:
			for box in result.boxes:
				x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
				conf = box.conf[0].cpu().numpy()
				detections.append({'bbox': (x1, y1, x2, y2), 'conf': float(conf)})
		return detections
