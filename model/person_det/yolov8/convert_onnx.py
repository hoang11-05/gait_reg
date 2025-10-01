from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO(r"D:\Gait_RECOGNITION\model\person_det\yolov8\yolov8n.pt")

# Export the model to ONNX format
model.export(format="onnx", dynamic=True, half=True, device="cpu") 