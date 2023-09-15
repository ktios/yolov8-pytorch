from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official detection model
model = YOLO('yolov8n-seg.pt')  # load an official segmentation model
model = YOLO('runs/detect/train4/weights/best.pt')  # load a custom model

# Track with the model
#results = model.track(source="https://youtu.be/Zgi9g1ksQHc", show=True)
results = model.track(source="/Users/hesijun01/video/travel.mp4", show=True, tracker="bytetrack.yaml")
