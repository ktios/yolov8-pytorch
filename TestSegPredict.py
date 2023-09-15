from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('runs/segment/train3/weights/best.pt')  # load a custom model

# Predict with the model
results = model('/Users/hesijun01/image/seg/000000000030.jpg', save=True)  # predict on an image
