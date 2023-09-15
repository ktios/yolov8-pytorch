from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load an official model
model = YOLO("runs/classify/train6/weights/best.pt")

results = model.predict('/Users/hesijun01/image/dog.jpeg', save=True)  # save plotted images