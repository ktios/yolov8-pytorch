from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model
model = YOLO('runs/pose/train/weights/best.pt')  # load a custom model

# Predict with the model
results = model('/Users/hesijun01/image/bus.jpg', save=True)  # predict on an image
print(results)
print("++++++++++++++++++++++++++++++++++++++++++")
for r in results:
    print(r.boxes)
    print("=====================================")
    print(r.keypoints)
