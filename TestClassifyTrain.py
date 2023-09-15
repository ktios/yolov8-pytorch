from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='mnist160', epochs=5, imgsz=64, device="mps")
# 评估模型
metrics = model.val()
# 导出模型
model.export(format="onnx")
