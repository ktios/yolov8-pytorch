from PIL import Image
from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")
#results = model("/Users/hesijun01/PycharmProjects/yolov3-pytorch/data/images/2.jpg")  # 对图像进行预测
# from PIL
im1 = Image.open("/Users/hesijun01/image/dog.jpeg")
results = model.predict(source=im1, save=True)  # save plotted images
for r in results:
    print(r.boxes)