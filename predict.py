from PIL import Image
from ultralytics import YOLO

model = YOLO("runs/detect/train4/weights/best.pt")
#results = model("/Users/hesijun01/PycharmProjects/yolov3-pytorch/data/images/2.jpg")  # 对图像进行预测
# from PIL
im1 = Image.open("/Users/hesijun01/image/dog.jpeg")
results = model.predict(source=im1, save=True)  # save plotted images
names_pre = results[0].names
cls_names = []
for r in results:
    cls_tensor_array = r.boxes.cls
    for cls in cls_tensor_array:
        cls_name = names_pre[int(cls)]
        if cls_name in cls_names:
            print(cls_name + "已经存在")
        else:
            print(cls_name)
