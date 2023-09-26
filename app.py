import base64
import json

from PIL import Image
from flask import Flask, request, render_template, make_response
from flask_cors import CORS
from flask_bootstrap import Bootstrap
from ultralytics import YOLO

# flask web service
app = Flask(__name__, template_folder="web")
CORS(app, resources={r"/*": {"origins": "https://yiyan.baidu.com"}})

# web UI
bootstrap = Bootstrap(app)


@app.route('/')
def detect():
    return render_template('detect.html')


@app.route('/detect/process', methods=['post'])
def upload():
    # step 1. receive image
    file = request.files['file']
    file_mame = file.filename
    save_path = "runs/img/" + file_mame

    if not file.filename:
        return render_template('error.html')  # check request

    # step 2. save origin image
    file.save(save_path)

    # step 3. load pre module
    model = YOLO("yolov8n.pt")
    model = YOLO("runs/detect/train4/weights/best.pt")

    im1 = Image.open(save_path)

    # step 4. detect image
    results = model.predict(source=im1, save=True)  # save plotted images

    # step 5. save processed image
    image_path = results[0].save_dir + "/" + file_mame
    with open(image_path, 'rb') as f:
        image_data = f.read()

    # cover imag data with base64 encode
    base64_image = base64.b64encode(image_data).decode('utf-8')

    return render_template('detectOut.html', img=base64_image)


@app.route('/getPicInfo', methods=['post'])
def getPicInfo():
    # step 1. receive image
    file = request.files['file']
    file_mame = file.filename
    save_path = "runs/img/" + file_mame

    if not file.filename:
        return render_template('error.html')  # check request

    # step 2. save origin image
    file.save(save_path)

    # step 3. load pre module
    model = YOLO("yolov8n.pt")
    model = YOLO("runs/detect/train4/weights/best.pt")

    im1 = Image.open(save_path)

    # step 4. detect image
    results = model.predict(source=im1, save=True)  # save plotted images
    predict_names = results[0].names
    cls_names = []
    for r in results:
        cls_tensor_array = r.boxes.cls
        for cls in cls_tensor_array:
            cls_name = predict_names[int(cls)]
            if cls_name in cls_names:
                print(cls_name + "已经存在")
            else:
                cls_names.append(cls_name)
    return make_json_response(cls_names)

def make_json_response(data, status_code=200):
    response = make_response(json.dumps(data), status_code)
    response.headers["Content-Type"] = "application/json"
    return response

@app.route("/.well-known/ai-plugin.json")
async def pluginManifest():
    host = request.host_url
    with open("ai-plugin.json") as f:
        text = f.read().replace("PLUGIN_HOST", host)
        return text, 200, {"Content-Type": "application/json"}

@app.route("/openapi.yaml")
async def openapiSpec():
    host = request.host_url
    with open("openapi.yaml") as f:
        text = f.read().replace("PLUGIN_HOST", host)
        return text, 200, {"Content-Type": "text/yaml"}


if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False, port=8080)
