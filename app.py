import base64

from PIL import Image
from flask import Flask, request, render_template
from flask_bootstrap import Bootstrap
from ultralytics import YOLO

# flask web service
app = Flask(__name__, template_folder="web")

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

if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=False, port=8080)
