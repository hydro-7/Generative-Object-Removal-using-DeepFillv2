from flask import Flask, render_template, request, redirect, url_for
import os
import uuid
import cv2
from PIL import Image

from objRemove import ObjectRemove
from models.deepFill import Generator
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models at startup
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
rcnn = maskrcnn_resnet50_fpn(weights=weights, progress=False).eval()
transforms = weights.transforms()

deepfill_weights_path = None
for f in os.listdir('models'):
    if f.endswith('.pth'):
        deepfill_weights_path = os.path.join('models', f)
deepfill = Generator(checkpoint=deepfill_weights_path, return_flow=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    if file.filename == '':
        return 'No image selected', 400

    raw_filename = f"{uuid.uuid4().hex}.jpg"
    raw_path = os.path.join(UPLOAD_FOLDER, f"input_{raw_filename}")
    final_path = os.path.join(UPLOAD_FOLDER, raw_filename)

    img = Image.open(file.stream).convert("RGB")
    img.save(raw_path, format="JPEG")

    model = ObjectRemove(
        segmentModel=rcnn,
        rcnn_transforms=transforms,
        inpaintModel=deepfill,
        image_path=raw_path
    )
    output = model.run()

    cv2.imwrite(final_path, output)  # BGR, no color shift

    os.remove(raw_path)

    return redirect(url_for('result', filename=raw_filename))

@app.route('/result')
def result():
    filename = request.args.get('filename')
    if not filename:
        return redirect('/')
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
