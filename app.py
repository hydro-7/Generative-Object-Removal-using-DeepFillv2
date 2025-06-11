from flask import Flask, render_template, request, send_file
import os
import uuid
import cv2
import torch
from PIL import Image
from io import BytesIO

from objRemove import ObjectRemove
from models.deepFill import Generator
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models at startup
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
rcnn = maskrcnn_resnet50_fpn(weights=weights, progress=False).eval()
transforms = weights.transforms()

# Load DeepFill model from weights
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

    # Save uploaded image temporarily
    image_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.png")
    file.save(image_path)

    # Inpaint with object removal
    model = ObjectRemove(
        segmentModel=rcnn,
        rcnn_transforms=transforms,
        inpaintModel=deepfill,
        image_path=image_path
    )
    output = model.run()

    # Convert BGR → RGB for PIL
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    output_pil = Image.fromarray(output_rgb)

    # Send inpainted image as downloadable file
    buffer = BytesIO()
    output_pil.save(buffer, format="PNG")
    buffer.seek(0)

    # Clean up uploaded file
    os.remove(image_path)

    return send_file(buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
