"""
Flask application for web interface.
"""

from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'frontend/static/uploads'

# Load the custom trained model
MODEL_PATH = r"C:\Users\jnb20\Desktop\Code\School\AI Project\models\traffic_light_model.pth"


def load_model():
    """Load the custom model and its state dict."""
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.Linear(model.fc.in_features, 2)
    )
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(img, image_size=(224, 224)):
    """Preprocess the input image to match the model input."""
    img = img.convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor


def does_img_contain_traffic_light(img, model):
    """Predict if the image contains a traffic light."""
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        output = model(img_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = probabilities.argmax(dim=1).item()
    return predicted_class == 1


# Load the model once when the app starts
model = load_model()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part in the request', 400
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            img = Image.open(file_path)
            result = does_img_contain_traffic_light(img, model)
            return render_template('result.html', result=result, filename=file.filename)
    return render_template('upload.html')


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
