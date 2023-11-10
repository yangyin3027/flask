from flask import Flask, jsonify, request, render_template

import io
import os
import signal
import sys
import json

import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

app = Flask(__name__)
imagenet_class_index = json.load(open('imagenet_class_index.json'))
model = models.densenet121(weights='IMAGENET1K_V1')
model.eval()

files_to_delete = set()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # get the file from the request
        file = request.files.get('my_image')
        if file is None or file.filename =='':
            return render_template('index.html', prediction='no image uploaded',
                                   img_path='static/'+file.filename)
        try:
            # convert that to bytes
            img_path = 'static/' + file.filename
            file.save(img_path)
            files_to_delete.add(img_path)
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            class_id, class_name = get_prediction(image_bytes=img_bytes)
            output = {'class_id': class_id,
                            'class_name': class_name}
            
            return render_template('index.html', prediction=output,
                                   img_path=img_path)
        except:
            return render_template('index.html', prediction='error in prediction',
                                   img_path='static/'+file.filename)
    else:
        return render_template('index.html')

def cleanup_files(signum, frame):
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f'Error deleting files: {str(e)}')
    
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_files)

if __name__ == '__main__':
   
    app.run()
    