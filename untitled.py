import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model('model/digit_recognizer')

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)
            img = prepare_image(file_path)
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction, axis=1)[0]
            return render_template('index.html', prediction=predicted_class, image_path=file_path)
    return render_template('index.html', prediction=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
