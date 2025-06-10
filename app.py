
import numpy as np
from PIL import Image
import cv2
import json

# TensorFlow and tf.keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from flask import Flask, request, render_template

model = MobileNetV2(weights='imagenet')

def model_predict(img, model):
 img = img.resize((224, 224))
 x = image.img_to_array(img)
 x = np.expand_dims(x, axis=0)
 x = preprocess_input(x, mode='tf')
 preds = model.predict(x)
 return preds

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=["GET","POST"])
def predict():
    file = request.files['file']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("image.jpeg", img)

    img = Image.open("./image.jpeg")
    
    preds = model_predict(img, model)
    pred_proba = "{:.3f}".format(np.amax(preds))
    pred_class = decode_predictions(preds, top=1)
    result = str(pred_class[0][0][1])
    result = result.replace('_', ' ').capitalize()
    return json.dumps({"result":result, "probability":pred_proba})
    
    return "API Classification d'image"

if __name__ == '__main__':
    app.run("0.0.0.0", port=5000, debug=True)

