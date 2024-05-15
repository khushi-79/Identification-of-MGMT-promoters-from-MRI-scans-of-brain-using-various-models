import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import preprocessing
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model =load_model('BrainTumor10EpochsCategorical.h5')

class_model = load_model('model.keras')

print('Model loaded. Check http://127.0.0.1:5000/')


# def get_className(classNo):
# 	if classNo==0:
# 		return "No Brain Tumor"
# 	elif classNo==1:
# 		return "Yes Brain Tumor"

def get_className(classProbabilities):
    classNo = np.argmax(classProbabilities)
    
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"
    else:
        return "Unknown Class"  # Handle the case where there are more classes than expected

def get_classType(classProbabilities):
    classType = np.argmax(classProbabilities)
    
    if classType == 0:
        return "Yes Brain tumor! Type is Glioma"
    elif classType == 1:
        return "Yes Brain tumor! Type is Meningioma"
    elif classType==3:
        return "Yes Brain tumor! Type is Pituitary"
    elif classType==2:
        return "No Brain tumor"
    else:
        return "Unknown Class"  # Handle the case where there are more classes than expected

def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    return result

def getClassify(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = class_model.predict(input_img)
    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result = get_className(value)
        return result
    return None

@app.route('/classify', methods=['GET', 'POST'])
def uploadClassImg():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getClassify(file_path)
        result = get_classType(value)
        return result
    return None

@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

@app.route('/classification')
def classification_page():
    return render_template('classification.html')


if __name__ == '__main__':
    app.run(debug=True)