import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10EpochsMultipleClasses.h5')


image = cv2.imread('pred/pred52.jpg')
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

input_img = np.expand_dims(img, axis=0)
result = model.predict(input_img)

predicted_class = np.argmax(result)

print("Predicted class:", predicted_class)
