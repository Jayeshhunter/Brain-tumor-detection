import cv2 
from PIL import Image
from keras.models import load_model
import numpy as np;

model = load_model('BrainTumor10Epochs.h5')

image = cv2.imread('pred\pred2.jpg')
image = Image.fromarray(image)
image = image.resize((64,64))

image = np.array(image)
input_img = np.expand_dims(image,axis=0)

result = (model.predict(input_img) > 0.5).astype("int32")
if result[0][0] == 0:
    print("Tumor not present")
else:
    print("Tumor present")
# print(image)