import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.python.keras.utils.data_utils import Sequence
import autokeras as ak
import os
from PIL import Image, ImageSequence, ImageDraw
from sklearn.model_selection import train_test_split
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train[:100]
# y_train = y_train[:100]
# print(x_train)  # (60000, 28, 28)
# print(y_train)  # (60000,)
# print(y_train[:3])  # array([7, 2, 1], dtype=uint8)
 
 
# get file names
folder_path = './Train'
input_tiff = []
for root, dirs, files in os.walk(os.path.abspath(folder_path)):
    for file in files:
        if(file.split('.')[-1] == 'tif'):
            input_tiff.append(os.path.join(root, file))
 
x_images = []
y_labels = []
 
# input_tiff = input_tiff[:5]
 
# for each image sequence, compute converted sequence and store it in same path
for j, path in enumerate(input_tiff):
    im = Image.open(path,'r')
    for i, page in enumerate(ImageSequence.Iterator(im)):
        y_labels.append(float(path.split('/')[-1].split('.tif')[0]))
        nparray = np.array(page)
        x_images.append(nparray)
 
x_images = np.asarray(x_images)
y_labels = np.asarray(y_labels)
X_train, X_test, y_train, y_test = train_test_split(x_images, y_labels, test_size=0.33, random_state=42)
# Initialize the image regressor.
reg = ak.ImageRegressor(
    overwrite=True,
    max_trials=1)
# Feed the image regressor with training data.
reg.fit(X_train, y_train, epochs=50)
 
 
model = reg.export_model()
 
try:
    model.save("model_autokeras", save_format="tf")
except Exception:
    model.save("model_autokeras.h5")
 
# # Predict with the best model.
# predicted_y = reg.predict(X_test)
# print(predicted_y)
 
# # Evaluate the best model with testing data.
# print(reg.evaluate(X_test, y_test))