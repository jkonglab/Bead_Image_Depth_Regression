from tensorflow.keras.models import load_model
import autokeras as ak
import os
import csv
import numpy as np
import time
from PIL import Image, ImageSequence, ImageDraw
from sklearn.model_selection import train_test_split
loaded_model = load_model("model_autokeras 2", custom_objects=ak.CUSTOM_OBJECTS)
print(loaded_model)
print(loaded_model.summary())

# paths = ['/Users/nikhil/Documents/MS/Bead Project/classification/Model 32/Experimental Datasets/Channel Step 32x32 Stacks']

paths = ['/Users/nikhil/Documents/MS/Bead Project/classification/Model 32/Experimental Datasets/test']

for folder_path in paths:
    for f in os.listdir(folder_path):
        if(f[0] == '.'):
            continue

        print(f)
        print(time.time())

        input_tiff = os.listdir(os.path.join(folder_path,f))
        preds = []
        for j, path in enumerate(input_tiff):
            x_images = []
            if(path[0] == '.'):
                continue
            im = Image.open(os.path.join(folder_path,f,path))

            for i, page in enumerate(ImageSequence.Iterator(im)):
                nparray = np.array(page)
                x_images.append(nparray)
            
            x_images = np.asarray(x_images)
            predicted_y = loaded_model.predict(x_images)
            preds.append(predicted_y)
        
        

        with open(f+'.csv', mode='w') as w_file:
            w_writer = csv.writer(w_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            
            for file_name, pred in zip(input_tiff,preds):
                row_lst = [file_name]
                for p in pred:
                    row_lst.append(p[0])
        
                w_writer.writerow(row_lst)

# # get file names
# folder_path = '/Users/nikhil/Documents/MS/Bead Project/Dataset-Final/Train'
# input_tiff = []
# for root, dirs, files in os.walk(os.path.abspath(folder_path)):
#     for file in files:
#         if(file.split('.')[-1] == 'tiff'):
#             input_tiff.append(os.path.join(root, file))

# x_images = []
# y_labels = []

# # input_tiff = input_tiff[:5]

# # for each image sequence, compute converted sequence and store it in same path
# for j, path in enumerate(input_tiff):
#     im = Image.open(path)
#     for i, page in enumerate(ImageSequence.Iterator(im)):
#         y_labels.append(float(path.split('/')[-1].split('.tiff')[0]))
#         nparray = np.array(page)
#         x_images.append(nparray)

# x_images = np.asarray(x_images)
# y_labels = np.asarray(y_labels)
# X_train, X_test, y_train, y_test = train_test_split(x_images, y_labels, test_size=0.33, random_state=42)


# predicted_y = loaded_model.predict(temp)
# print(predicted_y)