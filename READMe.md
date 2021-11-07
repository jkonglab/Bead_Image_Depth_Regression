# 3D Flow Mapping
The task of the project is to predict the depth of a particle (z level) based on a 2D image which is captured by a microscope.

## Dataset
#### Training Data
The training dataset consists of 110 tiff files where each file has a list of images of bead or particle. All the images in a tiff file are of same distance from the focal plane at which these images are captured from and it is denoted by the filename.

#### Experimental Data
The experimental set consists of two categories - Single Phase and Two Phase. Single phase is further Channel Step and Displacement Structure. Whereas two phase is further divided into straight channel and curved channel. Images are classified into these folders based on the enviornment they are captured in.
_Note: The Two phase data is more challenging to predict when compared to single phase._

### Installation
The entire codebase is based on python. The following libraries are required to train and test the code:
- numpy
- tensorflow
- autokeras
- scikit-learn
- os
- PIL

### Training
`train.py` file consists of code required to train the model. To train the model we use autokeras library. Autokeras has an Image Regression model which we use for this project. More details about Image Regression in autokeras can be found here: https://autokeras.com/tutorial/image_regression/
The model takes tiff images as input. The path to training dataset should be given in `folder_path` variable.
Once the training is complete, the model is stored in the same folder with name `model_autokeras.h5`

### Prediction/Inference
`prediction.py` file consists of code required for inference. The path to trained model should be assigned to `loaded_model` variable. And the list of folders for which prediction need to be calculated should be assigned to `path` variable.

The code will calculate predictions for each experimental images and generate a csv file which will be saved in the same folder with filename same as folder filename. A separate csv file will be generated for each folder.
