# 3D Flow Mapping
Tracking microfluidic flow in 3D (three dimensions) remains challenging due to the need for a specialized optical environment. In this project, we propose a simple method which uses brightfield microscopy and machine learning to track microfluidic flow in 3D. Firstly, we use Trackmate to track the particles in 2D. Then we use deep learning techniques to find the Z position of the seed based on the 2D top view of the seed. Through this project we demonstrates how our method is easily implemented to create high-resolution flow maps that guide the understanding and design of novel microfluidic structures.

## Experimental Setup
We used an inverted motorized microscope, a 20x/0.45NA objective, a fiber-optic LED illuminator and a high-speed camera to capture videos at 1920x1080 resolution with a 0.32µm x 0.32µm pixel size.

As tracer particles, we tested 1.02 µm and 3 µm diameter cross-linked polystyrene microbeads, 1.02 µm diameter silica microbeads, 2 µm glass microbeads, and 1 µm magnetic microbeads. Then we use TrackMate to detect bead paths.

## Dataset
#### Training Data
The training dataset consists of 110 tiff files where each file has a list of images of bead or particle. All the images in a tiff file are of same distance from the focal plane at which these images are captured from and it is denoted by the filename.

#### Experimental Data
The experimental set consists of two categories - Single Phase and Two Phase. Single phase is further Channel Step and Displacement Structure. Whereas two phase is further divided into straight channel and curved channel. Images are classified into these folders based on the enviornment they are captured in.
_Note: The Two phase data is more challenging to predict when compared to single phase._

## Neural Network Model
We formulate the depth prediction problem as a regression analysis. In contrast to a classification task that selects from a set of discrete classification labels, a regression task can produce continuous outcomes with enhanced interpretability. The regression model is built on a Resnet-50[1] convolutional neural network architecture with a depth of 50 layers, and is pre-trained on the ImageNet4 data set. Instead of connecting the output of one convolution block to the next block, Resnet-50 has new skip connections that connect the original input to the output of the current block for improved performance. Such new skip connections enable free flow of gradients and thereby help reduce the vanishing gradient problem. As Resnet-50 is a classification model, we tailor it to regression analysis by adding a flatten and regression head layer. The detailed architecture is presented below in Fig. 2. These models are trained using the machine learning library Autokeras[2]. We apply stratified partitioning to the reference images to select the training, validation, and testing sets with a ratio of 64:16:20.

For our particular task, at the core we used Resnet-50 architecture, where the last layers are modified to convert it into a Image regression model rather than image classification model. The next section talks more about the Resnet-50 architecture and why we considered this over other regression architectures.


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

The code will calculate predictions for each experimental images and generate a csv file which will be saved in the same folder with filename same as folder filename. A seperate csv file will be generated for each folder.

## License
This tool is available under the GNU General Public License (GPL) (https://www.gnu.org/licenses/gpl-3.0.en.html) and the LGPL (https://www.gnu.org/licenses/lgpl-3.0.en.html).
