# **Behavioral Cloning** 

## Writeup 

### This document describes the steps done to carry out this project .

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/CNN_Diagram.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 32, 64 and 128 (model.py lines 31-39)  with stride 2,2. After the classic Lenet5 is applied I included an extra convolution layers with filter 3x3 filters (due the image is quite small) with depths of 32 and 64

The model includes RELU layers to introduce nonlinearity in the CNN and also in the Full Connected Neuronal Network (FCNN) . The data is normalized in the model using a Keras lambda layer (code line 27). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting in the FCNN and also included L2 regularization. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 71-74). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 63).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and including the left and right cameras. Also Augmentation was including by mean of shifting the central image and increasing the steering rate.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Lenet5.  I thought this model might be appropriate because I tested with a simple Lenet5 and more or less it was working but not properly at all.

In order to know how well the model was working, I split my image and steering angle data into a training and validation set. I used this scheme as reference to avoid the overfitting problem.

To combat the overfitting, I modified the model including dropouts and L2 regularization.Then I realize that the validation result was getting better.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. In order to improve the driving behavior in these cases, I included the left and right cameras and also the augmentation by mean of shifting the center of the image to the left and to the right.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 27-61) consisted of a convolution neural network with the following as shown below: 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I was using the recordings provided by the course and no extra recordings were used.  To augment the data sat, I also shifted the images and angles but also applying the flipping and inverting the steering value.

It is important to remark that I used a generator in order to avoid memory problem. I was using an ad-hoc solution that instead of load the images, I was loading preprocessed pickles with a batch of images and steering. This solution it was quite fast in comparison with the one reading image by image. The difference was like 35 times more faster. The script that generate these pickle files is in "generate_data.py"

I finally randomly shuffled the data set and put 0.2% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by several test in the simulator I used an Adam optimizer so that manually training the learning rate wasn't necessary.

Results:

loss: 0.0238 - val_loss: 0.0161
