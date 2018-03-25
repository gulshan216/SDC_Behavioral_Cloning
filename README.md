# **Behavioral Cloning** 
---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_driving.png "Grayscaling"
[image3]: ./examples/left_recovery.png "Recovery Image"
[image4]: ./examples/right_recovery.png "Recovery Image"
[image5]: ./examples/recovery.png "Recovery Image"
[image6]: ./examples/augmented.png "Augmented Images"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---
#### 1. Required files to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 – A video recording of my vehicle driving autonomously one lap around the track 1
* video2.mp4 – A video recording of my vehicle driving autonomously one lap around the challenge track 2

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model architecture

I have used the NVIDIA model which uses strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers. We follow the five convolutional layers with three fully connected layers leading to an output control value.
The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers between the three fully connected layers in order to reduce overfitting 
The model was trained and validated on different data sets with data from track2 as well to ensure that the model was not overfitting. Also the data was augmented with random brightness change, shifts and rotation to ensure the data is more generalized and the model does not overfit. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...
My first step was to use a convolution neural network model similar to the NVIDIA model . I thought this model might be appropriate because it has been used to drive real autonomous vehicles.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.
To combat the overfitting, I modified the model to include Dropout layers between the fully connected layers.
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track during the curves. To improve the driving behavior in these cases, I added another lap of training data with the vehicle recovering from the left and right side of the road back to the center of the lane.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) uses first a lambda layer for normalization and then strided convolutions in the next three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers. We follow the five convolutional layers with three fully connected layers leading to an output control value. Dropout layers with the drop probability of 0.2 are added between the fully connected layers.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover back to the center of the lane if it is going away from the lane. These images show what a recovery looks like :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.
The training data was then preprocessed by cropping 60 pixels from the top that only contains the surroundings and not lane information and 25 pixels from the bottom which contains the unimportant car hood. Then the image was resized to a (64,64) image in order to decrease the training time for my model. Then the contrast of the image is improved by equalize histogram functionality in opencv. Then the image is converted to YUV color space as required by the NVIDIA model.
To augment the data sat, I also added random brightness change, translation and rotation to the images:

![alt text][image6]

Etc ....

After the collection process, I had 87000 number of data points. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as the Validation loss stops decreasing after this point. I used an adam optimizer so that manually training the learning rate wasn't necessary.
