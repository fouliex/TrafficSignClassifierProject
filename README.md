## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This project is to use Deep Neural Networks and Convolutional Neural Networks to classify Traffic Signs.
 A model is trained and validated so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
Furthermore, the model is tested on images of German traffic signs that you find on the web. These images are located under [TestImages](./TestImages)

A Ipython notebook is included and it contains all the information about the projects in details.



The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Data Set Summary and Exploration
### Basic Summary of the data set
The pandas library is use to calculate summary statistics of the traffic signs data set:
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is  1263
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

### Exploratory Visualization of the dataset
Here is an exploratory visualization of the data set. The graph shows the traffic sign label distribution of the validation,
training and test set. However as one can see the data is not uniformly distributed, a few traffic sings has less than 200 samples
in the training set and some sign labels has more than 1250 samples.  The Traffic sign model trained in this this project
is good at identifying some signs better then others.

![DataSetGraph](./misc/DatasetGraph.png)

## Design and Test the  Model Architecture

### Preprocessed the image data
A normalize() function is used to normalize image data between -1 and 1. No further pre-processing technique were done
on the data set.
```python
def normalize(data):
    normalize_data = ((data-128.0)/128.0) -1
    return normalize_data

X_train = normalize(X_train)
X_test = normalize(X_test)
X_valid = normalize(X_valid)

```
### Model Architecture
The Neural Network Architecture is based on the [LeNet-5](http://yann.lecun.com/exdb/lenet/) architecture.
The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x30 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x30 				|
| Convolution 5x5	    | 1x1 stride, same padding,output 10x10x64      |
| RELU					|												|
| Max pooling	      	| 2x2 stride,outputs 5x5x64 				    |
| Flatten				| input 5x5x64,output 1600(5*5*64)				|
| Fully Connected		| input 1600, output 1024                       | 
| RELU					|												|
| Drop out				| Probability Keep is 50%                       |
| Fully Connected       | input 1024,output 512                         |
| RELU					|												|
| Drop out				| Probability Keep is 50%                       |
| Fully Connected       | input 512,output 256                          |
| RELU					|												|
| Drop out				| Probability Keep is 50%                       |
| Fully Connected       | input 256,output number of classes 43         |

### Model Hyperparameters
The model hyperparameters are:
*  The Epoch is 10
* The Batch size is 128
* The learning rate is 0.001
* The Optimizer is the Adam Optimizer

#### Epoch
To tell the network how long we want it to train, we set an epoch number.An epoch is one trip through the entire 
training dataset. For this project, the training epochs} is set to 10.

#### Batch size
The Batch size is the number of training data in one forward/backward pass. It is the number of sample that is propagated
through the network. For this project the batch size is set to 128.The networks trains faster and update the weights after each mini-batches.
#### Learning rate
The learning rate has to do with how quickly the network abandons old beliefs for new ones.With a higher 
Learning Rate, the network  will changes its mind more quickly which is why a low learning rate of 0.001 is chosen for this model.

#### Adam Optimizer
Adam is an optimization algorithm that is use to update the weights iteratively based of the training data.

####
## Result
The final model results were:
* The Test set accuracy  is 94%
* The validation set accuracy is 96% 

### Test the Model on New Images
Here are  the five German traffic signs that were chosen from the web:

[//]: # (Image References)
[image1]: TestImages/traffic_sign_1.jpg
[image2]: TestImages/traffic_sign_2.jpg 
[image3]: TestImages/traffic_sign_3.jpg 
[image4]: TestImages/traffic_sign_4.jpg 
[image5]: TestImages/traffic_sign_5.jpg  


![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

### Model Prediction
 If the model predicted 1 out of 5 signs correctly,it would be 20% accurate on these new images.However, the model was able to predict all 5 images
 with an accuracy of 100%.

| Image			        |     Label	 | Description           | Prediction Label	 | Prediction Description|
|:---------------------:|:----------:|:---------------------:|:-----------------:|:---------------------:|  
| ![alt text][image1]   | 14         | Stop                  | 14                | Stop                  |
| ![alt text][image2]   | 17 		 | No entry 	         | 17                | No entry 	         |
| ![alt text][image3]	| 0			 | 0 Speed limit(20km/h) | 0                 | 0 Speed limit(20km/h) |
| ![alt text][image4]   | 40   		 | Roundabout mandatory  | 40                | Roundabout mandatory  |
| ![alt text][image5]	| 28		 | Children crossing     | 28                | Children crossing     |

### Softmax probabilities for each prediction
For images 1, 2 4 and 5 the top prediction has a softmax probability of 1.000 and the rest of the probabilities lower than 0.000.
For image 3 the top prediction result has a softmax probability of 0.844, 0.049 , 0.027 0.021 and 0.01. 
The top softmax is display in the  Jupyter Notebook.

## Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```
