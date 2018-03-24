## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This project is to use deep neural networks and Convolutional Neural Networks to classify Traffic Signs.
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

[DataSetPraph](./misc/DatasetGraph.png)



### Dependencies
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

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

