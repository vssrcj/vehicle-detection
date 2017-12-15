# Vehicle Detection

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

*made by [CJ](https://github.com/vssrcj)*

![Final Result Gif](./result.gif)

# Overview.

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

# Pipeline.

1. Features are extracted from thousand of training images.
2. A classifier is trained by passing these features to it.
3. The features of a video frame are extracted and passed to the classifier in order to make a prediction.
4. Boxes are drawn around the detected areas.

* *Note - any variables that can be set, are written in **bold**.*

## Feature extraction.

### 1. Color Histogram.
Combines each color channel in RGB into a histogram in the `get_color_histogram` function.

It works most effictive if the number of bins is set to **64**, and the bins range to **(0, 256)**.

Result of Car Color Histogram:
<div>
     <img src="/readme_images/histogram_cars.png" height="200">
</div>

Result of Non Car Color Histogram:
<div>
     <img src="/readme_images/histogram_non_cars.png" height="200">
</div>

### 2. Spatial Features.
The image can be converted into any color, and then transformed into a feature vector in the `bin_spatial` function.

The best result is achieved when the color space is set to **YCrCb** and the spatial size to **(32, 32)**.

Result of Car Color Spatial Features:
<div>
     <img src="/readme_images/color_spaces_cars.png" height="400">
</div>

Result of Non Car Color Spatial Features:
<div>
     <img src="/readme_images/color_spaces_non_cars.png" height="400">
</div>

### 3. Hog Features.
Using the hog function from the *scikit-image* module, a histogram is returned in the `get_hog_features` function.

The best result are achieved (after some experimentation) when the orient is **10**, the pixels per cell **8**, and the cells per block **2**.

Result of Car vs Non Car HOG:
<div>
     <img src="/readme_images/hog.png" height="200">
</div>

### Feature composition.
`single_img_features` combines the color histogram, the spatial features and the hog features of each image.  This data is then normalized and then scaled using the `StandardScaler` from *scikit-preprocesing*.

<div>
     <img src="/readme_images/features.png" height="200">
</div>

`get_scaled_features` the stack the car, and non-car features that will be fed to the classifier for training.

## Classifier.

### Loading data.

Thousands of 64x64 car and non-car images are loaded from [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) respectively.

### Training.
`train` uses a `LinearSVC` from *scikit-svm* to train the classifier.

**80%** of the data is used for training, **20%** for validation.

A test accuracy of *98.96%* is achieved.

## Sliding windows.
In order to detect cars on a image, the classifier will run on multiple *windows* of an image.  `slide_window` is how the windows are made:

3 sets of sliding windows are used:

**Small**
<div>
     <img src="/readme_images/small_windows.png" height="300">
</div>

**Medium**
<div>
     <img src="/readme_images/medium_windows.png" height="300">
</div>

**Large**
<div>
     <img src="/readme_images/large_windows.png" height="300">
</div>

## Finding matching windows.
`search_windows` run the classifiers on each window.  If a match is found, it draws a box around the window:

<div>
     <img src="/readme_images/windows.png" height="300">
</div>

## Applying a heatmap.
So far it does a good job of detecting cars, but it also detects a lot of false positives.

`get_fitted_boxes` mitigates this issue by applying a heatmap.  The heatap also allows to draw a nice single box around a detection.

The heatmap is made by overlapping the detected boxes.  A box will then only be drawn where **2** other boxes overlap:

<div>
     <img src="/readme_images/heatmap.png" height="300">
</div>

## Finding cars.
A video is then analyzed frame by frame in `find_boxes`.

Each frame (image) is passed to `get_fitted_boxes` to retrieved the detected cars (the single box surrounding a car).  These boxes are then drawn on the stream of images to produce the resulting video.

Simply processing each frame in isolation works, but it has some issues:
* The detected boxes are jittery (moves to much between frames).
* Some false positives persist.
* There are also false negatives (the boxes around cars vanish between frames).

These issues are solved by keeping a history of the detections / boxes:

Only if at least **3** out of the last **5** boxes around a car are within **30** pixels of a previous box, then it counts as a valid detection.

This means that:
* Eratic detections won't count.
* If detections are dropped between frames, a buffer of 2 exist, where the previous frame's detection will be used.
* The average of the detections are used, which helps smooths the box.

The result can be seen in:

<a href="/result.mp4">video.mp4</a>.

## Problems faced.
* Some false positives remains.  The heatmap threshold can be raised, the history count can be raised, or the history buffer can be lowered.  All of this will raise the risk of not detected an actual car though.
* The boxes surrounding the car are not fitted well.  Not allowing a box's width to be less than its height helps with this.

## Possible improvements.
Because of the amount of windows that needs to be analyzed, it is an expensive procedure to detect the cars.

This can be optimized by:
* Focusing the searching to areas around previously detected boxes.  Since a history of detections already exist, this won't be too difficult.  A detection is more likely to occur around a previous detection then on another part of the image.
* Then the rest of the image can only be scanned cursory with a low heatmap threshold to detect possible "new" cars.
* Subsequent frames can also scan different sections of the image.  For example, frames can alternate by scanning the left side of the image, then the right.
