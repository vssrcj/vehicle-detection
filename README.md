# Vehicle Detection

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

*made by [CJ](https://github.com/vssrcj)*

---
![Final Result Gif](./result.gif)
---
Overview
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.



## Classifier

### Color Histogram
Combines each color channel in RGB into a histogram as in `get_color_histogram`.

It works most effictive if the number of bins is set to **64**, and the bins range to **(0, 256)**.

Result of Car Color Histogram:
<div>
     <img src="/readme_images/histogram_cars.png" height="300">
</div>

Result of Non Car Color Histogram:
<div>
     <img src="/readme_images/histogram_non_cars.png" height="300">
</div>

### Spatial Features
The image can be converted into any color, and then transformed into a feature vector, as seen in `bin_spatial`.

The best result is achieved when the color space is set to **YCrCb** and the spatial size to **(32, 32)**.

Result of Car Color Spatial Features:
<div>
     <img src="/readme_images/color_spaces_cars.png" height="300">
</div>

Result of Non Car Color Spatial Features:
<div>
     <img src="/readme_images/color_spaces_non_cars.png" height="300">
</div>

### Hog Features
Using the hog function from the scikit-image module, a histogram is returned as in `get_hog_features`.

The best result is achieved when the orient is **10**, the pixels per cell **8**, and the cells per block **2**.

Result of Car vs Non Car HOG:
<div>
     <img src="/readme_images/hog.png" height="300">
</div>

## Extract Features
Combine the color histogram, the spatial features and the hog features of each image, and normalize it, as seen in `single_img_features`:
<div>
     <img src="/readme_images/features.png" height="300">
</div>

The car, and non car features are then stacked to get the features vectors that are fed to the classifier for training, as in `get_scaled_features`.

## Classifier

### Loading data

Thousands of 64x64 car and non-car images are loaded from [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) respectively.

### Training
A Linear SVC is used to train the classifier, as in `train`
80% of the data is used for training, 20% for validation.

A test accuracy of 98.96% is achieved.

## Sliding Windows
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
Run the classifier on each window.  If it is a match to a car, a box will be drawn around it, as in `search_windows`:

<div>
     <img src="/readme_images/search_windows.png" height="300">
</div>

## Applying heatmap
So far it does a good job of detecting cars, but it also detects a lot of false positives.

Applying a heatmap, as seen in `get_fitted_boxes` mitigates this issue, plus it allows to draw a nice single box around a detection.

The heatmap is made by overlapping the detected boxes.  Only if there are at least one overlap of detected boxes, then it will count as a detection:

<div>
     <img src="/readme_images/heatmap.png" height="300">
</div>

## Finding cars.
A video is then analyzed frame by frame in `find_boxes`.

Each frame (image) is basically passed to `get_fitted_boxes` to retrieved the boxes (detections).  These boxes are then drawn on the frame.

---
The process works, but the detections are jittery (moves to much between frames), some false positives persist, and there are also false negatives (cars are not detected between frames).

These issues are solved by keeping a history of detections.
Only if at least 3 out of the last 5 detections are within 30 pixels of the subsequent detections, then it counts as a valid detection.

Meaning if there is a once off detection, it won't count.  If there are some dropped detections, there is a buffer of 2 frames, where the previous frame's detection will be used.
The average of the detections are used, which helps smooths the box.

The result can be seen in:

<a href="/result.mp4">video.mp4</a>.

### Problems faced.
* Some false positives remains.  The heatmap threshold can be raised, the history count can be raised, or the buffer can be lowered.  All of this will raise the risk of false negatives though.
* The box surrounding a car are skewed sometimes.  Not allowing a box's width to be less than its height helps with this.

### Possible improvements.
It is an expensive procedure to detect cars, because of all the windows that are analyzed.
This can be optimized by:
* Limiting the search windows to less areas.
* Focusing the searching to areas around previously detected boxes.
* Divide the search areas on the images between frames.


[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

