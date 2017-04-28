## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/distorted_undistorted.png "distorted"
[image2]: ./test_images/straight_lines1.jpg "Original straight_lines1 image"
[image21]: ./examples/undistorted_straight_lines1.jpg "Undistorted straight_lines1 image"
[image3]: ./examples/binary.jpg "Binary of straight lines1 image"
[image31]: ./examples/orginal_binary.png "Fit Visual"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/warped_orginal_binary.png "warped original binary"
[image51]: ./examples/polynomial_fit.png "Histogram"
[image6]: ./examples/lane_detected.jpg "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./create_images.ipynb" and "image_helper.py" (from line 41-104).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Here is undistorted image:
![alt text][image21]

Based on the `objpoints` and `imgpoints`, camera matrix and distortion coefficients are calculated, which are used by `cv2.undistort()` function to produce undistorted image.

```python
# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

# undistort the image.
dst = cv2.undistort(img, mtx, dist, None, mtx)
```
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 217 through 257 in `image_helper.py`).  

I considered following thresholds and kernel size for creating the binary image.

```python
# read image
image = mpimg.imread(img)

# warp the image after distortion correction
dst = ih.undistort(image, mtx, dist)
warped, Minv = ih.warp(dst)

# apply the combined threshold
binary_warped = ih.pipeline(warped, (130,255),(50,255),(40,255))
binary_warped = ih.region_of_interest(binary_warped)
```

`ih.pipeline()` function is the combination of sobel, gradient magnitude, direction and color threshold applied.

```python
# Convert to HSV color space and separate the V channel
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
l_channel = hsv[:,:,1]
s_channel = hsv[:,:,2]

# Sobel x
sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

# Threshold x gradient
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

# Threshold saturation
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

# Threshold lightness
l_binary = np.zeros_like(l_channel)
l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

# use the combination of above three to created the binary image    
binary = np.zeros_like(sxbinary)
binary[((l_binary == 1) & (s_binary == 1) | (sxbinary==1))] = 1
```
Here's an result of my binary output for this step before warping. Lanes are clearly visible, additionaly i've applied `region masking` to mask unwanted information above the lanes using a polygon.

```python
shape = img.shape
vertices = np.array([[(0,0)
                    ,(shape[1],0)
                    ,(shape[1],0)
                    ,(6*shape[1]/7,shape[0])
                    ,(shape[1]/7,shape[0])
                    , (0,0)]]
                  ,dtype=np.int32)
mask = np.zeros_like(img)

#filling pixels inside the polygon defined by "vertices" with the fill color    
cv2.fillPoly(mask, vertices, 255)
```

![alt text][image3]

Here is the comparison of original undistorted and binary image.

![al text][image31]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 263 through 300 in the file `image_helper.py`.  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 587, 458      | 250, 0        |
| 293, 700      | 250, 720      |
| 1028, 700     | 950, 720      |
| 703, 458      | 950, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

My approach was windowing through the histogram.

![alt_text][image51]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 418 through 444 in my code in `image_helper.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 9 through 46 in my code in `video_helper.py` in the function `process_image()`.  

oveview of all the steps that were carried out are present in the python notebook `advanced_lane_finding.ipynb`.

Here is an example of my result on a test image:

![alt text][image6]


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

After calibrating the camera using the images present in "camera_cal" folder, I applied distortion correct and prepestive transformation (warping), followed by applied color and gradient thresholds, based on the threshold limits described above, now, this was a lot of experimentation phase, this step might fail depending upon the brightness and variation in the image background. It might be good idea to tune the kernel, threshold parameters depending upon the time of camera images (day or night) and weather pattern.
