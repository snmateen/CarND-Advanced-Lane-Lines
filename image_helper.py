# This python script contains functions that are needed for advanced lane finding project and it includes following topics

### camera calibration 
# 	- calibrate_camera(None) - returns None

### distortion correction
# 	- undistort(image) - returns dst (undistorted image)

### gradient and color threshold
#	- abs_sobel_thresh(image, orientation, kernel, threshold) - returns binary_image (gradient thresholded binary image)
#	- mag_threshold(image, kernel, threshold) - returns binary_image (gradient's magnitude thresholded binary image)
#	- dir_threshold(image, kernel, threshold) - returns binary_image (gradient's direction thresholded binary image)
#	- col_threshold(image, threshold) - returns binary_image (color thresholded binary image)

### combining gradient and color thresholds from above
#	- combined_threshold(image, kernel
#					   , sobel_threshold_x_min=0, sobel_threshold_x_max=255
#					   , sobel_threshold_y_min=0, sobel_threshold_y_max=255
#					   , mag_threshold_min=0, mag_threshold_max=255
#					   , dir_threshold_min=0, dir_threshold_max=np.pi/2
#					   , col_threshold_min=0, col_threshold_max=255) 
# 						- returns binary_image (sobel(x and y), magnitude, direction, color thresholded binary image)

#	- pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)) - returns binary_image (based on the pipeline design from the course)

### prespective transform
#	- warp(image, src, destination) - returns warped_image and Minv (inverse prespective transform matrix mapping)

### fitting polynomials
#	- fit_polynomial(binary_warped, nwindows=9, plotit=False) - returns left_fitx, right_fitx and ploty

### curvature calculation
#	- calculate_curvature(ploty, plotit=False) - , returns real_left_curverad, real_right_curverad, offset_from_center

# import modules
import numpy as np
import cv2, glob, pickle

import matplotlib.pyplot as plt

def calibrate_camera(images):
	"""
	This function loops through all the images in the "camera_cal" folder and 
	accumulates image and object points which will be later used for image calibrations
	
	input: list of images
	result: image and object points are stored as pickle file

	note: this function assumes that calibration images are present in "camera_cal" folder in the current directory
	"""
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane.

	# Step through the list and search for chessboard corners
	print("INFO: Looping through images")
	for fname in images:
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		# Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

		# If found, add object points, image points
		if ret == True:
			objpoints.append(objp)
			imgpoints.append(corners)
		print(".", end="")
	print("")

	# save the data in pickle file, which can later be loaded for camera calibration
	dist_pickle={}
	dist_pickle['objpoints'] = objpoints
	dist_pickle['imgpoints'] = imgpoints
	pickle.dump(dist_pickle, open( "wide_dist_pickle.p", "wb" ))
	print("Pickle dumped with object and image points!!!")

def undistort(img):
	"""
	This function calculates distortion coefficients and undistorts the given image
	Input: image, object and image points
	Output: undistored image (dst - destination)
	
	ret: return value
	mtx: camera matrix
	dist: distortion coefficients
	rvecs: rotation vectors estimated for each pattern view
	tvecs: translation vectors estimated for each pattern view
	
	dst: destination (undistorted image)
	"""
	# Read in the saved objpoints and imgpoints
	dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
	objpoints = dist_pickle["objpoints"]
	imgpoints = dist_pickle["imgpoints"]

	# Use cv2.calibrateCamera() and cv2.undistort()
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[1], img.shape[0]) ,None,None)
	dst = cv2.undistort(img, mtx, dist, None, mtx)
	return dst

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
	"""
	This function takes in image, orientation and threshold, basically it takes gradient of an image
	in the given x or y direction for the specified threshold
	"""
	thresh_min = thresh[0]
	thresh_max = thresh[1]
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
	# Apply x or y gradient with the OpenCV Sobel() function
	# and take the absolute value
	if orient == 'x':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
	if orient == 'y':
		abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
	
	# Rescale back to 8 bit integer
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	
	# Create a copy and apply the threshold
	binary_output = np.zeros_like(scaled_sobel)
	# Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
	binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

	# Return the result
	return binary_output

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
	"""
	This function returns the magnitude of gradient for given sobel kernel size and threshold value
	"""
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
	# Take both Sobel x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	
	# Calculate the gradient magnitude
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	
	# Rescale to 8 bit
	scale_factor = np.max(gradmag)/255 
	gradmag = (gradmag/scale_factor).astype(np.uint8) 
	
	# Create a binary image of ones where threshold is met, zeros otherwise
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

	# Return the binary image
	return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	"""
	This function returns direction of gradient for the given sobel kernel size and threshold value
	The direction of the gradient is simply the inverse tangent (arctangent) of the y gradient 
	divided by the x gradient: arctan(sobely/sobelx)
	"""
	# Convert to Grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	
	# Calculate the x and y gradients
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	
	# Take the absolute value of the gradient direction, 
	# apply a threshold, and create a binary image result
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

	# Return the binary image
	return binary_output

def col_threshold(img, thresh=(0, 255)):
	"""
	This function thresholds the S-channel of HLS
	"""
	# convert to HLS
	hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	
	# s channel
	s_channel = hls[:,:,2]
	
	# apply the threshold
	binary_output = np.zeros_like(s_channel)
	binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
	
	# return the binary image
	return binary_output

def combined_threshold(image, sobel_kernel=3
					   , sobel_threshold_x_min=0, sobel_threshold_x_max=255
					   , sobel_threshold_y_min=0, sobel_threshold_y_max=255
					   , mag_threshold_min=0, mag_threshold_max=255
					   , dir_threshold_min=0, dir_threshold_max=np.pi/2
					   , col_threshold_min=0, col_threshold_max=255):
	"""
	This function combines
	- abs sobel threshold (for x, y) - gradient of an image in the direction of x or y
	- magnitude of gradient
	- direction of gradient
	"""
	# Apply each of the thresholding functions
	gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=sobel_kernel, thresh=(sobel_threshold_x_min, sobel_threshold_x_max))
	grady = abs_sobel_thresh(image, orient='y', sobel_kernel=sobel_kernel, thresh=(sobel_threshold_y_min, sobel_threshold_y_max))
	mag_binary = mag_thresh(image, sobel_kernel=sobel_kernel, thresh=(mag_threshold_min, mag_threshold_max))
	dir_binary = dir_threshold(image, sobel_kernel=sobel_kernel, thresh=(dir_threshold_min, dir_threshold_max))
	col_binary = col_threshold(image, thresh=(col_threshold_min, col_threshold_max))

	# apply a threshold, and create a binary image result
	binary_output = np.zeros_like(dir_binary)
	binary_output[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (col_binary == 1)] = 1
	
	# Return the binary image
	return binary_output

def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
	"""
	Sample pipeline from the course, that applies
	- HLS convertion
	- seperation of L and S channel
	- take derivative of L channel
	- apply threshold to gradient of L channel in the direction of x
	- apply threshold to S channel 
	- stack above 2 channel
	- return binary image
	"""
	img = np.copy(img)
	
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
	
	# Threshold color channel
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
	
	# Stack each channel
	# Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
	# be beneficial to replace this channel with something else.
	color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
	
	# return the binary image
	return color_binary

def warp(img, src=None, dst=None):
	"""
	This function uses perspective tranform to warp the given image
	source and destination coordinates are defaulted if not provided.

	src - source coordinates
	dst - destination coordinates

	M - prespective transform mapping
	Minv - inverse of prespective transform mapping

	returns warped, Minv
	"""
	# define the image size using shape function
	img_size = (img.shape[1], img.shape[0])
	
	# four desired source and destination coordinates
	if src is None and dst is None:
		src = np.float32([[293, 700], 
						  [587, 458], 
						  [703, 458], 
						  [1028, 700]])
		dst = np.float32([[250, img_size[1]], 
						  [250, 0], 
						  [950, 0], 
						  [950,  img_size[1]]])

	# compute the perspective transform, M
	M = cv2.getPerspectiveTransform(src, dst)
	
	# compute the inverse by swapping the source and destination
	Minv = cv2.getPerspectiveTransform(dst, src)
	
	# create warped image - uses linear interpolation
	warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
	
	# return the warped image
	return warped, Minv

def fit_polynomial(binary_warped, nwindows=9, plotit=False):
	"""
	This function takes binary warped image and fit polynomial based on windowing on histogram logic

	returns:
	left_fitx - polynomial fit on left pixels
	right_fitx - polynomial fit on right pixels
	ploty - polynomial fit
	"""
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)

	# Create an output image to draw on and  visualize the result
	out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = nwindows

	# Set height of windows
	window_height = np.int(binary_warped.shape[0]/nwindows)

	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base

	# Set the width of the windows +/- margin
	margin = 100

	# Set minimum number of pixels found to recenter window
	minpix = 50

	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = binary_warped.shape[0] - (window+1)*window_height
		win_y_high = binary_warped.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
						  & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
						   & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:		
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# plot the data if requested
	if plotit:
		plt.figure(figsize=(15,5))
		plt.subplot(1,2,1)
		plt.plot(histogram)
		plt.ylim(0,150)
		
		plt.figure(figsize=(15,15))
		plt.subplot(1,2,2)
		plt.imshow(out_img / 255)
		plt.plot(left_fitx, ploty, color='yellow')
		plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
	
	return left_fitx, right_fitx, ploty

def calculate_curvature(image, leftx, rightx, ploty):
	"""
	This function calculates the curvature based on polynomial

	Input: ploty (polynomial fit on the lane lines)

	Output: real world curvature radius (in meters)
	left_curverad
	right_curverad
	offset
	"""
	y_eval = image.shape[0]
	
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension
	
	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

	# Calculate the new radii of curvature (in meters)
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

	# Offset from center
	x_offset = ((left_fit_cr[2] + right_fit_cr[2]) / 2) - (image.shape[1]*xm_per_pix/2)

	# return the pixel and real curvatures
	return left_curverad, right_curverad, x_offset

def draw_lanes(image, binary_warped_image, left_fitx, right_fitx, ploty, Minv):
	"""
	This function draws the lane lines on top of the original image
	"""
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(binary_warped_image).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	
	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))
	
	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
	
	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
	
	# Combine the result with the original image
	result = cv2.addWeighted(undistort(image), 1, newwarp, 0.3, 0)
	return result
