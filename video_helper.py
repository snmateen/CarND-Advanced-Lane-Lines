import numpy as np
from moviepy.editor import VideoFileClip
import cv2, pickle
import image_helper as ih
import matplotlib.image as mpimg

left_fit_g = []
right_fit_g = []
curvature_g = []

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]

img = mpimg.imread("./camera_cal/calibration1.jpg")

# Use cv2.calibrateCamera() and cv2.undistort()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[1], img.shape[0]) ,None,None)

def process_image(image):
	"""
	"""
	# warp the image after distortion correction
	dst = ih.undistort(image, mtx, dist)
	warped, Minv = ih.warp(dst)

	# apply the combined threshold
	binary_warped = ih.pipeline(warped, (180,255),(60,255),(50,255))
	binary_warped = ih.region_of_interest(binary_warped)
	
	# fit polynomial
	left_fit, right_fit, leftx, lefty, rightx, righty, ploty = ih.fit_polynomial(binary_warped, nwindows=15, plotit=False)
	
	# Smooth fits
	left_fit, right_fit = smooth_fits(left_fit, right_fit)

	# draw lanes on the road
	result = ih.draw_lanes(dst, binary_warped, left_fit, right_fit, ploty, Minv)
	
	# Calculate curvature
	left_curverad, right_curverad, offset_from_center = ih.calculate_curvature(result, leftx, lefty, rightx, righty)

	# Add curvature to image
	curvature_in_m = np.mean(left_curverad)
	curvature_in_m = smooth_curvature(curvature_in_m)
	curvature_text = 'Curvature : {:.2f}'.format(curvature_in_m)
	offset_text = 'Offset from Center : {:.2f}'.format(offset_from_center)
	cv2.putText(result, curvature_text, (200, 100), 0, 1.2, (255, 255, 0), 2)
	cv2.putText(result, offset_text, (200, 150), 0, 1.2, (255, 255, 0), 2)

	return result

def smooth_curvature(curvature, n=50):
	"""
	Smoothes the curvature over n images
	"""
	curvature_g.append(curvature)
	curvature_np = np.array(curvature_g)

	if len(curvature_g) > n:
		curvature = np.mean(curvature_np[-n:])

	return curvature

def smooth_fits(left_fit, right_fit, n=20):
	"""
	Smoothes the polynomial fits
	"""
	left_fit_g.append(left_fit)
	right_fit_g.append(right_fit)

	left_fit_np = np.array(left_fit_g)
	right_fit_np = np.array(right_fit_g)

	if len(left_fit_g) > n:
		left_fit = np.mean(left_fit_np[-n:, :], axis=0)
	if len(right_fit_g) > n:
		right_fit = np.mean(right_fit_np[-n:, :], axis=0)
	return left_fit, right_fit
