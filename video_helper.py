import numpy as np
from moviepy.editor import VideoFileClip
import cv2
import image_helper as ih

left_fit_g = []
right_fit_g = []
curvature_g = []
def process_image(image):
	"""
	"""
	# warp the image after distortion correction
	dst = ih.undistort(image)
	warped, Minv = ih.warp(dst)

	# apply the combined threshold
	binary_warped = ih.combined_threshold(warped,sobel_kernel=3
										   , sobel_threshold_x_min=50, sobel_threshold_x_max=255
										   , sobel_threshold_y_min=50, sobel_threshold_y_max=255
										   , mag_threshold_min=20, mag_threshold_max=255
										   , dir_threshold_min=-np.pi/2, dir_threshold_max=np.pi/2
										   , col_threshold_min=150, col_threshold_max=255)
	# fit polynomial
	left_fitx, right_fitx, ploty = ih.fit_polynomial(binary_warped=binary_warped, nwindows=15, plotit=False)
	
	# draw lanes
	output_image = ih.draw_lanes(image, binary_warped, left_fitx, right_fitx, ploty, Minv)

	# Smooth fits
	left_fit, right_fit = smooth_fits(left_fitx, right_fitx)

	# Calculate curvature
	left_curverad, right_curverad, offset_from_center = ih.calculate_curvature(output_image, left_fit, right_fit, ploty)

	# draw lanes on the road
	result = ih.draw_lanes(dst, binary_warped, left_fit, right_fit, ploty, Minv)

	# Add curvature to image
	curvature_in_m = np.mean(left_curverad)
	curvature_in_m = smooth_curvature(curvature_in_m)
	curvature_text = 'Curvature : {:.2f}'.format(curvature_in_m)
	offset_text = 'Offset from Center : {:.2f}'.format(offset_from_center)
	cv2.putText(result, curvature_text, (200, 100), 0, 1.2, (255, 255, 0), 2)
	cv2.putText(result, offset_text, (200, 200), 0, 1.2, (255, 255, 0), 2)

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
