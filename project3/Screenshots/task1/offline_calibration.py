#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import yaml
import os
# Adjust dimensions according to your actual chessboard
chessboard_size = (7, 10)  # Number of inner corners: an 8×11 chessboard has 7×10 inner corners
square_size = 0.015  # Square size 15mm, converted to meters
# Prepare object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size  # Convert to actual dimensions
# Store object points and image points for all images
objpoints = []  # 3D coordinates in real world space
imgpoints = []  # 2D coordinates in image plane
# Read all images
images_path = os.path.expanduser('~/camera_calibration_ws/chessboard_images/')
images = glob.glob(images_path + '*.jpg')
if len(images) == 0:
    print(f"Error: No image files found in the directory {images_path}")
    exit()
print(f"Found {len(images)} images")
# Create a directory to save processed images
debug_dir = os.path.join(images_path, 'debug')
os.makedirs(debug_dir, exist_ok=True)
# Find corners
print("Starting image processing...")
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adjust image processing
    # Try to enhance contrast
    gray = cv2.equalizeHist(gray)
    
    # Save processed grayscale image for debugging
    debug_gray_path = os.path.join(debug_dir, f'gray_{idx:02d}.jpg')
    cv2.imwrite(debug_gray_path, gray)
    
    # Try different parameters to find chessboard corners
    found = False
    for flags in [cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
                 cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK]:
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, flags)
        if ret:
            found = True
            break
    
    # If found, add to our point lists
    if found:
        # Further optimize corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        objpoints.append(objp)
        imgpoints.append(corners2)
        
        # Draw corners and save
        img_with_corners = img.copy()
        cv2.drawChessboardCorners(img_with_corners, chessboard_size, corners2, ret)
        output_path = os.path.join(debug_dir, f'corners_{idx:02d}.jpg')
        cv2.imwrite(output_path, img_with_corners)
        
        print(f"Processing image {idx+1}/{len(images)} - Corners found")
    else:
        print(f"Processing image {idx+1}/{len(images)} - Corners not found")
        # Save original image for inspection
        output_path = os.path.join(debug_dir, f'failed_{idx:02d}.jpg')
        cv2.imwrite(output_path, img)
print(f"Found {len(objpoints)} valid images for calibration")
if len(objpoints) < 5:
    print("Error: Insufficient valid images for calibration (at least 5 needed)")
    print("Please check if the chessboard format is correct and ensure it is fully visible in the photos")
    print("You can check the images in the debug directory to understand the problem")
    exit()
# Calibration
print("Starting calibration...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("\nCamera calibration results:")
print("Camera matrix:")
print(mtx)
print("\nDistortion coefficients:")
print(dist)
# Calculate error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print(f"\nTotal reprojection error: {mean_error/len(objpoints)}")
# Extract intrinsic parameters
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]
k1, k2, p1, p2, k3 = 0, 0, 0, 0, 0
if len(dist[0]) >= 5:
    k1, k2, p1, p2, k3 = dist[0][:5]
elif len(dist[0]) == 4:
    k1, k2, p1, p2 = dist[0]

# Save in ROS format with custom formatting
output_path = os.path.expanduser('~/camera_calibration_ws/camera_calibration.yaml')
with open(output_path, 'w') as f:
    # Write camera matrix with specified format
    f.write("camera_matrix:\n")
    f.write(f"  rows: 3\n")
    f.write(f"  cols: 3\n")
    f.write(f"  data: [{fx}, 0, {cx}, 0, {fy}, {cy}, 0, 0, 1]\n")
    
    # Write distortion model
    f.write(f"distortion_model: \"plumb_bob\"\n")
    
    # Write distortion coefficients with the comment about simulation
    f.write("distortion_coefficients:\n")
    f.write("  data: [0, 0, 0, 0, 0]  # no distortion in sim\n")

print(f"Calibration results saved to {output_path}")

# Compare with ideal parameters
print("\nComparison with ideal parameters:")
print("Ideal distortion coefficients: [0, 0, 0, 0, 0]")
print(f"Actual distortion coefficients: [{k1:.8f}, {k2:.8f}, {p1:.8f}, {p2:.8f}, {k3:.8f}]")
# If calibration is successful, generate example corrected images
if len(objpoints) >= 5:
    print("\nGenerating example corrected images...")
    for idx, fname in enumerate(images[:min(3, len(images))]):
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        # Correct image
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # Crop ROI region
        x, y, w, h = roi
        if all(v > 0 for v in [x, y, w, h]):
            dst = dst[y:y+h, x:x+w]
        
        # Save corrected image
        output_path = os.path.join(debug_dir, f'undistorted_{idx:02d}.jpg')
        cv2.imwrite(output_path, dst)
        
        # Display side-by-side comparison
        compare = np.hstack((img, dst))
        cv2.imwrite(os.path.join(debug_dir, f'compare_{idx:02d}.jpg'), compare)
        
        print(f"Generated corrected image: {output_path}")
print("\nCalibration completed!")
