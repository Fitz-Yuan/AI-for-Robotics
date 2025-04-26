#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import time
import yaml
# Chessboard dimensions
chessboard_size = (7, 6)
square_size = 0.025  # Square size in meters
# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp = objp * square_size  # Convert to actual dimensions
# Store object points and image points for all images
objpoints = []  # 3D coordinates in real world space
imgpoints = []  # 2D coordinates in image plane
# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
# Get camera resolution
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {frame_width}x{frame_height}")
# Set capture flag
capture_flag = False
images_taken = 0
min_images = 10
last_capture_time = time.time() - 5  # Initialize to 5 seconds ago
capture_interval = 2  # Minimum interval between captures (seconds)
print("Ready to capture calibration images")
print("Press space to capture an image, press ESC to end capture")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot get image")
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Display real-time image
    display_img = frame.copy()
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    # If corners are found, add to object points and image points
    if ret:
        cv2.drawChessboardCorners(display_img, chessboard_size, corners, ret)
        
        # Show that the chessboard is detected
        cv2.putText(display_img, "Chessboard Detected!", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # If space is pressed and time since last capture exceeds the interval, capture the image
        current_time = time.time()
        if capture_flag and (current_time - last_capture_time) > capture_interval:
            objpoints.append(objp)
            imgpoints.append(corners)
            images_taken += 1
            last_capture_time = current_time
            print(f"Image captured {images_taken}")
            capture_flag = False  # Reset flag
    
    # Display the number of captured images
    cv2.putText(display_img, f"Images: {images_taken}/{min_images}", (frame_width - 200, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display results
    cv2.imshow('Camera Calibration', display_img)
    
    # Keyboard event handling
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break
    elif key == 32:  # Space key
        capture_flag = True
    
    # If enough images have been collected, ask whether to end
    if images_taken >= min_images:
        cv2.putText(display_img, "Min images reached. Press 'q' to calibrate or continue for more images.", 
                    (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow('Camera Calibration', display_img)
        if key == ord('q'):
            break
# Release camera
cap.release()
cv2.destroyAllWindows()
print("Starting camera calibration...")
if len(objpoints) < 5:
    print("Error: Insufficient number of images, at least 5 images needed")
    exit()
# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (frame_width, frame_height), None, None)
print("\nCamera calibration results:")
print("Camera matrix:")
print(mtx)
print("\nDistortion coefficients:")
print(dist)
# Calculate reprojection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print("\nTotal reprojection error: {}".format(mean_error/len(objpoints)))
# Generate ROS-compatible YAML format
calibration_data = {
    'image_width': frame_width,
    'image_height': frame_height,
    'camera_name': 'camera',
    'camera_matrix': {
        'rows': 3,
        'cols': 3,
        'data': mtx.flatten().tolist()
    },
    'distortion_model': 'plumb_bob',
    'distortion_coefficients': {
        'rows': 1,
        'cols': 5,
        'data': dist.flatten().tolist()
    }
}
# Save calibration results
with open('camera_calibration.yaml', 'w') as f:
    yaml.dump(calibration_data, f)
print("Calibration results saved to camera_calibration.yaml")
# Test calibration effect - load a test image
if len(imgpoints) > 0:
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        # Undistort image
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        
        # Crop ROI region
        x, y, w, h = roi
        if all(v > 0 for v in [x, y, w, h]):
            dst = dst[y:y+h, x:x+w]
        
        # Display original image and undistorted image
        cv2.imshow('Original Image', frame)
        cv2.imshow('Calibrated Image', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
