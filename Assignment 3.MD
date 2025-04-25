# Assignment 3

## Task 1: Camera Calibration (ROS)

In this task, we performed camera calibration to determine the intrinsic parameters of our camera. The process involves the following steps:

1. **Image Acquisition**: Multiple images of a chessboard pattern were captured from different angles and orientations using the camera.

2. **Pattern Detection**: The calibration algorithm automatically detects the chessboard corners in each image, providing precise reference points for calculation.

3. **Parameter Estimation**: Using these reference points, the algorithm computes the camera's intrinsic parameters, including:
   - Focal length
   - Principal point (optical center)
   - Distortion coefficients

4. **Configuration File**: The calibration results are stored in a config.yaml file, containing essential camera parameters such as:
   - Camera matrix
   - Distortion coefficients
   - Rectification matrix
   - Projection matrix

5. **Visualization**: Calibrated images demonstrate the correction of lens distortions.

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task1_2.png?raw=true" alt="Chessboard Calibration" width="500">
</p>

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task1_3.png?raw=true" alt="Config File Part 1" width="500">
</p>

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task1_4.png?raw=true" alt="Config File Part 2" width="500">
</p>

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task1_5.jpg?raw=true" alt="Calibrated Images" width="500">
</p>

The calibration enables accurate 3D reconstruction from 2D images by correcting for lens distortions and establishing the relationship between pixel coordinates and real-world coordinates.


## Task 2: Object Pose Estimation Using Calibrated Camera

In this task, we used our calibrated camera to estimate the pose of objects in a simulated maze environment. The process includes using the PnP (Perspective-n-Point) algorithm to determine object positions relative to the camera coordinate system.

### Environment Setup in Gazebo

Here is the maze environment with colored blocks placed at different locations. And you can see the views from the robot's camera.

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task2_1.png?raw=true" alt="Gazebo Environment View 1" width="700">
</p>

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task2_2.png?raw=true" alt="Gazebo Environment View 2" width="700">
</p>

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task2_3.png?raw=true" alt="Gazebo Environment View 3" width="700">
</p>

### ArUco Marker Generation

To simplify pose estimation, I generated ArUco markers as the references:

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task2_4.png?raw=true" alt="ArUco Marker Generation" width="700">
</p>

These markers provide easily detectable features with known geometry.

### PnP Pose Estimation Results

Using the solvePnP algorithm with the calibrated camera parameters, I successfully estimated the pose of a cube in the scene:

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task2_5.png" alt="PnP Pose Estimation Results" width="700">
</p>


## Task 3: Showcase StellaVSLAM

In this task, I implemented and demonstrated StellaVSLAM, a visual SLAM system based on ORB features for 3D environment mapping and localization. StellaVSLAM supports various camera models and can work with monocular, stereo, and RGB-D inputs.

### StellaVSLAM Setup and Configuration

I set up the StellaVSLAM environment using Docker, which provided all the necessary dependencies and the PangolinViewer for visualization:

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task3_1.png?raw=true" alt="StellaVSLAM Setup" width="700">
</p>

The terminal output shows the Docker container being initialized with the required components, including:
- Eigen (3.3.7)
- OpenCV (4.7.0)
- Pangolin viewer
- StellaVSLAM core libraries

### Equirectangular Dataset Testing

Following the documentation, I ran StellaVSLAM with the provided equirectangular dataset (aist_living_lab_1). The system successfully tracked features and created a 3D map of the environment:

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task3_2.png?raw=true" alt="Docker Build Process" width="700">
</p>

The PangolinViewer displays:
- Left: Map viewer showing the 3D point cloud and camera trajectory
- Right: Frame viewer showing the current camera view with tracked features

### Custom Environment Mapping

I then applied StellaVSLAM to the own indoor environment using a calibrated webcam with parameters obtained from Task 1:

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task3_3.png?raw=true" alt="StellaVSLAM with Custom Environment" width="700">
</p>

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task3_4.png?raw=true" alt="StellaVSLAM with Custom Environment" width="700">
</p>

### Youtube Video Link

For a complete explanation of StellaVSLAM running in the environment, please watch my videos:

1. Reproduce the aist_living_lab_1 video: https://youtu.be/RUDCPeeL3o8?si=MWS9tlHxu1xW6MKt
2. Explore my own environement: https://youtu.be/Dcy45Ht2wFQ?si=yUPx4mYagFHHJTh_


## Task 4: Understanding Visual SLAM

In this task, I implemented a visual odometry system in Python to understand the fundamental principles of VSLAM. This explains how a camera's motion can be tracked by analyzing the visual features in sequential frames.

### Visual Odometry Implementation

My Python-based visual odometry system successfully demonstrates the core principles of VSLAM using the KITTI dataset. Using ORB feature detection and tracking combined with essential matrix estimation, I've tracked camera motion through diverse environments with high accuracy. The system proves robust across various environments from the KITTI benchmark as shown in the images below, where green points represent detected features and the trajectory plots show the estimated camera path. 

<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task4_1.png?raw=true" alt="Feature Detection on KITTI" width="350" style="display:inline-block; margin:5px">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task4_2.png?raw=true" alt="Urban Trajectory on KITTI" width="350" style="display:inline-block; margin:5px">
</p>
<p align="center">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task4_3.png?raw=true" alt="Park Features on KITTI" width="350" style="display:inline-block; margin:5px">
  <img src="https://github.com/Fitz-Yuan/AI-for-Robotics/blob/main/project3/Screenshots/Task4_4.png?raw=true" alt="Residential Area on KITTI" width="350" style="display:inline-block; margin:5px">
</p>

### Youtube Video Link

For a complete explanation of the VSLAM implementation, please watch my video:

https://youtu.be/MlPLxZs3cgk?si=26GjUq7cfRW_3Mbr
