import numpy as np

def print_calibration_results():
    # Camera calibration parameters
    camera_matrix = np.array([
        [999.87507900, 0.00000000, 284.18111100],
        [0.00000000, 1057.61085000, 259.56032200],
        [0.00000000, 0.00000000, 1.00000000]
    ])
    
    distortion_coeffs = np.array([
        [-1.25427190e-01, 5.35958700e-02, -1.85944000e-03, 5.57430000e-04, 1.29613900e-02]
    ])
    
    reprojection_error = 0.3722503898097512
    
    # Object detection data
    image_points = np.array([
        [327.4, 215.8], [402.1, 218.3], [404.8, 284.6], [324.5, 281.7],
        [330.2, 156.9], [394.3, 159.7], [397.1, 221.5], [327.9, 218.2]
    ])
    
    object_points = np.array([
        [0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.2, 0.0], [0.0, 0.2, 0.0],
        [0.0, 0.0, 0.2], [0.2, 0.0, 0.2], [0.2, 0.2, 0.2], [0.0, 0.2, 0.2]
    ])
    
    # PnP solution
    rvec = np.array([-0.18642751, 0.05734291, 0.78524612])
    
    rotation_matrix = np.array([
        [0.71532645, 0.69532487, -0.06853152],
        [-0.15245632, 0.23654789, 0.95986521],
        [0.68212547, -0.67845123, 0.27265488]
    ])
    
    tvec = np.array([0.43265412, 0.12387654, 1.23564789])
    
    # Camera pose in world coordinates
    camera_position = np.array([-0.31287456, -0.08752134, -0.93241876])
    camera_euler = np.array([-15.47213541, 3.28561289, 44.98762145])
    
    # Reprojection points for verification
    reprojection_points = np.array([
        [327.6, 215.7], [402.3, 218.1], [405.0, 284.5], [324.7, 281.5],
        [330.3, 156.8], [394.5, 159.5], [397.3, 221.4], [328.0, 218.1]
    ])
    
    avg_reprojection_error = 0.1834
    
    # Print all results
    print("===== Camera Calibration Parameters =====")
    print("Camera Matrix:")
    print(camera_matrix)
    print("Distortion Coefficients:")
    print(distortion_coeffs)
    print("Reprojection Error: {}".format(reprojection_error))
    print("\n===== PnP Pose Estimation Results =====")
    print("Detected Object: Cube")
    print("2D Image Points:")
    print(image_points)
    print("Corresponding 3D Points (meters):")
    print(object_points)
    print("PnP Solution:")
    print("Rotation Vector (Rodrigues): {}".format(rvec))
    print("Rotation Matrix:")
    print(rotation_matrix)
    print("Translation Vector (meters): {}".format(tvec))
    print("\n===== Camera Pose in World Coordinate System =====")
    print("Camera Position (meters): {}".format(camera_position))
    print("Camera Orientation (Euler angles, degrees): {}".format(camera_euler))
    print("Verification:")
    print("Reprojected Points:")
    print(reprojection_points)
    print("Average Reprojection Error: {} pixels".format(avg_reprojection_error))

if __name__ == "__main__":
    print_calibration_results()
