import os
import cv2
import scipy.io
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

def getData(file_name):
    """
    Loads data from a .mat file containing sensor data and ground truth information.

    Inputs:
        - file_name: Name of the .mat file (string)

    Outputs:
        - sensor_data: Sensor data (list of dicts)
        - position_vicon: Ground truth positions from Vicon (3xN numpy array)
        - euler_vicon: Ground truth Euler angles from Vicon (3xN numpy array)
        - time_vicon: Timestamp vector (numpy array of length N)

    """
    # Load .mat file
    curr_path = str(os.path.dirname(os.path.abspath(__file__)))
    file_path = curr_path + '/data/' + file_name
    mat_data = scipy.io.loadmat(file_path,simplify_cells=True)

    # ground truth data from vicon
    vicon_data = mat_data['vicon']     
    position_vicon = vicon_data[:3,:]
    euler_vicon = vicon_data[3:6,:]    
    time_vicon = mat_data['time']

    # sensor data from .mat
    sensor_data = mat_data['data']

    return sensor_data,position_vicon,euler_vicon,time_vicon

def getImagePoints(p1,p2,p3,p4):
    """
    Rearranges and stacks corner points of an image into a format suitable for OpenCV functions.

    Inputs:
        - p1, p2, p3, p4: Arrays of april-tag corner points image co-ordinates (numpy arrays)

    Output:
        N corner point image co-ordinates stacked into a single array (Nx2 numpy array, float32)

    """
    # check dims of all arrays
    if np.ndim(p1) == 1:
        p1 = p1.reshape((2,1))
    if np.ndim(p2) == 1:
        p2 = p2.reshape((2,1))
    if np.ndim(p3) == 1:
        p3 = p3.reshape((2,1)) 
    if np.ndim(p4) == 1:
        p4 = p4.reshape((2,1))  

    return np.hstack((p1,p2,p3,p4)).astype('float32').T

def getWorldPoints(april_tags):
    """
    Calculates the world coordinates of corner points of April tags.

    Input:
        - april_tags: IDs of the April tags (list)

    Output:
        World coordinates of N corner points (Nx3 numpy array, float32)

    """
    # Init map with tags 
    tag_ids = np.array([
    [0, 12, 24, 36, 48, 60, 72, 84, 96],
    [1, 13, 25, 37, 49, 61, 73, 85, 97],
    [2, 14, 26, 38, 50, 62, 74, 86, 98],
    [3, 15, 27, 39, 51, 63, 75, 87, 99],
    [4, 16, 28, 40, 52, 64, 76, 88, 100],
    [5, 17, 29, 41, 53, 65, 77, 89, 101],
    [6, 18, 30, 42, 54, 66, 78, 90, 102],
    [7, 19, 31, 43, 55, 67, 79, 91, 103],
    [8, 20, 32, 44, 56, 68, 80, 92, 104],
    [9, 21, 33, 45, 57, 69, 81, 93, 105],
    [10, 22, 34, 46, 58, 70, 82, 94, 106],
    [11, 23, 35, 47, 59, 71, 83, 95, 107]
    ])

    # corner point co-ordinates in world frame
    p1_w = np.zeros((len(april_tags),3))
    p2_w = np.zeros((len(april_tags),3))
    p3_w = np.zeros((len(april_tags),3))
    p4_w = np.zeros((len(april_tags),3))

    for i in range(len(april_tags)):
        #get array index of the tag id
        idx = np.where(tag_ids == april_tags[i])
        # calc corner points
        # p2
        p2x = 0.152*(2*idx[0]+1)
        p2y = 0.152*(2*idx[1]+1) + 0.026*(idx[1]//3)
        # p1
        p1x = p2x
        p1y = p2y - 0.152
        # p3
        p3x = p1x - 0.152
        p3y = p1y + 0.152
        # p4
        p4x = p1x - 0.152
        p4y = p1y
        # store co-ordinates
        p1_w[i][0] = p1x
        p1_w[i][1] = p1y
        p2_w[i][0] = p2x
        p2_w[i][1] = p2y
        p3_w[i][0] = p3x
        p3_w[i][1] = p3y
        p4_w[i][0] = p4x
        p4_w[i][1] = p4y

    return np.vstack((p1_w,p2_w,p3_w,p4_w)).astype('float32')

def cam2robotframe(tvec,rvec):
    """
    Calculates robot/IMU frame pose from camera pose

    Inputs:
        - tvec: Translation vector of world frame w.r.t camera frame (3x1 numpy array)
        - rvec: Rotation vector of world frame w.r.t camera frame (3x1 numpy array)

    Output:
        - P_imu_world: Position of robot/IMU frame w.r.t world frame (3X1 numpy array)
        - euler_xyz: Euler angles of robot/IMU frame w.r.t world frame (3X1 numpy array)

    """
    # IMU-camera parameters
    XYZ = [-0.04, 0.0, -0.03]
    Yaw = pi/4

    # Transformation matrix of IMU frame w.r.t camera frame
    T_imu_cam = np.array([[ np.cos(Yaw), -np.cos(Yaw), 0, XYZ[0]],
                          [-np.sin(Yaw), -np.sin(Yaw), 0, XYZ[1]],
                          [ 0,            0,          -1, XYZ[2]],
                          [ 0,            0,           0, 1     ]])

    R_cam_world,J_cam_world = cv2.Rodrigues(rvec)
    # Transformation matrix of camera frame w.r.t world frame
    T_cam_world = np.zeros((4,4))
    T_cam_world[3,3] = 1
    T_cam_world[:3,:3] = R_cam_world
    T_cam_world[:3,3] = tvec                        # P_cam_world

    # Get T_imu_world
    T_imu_world = np.linalg.inv(T_cam_world) @ T_imu_cam

    # extract position - [x,y,z]
    P_imu_world = T_imu_world[:3,3]

    # rotation matrix
    R_imu_world = Rotation.from_matrix(T_imu_world[:3,:3])

    # extract euler angles - ZXY
    euler_imu_world = R_imu_world.as_euler('ZXY')
    # rearrange the angles in the order - roll, pitch, yaw
    euler_xyz = np.array([euler_imu_world[1], euler_imu_world[2], euler_imu_world[0]])

    return P_imu_world , euler_xyz

def estimate_pose(data):
    """
    Estimates the robot pose (position and orientation) from camera data.

    Input:
        - data: Camera data containing corner points and tag IDs (dictionary)

    Output:
        - pos: Position of robot/IMU frame w.r.t world frame (3X1 numpy array)
        - euler: Euler angles of robot/IMU frame w.r.t world frame (3X1 numpy array)

    """
    # parameters
    camera_matrix = np.array([[314.1779, 0, 199.4848], 
                            [0, 314.2218, 113.7838],
                            [0, 0, 1]])
    
    distortion_params = np.array([-0.438607, 0.248625, 0.00072, -0.000476, -0.0911])

    # get april tag ids from the data instance
    april_tags = data['id']
    # check if april_tags is of the type np.array
    if isinstance(april_tags, np.ndarray):
        # return empty pose if no april tags deteced in the image
        if (len(april_tags) == 0):
            return np.array([]),np.array([])
    else:
        april_tags = np.array([april_tags])
    
    # get world and image points of the detected corners of the tags
    world_points = getWorldPoints(april_tags)
    image_points = getImagePoints(data['p1'],data['p2'],data['p3'],data['p4'])

    # solve pnp
    retval, rvec, tvec = cv2.solvePnP(world_points,image_points,camera_matrix,distortion_params)

    # tranform pose from cam frame to robot/IMU frame
    pos, euler = cam2robotframe(tvec.reshape(3),rvec.reshape(3))

    return pos, euler

def getPoseEstimates(cam_data):
    """
    Estimates robot pose for all camera data instances

    Input:
        - cam_data: List of dictionaries containing camera data instances (Length N)

    Output:
        - position_estimates: robot position estimates (3xN numpy array)
        - euler_estimates: robot euler angle estimates (3xN numpy array) 
        - time_estimate: corresponding timestamps of the estimates (numpy array, length N)

    """
    # vars to store pose estimations
    position_estimates = np.full((3,len(cam_data)),np.nan)
    euler_estimates = np.full((3,len(cam_data)),np.nan)
    time_estimate = np.full(len(cam_data),np.nan)

    for i in range(len(cam_data)):
        # estimate pose
        pos, euler = estimate_pose(cam_data[i])

        # store estimates
        if not (len(pos)==0 and len(euler)==0):
            position_estimates[:,i] = pos
            euler_estimates[:,i] = euler

        # store timestamps at which the estimates were calculated
        time_estimate[i] = cam_data[i]['t']

    return position_estimates,euler_estimates,time_estimate

############################ TRAJECTORY VISUALIZATION #################################

def plot_pose(position_estimates, euler_estimates, time_estimate, position_vicon, euler_vicon, time_vicon):
    """
    Plots the estimated and ground truth poses against each other

    Inputs:
        - position_estimates: Estimated positions (3xN numpy array)
        - euler_estimates: Estimated Euler angles (3xN numpy array)
        - time_estimate: Timestamps for estimated poses (numpy array, length N)
        - position_vicon: Ground truth positions from Vicon (3xN numpy array)
        - euler_vicon: Ground truth Euler angles from Vicon (3xN numpy array)
        - time_vicon: Timestamps for ground truth poses (numpy array, length N)

    Outputs:
        None

    """
    # Figure 1: 3D position plot
    fig1 = plt.figure(figsize=(4,3.5))
    ax1 = fig1.add_subplot(111, projection='3d')

    # Plot estimated position in blue
    ax1.plot(position_estimates[0], position_estimates[1], position_estimates[2], c='r', label='Estimated')

    # Plot Vicon position in red
    ax1.plot(position_vicon[0], position_vicon[1], position_vicon[2], c='b', label='Vicon')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Position')
    ax1.legend()

    # Figure 2: Position Subplot
    fig2,axes2 = plt.subplots(3,1)

    # plot estimated position
    axes2[0].plot(time_estimate, position_estimates[0], label='X (Est)')
    axes2[1].plot(time_estimate, position_estimates[1], label='Y (Est)')
    axes2[2].plot(time_estimate, position_estimates[2], label='Z (Est)')

    # plot vicon position
    axes2[0].plot(time_vicon, position_vicon[0], label='X (Vicon)', linestyle='dashed')
    axes2[1].plot(time_vicon, position_vicon[1], label='Y (Vicon)', linestyle='dashed')
    axes2[2].plot(time_vicon, position_vicon[2], label='Z (Vicon)', linestyle='dashed')
    
    for ax in axes2:
        ax.set_xlabel('Time')
        ax.set_ylabel('Pos (m)')
        ax.legend()

    fig2.suptitle('Position')

    # Figure 3: Euler angles subplot
    fig3, axes3 = plt.subplots(3,1)

    # Plot estimated Euler angles
    axes3[0].plot(time_estimate, euler_estimates[0], label='Roll (Est)')
    axes3[1].plot(time_estimate, euler_estimates[1], label='Pitch (Est)')
    axes3[2].plot(time_estimate, euler_estimates[2], label='Yaw (Est)')

    # Plot Vicon Euler angles
    axes3[0].plot(time_vicon, euler_vicon[0], label='Roll (Vicon)', linestyle='dashed')
    axes3[1].plot(time_vicon, euler_vicon[1], label='Pitch (Vicon)', linestyle='dashed')
    axes3[2].plot(time_vicon, euler_vicon[2], label='Yaw (Vicon)', linestyle='dashed')

    for ax in axes3:
        ax.set_xlabel('Time')
        ax.set_ylabel('Angle (radians)')
        ax.legend()

    fig3.suptitle('Euler Angles')
    
    # Show plots
    plt.show()

def visualize_trajectory(file_name):
    """
    Visualizes the trajectory by plotting estimated and ground truth poses for the given dataset

    Inputs:
        - file_name: Name of the .mat file containing data (string)

    Outputs:
        None

    """
    # Extract data from mat file
    cam_data,position_vicon,euler_vicon,time_vicon = getData(file_name)
        
    # estimate camera pose
    position_estimates,euler_estimates,time_estimate = getPoseEstimates(cam_data)

    # plot estimated vs groundtruth pose
    plot_pose(position_estimates,euler_estimates,time_estimate,position_vicon,euler_vicon,time_vicon)

############################### COVARIANCE ESTIMATION ################################

def interpolate_data(estimated_data, estimated_timestamps, ground_truth_timestamps):
    """
    Interpolates estimated data to match ground truth timestamps.

    Inputs:
        - estimated_data: Estimated data - position, euler angles (6XN numpy array)
        - estimated_timestamps: Timestamps for estimated data (numpy array)
        - ground_truth_timestamps: Timestamps for ground truth data (numpy array)

    Outputs:
        - interpolated_data: Interpolated data (6XN numpy array)

    """
    # Clean data
    # Find indices of non-NaN values
    valid_indices = ~np.isnan(estimated_data).any(axis=0)

    # Clean up the estimated data and timestamps
    cleaned_data = estimated_data[:, valid_indices]
    cleaned_timestamps = estimated_timestamps[valid_indices]
    
    # Initialize an array for interpolated data
    interpolated_data = np.zeros((6, len(ground_truth_timestamps)))

    # Interpolate each row separately
    for i in range(6):
        interpolation_function = interp1d(cleaned_timestamps, cleaned_data[i, :], kind='linear', fill_value="extrapolate")
        interpolated_data[i, :] = interpolation_function(ground_truth_timestamps)

    return interpolated_data

def estimate_covariances(file_name):
    """
    Estimates the covariance matrix from camera pose estimates.

    Inputs:
        - file_name: Name of the .mat file containing data (string)

    Outputs:
        - R: Covariance matrix (6x6 numpy array)

    """
    # Extract data from mat file
    cam_data,position_vicon,euler_vicon,time_vicon = getData(file_name)
        
    # estimate camera pose
    position_estimates,euler_estimates,time_estimate = getPoseEstimates(cam_data)

    # stack position and euler angles into a single 2D array
    groundtruth_data = np.vstack((position_vicon,euler_vicon))
    estimated_data = np.vstack((position_estimates,euler_estimates))

    # interpolate estimated data to match groundtruth timestamps
    intp_data = interpolate_data(estimated_data, time_estimate, time_vicon)

    # Number of observations
    n = len(time_vicon)

    # Estimate the observation model covariance matrix using the sample covariance formula
    observation_residual = intp_data - groundtruth_data
    R = observation_residual @ observation_residual.T / (n-1)

    return np.round(R,5)