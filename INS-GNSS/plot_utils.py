import matplotlib.pyplot as plt
from haversine import haversine, Unit
import numpy as np

def plot_pose(position_estimates, euler_estimates, time_estimate, position_vicon, euler_vicon, time_vicon):
    """
    Plots the estimated and ground truth poses against each other

    Inputs:
        - position_estimates: Estimated positions (3xN numpy array)
        - euler_estimates: Estimated Euler angles (3xN numpy array)
        - time_estimate: Timestamps for estimated poses (numpy array, length N)
        - position_vicon: Ground truth positions (3xN numpy array)
        - euler_vicon: Ground truth Euler angles (3xN numpy array)
        - time_vicon: Timestamps for ground truth poses (numpy array, length N)

    Outputs:
        None

    """
    # Figure 1: 3D position plot
    fig1 = plt.figure(figsize=(4,3.5))
    ax1 = fig1.add_subplot(111, projection='3d')

    # Plot estimated position in blue
    ax1.plot(position_estimates[0], position_estimates[1], position_estimates[2], c='r', label='Estimated')

    # Plot ground truth position in red
    ax1.plot(position_vicon[0], position_vicon[1], position_vicon[2], c='b', label='Ground truth')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Position')
    ax1.legend()

    # Figure 2: Position Subplot
    fig2,axes2 = plt.subplots(3,1)

    # plot estimated position
    axes2[0].plot(time_estimate, position_estimates[0], label='Estimated')
    axes2[1].plot(time_estimate, position_estimates[1], label='Estimated')
    axes2[2].plot(time_estimate, position_estimates[2], label='Estimated')

    # plot ground truth position
    axes2[0].plot(time_vicon, position_vicon[0], label='Ground truth', linestyle='dashed')
    axes2[1].plot(time_vicon, position_vicon[1], label='Ground truth', linestyle='dashed')
    axes2[2].plot(time_vicon, position_vicon[2], label='Ground truth', linestyle='dashed')


    
    for ax in axes2:
        ax.set_xlabel('Time')
        ax.legend()

    axes2[0].set_ylabel('Longitude (Deg)')
    axes2[1].set_ylabel('Latitude (Deg)')
    axes2[2].set_ylabel('Altitude (m)')

    fig2.suptitle('Position')

    # Figure 3: Euler angles subplot
    fig3, axes3 = plt.subplots(3,1)

    # Plot estimated Euler angles
    axes3[0].plot(time_estimate, euler_estimates[0], label='Estimated')
    axes3[1].plot(time_estimate, euler_estimates[1], label='Estimated')
    axes3[2].plot(time_estimate, euler_estimates[2], label='Estimated')

    # Plot ground truth Euler angles
    axes3[0].plot(time_vicon, euler_vicon[0], label='Ground truth', linestyle='dashed')
    axes3[1].plot(time_vicon, euler_vicon[1], label='Ground truth', linestyle='dashed')
    axes3[2].plot(time_vicon, euler_vicon[2], label='Ground truth', linestyle='dashed')


    for ax in axes3:
        ax.set_xlabel('Time')
        ax.legend()

    axes3[0].set_ylabel('Roll (Deg)')
    axes3[1].set_ylabel('Pitch (Deg)')
    axes3[2].set_ylabel('yaw (Deg)')


    fig3.suptitle('Euler Angles')
    
    # Show plots
    plt.show()

def plot_error(pose_est,pose_truth,angles_est,angles_truth,time):
    # calc error
    err_pose = pose_est - pose_truth
    err_angles = angles_est - angles_truth

    # plot error
    # Figure 1: Position Error Subplot
    fig1,axes1 = plt.subplots(3,1)

    # plot estimated position
    axes1[0].plot(time, err_pose[0], label='Longitude')
    axes1[0].set_xlabel('Time (s)')
    axes1[0].set_ylabel('Error (deg)')
    axes1[0].legend()

    axes1[1].plot(time, err_pose[1], label='Latitude')
    axes1[1].set_xlabel('Time (s)')
    axes1[1].set_ylabel('Error (deg)')
    axes1[1].legend()

    axes1[2].plot(time, err_pose[2], label='Altitude')
    axes1[2].set_xlabel('Time (s)')
    axes1[2].set_ylabel('Error (m)')
    axes1[2].legend()
            
    fig1.suptitle('Error in Position')

    # Figure 2: Orientation error subplot
    fig2,axes2 = plt.subplots(3,1)

    # plot estimated position
    axes2[0].plot(time, err_angles[0], label='Phi Error')
    axes2[1].plot(time, err_angles[1], label='Theta Error')
    axes2[2].plot(time, err_angles[2], label='Psi Error')
    
    for ax in axes2:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error (deg)')
        ax.legend()

    fig2.suptitle('Error in Orientation')

    # Show plots
    plt.show()

    # calc avg errors
    err_pose_avg = np.mean(np.abs(err_pose),axis=1)
    err_ang_avg = np.mean(np.abs(err_angles),axis=1)
    print('Average error in Longitude (deg):,', err_pose_avg[0])
    print('Average error in Latitude (deg):,', err_pose_avg[1])
    print('Average error in Altutude (m):,', err_pose_avg[2])
    print('Average error in Roll (deg):,', err_ang_avg[0])
    print('Average error in Pitch (deg):,', err_ang_avg[1])
    print('Average error in Yaw (deg):,', err_ang_avg[2])



def wrap_angle(angle):
    while angle <= -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return angle

def plot_haversine(pose_est,pose_truth,time):
    #calc haversine
    dist = np.full(pose_est.shape[1],np.nan)
    for i in range(pose_est.shape[1]):
        long = wrap_angle(pose_est[0,i])
        lat = wrap_angle(pose_est[1,i])

        dist[i] = haversine((long,lat),pose_truth[0:2,i],unit=Unit.DEGREES)

    # plot
    plt.figure()
    plt.plot(time,dist,label='haversine dist')
    plt.xlabel('Time (s)')
    plt.ylabel('Dist (deg)')
    plt.title('Haversine distance b/w Estimated and Truth position')
    plt.legend()

    plt.show()
    print('Average haversine dist (deg): ', np.mean(dist))

