import matplotlib.pyplot as plt

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
    axes2[0].plot(time_estimate, position_estimates[0], label='X (Est)')
    axes2[1].plot(time_estimate, position_estimates[1], label='Y (Est)')
    axes2[2].plot(time_estimate, position_estimates[2], label='Z (Est)')

    # plot ground truth position
    axes2[0].plot(time_vicon, position_vicon[0], label='X (Ground truth)', linestyle='dashed')
    axes2[1].plot(time_vicon, position_vicon[1], label='Y (Ground truth)', linestyle='dashed')
    axes2[2].plot(time_vicon, position_vicon[2], label='Z (Ground truth)', linestyle='dashed')
    
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

    # Plot ground truth Euler angles
    axes3[0].plot(time_vicon, euler_vicon[0], label='Roll (Ground truth)', linestyle='dashed')
    axes3[1].plot(time_vicon, euler_vicon[1], label='Pitch (Ground truth)', linestyle='dashed')
    axes3[2].plot(time_vicon, euler_vicon[2], label='Yaw (Ground truth)', linestyle='dashed')

    for ax in axes3:
        ax.set_xlabel('Time')
        ax.set_ylabel('Angle (deg)')
        ax.legend()

    fig3.suptitle('Euler Angles')
    
    # Show plots
    plt.show()