import os
import numpy as np
import scipy.io
import pose_estimation as pe
from scipy.spatial.transform import Rotation
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.linalg import cholesky
from ukf import prediction
## PARAMS
FILENAME = 'studentdata1.mat'
N_STATE = 15                    # state vector length
N_OBS = 6                       # observation vector length

def getSigmaPts(mean, covariance, kappa=1):
    n = len(mean)
    alpha = 1
    beta = 2
    lambda_ = alpha**2 * (n + kappa) - n
    
    # Calculate sigma points
    sigma_points = np.zeros((2 * n + 1, n))
    weights_mean = np.zeros(2 * n + 1)
    weights_covariance = np.zeros(2 * n + 1)

    sigma_points[0] = mean
    weights_mean[0] = lambda_ / (n + lambda_)
    weights_covariance[0] = weights_mean[0] + (1 - alpha**2 + beta)

    sqrt_covariance = cholesky((n + lambda_) * np.round(covariance,4))

    for i in range(n):
        sigma_points[i+1] = mean + sqrt_covariance[:,i]
        sigma_points[i+1+n] = mean - sqrt_covariance[:,i]

        weights_mean[i+1] = 1/(2*(n + lambda_))
        weights_covariance[i+1] = weights_mean[i+1]

        weights_mean[i+1+n] = 1/(2*(n + lambda_))
        weights_covariance[i+1+n] = weights_mean[i+1+n]

    return sigma_points, weights_mean, weights_covariance

def process_model(x_prev,dt,u_w,u_a):
    p = x_prev[:3]
    q = x_prev[3:6]

    # chnge in linear velocity
    p_dot = x_prev[6:9]

    # change in angular velocity
    G_q = np.array([[np.cos(q[1]), 0, -np.cos(q[0])*np.sin(q[1])],
                    [0,            1,  np.sin(q[0])             ],
                    [np.sin(q[1]), 0,  np.cos(q[0])*np.cos(q[1])]])     
    q_dot = np.linalg.inv(G_q) @ u_w

    # change in acceleration
    g = np.array([0, 0, -9.81])
    phi, theta, psi = q
    # Compute individual rotation matrices
    R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi), np.cos(psi), 0],
                    [0, 0, 1]])

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])

    R_y = np.array([[np.cos(phi), 0, np.sin(phi)],
                    [0, 1, 0],
                    [-np.sin(phi), 0, np.cos(phi)]])
    # Compute composite rotation matrix
    Rot_q = R_z @ R_x @ R_y
    # Rot_q = Rotation.from_euler('zxy',q, degrees=False).as_matrix()
    p_ddot = g + Rot_q @ u_a

    # change in gyroscope bias
    Q_g = 1e-3*np.eye(3)    # covariace
    mu_g = np.zeros(3)      # mean
    bg_dot = np.random.multivariate_normal(mu_g,Q_g)

    # change in accelerometer bias
    Q_a = 1e-3*np.eye(3)    # covariance
    mu_a = np.zeros(3)      # mean
    ba_dot = np.random.multivariate_normal(mu_a,Q_a)

    # change in state
    x_dot = np.hstack((p_dot,q_dot,p_ddot,bg_dot,ba_dot))

    return x_prev + x_dot*dt

def obs_model(X_prev):
    # init measurement matrix
    C = np.hstack((np.eye(N_OBS),np.zeros((N_OBS,len(X_prev)-N_OBS))))

    Z_est = C @ X_prev

    return Z_est

# Load .mat file
# file_name = 'studentdata1.mat'
curr_path = str(os.path.dirname(os.path.abspath(__file__)))
file_path = curr_path + '/data/' + FILENAME
mat_data = scipy.io.loadmat(file_path,simplify_cells=True)

# extract data
sensor_data = mat_data['data']
vicon_data = mat_data['vicon']
time_vicon = mat_data['time']

# Init UKF
t_prev = time_vicon[0]
dt = time_vicon[1] - t_prev
# points = MerweScaledSigmaPoints(N_STATE, alpha=.1, beta=2., kappa=-1)
points = MerweScaledSigmaPoints(N_STATE, alpha=1, beta=2., kappa=1)
ukf = UnscentedKalmanFilter(N_STATE,N_OBS,dt,obs_model,process_model,points)
# init state vector
ukf.x = np.hstack((vicon_data[:9,0],0*np.ones(6)))
# init covariance matrix
ukf.P *= 1e-3
# init Q
ukf.Q = 1e-3*np.eye(N_STATE)
ukf.R = 1e-3*np.eye(N_OBS)

# store filtered states
X_filtered = np.full((15,vicon_data.shape[1]),np.nan)
t_filtered = np.full(len(time_vicon),np.nan)

for i in range(len(sensor_data)):
    data = sensor_data[i]
    t_curr = data['t']
    dt = t_curr - t_prev
    # parse imu data data
    w = data['rpy']
    a = data['acc']

        # X_s, weights_mean, weights_covariance = getSigmaPts(ukf.x,ukf.P)
    # # pass sigma points through process model
    # X_s_est = np.full(X_s.shape,np.nan)
    # for i in range(X_s.shape[0]):
    #     X_s_est[i] = process_model(X_s[i],dt,w,a)
    #predict
    X_est,P_est,X_s_est = prediction(ukf.x,ukf.P,np.vstack((w,a)),dt)
    ukf.predict(dt=dt,u_w=w,u_a=a)
    
    # get camera pose estimates
    pos, euler = pe.estimate_pose(data)
    #update
    ukf.update(np.hstack((pos,euler)))
    
    X_filtered[:,i] = ukf.x
    t_filtered[i] = t_curr
    t_prev = t_curr

pos_filtered = X_filtered[:3,:]
euler_filtered = X_filtered[3:6,:]

pe.plot_pose(pos_filtered,euler_filtered,t_filtered,vicon_data[:3,:],vicon_data[3:6,:],time_vicon)
