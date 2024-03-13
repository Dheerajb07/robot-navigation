import os
import numpy as np
import scipy.io
import pose_estimation as pe
from scipy.spatial.transform import Rotation
from scipy.linalg import sqrtm
from scipy.linalg import cholesky

## PARAMS
FILENAME = 'studentdata1.mat'
N_STATE = 15                    # state vector length
N_OBS = 6                       # observation vector length

def getSigmaPts(mean, covariance, kappa=0):
    n = len(mean)
    alpha = 1e-3
    beta = 2.0    
    lambda_ = alpha**2 * (n + kappa) - n
    
    # Calculate sigma points
    sigma_points = np.zeros((2 * n + 1, n))
    weights_mean = np.zeros(2 * n + 1)
    weights_covariance = np.zeros(2 * n + 1)

    sigma_points[0] = mean
    weights_mean[0] = lambda_ / (n + lambda_)
    weights_covariance[0] = weights_mean[0] + (1 - alpha**2 + beta)

    sqrt_covariance = sqrtm((n + lambda_) * covariance)

    for i in range(n):
        sigma_points[i+1] = mean + sqrt_covariance[i]
        sigma_points[i+1+n] = mean - sqrt_covariance[i]

        weights_mean[i+1] = 1/(2*(n + lambda_))
        weights_covariance[i+1] = weights_mean[i+1]

        weights_mean[i+1+n] = 1/(2*(n + lambda_))
        weights_covariance[i+1+n] = weights_mean[i+1+n]

    return sigma_points, weights_mean, weights_covariance

def process_model(x_prev,u_w,u_a):
    p = x_prev[:3]
    q = x_prev[3:6]

    # chnge in linear velocity
    p_dot = x_prev[6:9]

    # change in angular velocity
    G_q = np.array([[np.cos(q[1]), 0, -np.cos(q[0])*np.sin(q[1])],
                    [0,            1,  0                        ],
                    [np.sin(q[1]), 0,  np.cos(q[0]*np.cos(q[1]))],])   
    q_dot = np.linalg.inv(G_q) @ u_w

    # change in acceleration
    g = np.array([0, 0, -9.81])
    Rot_q = Rotation.from_euler('zxy',q, degrees=False).as_matrix()
    p_ddot = g + Rot_q @ u_a

    # change in gyroscope bias
    Q_g = 1e-6*np.eye(3)    # covariace
    mu_g = np.zeros(3)      # mean
    bg_dot = np.random.multivariate_normal(mu_g,Q_g)

    # change in accelerometer bias
    Q_a = 1e-6*np.eye(3)         # covariance
    mu_a = np.zeros(3)      # mean
    ba_dot = np.random.multivariate_normal(mu_a,Q_a)

    # change in state
    x_dot = np.hstack((p_dot,q_dot,p_ddot,bg_dot,ba_dot))

    return x_dot

def prediction(X_prev,P_prev,u_curr):
    # get sigma points
    X_s, W_m, W_c = getSigmaPts(X_prev,P_prev)

    # pass sigma points through process model
    X_s_est = np.full(X_s.shape,np.nan)
    for i in range(X_s.shape[0]):
        X_s_est[i] = process_model(X_s[i],u_curr[0],u_curr[1])

    # Compute the predicted mean and covariance using the weighted sum
    X_est = W_m @ X_s_est
    residual = X_s_est - X_est.reshape(1, -1)
    P_est = (W_c * residual.T ) @ residual

    return X_est, P_est, X_s_est

def obs_model(X_prev,R):
    n_obs = 6
    # init measurement matrix
    C = np.hstack((np.eye(n_obs),np.zeros((n_obs,len(X_prev)-n_obs))))

    Z_est = C @ X_prev + np.random.multivariate_normal(np.zeros(n_obs),R)

    return Z_est

def update(X_est,P_est,Z_m,X_s_est):
    n_obs = 6
    # measurement model covariance
    R = 1e-3*np.eye(n_obs)
    
    # get sigma points with estimated mean and covariance
    Z_s, W_m, W_c = getSigmaPts(X_est,P_est)

    # pass pts through measurement model
    Z_s_est = np.full((Z_s.shape[0],n_obs),np.nan)
    for i in range(Z_s.shape[0]):
        Z_s_est[i] = obs_model(Z_s[i],R)

    # mean of estimated sigma points
    Z_est = W_m @ Z_s_est

    # innovation matrix
    residual_z = Z_s_est - Z_est.reshape(1,-1)
    S = W_c*residual_z.T @ residual_z + R
    
    # kalman gain
    residual_x = X_s_est - X_est.reshape(1,-1)
    Pxz = W_c * residual_x.T @ residual_z
    K = Pxz @ np.linalg.inv(S)

    # update state estimates
    X_updated = X_est + K @ (Z_m - Z_est)
    P_updated = P_est - K @ S @ K.T

    return X_updated, P_updated

# Load .mat file
file_name = 'studentdata1.mat'
curr_path = str(os.path.dirname(os.path.abspath(__file__)))
file_path = curr_path + '/data/' + file_name
mat_data = scipy.io.loadmat(file_path,simplify_cells=True)

# extract data
sensor_data = mat_data['data']
vicon_data = mat_data['vicon']
time_vicon = mat_data['time']

# init state vector
X_prev = np.hstack((vicon_data[:9,0],1e-3*np.ones(6)))
# init covariance matrix
P_prev = 1e-1*np.eye(len(X_prev))
# init time
t_prev = time_vicon[0]

# filtered states
X_filtered = np.full((15,vicon_data.shape[1]),np.nan)
t_filtered = np.full(len(time_vicon),np.nan)

# imu reading noise
n_bg = 0
n_ba = 0
for i in range(len(sensor_data)):
    data = sensor_data[i]
    # parse imu data data
    w = data['rpy']
    a = data['acc']
    w_hat = w + X_prev[9:12] + n_bg
    Rot = Rotation.from_euler('zxy', X_prev[3:6], degrees=False).as_matrix()
    g = np.array([0,0,-9.81])
    a_hat = Rot.T @ (a-g) + X_prev[12:] + n_ba

    u_curr = np.vstack((w_hat,a_hat))
    t_filtered[i] = data['t']
    # get camera pose estimates
    pos, euler = pe.estimate_pose(data)

    X_est, P_est, X_s_est = prediction(X_prev,P_prev,u_curr)
    X_curr, P_curr = update(X_est,P_est,np.hstack((pos,euler)),X_s_est)

    X_filtered[:,i] = X_curr

    X_prev = X_curr
    P_prev = P_curr

pos_filtered = X_filtered[:3,:]
euler_filtered = X_filtered[3:6,:]

pe.plot_pose(pos_filtered,euler_filtered,t_filtered,vicon_data[:3,:],vicon_data[3:6,:],time_vicon)
