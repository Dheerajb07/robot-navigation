import os
import numpy as np
import earth
from scipy.spatial.transform import Rotation
from scipy.linalg import cholesky
from plot_pose import plot_pose
from tqdm import tqdm

def skew(v):
    v1, v2, v3 = v
    V = np.array([[0, -v3, v2],
                  [v3, 0, -v1],
                  [-v2, v1, 0]])
    return V

def wrap_to_360(angle):
    return angle % 360

def propagation_model(X_prev,w_i_b,f_b,dt):
    # extract prev states
    L,l,h = X_prev[:3]
    q = X_prev[3:6]
    v_n = X_prev[6:9]
    v_N,v_E,v_D = v_n

    # attitude
    b_g = X_prev[12:]
    w_i_b = w_i_b - b_g
    
    w_e = earth.RATE*np.array([0,0,1])
    omega_i_e = skew(w_e)
    
    ###### ATTITUDE UPDATE
    R_N,R_E,R_P = earth.principal_radii(L,h)

    w_e_n = np.zeros(3)
    w_e_n[0] = v_E / (R_E + h)
    w_e_n[1] = -v_N / (R_N + h)
    w_e_n[2] = -v_E * np.tan(np.deg2rad(L)) / (R_E + h)

    omega_e_n = skew(w_e_n)
    omega_i_b = skew(w_i_b)

    R_prev = Rotation.from_euler('xyz',q,degrees=True).as_matrix()

    R_curr = R_prev @ (np.eye(3) + omega_i_b * dt) - (omega_i_e + omega_e_n) @ R_prev * dt

    q_curr = Rotation.from_matrix(R_curr).as_euler('xyz',degrees=True)
    q_curr = wrap_to_360(q_curr)

    ###### VELOCITY UPDATE
    b_a = X_prev[9:12]
    f_b = f_b - b_a

    f_n = 0.5 * (R_prev + R_curr) @ f_b
    g_n = earth.gravity_n(L,h) 
    v_n_curr = v_n + dt * (f_n + g_n - (omega_e_n + 2*omega_i_e) @ v_n)

    ###### POSITION UPDATE
    v_N_curr,v_E_curr,v_D_curr = v_n_curr
    
    h_curr = h - dt/2 * (v_D + v_D_curr)

    L_curr_rad = np.deg2rad(L) + dt/2 * (v_N/(R_N + h) + v_N_curr/(R_N + h_curr))
    L_curr = np.rad2deg(L_curr_rad)

    R_N_curr,R_E_curr,R_P_curr = earth.principal_radii(L_curr,h_curr)
    l_curr_rad = np.deg2rad(l) + dt/2 * (v_E/((R_E + h)*np.cos(np.deg2rad(L))) + v_E_curr/((R_E_curr + h_curr)*np.cos(np.deg2rad(L_curr))))
    l_curr = np.rad2deg(l_curr_rad)

    X_curr = np.hstack((L_curr,l_curr,h_curr,q_curr,v_n_curr,b_a,b_g))

    return X_curr

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

    sqrt_covariance = cholesky((n + lambda_) * covariance)

    for i in range(n):
        sigma_points[i+1] = mean + sqrt_covariance[i]
        sigma_points[i+1+n] = mean - sqrt_covariance[i]

        weights_mean[i+1] = 1/(2*(n + lambda_))
        weights_covariance[i+1] = weights_mean[i+1]

        weights_mean[i+1+n] = 1/(2*(n + lambda_))
        weights_covariance[i+1+n] = weights_mean[i+1+n]

    return sigma_points, weights_mean, weights_covariance

def prediction(X_prev,P_prev,w_curr,a_curr,dt):
    # process noise     
    Q = np.diag([100,100,10000,0.5,0.5,0.5,1,1,1,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3])
    # Q = 1e-2*np.eye(N_STATE)                          
    # Q = 1e3*np.eye(N_STATE)
    # get sigma points
    X_s, W_m, W_c = getSigmaPts(X_prev,P_prev)

    # pass sigma points through process model
    X_s_est = np.full(X_s.shape,np.nan)
    for i in range(X_s.shape[0]):
        X_s_est[i] = propagation_model(X_s[i],w_curr,a_curr,dt)

    # Compute the predicted mean and covariance using the weighted sum
    X_est = W_m @ X_s_est
    # residual_1 = X_s_est - X_est.reshape(1, -1)
    P_est = np.zeros((N_STATE,N_STATE))
    for i in range(X_s_est.shape[0]):
        residual = (X_s_est[i,:] - X_est).reshape((N_STATE,1))
        P_est += W_c[i] * residual @ residual.T
    P_est += Q
    # P_est_1 = (W_c * residual_1.T) @ residual_1

    return X_est, P_est, X_s_est

def obs_model(X_prev,R):
    p = X_prev[:3]
    pdot= X_prev[6:9]
    Z_est = np.hstack((p,pdot)) + np.random.multivariate_normal(np.zeros(N_OBS),R)
    return Z_est

def update(X_est,P_est,Z_m,X_s_est):
    # measurement model covariance
    # R = np.diag([1,1,10,2.5,2.5,2.5])
    R = 1e-3*np.eye(N_OBS)
    
    # get sigma points with estimated mean and covariance
    Z_s, W_m, W_c = getSigmaPts(X_est,P_est)

    # pass pts through measurement model
    Z_s_est = np.full((Z_s.shape[0],N_OBS),np.nan)
    for i in range(Z_s.shape[0]):
        Z_s_est[i] = obs_model(Z_s[i],R)

    # mean of estimated sigma points
    Z_est = W_m @ Z_s_est

    # innovation matrix
    S = np.zeros((N_OBS,N_OBS))
    Pxz = np.zeros((N_STATE,N_OBS))
    for i in range(Z_s_est.shape[0]):
        residual_z = (Z_s_est[i,:] - Z_est).reshape((N_OBS,1))
        residual_x = (X_s_est[i,:] - X_est).reshape((N_STATE,1))

        S += W_c[i] * residual_z @ residual_z.T
        Pxz += W_c[i] * residual_x @ residual_z.T
        
    # residual_z = Z_s_est - Z_est.reshape(1,-1)
    # S = W_c*residual_z.T @ residual_z + R
    # residual_x = X_s_est - X_est.reshape(1,-1)
    # Pxz = W_c * residual_x.T @ residual_z
    
    # kalman gain       
    K = Pxz @ np.linalg.inv(S)

    # update state estimates
    X_updated = X_est + K @ (Z_m - Z_est)
    P_updated = P_est - K @ S @ K.T

    return X_updated, P_updated

N_STATE = 15
N_OBS = 6
# load data
curr_path = str(os.path.dirname(os.path.abspath(__file__)))
file_name = '/trajectory_data.csv'
file_path = curr_path + file_name
data = np.genfromtxt(file_path, delimiter=',')

# extract data
# time
time = data[1:,0]
# ground truth
ground_truth = data[1:,1:7] # lat,lon,alt,roll,pitch,yaw
# imu data
w_imu = data[1:,7:10]
acc_imu = data[1:,10:13]
# gnss data
gnss_NED = data[1:,13:]
# vel_NED = data[1:,16:]

# init filter
t_prev = time[0]
p = ground_truth[0,0:3].copy()
q = ground_truth[0,3:6].copy()

X_prev = np.hstack((p,q,np.zeros(9)))
P_prev = 1e-2*np.eye(N_STATE)
# P_prev = np.diag([1e-2,1e-2,1e-2,1e-2,1e-2,1e-2,10,10,10,1e-2,1e-2,1e-2,1e-2,1e-2,1e-2])

X_curr = np.full((N_STATE,len(time)),np.nan)
# init filtered state var
X_filtered = np.full((N_STATE,len(time)),np.nan)

for i in tqdm(range(len(time))):
    # time
    t_curr = time[i]
    dt = t_curr-t_prev
    # retreive imu data
    w_curr = w_imu[i]
    a_curr = acc_imu[i]

    # GNSS measurements
    z_curr = gnss_NED[i]

    X_est, P_est, X_s_est = prediction(X_prev,P_prev,w_curr,a_curr,dt)
    X_curr, P_curr = update(X_est,P_est,z_curr,X_s_est)

    X_filtered[:,i] = X_curr
    t_prev = t_curr
    X_prev = X_curr
    P_prev = P_curr

pos_filtered = X_filtered[:3,:]
euler_filtered = X_filtered[3:6,:]
# np.save()
plot_pose(pos_filtered,euler_filtered,time,ground_truth[:,:3].T,ground_truth[:,3:6].T,time)
