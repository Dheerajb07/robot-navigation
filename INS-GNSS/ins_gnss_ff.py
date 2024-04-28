import os
import numpy as np
import earth
from scipy.spatial.transform import Rotation
from plot_pose import plot_pose
from tqdm import tqdm

from ukf import UKF
from haversine import haversine, Unit

N_STATE = 12
N_OBS = 3

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
    e = X_prev[9:]
    
    ###### ATTITUDE UPDATE
    w_e = earth.RATE*np.array([0,0,1])
    omega_i_e = skew(w_e)
    
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

    X_curr = np.hstack((L_curr,l_curr,h_curr,q_curr,v_n_curr,e))

    return X_curr

def obs_model(X_prev,R):
    e = X_prev[9:]
    Z_est = e + np.random.multivariate_normal(np.zeros(len(R)),R)
    return Z_est

def wrap_angle(angle):
    while angle <= -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return angle

def ins_gnss_fb():
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
    ukf = UKF(N_STATE,N_OBS,propagation_model,obs_model)
    # process noise     
    Q = np.diag([10,10,10000,100,100,100,100,100,100,100,100,10000])
    # Q = 10*np.eye(N_STATE)
    # measurement model covariance
    R = 1e-3*np.diag([1,1,100])

    p = ground_truth[0,0:3].copy()
    q = ground_truth[0,3:6].copy()
    X_prev = np.hstack((p,q,np.zeros(N_STATE-6)))
    P_prev = np.diag([1,1,100,1,1,1,2,2,2,100,100,1000])
    t_prev = time[0]
    # P_prev = 1e-2*np.eye(N_STATE)

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

        X_est, P_est, X_s_est = ukf.prediction(X_prev,P_prev,w_curr,a_curr,dt,Q)
    
        # GNSS measurements
        p_m = gnss_NED[i][:3]
        
        # calc error
        p_est = X_est[:3]
        err = np.zeros(N_OBS)
        # wrap long and lat to -90 to 90
        p_est[0] = wrap_angle(p_est[0])
        p_est[1] = wrap_angle(p_est[1])
        dist = haversine(p_est[0:2],p_m[0:2],unit=Unit.DEGREES)
        err[0] = dist
        err[1] = dist
        err[2] = p_est[2] - p_m[2]

        X_curr, P_curr = ukf.update(X_est,P_est,err,X_s_est,R)

        X_filtered[:,i] = X_curr
        t_prev = t_curr
        X_prev = X_curr
        P_prev = P_curr

    pos_filtered = X_filtered[:3,:]
    euler_filtered = X_filtered[3:6,:]

    plot_pose(pos_filtered,euler_filtered,time,ground_truth[:,:3].T,ground_truth[:,3:6].T,time)

ins_gnss_fb()