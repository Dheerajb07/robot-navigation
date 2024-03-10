import os
import numpy as np
import scipy.io

# Load .mat file
file_name = 'studentdata0.mat'
curr_path = str(os.path.dirname(os.path.abspath(__file__)))
file_path = curr_path + '/data/' + file_name
mat_data = scipy.io.loadmat(file_path,simplify_cells=True)

sensor_data = mat_data['data']

# init state vector
x_init = np.zeros(15)

def process_model(x_prev,u_w,u_a):
    p = x_prev[:3]
    q = x_prev[3:6]

    # chnge in linear velocity
    p_dot = x_prev[6:9]

    # change in angular velocity
    G_q = np.array([[np.cos(q[1]), 0, -np.cos(q[0])*np.sin(q[1])]
                    [0,            1,  0                        ]
                    [np.sin(q[1]), 0,  np.cos(q[0]*np.cos(q[1]))]])   
    q_dot = np.linalg.inv(G_q) @ u_w

    # change in acceleration
    g = np.array([0, 0, -9.81])
    R_q = np.array([])
    p_ddot = g + R_q @ u_a

    # change in gyroscope bias
    Q_g = np.eye(3)         # covariace
    mu_g = np.zeros(3)      # mean
    bg_dot = np.random.multivariate_normal(mu_g,Q_g)

    # change in accelerometer bias
    Q_a = np.eye(3)         # covariance
    mu_a = np.zeros(3)      # mean
    ba_dot = np.random.multivariate_normal(mu_a,Q_a)

    # change in state
    x_dot = np.hstack((p_dot,q_dot,p_ddot,bg_dot,ba_dot))

    return x_dot