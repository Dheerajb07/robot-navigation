import os
import numpy as np
import scipy.io
import pose_estimation as pe
import sympy as sp

def process_model_sym():
    # Init symbolic variables
    p = sp.Matrix([sp.Symbol('x'), sp.Symbol('y'), sp.Symbol('z')])
    q = sp.Matrix([sp.Symbol('phi'), sp.Symbol('theta'), sp.Symbol('psi')])
    
    p_dot = sp.Matrix([sp.Symbol('x_dot'), sp.Symbol('y_dot'), sp.Symbol('z_dot')])

    # bias variables
    bg = sp.Matrix([sp.Symbol('bg1'),sp.Symbol('bg2'),sp.Symbol('bg3')])
    ba = sp.Matrix([sp.Symbol('ba1'),sp.Symbol('ba2'),sp.Symbol('ba3')])

    # state vector
    X = sp.Matrix([p,q,p_dot,bg,ba])

    # inputs
    u_w = sp.Matrix([sp.Symbol('u_w1'), sp.Symbol('u_w2'), sp.Symbol('u_w3')])
    u_a = sp.Matrix([sp.Symbol('u_a1'), sp.Symbol('u_a2'), sp.Symbol('u_a3')])
    # input vector
    # U = sp.Matrix([u_w,u_a])
    
    phi, theta, psi = q
    # change in angular velocity
    G_q = sp.Matrix([[sp.cos(theta), 0, -sp.cos(phi)*sp.sin(theta)],
                    [0,            1,  sp.sin(phi)             ],
                    [sp.sin(theta), 0,  sp.cos(phi)*sp.cos(theta)]])   
    q_dot = G_q.inv() @ u_w

    # change in acceleration
    g = sp.Matrix([0, 0, -9.81])
    # Compute individual rotation matrices
    R_z = sp.Matrix([[sp.cos(psi), -sp.sin(psi), 0],
                    [sp.sin(psi), sp.cos(psi), 0],
                    [0, 0, 1]])

    R_x = sp.Matrix([[1, 0, 0],
                    [0, sp.cos(theta), -sp.sin(theta)],
                    [0, sp.sin(theta), sp.cos(theta)]])

    R_y = sp.Matrix([[sp.cos(phi), 0, sp.sin(phi)],
                    [0, 1, 0],
                    [-sp.sin(phi), 0, sp.cos(phi)]])
    # Compute composite rotation matrix
    R_q = R_z @ R_x @ R_y
    p_ddot = g + R_q @ u_a

    # change in gyroscope bias
    bg_dot = sp.Matrix([sp.Symbol('Ng1'), sp.Symbol('Ng2'), sp.Symbol('Ng3')])

    # change in accelerometer bias
    ba_dot = sp.Matrix([sp.Symbol('Na1'), sp.Symbol('Na2'), sp.Symbol('Na3')])

    # change in state
    X_dot = sp.Matrix([p_dot, q_dot, p_ddot, bg_dot, ba_dot])

    return X_dot,X

def obs_model_sym():
    Z = sp.Matrix([sp.symbols('x y z phi theta psi')])
    return Z

class EKF():
    def __init__(self,N_STATE,N_OBS,process_model,obs_model):
        self.N_STATE = N_STATE
        self.N_OBS = N_OBS
        # init state and covariance matrices
        self.X = np.zeros(N_STATE)
        self.P = np.eye(N_STATE)
        self.X_est = np.zeros(N_STATE)
        self.P_est = np.eye(N_STATE)
        # init process & measurement noise
        self.Q = np.eye(N_STATE)
        self.R = np.eye(N_OBS)
        # init Kalman gain
        self.K = np.zeros((N_STATE,N_OBS))
        # init state,input and measurement matrices
        self.A = np.zeros((N_STATE,N_STATE))
        self.C = np.zeros((N_STATE,N_OBS))
        
        # calc symbolic model matrices
        self.f_sym,self.x_sym = process_model()
        self.g_sym = obs_model()
        self.A_sym = self.f_sym.jacobian(self.x_sym)
        self.C = np.array(self.g_sym.jacobian(self.x_sym),dtype=float)

    def eval_f(self,x,u):
        # change in gyroscope bias
        Q_g = 1e-6*np.eye(3)    # covariace
        mu_g = np.zeros(3)      # mean
        bg_dot = np.random.multivariate_normal(mu_g,Q_g)
        # change in accelerometer bias
        Q_a = 1e-6*np.eye(3)    # covariance
        mu_a = np.zeros(3)      # mean
        ba_dot = np.random.multivariate_normal(mu_a,Q_a)

        return np.array(self.f_sym.subs({
            'phi': x[3],
            'theta': x[4],
            'psi': x[5],
            'x_dot': x[6],
            'y_dot': x[7],
            'z_dot': x[8],
            'u_w1': u[0],
            'u_w2': u[1],
            'u_w3': u[2],
            'u_a1': u[3],
            'u_a2': u[4],
            'u_a3': u[5],
            'Ng1': bg_dot[0],
            'Ng2': bg_dot[1],
            'Ng3': bg_dot[2],
            'Na1': ba_dot[0],
            'Na2': ba_dot[1],
            'Na3': ba_dot[2],
            }),dtype=float).reshape(-1)

    def eval_A(self,x,u):
        return np.array(self.A_sym.subs({
            'phi': x[3],
            'theta': x[4],
            'psi': x[5],
            'x_dot': x[6],
            'y_dot': x[7],
            'z_dot': x[8],
            'u_w1': u[0],
            'u_w2': u[1],
            'u_w3': u[2],
            'u_a1': u[3],
            'u_a2': u[4],
            'u_a3': u[5],
            'Ng1': np.random
            }),dtype=float)

    def predict(self,u,dt):
        # evaluate symbolic vars
        self.f = self.eval_f(self.X,u)
        self.A = self.eval_A(self.X,u)
        
        # predict states
        self.X_est = self.X + self.f*dt
        
        # predict covariance
        self.F = np.eye(self.N_STATE) + self.A*dt
        self.P_est = self.F @ self.P @ self.F.T + self.Q

    def update(self,z):
        # compute kalman gain
        self.K = self.P_est @ self.C.T @ np.linalg.inv(self.C @ self.P_est @ self.C.T + self.R)
        # update estimate
        self.X = self.X_est + self.K @ (z - self.C @ self.X_est)
        # update uncertainity
        self.P = (np.eye(self.N_STATE) - self.K @ self.C) @ self.P_est

def ExtendedKalmanFilter(FILENAME):
    ## PARAMS
    N_STATE = 15           # state vector length
    N_OBS = 6              # observation vector length

    # Load .mat file
    curr_path = str(os.path.dirname(os.path.abspath(__file__)))
    file_path = curr_path + '/data/' + FILENAME
    mat_data = scipy.io.loadmat(file_path,simplify_cells=True)

    # extract data
    sensor_data = mat_data['data']
    vicon_data = mat_data['vicon']
    time_vicon = mat_data['time']

    # init EKF
    ekf = EKF(N_STATE,N_OBS,process_model_sym,obs_model_sym)
    # init process noise
    ekf.Q *= 1e-3
    # init measurement noise
    ekf.R *= 0.5*1e-2
    # init state vector
    ekf.X = np.hstack((vicon_data[:9,0],0*np.ones(6)))
    # init covariance matrix
    ekf.P *= 1e-3
    # init time
    t_prev = time_vicon[0]

    # filtered states
    X_filtered = np.full((15,vicon_data.shape[1]),np.nan)
    t_filtered = np.full(len(time_vicon),np.nan)

    for i in range(len(sensor_data)):
        data = sensor_data[i]
        # time
        t_curr = data['t']
        dt = t_curr - t_prev 
        # retreive control input, measurement and time for current step
        if FILENAME=='studentdata0.mat':
            w = data['drpy']
        else:
            w = data['omg']
        a = data['acc']
        u_curr = np.hstack((w,a))
        # get camera pose estimates
        pos, euler = pe.estimate_pose(data)
        z_curr = np.hstack((pos,euler))

        if len(z_curr)==0:
            X_filtered[:,i] = np.full(len(ekf.X),np.nan)
            t_filtered[i] = t_curr
            t_prev = t_curr
            continue

        ekf.predict(u_curr,dt)
        ekf.update(z_curr)

        X_filtered[:,i] = ekf.X
        t_filtered[i] = t_curr
        t_prev = t_curr

    pos_filtered = X_filtered[:3,:]
    euler_filtered = X_filtered[3:6,:]

    pe.plot_pose(pos_filtered,euler_filtered,t_filtered,vicon_data[:3,:],vicon_data[3:6,:],time_vicon)