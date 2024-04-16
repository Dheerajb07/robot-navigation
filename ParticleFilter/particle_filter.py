import numpy as np
import os
import scipy.io

def process_model(x_prev,u_w,u_a,dt):
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
    R_q = R_z @ R_x @ R_y
    # Rot_q = Rotation.from_euler('zxy',q, degrees=False).as_matrix()
    p_ddot = g + R_q @ u_a

    # change in gyroscope bias
    Q_g = 1e-6*np.eye(3)    # covariace
    mu_g = np.zeros(3)      # mean
    bg_dot = np.random.multivariate_normal(mu_g,Q_g)

    # change in accelerometer bias
    Q_a = 1e-6*np.eye(3)    # covariance
    mu_a = np.zeros(3)      # mean
    ba_dot = np.random.multivariate_normal(mu_a,Q_a)

    # change in state
    x_dot = np.hstack((p_dot,q_dot,p_ddot,bg_dot,ba_dot))

    return x_prev + x_dot*dt

def obs_model(X_prev,R):
    n_obs = 6
    # init measurement matrix
    C = np.hstack((np.eye(n_obs),np.zeros((n_obs,len(X_prev)-n_obs))))

    Z_est = C @ X_prev + np.random.multivariate_normal(np.zeros(n_obs),R)

    return Z_est

class PF():
    def __init__(self,N_STATE,N_OBS,N_PARTICLES,process_model,obs_model):
        # store params
        self.N_STATE = N_STATE
        self.N_OBS = N_OBS
        self.N_PARTICLES = N_PARTICLES

        self.prcoess_model = process_model
        self.obs_model = obs_model
        # init process & measurement noise
        self.Q = np.eye(N_STATE)
        self.R = np.eye(N_OBS)
        # init particles and weights
        self.init_particles()
    
    def init_particles(self):
        # Initialize an empty array to store particles
        self.X = np.zeros((self.N_PARTICLES, self.N_STATE)) 

        # Initialize position (x, y, z) within a reasonable range
        self.X[:, :3] = np.random.uniform(low=0, high=3, size=(self.N_PARTICLES,3))
        
        # Initialize orientation (roll, pitch, yaw) within a reasonable range
        self.X[:, 3:6] = np.random.uniform(low=-np.pi, high=np.pi, size=(self.N_PARTICLES,3))
        
        # Initialize linear velocity (vx, vy, vz) within a reasonable range
        self.X[:, 6:9] = np.random.uniform(low=-5, high=5, size=(self.N_PARTICLES,3))
        
        # Initialize angular velocity bias (bwx, bwy, bwz) within a reasonable range
        self.X[:, 9:12] = np.random.uniform(low=-0.01, high=0.01, size=(self.N_PARTICLES,3))
        
        # Initialize accelerometer bias (bax, bay, baz) within a reasonable range
        self.X[:, 12:] = np.random.uniform(low=-0.01, high=0.01, size=(self.N_PARTICLES,3))

        # Init equal weights to all particles
        self.W = np.ones(self.N_PARTICLES)/self.N_PARTICLES

    # def preditc(self,u,dt):
        # sample noisy inputs

## PARAMS
N_STATE = 15           # state vector length
N_OBS = 6              # observation vector length
FILENAME = 'studentdata1.mat'

# Load .mat file
curr_path = str(os.path.dirname(os.path.abspath(__file__)))
file_path = curr_path + '/data/' + FILENAME
mat_data = scipy.io.loadmat(file_path,simplify_cells=True)

# extract data
sensor_data = mat_data['data']
vicon_data = mat_data['vicon']
time_vicon = mat_data['time']

# init PF
# number of partciles
M = 1000
pf = PF(N_STATE,N_OBS,M,process_model,obs_model)
# init process noise
pf.Q *= 1e-3
# init measurement noise
pf.R *= 0.5*1e-2
# init time
t_prev = time_vicon[0]

