import numpy as np
import os
import scipy.io
from scipy import stats
import pose_estimation as pe
from scipy.spatial.transform import Rotation
from numpy import sin,cos
class PF():
    def __init__(self,N_STATE,N_OBS,N_INPUTS,N_PARTICLES):
        # store params
        self.N_STATE = N_STATE
        self.N_OBS = N_OBS
        self.N_INPUTS = N_INPUTS
        self.N_PARTICLES = N_PARTICLES

        # init input, process & measurement noise
        self.Q = np.eye(N_STATE)
        self.Q_u = np.eye(N_INPUTS)
        self.R = np.eye(N_OBS)
        # init particles and weights
        self.init_particles()
    
    def init_particles(self):
        # Initialize an empty array to store particles
        self.X = np.zeros((self.N_PARTICLES, self.N_STATE)) 

        # Initialize position (x, y, z) within a reasonable range
        self.X[:, :3] = np.random.uniform(low=0, high=3, size=(self.N_PARTICLES,3))
        
        # Initialize orientation (roll, pitch, yaw) within a reasonable range
        self.X[:, 3:6] = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(self.N_PARTICLES,3))
        
        # Initialize linear velocity (vx, vy, vz) within a reasonable range
        self.X[:, 6:9] = np.random.uniform(low=-1, high=1, size=(self.N_PARTICLES,3))
        
        # Initialize angular velocity bias (bwx, bwy, bwz) within a reasonable range
        self.X[:, 9:12] = np.random.uniform(low=-0.5, high=0.5, size=(self.N_PARTICLES,3))
        
        # Initialize accelerometer bias (bax, bay, baz) within a reasonable range
        self.X[:, 12:] = np.random.uniform(low=-0.5, high=0.5, size=(self.N_PARTICLES,3))

        # Init equal weights to all particles
        self.W = np.ones(self.N_PARTICLES)/self.N_PARTICLES

    def predict(self,u,dt):       
        # Extracting components from X_prev matrix
        q = self.X[:, 3:6]
        phi, theta, psi = q.T  # Transpose for broadcasting

        # Change in linear velocity
        p_dot = self.X[:, 6:9]

        # Add noise to inputs
        u_hat = u + np.random.multivariate_normal(np.zeros(len(self.Q_u)), self.Q_u, size=self.X.shape[0])
        u_w = u_hat[:,:3]
        u_a = u_hat[:,3:]
        # Change in angular velocity
        G_q = np.array([[np.cos(theta)        ,  np.zeros_like(theta) , -np.cos(phi) * np.sin(theta)],
                        [np.zeros_like(theta) ,  np.ones_like(theta)  ,  np.sin(phi)                ],
                        [np.sin(theta)        ,  np.zeros_like(theta) ,  np.cos(phi) * np.cos(theta)]])
        
        q_dot = (np.linalg.inv(G_q.transpose(2,0,1)) @ u_w[:,:,np.newaxis]).squeeze(axis=-1)

        g = np.array([0, 0, -9.81])
        # Compute composite rotation matrix
        R_q = np.array([[np.cos(psi) * np.cos(theta) - np.sin(phi) * np.sin(psi) * np.sin(theta),
                        -np.cos(phi) * np.sin(psi), np.cos(psi) * np.sin(theta) + np.cos(theta) * np.sin(phi) * np.sin(psi)],
                        [np.cos(theta) * np.sin(psi) + np.cos(psi) * np.sin(phi) * np.sin(theta),
                        np.cos(phi) * np.cos(psi), np.sin(psi) * np.sin(theta) - np.cos(psi) * np.cos(theta) * np.sin(phi)],
                        [-np.cos(phi) * np.sin(theta), np.sin(phi), np.cos(phi) * np.cos(theta)]])
        # Change in acceleration
        p_ddot = g + (R_q.transpose(2,0,1) @ u_a[:,:,np.newaxis]).squeeze(axis=-1)

        # Change in gyroscope bias
        Q_g = 1e-4 * np.eye(3)  # covariance
        mu_g = np.zeros(3)  # mean
        bg_dot = np.random.multivariate_normal(mu_g, Q_g, size=self.X.shape[0])

        # Change in accelerometer bias
        Q_a = 1e-4 * np.eye(3)  # covariance
        mu_a = np.zeros(3)  # mean
        ba_dot = np.random.multivariate_normal(mu_a, Q_a, size=self.X.shape[0])

        # Change in state
        x_dot = np.hstack((p_dot, q_dot, p_ddot, bg_dot, ba_dot)) + np.random.multivariate_normal(np.zeros(len(self.Q)), self.Q, size=self.X.shape[0])

        self.X += x_dot * dt

    def update(self,z):
        # pass particles through observation model
        z_est = self.X[:,:6]
            
        # update weight
        self.W = self.multivariate_gaussian_pdf(z_est,z,self.R)
        
        # normalize weights
        if not np.sum(self.W)==0:
            self.W /= np.sum(self.W)
        else:
            self.W = np.ones(self.N_PARTICLES)/self.N_PARTICLES

    def multivariate_gaussian_pdf(self,x,z,covariance):
        # Dimension of the distribution
        n = len(z)

        # Calculate the determinant and inverse of the covariance matrix
        det_covariance = np.linalg.det(covariance)
        inv_covariance = np.linalg.inv(covariance)

        # Calculate the exponent term of the PDF
        exponent = -0.5 * np.sum(((x - z) @ inv_covariance) * (x - z),axis=1)
        # Calculate the normalization constant
        normalization = 1.0 / np.sqrt((2 * np.pi) ** n * np.sqrt(det_covariance))

        # Calculate the PDF value
        pdf_value = normalization * np.exp(exponent)

        return pdf_value

    def estimate(self):
        # weighted average
        return np.average(self.X,weights=self.W,axis=0)
        # return np.mean(self.X,axis=0)
        # idx = np.argmax(self.W)
        # return self.X[idx]
    
    def resample(self):
        '''
        Low Variance Resampling
        '''
        X_new = np.full((self.N_PARTICLES,self.N_STATE),np.nan)
        # sample random number
        r = np.random.uniform(0,1/self.N_PARTICLES)
        # cumulative sum
        c = self.W[0]
        # index
        j = 0
        for i in range(self.N_PARTICLES):
            u = r + i/self.N_PARTICLES
            while (u > c):
                j += 1
                c += self.W[j]
            # store particle
            X_new[i] = self.X[j]
        
        # set resampled particle list
        self.X = X_new
        self.W = np.ones(self.N_PARTICLES)/self.N_PARTICLES
    
def ParticleFilter(N_PARTICLES,FILENAME):
    ## PARAMS
    N_STATE = 15           # state vector length
    N_OBS = 6              # observation vector length
    N_INPUTS = 6           # no of inputs

    # Load .mat file
    curr_path = str(os.path.dirname(os.path.abspath(__file__)))
    file_path = curr_path + '/data/' + FILENAME
    mat_data = scipy.io.loadmat(file_path,simplify_cells=True)

    # extract data
    sensor_data = mat_data['data']
    vicon_data = mat_data['vicon']
    time_vicon = mat_data['time']

    # init PF
    pf = PF(N_STATE,N_OBS,N_INPUTS,N_PARTICLES)
    # init process noise
    pf.Q = 500*np.diag([0.01,0.01,0.01,0.001,0.001,0.001,0.1,0.1,0.1,0.001,0.001,0.001,0.001,0.001,0.001]) # [0.01] * 3 + [0.002] * 3 + [0.1] * 3 + [0.001] * 3 + [0.001] * 3) * 500
    # init measurement noise
    pf.R = np.diag([0.005, 0.005, 0.01, 0.005, 0.01, 0.001])
    # init input noise
    pf.Q_u *= 1e-3
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
            X_filtered[:,i] = np.full(len(pf.X),np.nan)
            t_filtered[i] = t_curr
            t_prev = t_curr
            continue

        pf.predict(u_curr,dt)
        pf.update(z_curr)
        X_curr = pf.estimate()
        pf.resample()

        X_filtered[:,i] = X_curr
        t_filtered[i] = t_curr
        t_prev = t_curr
        # print('Pos: ', X_curr[:3],'Euler: ', X_curr[3:6])

    pos_filtered = X_filtered[:3,:]
    euler_filtered = X_filtered[3:6,:]

    pe.plot_pose(pos_filtered,euler_filtered,t_filtered,vicon_data[:3,:],vicon_data[3:6,:],time_vicon)

FILENAME = 'studentdata1.mat'
# number of partciles
M = 250

ParticleFilter(M,FILENAME)