import numpy as np
from scipy.linalg import cholesky
from scipy.linalg import sqrtm

class UKF():
    def __init__(self,N_STATE,N_OBS,propagation_model,obs_model) -> None:
        self.N_STATE = N_STATE
        self.N_OBS = N_OBS
        self.propagation_model = propagation_model
        self.obs_model = obs_model

    def getSigmaPts(self,mean, covariance, kappa=1):
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

    def prediction(self,X_prev,P_prev,w_curr,a_curr,dt,Q):
        # get sigma points
        X_s, W_m, W_c = self.getSigmaPts(X_prev,P_prev)

        # pass sigma points through process model
        X_s_est = np.full(X_s.shape,np.nan)
        for i in range(X_s.shape[0]):
            X_s_est[i] = self.propagation_model(X_s[i],w_curr,a_curr,dt)

        # Compute the predicted mean and covariance using the weighted sum
        X_est = W_m @ X_s_est
        # residual_1 = X_s_est - X_est.reshape(1, -1)
        P_est = np.zeros((self.N_STATE,self.N_STATE))
        for i in range(X_s_est.shape[0]):
            residual = (X_s_est[i,:] - X_est).reshape((self.N_STATE,1))
            P_est += W_c[i] * residual @ residual.T
        P_est += Q
        # P_est_1 = (W_c * residual_1.T) @ residual_1

        return X_est, P_est, X_s_est

    def update(self,X_est,P_est,Z_m,X_s_est,R):
        # get sigma points with estimated mean and covariance
        Z_s, W_m, W_c = self.getSigmaPts(X_est,P_est)

        # pass pts through measurement model
        Z_s_est = np.full((Z_s.shape[0],self.N_OBS),np.nan)
        for i in range(Z_s.shape[0]):
            Z_s_est[i] = self.obs_model(Z_s[i],R)

        # mean of estimated sigma points
        Z_est = W_m @ Z_s_est

        # innovation matrix
        S = np.zeros((self.N_OBS,self.N_OBS))
        Pxz = np.zeros((self.N_STATE,self.N_OBS))
        for i in range(Z_s_est.shape[0]):
            residual_z = (Z_s_est[i,:] - Z_est).reshape((self.N_OBS,1))
            residual_x = (X_s_est[i,:] - X_est).reshape((self.N_STATE,1))

            S += W_c[i] * residual_z @ residual_z.T
            Pxz += W_c[i] * residual_x @ residual_z.T
            
        # kalman gain       
        K = Pxz @ np.linalg.inv(S)

        # update state estimates
        X_updated = X_est + K @ (Z_m - Z_est)
        P_updated = P_est - K @ S @ K.T

        return X_updated, P_updated
    
