import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def kalman_filter(x_prev,u_curr,P_prev,y_curr,dt,flag):
  # init mass
  m = 0.027 # kg

  # init process matrices
  A = np.array([[1,0,0,dt,0,0],
                [0,1,0,0,dt,0],
                [0,0,1,0,0,dt],
                [0,0,0,1,0,0],
                [0,0,0,0,1,0],
                [0,0,0,0,0,1]])

  B = np.vstack((np.eye(3)*dt**2/(2*m),np.eye(3)*dt/m))

  # init measurement matrix
  if flag:
    # position measurements
    C = np.hstack((np.eye(3),np.zeros((3,3))))
  else:
    # velocity measurements
    C = np.hstack((np.zeros((3,3)),np.eye(3)))

  # init noise matrices
  Q = np.eye(6)
  R = np.eye(3)

  # prediction
  # predict state
  x_pred = A @ x_prev + B @ u_curr
  # preditc uncertainity
  P_pred = A @ P_prev @ A.T + Q

  # update
  # compute kalman gain
  K = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + R)
  # update estimate
  x= x_pred + K @ (y_curr - C @ x_pred)
  # update uncertainity
  P = (np.eye(6) - K @ C) @ P_pred

  return x,P

# import data from txt file into np array
data_file = '/kalman_filter_data_low_noise.txt'
data_file_path = os.path.dirname(os.path.abspath(__file__)) + data_file
data = np.genfromtxt(data_file_path, delimiter=',', dtype=None, encoding=None)

# var to store state
x = np.zeros((6,data.shape[0]))

# initial time
t_prev = data[0,0]

# initial position estimate
x[0:3,0] = data[0,4:]
# initial covariance/uncertainity
P_prev = 0.1*np.eye(6)

# loop over data
for i in range(1,data.shape[0]):
  u_curr = data[i,1:4].reshape(3,1)
  y_curr = data[i,4:].reshape(3,1)
  t_curr = data[i,0]
  dt = t_curr - t_prev
  # kalman filter
  [x_hat,P] = kalman_filter(x[:,i-1].reshape(6,1),u_curr,P_prev,y_curr,dt,True)

  # store current values
  x[:,i] = x_hat.reshape(1,-1)[0]
  P_prev = P
  t_prev = t_curr

  # Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.plot(x[0,:], x[1,:], x[2,:])


# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()