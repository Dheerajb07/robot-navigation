import numpy as np
import sympy as sp
from sympy.stats import Normal,sample

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
    U = sp.Matrix([u_w,u_a])
    
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
    Q_g = 1e-3    # covariace
    mu_g = 0      # mean
    bg_dot1 = Normal('Ng1',mu_g,Q_g)
    bg_dot2 = Normal('Ng2',mu_g,Q_g)
    bg_dot3 = Normal('Ng3',mu_g,Q_g)
    bg_dot = sp.Matrix([bg_dot1,bg_dot2,bg_dot3])

    # change in accelerometer bias
    Q_a = 1e-3    # covariance
    mu_a = 0     # mean
    ba_dot1 = Normal('Na1',mu_a,Q_a)
    ba_dot2= Normal('Na2',mu_a,Q_a)
    ba_dot3 = Normal('Na3',mu_a,Q_a)
    ba_dot = sp.Matrix([ba_dot1,ba_dot2,ba_dot3])
    
    # change in state
    X_dot = sp.Matrix([p_dot, q_dot, p_ddot, bg_dot, ba_dot])

    return X_dot,X

def obs_model_sym():
    Z = sp.Matrix([sp.symbols('x y z phi theta psi')])
    return Z

X_prev=np.zeros(15)
F,X = process_model_sym()
A = F.jacobian(X)
F.subs(('Ng1 Ng2 Ng3',[0,0,0]))


# print(sample(F[9]))
print(A)
# g = obs_model_sym()
# C = g.jacobian(X)

