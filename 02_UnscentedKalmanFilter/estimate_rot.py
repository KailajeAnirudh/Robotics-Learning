import numpy as np
from scipy import io
from quaternion import Quaternion
import math
import scipy
from scipy.constants import g

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]

    #Initialize Sensor Models
    ax_bias, ay_bias, az_bias, gx_bias, gy_bias, gz_bias = np.array([511.61372013, 500.3814942 , 502.54967008, 371.85559987, 375.26009382, 375.08646026])
    ax_sensitivity, ay_sensitivity, az_sensitivity, gx_sensitivity, gy_sensitivity, gz_sensitivity = np.array([ -34.3384566 ,  -33.9624254 ,   33.94389528, -183.63930862, -147.10729754, -180.10267387])

    ax = get_val(accel[0], ax_bias, ax_sensitivity)
    ay = get_val(accel[1], ay_bias, ay_sensitivity)
    az = get_val(accel[2], az_bias, az_sensitivity)
    gx = get_val(gyro[0], gx_bias, gx_sensitivity)
    gy = get_val(gyro[2], gy_bias, gy_sensitivity)
    gz = get_val(gyro[1], gz_bias, gz_sensitivity)
    sensor_values = np.vstack((ax, ay, az, gx, gy, gz)).T

    t = imu['ts'].flatten()

    #Initialize State
    # Q = np.diag([0.5, 0.4, 0.2, 4, 4.5, 7]).astype(float)
    # R =  np.diag([1.5, 2.5, 3.5, 15, 3, 0.1]).astype(float)

    # Q = np.diag([225, 225, 1815, 150, 150, 225]).astype(float)/34
    Q = np.diag([225/34, 225/34, 1815/34, 310/34, 270/34, 574/34]).astype(float)
    # R =  np.diag([2300,2300,2300,1000,1000,800]).astype(float)/100
    R = 5*np.diag([2.12152214, 2.58091551, 1.4783665 , 0.25833854, 0.31113115,
       0.27795984])

    #Initialize State Noise
    mu_kk = np.zeros(7,)
    mu_kk[0] = 1
    mu_kk[4:] = [gx[0], gy[0], gz[0]]
    Sigma_kk = np.eye(6)
    euler_pred = np.zeros((T, 3))
    quats = []
    ang_vels = []
    covs = []

    for i in range(1, T):
        dt = t[i] - t[i-1]
        X_sigma_kk = propagateStateNoise(mu_kk, Sigma_kk, R, dt)
        Y_sigma_k1k = propagateSigmapoints(mu_kk, X_sigma_kk, dt)
        mu_k1k, Sigma_k1k = get_state_mean(Y_sigma_k1k, mu_kk)
        mu_k1k1, Sigma_k1k1 = measurement_update(sensor_values[i], mu_k1k, Sigma_k1k, R, Q, dt)
        mu_kk = mu_k1k1
        Sigma_kk = Sigma_k1k1
        q = Quaternion(mu_k1k1[0], mu_k1k1[1:4])
        q.normalize()
        euler_pred[i] = q.euler_angles()
        quats.append(q)
        ang_vels.append(mu_k1k1[4:])
        covs.append(Sigma_k1k1)
    
    roll, pitch, yaw = euler_pred.T

    # roll, pitch, yaw are numpy arrays of length T
    return roll,pitch,yaw

def propagateStateNoise(x_k, Sigma_kk, R, dt):
    """
    This function propagates the state noise using Process Noise and state covariance
    Input:
        x_k: (7,) vector indicating State at time k
        Sigma_kk: (6,6) matrix  of State Covariance
        R: (6,6) vector of Process Noise
        dt: time step according to the IMU
    Output:
        X_Sigma: (12,7) matrix containing Sigma points representing the state noise
    """
    n, _ = Sigma_kk.shape
    S = scipy.linalg.sqrtm(Sigma_kk+ R*dt)
    sigma_vectors = np.hstack((S*np.sqrt(2*n), -S*np.sqrt(2*n)))
    x_k_quat = Quaternion(float(x_k[0]), x_k[1:4])
    q = Quaternion()
    X_sigma = np.zeros((2*n, 7))
    for i in range(sigma_vectors.shape[1]):
        q.from_axis_angle(sigma_vectors[:3, i])
        # q.normalize()
        # x_k_quat.normalize()
        quat_product = x_k_quat*q
        # quat_product.normalize()
        X_sigma[i, :4] = (quat_product).q
        X_sigma[i, 4:] = x_k[4:] + sigma_vectors[3:, i]
    return X_sigma


def propagateSigmapoints(x_kk, X_sigmapts, dt):
    """
    This function propagates the Sigma points using the gyro data
    Input:
        x_kk: (7,) vector indicating State at time k
        X_sigmapts: (12,7) matrix containing Sigma points representing the state noise
        dt: time interval
    Output:
        Y_sigmapts: (12,7) matrix containing Sigma points representing the state noise
    """

    Y_sigmapts = np.zeros_like(X_sigmapts)
    alpha_delta = np.linalg.norm(X_sigmapts[:, 4:], axis = 1)*dt
    axis = X_sigmapts[:, 4:]/np.linalg.norm(X_sigmapts[:, 4:], axis = 1).reshape(-1,1)
    for i in range(X_sigmapts.shape[0]):
        q_delta = Quaternion(scalar = np.cos(alpha_delta[i]/2), vec = np.sin(alpha_delta[i]/2)*axis[i])
        # q_delta.normalize()
        quat_product = Quaternion(float(X_sigmapts[i, 0]), X_sigmapts[i, 1:4])*q_delta
        # quat_product.normalize()
        Y_sigmapts[i, :4] = (quat_product).q
        Y_sigmapts[i, 4:] = X_sigmapts[i, 4:]
    return Y_sigmapts
   
def get_mean_quaternion(qs, q0, epsilon=2e-3, max_iter=60):
    """
    This function computes the mean quaternion from a set of quaternions
    Input:
        qs: (N,) list of quaternions of Quarternion class instances
        q0: initial guess of the mean quaternion (Quaternion class instance)
        epsilon: threshold for convergence
        max_iter: maximum number of iterations
    Output:
        Q_t: mean quaternion (Quaternion class instance)
        QCov: (3,3) matrix - Covariance of axis orientation error with respect to the mean quaternion axes
    """
    Q_t = q0
    # Q_t.normalize()
    num_quats = len(qs)
    error = np.zeros((num_quats, 3))
    e = np.array([1, 1, 1])
    for i in range(max_iter):
        if np.linalg.norm(e) < epsilon:
            break
        e = np.zeros(3)
        e_i_axes = np.zeros((num_quats, 3))
        for idx, q_i in enumerate(qs):
            e_i = q_i * Q_t.inv()
            e_i_axes[idx, :] = e_i.axis_angle()
            e += e_i.axis_angle()
        e = e/num_quats
        e_Q = Quaternion(1, [0,0,0])
        e_Q.from_axis_angle(e)
        Q_t = e_Q * Q_t
        
        Qcov = np.cov((e_i_axes).T, bias = True)/(2*num_quats)
    return Q_t, Qcov

def get_state_mean(Y_sigmapts, x_kk):
    """
    This function calculates the mean, and the covariance of the transformed Sigma points
    Input:
        Y_sigmapts: (12, 7) matrix containing Sigma points representing the state noise
        x_kk: (7,) vector indicating State at time k
    Output:
        mu_k1k: (7,) vector indicating State at time k
        Sigma_k1k: (3,3) matrix  of State Covariance
    """
    mu_k1k = np.zeros(7)
    Sigma_k1k = np.zeros((6,6))
    qs = [Quaternion(float(Y_sigmapts[i, 0]), Y_sigmapts[i, 1:4]) for i in range(Y_sigmapts.shape[0])]

    q_mean, axesCovar = get_mean_quaternion(qs, Quaternion(x_kk[0], x_kk[1:4]))

    mu_k1k[:4] = q_mean.q
    mu_k1k[4:] = Y_sigmapts[:, 4:].mean(axis = 0)

    Sigma_k1k[:3, :3] = axesCovar
    Sigma_k1k[3:, 3:] = np.cov((Y_sigmapts[:, 4:] - mu_k1k[4:]).T, bias= True)/(2*Y_sigmapts.shape[0])

    return mu_k1k, Sigma_k1k
    
    
def measurementModel(x_k, Q):
    """
    This function calculates the measurement model
    Input:
        x_k: (7,) vector indicating State at time k
        Q: (6,6) vector of Measurement Noise
    Output:
        z_k: (6,) vector indicating the measurement at time k
    """
    #Accelerometer Model
    z_k = np.zeros(6,)
    g_quat = Quaternion(scalar = 0.0, vec = [0,0,g])
    g_prime = (Quaternion(x_k[0], x_k[1:4]).inv()*g_quat*Quaternion(x_k[0], x_k[1:4])).vec()
    z_k[:3] = g_prime

    #Gyroscope Model
    z_k[3:] = x_k[4:]
    if np.all(Q != 0):
        z_k += np.random.multivariate_normal(np.zeros(6), Q, 1).reshape(z_k.shape)

    return z_k   

def measurement_update(sensor_values, mu_k1k, Sigma_k1k, R, Q, dt):
    """
    This function performs the measurement update step
    Input:
        sensor_values: (6,) vector indicating the measurement at time k
        mu_k1k: (7,) vector indicating State mean after dynamics propagation
        Sigma_k1k: (6,6) matrix  of State Covariance after dynamics propagation
        R: (6,6) matrix of Measurement Noise
        Q: (6,6) matrix of Process Noise
    Output:
        mu_k1k1: (7,) vector indicating State mean after measurement update
        Sigma_k1k1: (6,6) matrix  of State Covariance after measurement update
    """
    X_sigma_k1k = propagateStateNoise(mu_k1k, Sigma_k1k, 0*R, dt)
    Z_sigma_k1k = np.zeros((X_sigma_k1k.shape[0], 6))
    
    for i in range(X_sigma_k1k.shape[0]):
        Z_sigma_k1k[i] = measurementModel(X_sigma_k1k[i], np.zeros((6,6)))

    
    Z_sigma_k1k = Z_sigma_k1k.T
    Z_sigma_mean = Z_sigma_k1k.mean(axis = 1).reshape(6,1)
    Z_sigma_diff = Z_sigma_k1k - Z_sigma_mean
    P_zz = (Z_sigma_diff @ Z_sigma_diff.T/(2*X_sigma_k1k.shape[0])) + Q
    

    AxisVectorRep_Sigmapts = np.zeros((X_sigma_k1k.shape[0], 6))
    AxisVectorRep_Sigmapts[:, :] = X_sigma_k1k[:, 1:]
    # for i in range(X_sigma_k1k.shape[0]):
    #     AxisVectorRep_Sigmapts[i][:3] = Quaternion(X_sigma_k1k[i, 0], X_sigma_k1k[i, 1:4]).axis_angle()
    AxisVectorRep_Sigmapts_mean = AxisVectorRep_Sigmapts.mean(axis = 0)

    
    # P_xz = np.zeros((6,6))
    P_xz = (AxisVectorRep_Sigmapts-AxisVectorRep_Sigmapts_mean).T @ (Z_sigma_k1k-Z_sigma_mean).T/(2*X_sigma_k1k.shape[0])

    K = P_xz @ np.linalg.inv(P_zz)

    StateUpdate = K @ (sensor_values.reshape(6,1) - Z_sigma_mean.reshape(6,1))
    mu_k1k1 = np.zeros(7)
    q = Quaternion()
    q.from_axis_angle(StateUpdate[:3].flatten())
    mu_k1k1[:4] = (q*Quaternion(mu_k1k[0], mu_k1k[1:4])).q
    mu_k1k1[4:] = StateUpdate[3:].flatten() + mu_k1k[4:].flatten()
 
    Sigma_k1k1 = Sigma_k1k - K @ P_zz @ K.T
    return mu_k1k1, Sigma_k1k1

def get_val(raw, bias, alpha):
    return (raw-bias)*3300/(1023*alpha)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    num = 3
    vicon = io.loadmat("vicon/viconRot" + str(num) + ".mat")
    roll,pitch,yaw = estimate_rot(num)

    vicon2Sens = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    r = []
    p = []
    y = []
    quat = Quaternion()
    for i in range(vicon["rots"].shape[-1]):
        R = vicon["rots"][:, :, i].reshape(3, 3)
        quat.from_rotm(R)
        ang = quat.euler_angles()
        r.append(float(ang[0]))
        p.append(float(ang[1]))
        y.append(float(ang[2]))
    r = np.array(r)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(r[:roll.shape[0]])
    plt.title("Roll")
    plt.plot(roll)

    plt.subplot(3,1,2)
    plt.plot(p[:roll.shape[0]])
    plt.title("Pitch")
    plt.plot(pitch)

    plt.subplot(3,1,3)
    plt.plot(y[:roll.shape[0]])
    plt.plot(yaw)
    plt.title("Yaw")
    # plt.plot(ey,'--')
    plt.show()