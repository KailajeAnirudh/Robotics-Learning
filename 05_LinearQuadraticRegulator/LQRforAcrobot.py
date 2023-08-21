import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp


def wrap(q):
    return (q + np.pi) % (2*np.pi) - np.pi

# From Russ Tedrake's notes


class Acrobot():

    def __init__(self, params):

        self.dim = 2

        # Define params for the system
        self.p = params
        self.g = 10
        self.m1, self.m2, self.l1, self.l2, self.I1, self.I2, self.umax = \
            self.p['m1'], self.p['m2'], self.p['l1'], self.p['l2'], self.p['I1'], self.p['I2'], self.p['umax']
        self.g, self.xf, self.dt = self.p['g'], self.p['xf'], self.p['dt']

        # Hyperparameters for controllers

        # TODO Student's code here
        self.K = np.array([0, 0, 0])
        self.Q = np.eye(4)
        self.R = np.eye(1)

    def get_M(self, x):
        M = np.array([[self.I1 + self.I2 + self.m2*self.l1**2 + 2*self.m2*self.l1*self.l2/2*np.cos(x[1]),
                       self.I2 + self.m2*self.l1*self.l2/2*np.cos(x[1])],
                      [self.I2 + self.m2*self.l1*self.l2/2*np.cos(x[1]), self.I2]])
        return M

    def get_C(self, x):
        q, dq = x[:self.dim], x[self.dim:]
        C = np.array([[-2*self.m2*self.l1*self.l2/2*np.sin(q[1])*dq[1], -self.m2*self.l1*self.l2/2*np.sin(q[1])*dq[1]],
                      [self.m2*self.l1*self.l2/2*np.sin(q[1])*dq[0], 0]])
        return C

    def get_G(self, x):
        G = np.array([(self.m1*self.l1/2 + self.m2*self.l1)*self.g*np.sin(x[0]) + self.m2*self.g*self.l2/2*np.sin(x[0]+x[1]),
                      self.m2*self.g*self.l2/2*np.sin(x[0]+x[1])])
        return G

    def get_B(self,):
        B = np.array([0, 1])
        return B

    def dynamics_step(self, x, u):
        q, dq = x[:self.dim], x[self.dim:]

        M = self.get_M(x)

        C = self.get_C(x)

        G = self.get_G(x)
        B = self.get_B()

        xdot = np.hstack([dq,
                          np.dot(np.linalg.inv(M), B*u - np.dot(C, dq) - G)])

        return xdot

    def simulate(self, x, dt):
        # Simulate the open loop acrobot for one step
        def f(t, x):
            u = self.get_control_efforts(x)
            return self.dynamics_step(x, u)
        sol = solve_ivp(f, (0, dt), x, first_step=dt)
        return sol.y[:, -1].ravel()

    def get_control_efforts(self, x):
        """
        Calculate the control efforts applied to the acrobot

        params:
            x: current state, np.array of size (4,)
        returns:
            u: control effort applied to the robot, a scalar number
        """

        # TODO student's code here

        u = 0
        K, P = self.get_lqr_term()

        if (x-self.xf) @ P @ (x-self.xf) < 1.11313859e+05:
            u = -K @ (x - self.xf)
        else:
            x[0] = wrap(x[0])
            x[1] = wrap(x[1])
            u = self.get_swingup_input(x)

        u = np.clip(u, -self.umax, self.umax)

        return u

    def get_linearized_dynamics(self):
        """
        Calculate the linearized dynamics around the terminal condition such that

        x_dot approximately equals to Alin @ x + Blin @ u

        returns:
            Alin: 4X4 matrix 
            Blin: 4X1 matrix 
        """

        # TODO student's code here

        Alin = 0.0 * np.eye(4)
        Blin = np.zeros((4, 1))

        Alin[:2, 2:] = np.eye(2)
        dTdq = np.array([[self.g*(self.m1*self.l1*0.5 + self.m2*self.l1 + self.m2*self.l2*0.5), self.g*self.m2*self.l2*0.5],
                         [self.g*self.m2*self.l2*0.5, self.g*self.m2*self.l2*0.5]])
        Alin[2:, :2] = np.linalg.inv(self.get_M(self.xf)) @ dTdq

        Blin[2:, 0] = np.linalg.inv(self.get_M(self.xf)) @ self.get_B()

        return Alin, Blin

    def get_lqr_term(self):
        """
        Calculate the lqr terms for linearized dynamics:

        The control input and cost to go can be calculated as:
            u(dx) = -K dx
            V(dx) = dx^T P @ dx

        returns:
            K: 1X4 matrix
            P: 4x4 matrix
        """

        # TODO student's code here

        P = np.eye(4)
        K = np.zeros((1, 4))

        Alin, Blin = self.get_linearized_dynamics()
        P = sp.linalg.solve_continuous_are(Alin, Blin, self.Q, self.R)
        # print("P: ", P)
        # print("Eignevalues of P: ", np.linalg.eigvals(P))
        K = np.linalg.inv(self.R) @ Blin.T @ P

        return K, P

    # Spong's paper, energy based swingup
    def get_swingup_input(self, x):
        """
        Calculate the swingup input using energy shaping method

        params:
            x: curret state, (4,) np.arrary

        returns:
            u: the input applied to the robot, a scalar number
        """

        # TODO student's code here

        k1 = 2.0
        k2 = 2.0
        k3 = 2.0
        E_prime = self.energy(x) - self.energy(self.xf)
        u_prime = x[2]*E_prime
        q2dd = -k1*x[1] - k2*x[3] + k3*u_prime

        M = self.get_M(x)
        C = self.get_C(x)
        G = self.get_G(x)
        Tau = -C @ x[2:] - G

        u = (M[1, 1] - M[1, 0]*M[0, 1]/M[0, 0])*q2dd + \
            np.array([M[1, 0]/M[0, 0], -1]).reshape((1, 2)
                                                    ) @ Tau.reshape((2, 1))

        return u[0, 0]

    def energy(self, x):
        q, dq = x[:self.dim], x[self.dim:]
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s2, c2 = np.sin(q[1]), np.cos(q[1])

        T1 = 0.5*self.I1*dq[0]**2
        T2 = 0.5*(self.m2*self.l1**2 + self.I2 + 2*self.m2*self.l1*self.l2/2*c2)*dq[0]**2 \
            + 0.5*self.I2*dq[1]**2 \
            + (self.I2 + self.m2*self.l1*self.l2/2*c2)*dq[0]*dq[1]
        U = -self.m1*self.g*self.l1/2*c1 - self.m2*self.g * \
            (self.l1*c1 + self.l2/2*np.cos(q[0]+q[1]))
        return T1 + T2 + U


def plot_trajectory(t: np.ndarray, knees: np.ndarray, toes: np.ndarray, params: dict, dt: float):
    l1 = params['l1']
    l2 = params['l2']
    threshold = 0.5

    fig = plt.figure(1)
    ax = plt.axes()

    def draw_frame(i):
        ax.clear()
        ax.set_xlim(-(l1+l2+threshold), (l1+l2+threshold))
        ax.set_ylim(-(l1+l2+threshold), (l1+l2+threshold))
        ax.axhline(y=0, color='r', linestyle='-')
        ax.plot(0, 0, 'bo', label="joint")
        ax.plot(knees[0][i], knees[1][i], 'bo')
        ax.plot(knees[0][i]/2, knees[1][i]/2, 'go', label="center of mass")
        ax.plot([0, knees[0][i]], [0, knees[1][i]], 'k-', label="linkage")
        ax.plot(toes[0][i], toes[1][i], 'bo')
        ax.plot((toes[0][i] + knees[0][i])/2,
                (toes[1][i] + knees[1][i])/2, 'go')
        ax.plot([knees[0][i], toes[0][i]], [knees[1][i], toes[1][i]], 'k-')
        ax.legend()
        ax.set_title("{:.1f}s".format(t[i]))

    anim = animation.FuncAnimation(
        fig, draw_frame, frames=t.shape[0], repeat=False, interval=dt*1000)
    anim.save('acrobot.gif', writer='imagemagick', fps=1/dt)

    return anim, fig


def test_acrobot():

    dt = 0.05
    x0 = np.array([0.001, 0, 0, 0])
    # x0 = np.array([np.pi-0.18, 0, 0.2, 0])
    xf = np.array([np.pi, 0, 0, 0])
    t0 = 0
    tf = 25
    t = np.arange(t0, tf, dt)

    p = {'l1': 0.5, 'l2': 1,
         'm1': 8, 'm2': 8,
         'I1': 2, 'I2': 8,
         'umax': 25,
         'g': 10,
         'dt': dt,
         'xf': xf,
         }

    acrobot = Acrobot(params=p)

    xs = np.zeros((t.shape[0], x0.shape[0]))
    xs[0] = x0

    for i in range(1, xs.shape[0]):
        xs[i] = acrobot.simulate(xs[i-1], dt)

    xs[:, 0] = np.arctan2(np.sin(xs[:, 0]), np.cos(xs[:, 0]))
    xs[:, 1] = np.arctan2(np.sin(xs[:, 1]), np.cos(xs[:, 1]))
    e = np.array([acrobot.energy(x) for x in xs])

    knee = p['l1']*np.cos(xs[:, 0]-np.pi/2), p['l1']*np.sin(xs[:, 0]-np.pi/2)
    toe = p['l1']*np.cos(xs[:, 0]-np.pi/2) + p['l2']*np.cos(xs[:, 0]+xs[:, 1]-np.pi/2), \
        p['l1']*np.sin(xs[:, 0]-np.pi/2) + p['l2'] * \
        np.sin(xs[:, 0]+xs[:, 1]-np.pi/2)

    anim, fig = plot_trajectory(t, knee, toe, p, dt)

    plt.figure(2)
    plt.clf()
    plt.plot(knee[0], knee[1], 'k.-', lw=0.5, label='knee')
    plt.plot(toe[0], toe[1], 'b.-', lw=0.5, label='toe')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(np.linspace(-2, 2, 100),
             (p['l1']+p['l2'])*np.ones(100), 'r', lw=1,
             label='max height')
    plt.title("position")
    plt.legend()

    plt.figure(3)
    plt.clf()
    plt.plot(t, e, 'k.-', lw=0.5)
    plt.axhline(acrobot.energy(
        np.array([np.pi, 0, 0, 0])), linestyle='--', color="r")
    plt.title('E')

    plt.show()
    plt.close()

    return xs, e


test_acrobot()
