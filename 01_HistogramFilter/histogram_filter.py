import numpy as np
import matplotlib.pyplot as plt
import os


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        '''
        size = cmap.shape
        
        T_right = np.zeros((size[0]*size[1],size[0]*size[1]))
        for x in range(size[1]):
            for y in range(size[0]):
                if y==size[1]-1:
                    T_right[x*size[1]+y, x*size[1]+y] = 1
                else:
                    T_right[x*size[1]+y,x*size[1]+y] = 0.1
                    T_right[x*size[1]+y,x*size[1]+y+1] = 0.9
                    
        T_left = np.zeros((size[0]*size[1],size[0]*size[1]))
        for x in range(size[1]):
            for y in range(size[0]):
                if y==0:
                    T_left[x*size[1]+y, x*size[1]+y] = 1
                else:
                    T_left[x*size[1]+y,x*size[1]+y] = 0.1
                    T_left[x*size[1]+y,x*size[1]+y-1] = 0.9

        T_up = np.zeros((size[0]*size[1],size[0]*size[1]))
        for x in range(size[1]):
            for y in range(size[0]):
                if x==0:
                    T_up[x*size[1]+y, x*size[1]+y] = 1
                else:
                    T_up[x*size[1]+y,x*size[1]+y] = 0.1
                    T_up[(x-1)*size[1]+y,x*size[1]+y] = 0.9
        T_up = T_up.T
                    
        T_down = np.zeros((size[0]*size[1],size[0]*size[1]))
        for x in range(size[1]):
            for y in range(size[0]):
                if x==size[0]-1:
                    T_down[x*size[1]+y, x*size[1]+y] = 1
                else:
                    T_down[x*size[1]+y,x*size[1]+y] = 0.1
                    T_down[(x+1)*size[1]+y,x*size[1]+y] = 0.9
        T_down = T_down.T

        M_1 = cmap*0.8+0.1
        M_0 = (-1*(cmap-1))*0.8+0.1

        if np.all(action == np.array([1,0])):
            T = T_right
        if np.all(action == np.array([-1,0])):
            T = T_left
        if np.all(action == np.array([0,1])):
            T = T_up
        if np.all(action == np.array([0,-1])):
            T = T_down
        if observation == 0:
            M = M_0
        if observation == 1:
            M = M_1
        
        alpha_k = belief.flatten()
        alpha_k1 = np.multiply(M.flatten(), alpha_k.T@T)
        return ((1/(alpha_k1.sum()))*alpha_k1).reshape(size)
    
if __name__ == '__main__':
    data = np.load(open('./starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']
    filter = HistogramFilter()

    if os.path.exists('./gif') == False:
        os.mkdir('./gif')

    pi_x = np.ones_like(cmap)
    pi_x = pi_x/pi_x.sum()

    state_estimates = np.zeros((cmap.shape[0], cmap.shape[1], len(actions)))
    for i, action in enumerate(actions):
        if i == 0:
            belief = pi_x
        else:
            belief = state_estimates[:,:, i-1]
        state_estimates[:,:, i] = filter.histogram_filter(cmap, belief, action, observations[i])
        fig = plt.figure()
        plt.imshow(state_estimates[:,:, i], cmap='gray')
        plt.savefig('./gif/state_estimate_{}.png'.format(i))
        plt.close(fig)
