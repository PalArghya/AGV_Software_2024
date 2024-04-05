import numpy as np

from utils import minimized_angle


class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta
        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.
        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # YOUR IMPLEMENTATION HERE
        G=env.G(self.mu,u)
        V=env.V(self.mu,u)
        updated_mu = env.forward(self.mu, u)
        M = env.noise_from_motion(u, self.alphas)
        updated_sigma = np.dot(np.dot(G,self.sigma),G.transpose()) + np.dot(np.dot(V,M),V.transpose())
        updated_z = env.observe (updated_mu, marker_id)
        H = env.H(updated_mu, marker_id)
        S = np.dot(np.dot (H, updated_sigma), H.transpose()) + self.beta
        P = np.dot(updated_sigma.dot(H.transpose()).reshape(3,1),np.linalg.inv(S))
        L = (np.dot(updated_sigma, H.transpose()))
        K = np.dot(L,P)
        self.mu = updated_mu + K.dot(z - minimized_angle(updated_z-z))
        #self.sigma=(np.identity(3)-K.dot(H.reshape(1,3))).dot(updated_sigma)
        self.sigma = np.dot(np.identity(3)- np.dot(K,H.reshape(1,3)), updated_sigma)
        return self.mu, self.sigma