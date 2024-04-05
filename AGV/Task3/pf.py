import numpy as np

from utils import minimized_angle


class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov)
        self.weights = np.zeros_like(self.num_particles) / self.num_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # YOUR IMPLEMENTATION HERE
        for i in range(self.num_particles):
            u_noisy = env.sample_noisy_action(u, self.alphas)
            self.particles[i, :] = env.forward(self.particles[i, :], u_noisy).ravel()
        
        weights = np.ones(self.num_particles)
        for j in range(self.num_particles):
            z_noisy = env.sample_noisy_observation(self.particles[j, :], marker_id, self.beta)
            weights[j] = env.likelihood(z-z_noisy, self.beta)
        weights /= weights.sum()
        self.particles, _ = self.resample(self.particles, weights)

        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov    
    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        new_particles, new_weights = particles, weights
        # YOUR IMPLEMENTATION HERE
        beta = 0.00
        num_particles = len(weights)
        new_particles = np.zeros_like(particles)
        new_weights = np.zeros_like(num_particles) / num_particles
        resampling_particle = np.random.choice(num_particles, 1)[0]
        for i in range(self.num_particles):
            beta += np.random.uniform(0, 2 * weights.max())
            while beta > weights[resampling_particle]:
                beta = beta-weights[resampling_particle]
                resampling_particle = (resampling_particle + 1) % num_particles

            new_particles[i, :] = particles[resampling_particle, :]

        return new_particles, new_weights

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = particles.mean(axis=0)
        mean[2] = np.arctan2(
            np.cos(particles[:, 2]).sum(),
            np.sin(particles[:, 2]).sum()
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles

        return mean.reshape((-1, 1)), cov