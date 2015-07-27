"""
Here the time series generators such as AR, MA and ARMA will go for the moment.
"""
import numpy as np


class AR(object):
    """
    This is a class that represents an AR process.
    We will make the conceptualization here that an AR can be
    characterized by a vector of phis in the following expression:

    z(t) = phi_0 + phi_1 * z(t - 1) + ... phi_k * z(t - k) + a_t

    Where alpha is random noise

    NOte: check whether the variance is properly transcipred (sigma^2)
    """

    def __init__(self, phi, variance=1.0, dt=1.0, Tmax=100):
        # General parameters of the series
        self.phi = phi
        self.variance = variance

        # Intialize time parameters
        self.dt = dt
        self.Tmax = Tmax
        self.klags = phi.size

        # Put time in proper units
        self.NTmax = int(self.Tmax * 1.0 / self.dt)
        self.time = np.arange(self.NTmax) * self.dt

        # Initialize series
        self.series = np.zeros(self.NTmax)

    def initial_conditions(self, initial_conditions):
        """
        This sets the initial conditions of the series
        """

        self.series[0:self.klags] = initial_conditions

    def construct_series(self):
        """
        This is the method that actually construct the series
        A natural optimization is to vectorize the noise
        """

        for t in range(self.klags, self.NTmax):
            self.series[t] = np.dot(self.phi, self.series[t - self.klags:t])
            self.series[t] += np.random.normal(0, self.variance)

        return self.series


class MixAr(AR):
    """
    This class is constructed with an AR process but mixing a
    sideckick function beta
    """

    def __init__(self, phi, variance=1.0, dt=1.0, Tmax=100, beta=None):

        """
        Overrides the initialization but also gets the sideckick function
        beta
        """
        super(AR, self).__init__(phi=phi, variance=variance, Tmax=Tmax)
        self.beta = beta

    def construct_series(self):
        """
        Overides the construct series method in order to implement
        the filtering by a sidekick function.
        """

        for t in range(self.klags, self.NTmax):
            # First  do the filtering with the past
            self.series[t] = np.dot(self.phi, self.series[t - self.klags:t])
            # Then add random noise and sidekick function
            self.series[t] += np.random.normal(0, self.variance) + self.beta[t]

        return self.series


def main():
    pass

if __name__ == '__main__':
    main()
