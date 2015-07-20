import numpy as np


class SpatioTemporalSignal:
    """
    This Class is used to create a group of N signals that interact in space
    and time. The way that this interaction is carrieud out can be fully
    determined by the user through a SpatioTemporal matrix of vectors that
    specify how the signals mix.
    """

    def __init__(self, dt=0.1, delay=10, Tmax=100, Nseries=2):
        # Intialize time parameters
        self.dt = dt
        self.delay = delay
        self.Tmax = Tmax
        self.Nseries = Nseries

        # Put time in proper units
        self.NTmax = int(self.Tmax * 1.0 / self.dt)
        self.Ndelay = int(self.delay * 1.0 / self.dt)
        self.time = np.arange(self.NTmax) * self.dt

        # Initialize series
        self.series = np.zeros((self.Nseries, self.NTmax))

        # Intialize interaction
        self.interaction = np.zeros((self.Nseries, self.Nseries, self.NTmax))

    def construct_series(self):
        """
        This is the function that construct the series with a given interaction
        """

        for t in range(self.NTmax - 1):
            print 'Time t', t

            # First let's set the correct delay
            if t + 1 > self.Ndelay:
                delay_aux = self.Ndelay
            else:
                delay_aux = t + 1

            # Update signal_index
            for series_idx in xrange(self.Nseries):
                # Intialize vector to save time contribuionts
                vec_aux = np.zeros(self.Nseries)
                # Accomulate time contributions
                for delay_index in range(delay_aux):
                    # aux1 += x[delay_index] * a[t - delay_index]
                    # aux2 += y[delay_index] * b[t - delay_index]
                    aux1 = self.series[:, delay_index]
                    aux2 = self.interaction[series_idx, :, delay_index]
                    vec_aux += aux1 * aux2

                # Combine time contributions and normalize
                self.series[series_idx, t + 1] = np.mean(vec_aux) / delay_aux

    def interaction(self, interaction_matrix):
        """
        This function is used whenever the user wants
        to pass a particular interaction matrix
        """

        self.interaction = interaction_matrix


def main():
    print 'This is all right'
    return SpatioTemporalSignal()

if __name__ == '__main__':
    x = main()
