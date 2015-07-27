import numpy as np


class SpatioTemporalSignal(object):
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

    def set_initial_conditions(self, initial):
        """
        Set the initial conditions
        """
        self.series[..., 0] = initial

    def construct_series(self):
        """
        This is the function that construct the series with a given interaction
        Doesn't work for one dimensional series
        """

        for t in range(self.NTmax - 1):
            print '------------'
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
                    aux1 = self.series[:, t - delay_index]
                    aux2 = self.interaction[esries_idx, :, delay_index]
                    vec_aux += aux1 * aux2
                    # print 'vec_aux', vec_aux

                # Combine time contributions and normalize
                self.series[series_idx, t + 1] = np.sum(vec_aux) / (delay_aux)

    def construct_series_verbose(self):
        """
        This is the function that construct the series with a given interaction
        """

        for t in range(self.NTmax - 1):
            print '------------'
            print 'Time t', t

            # First let's set the correct delay
            if t + 1 > self.Ndelay:
                delay_aux = self.Ndelay
            else:
                delay_aux = t + 1

            # Update signal_index

            for series_idx in xrange(self.Nseries):
                print 'series_idx', series_idx
                print 'delay_aux of delay', delay_aux, self.Ndelay
                # Intialize vector to save time contribuionts
                vec_aux = np.zeros(self.Nseries)
                # Accomulate time contributions
                for delay_index in range(delay_aux):
                    aux1 = self.series[:, t - delay_index]
                    aux2 = self.interaction[series_idx, :, delay_index]
                    print 'series', aux1
                    print 'interactions', aux2
                    vec_aux += aux1 * aux2
                    # print 'vec_aux', vec_aux

                # Combine time contributions and normalize
                print 'Contribution ', vec_aux
                print 'Total contribution (BN) ', np.sum(vec_aux)
                self.series[series_idx, t + 1] = np.sum(vec_aux) / (delay_aux)
                print 'next value series', self.series[series_idx, t + 1]

    def set_interaction(self, interaction_matrix):
        """
        This function is used whenever the user wants
        to pass a particular interaction matrix
        """

        self.interaction = interaction_matrix


class TrigonometricMix(SpatioTemporalSignal):
    """
    This should allow us to initialize mixed signals easier
    """
    def __init__(self, dt=0.1, delay=10, Tmax=100, Nseries=2,
                 phase_m=None, frequency_m=None):
        """
        Overrides the initialization but also gets the frequency
        and phases matrix that are sufficient to determine a
        trignometric mix.
        """
        super(TrigonometricMix, self).__init__(dt, delay,
                                               Tmax, Nseries)
        self.phase_matrix = phase_m
        self.frequency_matrix = frequency_m

        # Create trigonometric matrix
        aux = []
        for phase, frequency in zip(self.phase_matrix.flatten(),
                                    self.frequency_matrix.flatten()):

            aux.append(np.cos(frequency * self.time + frequency))

        # Transform to array and reshape
        aux = np.array(aux)
        self.interaction = aux.reshape((self.Nseries,
                                        self.Nseries, self.NTmax))


def main():
    print 'This is all right'
    return SpatioTemporalSignal()

if __name__ == '__main__':
    x = main()
