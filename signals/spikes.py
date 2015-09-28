import numpy as np


def poisson_generator(rate, t_start=0, t_stop=1000.0):
    """
    Returns a SpikeTrain whose spikes are a realization of a Poisson process
    with the given rate (Hz) and stopping time t_stop (milliseconds).

        Inputs:
            rate    - the rate of the discharge (in Hz)
            t_start - the beginning of the SpikeTrain (in ms)
            t_stop  - the end of the SpikeTrain (in ms)
            array   - if True, a numpy array of sorted spikes is returned,
                      rather than a SpikeTrain object.

        Examples:
            >> gen.poisson_generator(50, 0, 1000)
            >> gen.poisson_generator(20, 5000, 10000)


    Inspired mostly in the NeuroTools library.
    """

    # First we get approximately the number of spikes
    number = int((t_stop-t_start)/1000.0*2.0*rate)

    if number > 1:
        # Generate the inter spike intervals
        isi = np.random.exponential(1.0 / rate, number) * 1000.0
        # Accumulate them (1, 2, 4) -> (1, 3, 7)
        spikes = np.add.accumulate(isi)
    else:
        spikes = np.array([])

    # Here we set the 0 equal to t_star or move the scale.
    spikes += t_start

    # If the stop time is bigger than the last spike occurence time
    if spikes[-1] < t_stop:
        extra_spikes = []

        # We add spikes at the end until we cross t_stop
        t_last = spikes[-1] + np.random.exponential(1.0 / rate, 1)[0] * 1000.0
        while(t_last < t_stop):
            extra_spikes.append(t_last)
            t_last += np.random.exponential(1.0 / rate, 1)[0] * 1000.0

        # We add the extra spikes at the end
        spikes = np.concatenate((spikes, extra_spikes))

    # Otherwise we need to get rid of the extra spikes
    else:
        # Get the index of the last spikes that is smaller than t_stop
        i = np.searchsorted(spikes, t_stop)
        spikes = np.resize(spikes, (i,))

    return spikes


def inh_poisson_generator(rate, t, t_stop):
    """
        Returns a SpikeTrain whose spikes are a realization of an inhomogeneous
        poisson process (dynamic rate). The implementation uses the thinning
        method, as presented in the references.

    Inputs:
    rate - an array of the rates (Hz) where rate[i] is active on interval
    [t[i],t[i+1]]
    t - an array specifying the time bins (in milliseconds) at which to
        specify the rate
    t_stop - length of time to simulate process (in ms)
    array  - if True, a numpy array of sorted spikes is returned,
              rather than a SpikeList object.

        Note:
            t_start=t[0]

        References:

        Eilif Muller, Lars Buesing, Johannes Schemmel, and Karlheinz Meier
        Spike-Frequency Adapting Neural Ensembles: Beyond Mean Adaptation and
        Renewal Theories Neural Comput. 2007 19: 2958-3010.
        Examples:
            >> time = arange(0,1000)
            >> stgen.inh_poisson_generator(time,sin(time), 1000)

    """
    # Transform things into arrays
    rate = np.asarray(rate)
    t = np.asarray(t)

    # get max rate and generate poisson process to be thinned
    rmax = np.max(rate)
    ps = poisson_generator(rmax, t_start=t[0], t_stop=t_stop)

    # Return empty array in absence of spikes
    if(len(ps) == 0):
        return np.array([])

    # Get a uniform random variable with as many elements as spikes
    rn = np.random.uniform(0, 1, len(ps))
    # Gets the indexes of where the spikes have to be inserted in time
    idx = np.searchsorted(t, ps) - 1

    spike_rate = rate[idx]
    spike_train = ps[rn < spike_rate / rmax]

    return spike_train


def sparse_to_dense(sparse_spike_train, dt, t_stop):
    """
    Transforms a sparse spike train to a dense one where
    at each element we will have 1 if there was a spike
    at index * dt, (index + 1 * dt)
    """

    dense_spikes = np.zeros(int(t_stop / dt))
    for spike in sparse_spike_train:
        index = int(spike / dt)
        dense_spikes[index] = 1

    return dense_spikes
