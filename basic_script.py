import numpy as np
import matplotlib.pyplot as plt

# First the paramters
dt = 0.1  # Resoultion
delay = 10   # In s
Tmax = 100  # In s

NTmax = int(Tmax * 1.0 / dt)
Ndelay = int(delay * 1.0 / dt)

# Time
time = np.arange(NTmax) * dt
x = np.zeros(time.size)
y = np.zeros(time.size)

# Initial values
x[0] = 0
y[0] = 1

# Cross terms
a = np.cos(time)
b = np.sin(time)
c = np.cos(time)
d = np.sin(time)

# Running Options
verbose = False
plot = True

# Calculate the term
for t in range(NTmax - 1):
    print '----------- Script'
    print 'Time t', t
    # Update x
    if verbose:
        print 'x'
    aux1 = 0
    aux2 = 0
    if t + 1 > Ndelay:
        delay_aux = Ndelay
    else:
        delay_aux = t + 1
    for index in range(delay_aux):
        if verbose:
            print 'series ', x[index], y[index]
            print 'interactions', a[t - index], b[t - index]
        aux1 += x[index] * a[t - index]
        aux2 += y[index] * b[t - index]

    aux3 = aux1 + aux2
    if verbose:
        print 'Contribution ', aux1, aux2
        print 'Total contribution (BN) ', aux3
    x[t + 1] = (aux1 + aux2) / delay_aux

    # Update y
    if verbose:
        print 'y'
    aux1 = 0
    aux2 = 0
    if t + 1 > Ndelay:
        delay_aux = Ndelay
    else:
        delay_aux = t + 1
    for index in range(delay_aux):
        if verbose:
            print 'series ', x[index], y[index]
            print 'interactions', c[t - index], d[t - index]
        aux1 += y[index] * c[t - index]
        aux2 += x[index] * d[t - index]

    aux3 = aux1 + aux2
    if verbose:
        print 'Contribution ', aux1, aux2
        print 'Total contribution (BN) ', aux3
    y[t + 1] = (aux1 + aux2) / delay_aux


# Plot
if plot:
    plt.subplot(2, 1, 1)
    plt.plot(time, x)
    plt.plot(time, y)

    plt.subplot(2, 1, 2)
    plt.plot(time, a)
    plt.plot(time, b)

    plt.show()
