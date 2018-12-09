from random import randint
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import random
import scipy.optimize as optimize
from matplotlib import pylab

# Define function for calculating a power law
powerlaw = lambda x, amp, index: amp * (x**index)

##########
# Generate data points with noise
##########
num_points = 94

# Note: all positive, non-zero data
ydata = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 4, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 3, 1, 2, 4, 2, 3, 6, 2, 3, 4, 3, 4, 7, 2, 4, 5, 6, 6, 3, 3, 7, 6, 7, 4, 6, 16, 8, 9, 15, 21, 18, 24, 23, 18, 31, 27, 36, 38, 41, 53, 51, 73, 92, 95, 119, 132, 204, 238, 315, 370, 556, 811, 1208, 1900, 3306]
xdata = [384, 329, 263, 248, 216, 192, 176, 173, 165, 158, 155, 133, 132, 128, 126, 125, 114, 111, 107, 101, 98, 93, 91, 90, 89, 86, 82, 80, 77, 76, 73, 72, 71, 70, 68, 67, 66, 65, 63, 62, 61, 60, 59, 57, 55, 54, 53, 52, 51, 49, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]

yerr = [0.2 * i for i in ydata]                      # simulated errors (10%)

ydata += np.random.randn(num_points) * yerr       # simulated noisy data

##########
# Fitting the data -- Least Squares Method
##########

# Power-law fitting is best done by first converting
# to a linear equation and then fitting to a straight line.
# Note that the `logyerr` term here is ignoring a constant prefactor.
#
#  y = a * x^b
#  log(y) = log(a) + b*log(x)
#

logx = np.log10(xdata)
logy = np.log10(ydata)
logyerr = yerr / ydata

# define our (line) fitting function
fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

pinit = [1.0, -1.0]
out = optimize.leastsq(errfunc, pinit,
                       args=(logx, logy, logyerr), full_output=1)

pfinal = out[0]
covar = out[1]
#print pfinal
#print covar

index = pfinal[1]
amp = 10.0**pfinal[0]

indexErr = np.sqrt( covar[1][1] )
ampErr = np.sqrt( covar[0][0] ) * amp

##########
# Plotting data
##########

plt.clf()
plt.subplot(2, 1, 1)
plt.plot(xdata, powerlaw(xdata, amp, index))     # Fit
plt.errorbar(xdata, ydata, yerr=yerr, fmt='k.')  # Data
plt.text(5, 6.5, 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr))
plt.text(5, 5.5, 'Index = %5.2f +/- %5.2f' % (index, indexErr))
plt.title('Best Fit Power Law')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(1, 500)

plt.subplot(2, 1, 2)
plt.loglog(xdata, powerlaw(xdata, amp, index), label='Least square fit')
plt.errorbar(xdata, ydata, yerr=yerr, fmt='k.', label='Distribution')  # Data
plt.xlabel('X (log scale)')
plt.ylabel('Y (log scale)')
plt.xlim(1.0, 500)

plt.legend(loc='upper right')
plt.show()