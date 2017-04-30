# -*- coding: utf-8 -*-
"""
@author: portier, salmon
"""
import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from os import mkdir, path
import scipy.stats
from scipy.stats import pearsonr
from scipy.stats import norm
# For timing comparisons:
from timeit import time
t = time.time()

dirname = "../srcimages/"
if not path.exists(dirname):
    mkdir(dirname)
imageformat = '.pdf'


def my_saving_display(fig, dirname, filename, imageformat):
    """saving faster"""
    dirname + filename + imageformat
    image_name = dirname + filename + imageformat
    fig.savefig(image_name)


###############################################################################
# Part: Sample Mean
###############################################################################


###############################################################################
# Q1

np.random.seed(1)  # Seed intialization.
alpha = 2.
beta = 5.
n = 500

X = np.random.beta(alpha, beta, n)
fig = plt.figure()
plt.hist(X, bins=25)
plt.show()


###############################################################################
# Q2
d = 2
X = np.random.beta(alpha, beta, size=(d, n))
X_mean = np.mean(X, axis=1)

# Plot / visualization
fig = plt.figure()
plt.plot(X[0, :], X[1, :], 'bo', label='obs', markersize=4)
plt.plot(X_mean[0], X_mean[1], 'ro', label='mean_obs', markersize=7)
plt.title('Observed data')
plt.axis([-0.05, 1, -0.05, 1])
plt.legend(numpoints=1)
filename = 'fig1'
plt.show()

my_saving_display(fig, dirname, filename, imageformat)

# NB: Many methods are available to generate such random numbers
# First method
d = 2
t_init = time.time()
X = np.zeros([d, n])
for k in range(d):
    X[k, :] = np.random.beta(alpha, beta, n)

t_end = time.time()
print("Timing for random array generation with a loop: " + "\n" +
      str(t_end - t_init))

# Second method
t_init = time.time()
X = np.zeros([d, n])
X_vector = np.random.beta(alpha, beta, n * d)
X = X_vector.reshape([d, n])
t_end = time.time()
print("Timing for random array generation with reshaping: " + "\n" +
      str(t_end - t_init))

# third method
t_init = time.time()
X = np.random.beta(alpha, beta, size=(d, n))
t_end = time.time()
print("Timing for random array generation with correct array generation: " +
      "\n" + str(t_end - t_init))


# NB : Other libraries such as ''random'' can also be used


###############################################################################
# Q3
# NB: First, one can consider the computational time
t_init = time.time()
Xstar = X[:, np.random.choice(range(n), n)]
t_end = time.time()
print("Timing for one replication with range: " + "\n" +
      str(t_end - t_init))

t_init = time.time()
Xstar = X[:, np.random.randint(n, size=n)]
t_end = time.time()
print("Timing for one replication with randint: " + "\n" +
      str(t_end - t_init))

# In Ipython, one can launch the following line, without the comment signs,
# to get a time evalution of the different methods:
# %timeit Xstar = X[:, np.random.randint(n, size=n)]


Xstarbar = np.mean(Xstar, axis=1)

# NB: Graph of the observations, the mean and the bootstrap mean
fig = plt.figure()
plt.plot(X[0, :], X[1, :], 'ow', label='obs', markersize=2)
plt.plot(X_mean[0], X_mean[1], 'or', label='mean_obs', markersize=7)
plt.plot(Xstar[0, :], Xstar[1, :], 'ob', label='boot', markersize=2)
plt.plot(Xstarbar[0], Xstarbar[1], 'ob', label='mean_boot', markersize=7)
plt.title('Bootstrapped sample')
plt.axis([-0.05, 1, -0.05, 1])
plt.legend(numpoints=1)
filename = 'fig2'
plt.show()

my_saving_display(fig, dirname, filename, imageformat)


# Bootstrap with 500 replicas
B = 500
Xmeans_boot = np.zeros([d, B])

for b in range(B):
    Xmeans_boot[:, b] = np.mean(X[:, np.random.randint(n, size=n)], axis=1)


fig = plt.figure()
plt.plot(X[0, :], X[1, :], 'bo', label='obs', markersize=2)
plt.plot(Xmeans_boot[0, 0], Xmeans_boot[1, 0], 'ro', label='mean bts',
         markersize=7)
for b in range(B):
    plt.plot(Xmeans_boot[0, b], Xmeans_boot[1, b], 'ro', markersize=7)
plt.plot(X_mean[0], X_mean[1], 'bo', label='mean obs', markersize=7)

plt.title('Bootstrap of the mean')
plt.axis([-0.05, 1, -0.05, 1])
plt.legend(numpoints=1)
filename = 'fig3'
plt.show()

my_saving_display(fig, dirname, filename, imageformat)


###############################################################################
# Q4
# Bootstrap estimate of the covariance (of the mean estimator)
v_boot = np.cov(Xmeans_boot)

# Bootstrap bias
b_boot = np.mean(Xmeans_boot) - X_mean

# Jackknife
Xmeans_jack = np.zeros([2, n])

t_init = time.time()
for i in range(n):
    Xi = np.delete(X, (i), axis=1)
    Xmeans_jack[:, i] = np.mean(Xi, axis=1)

b_jack = (n - 1) * (np.mean(Xmeans_jack, axis=1) - X_mean)
v_jack = (n - 1) * np.cov(Xmeans_jack, bias=True)

# Verifying the course formula
print(np.cov(X, bias=False) / n)
print(v_jack)

# Testing numerical equality:
print("Is the formula true? " +
      str(np.allclose(np.cov(X, bias=False) / n, v_jack)))


###############################################################################
# Q5

# As the variance of a beta distribution with parameter a b is
# (a * b)/((a + b)**2*(a+b+1))
# This is what we find in the diagonal of the covariance matrix
# Non-diagonal terms are 0 because the two components are independent

v_true = np.eye(2, 2) * (1. / n) * (2 * 5) / ((2. + 5.)**2 * (2 + 5 + 1))

print('The error of the bootstrap approximation is ' +
      str(np.sum(np.abs(np.triu(v_boot) - np.triu(v_true)))) + '\n' +
      'The error of the Jackknife approximation is ' +
      str(np.sum(np.abs(np.triu(v_jack) - np.triu(v_true)))))

# NB: Note that we could run the procedure many times to have a better idea on
# which is the best between Jackknife and Bootstrap here


###############################################################################
# Part: Correlation coefficient
###############################################################################


###############################################################################
# Q6

np.random.seed(1)
eps = .1
n = 300
X = np.zeros([2, n])
X[0, :] = np.random.uniform(0, 1, n)
U = np.random.uniform(-eps, eps, n)
X[1, :] = X[0, :] + U

# NB: one can visualize the data
fig = plt.figure()
plt.plot(X[0, :], X[1, :], 'or')
plt.show()


###############################################################################
# Q7

# the underlying true
var_1 = 1. / 12
var_2 = 1. / 12 + (2 * eps) ** 2 / 12.
cov_12 = var_1
c_true = cov_12 / np.sqrt(var_1 * var_2)

# the estimator
c_hat = pearsonr(X[0, :], X[1, :])[0]


print("Comapare theoretical and estimated values: " + "\n" +
      "Theory    : " + str(c_true) + "\n" +
      "Estimated : " + str(c_hat) + "\n")

# NB: One can compare different methods to compute the estimated covariance
# in term of computational time

# Method 1
t_init = time.time()
c_hat = np.corrcoef(X[0, :], X[1, :])[0, 1]
t_end = time.time()
print("Timing to evaluate correlation with corrcoef: " + "\n" +
      str(t_end - t_init))

# Method 2
t_init = time.time()
mean0 = np.mean(X[0, :])
mean1 = np.mean(X[1, :])
num = np.mean(X[0, :] * X[1, :]) - mean0 * mean1
var0 = np.mean(X[0, :] ** 2) - mean0 ** 2
var1 = np.mean(X[1, :] ** 2) - mean1 ** 2
c_manual = num / np.sqrt(var0 * var1)
t_end = time.time()
print("Timing to evaluate correlation with true formula: " + "\n" +
      str(t_end - t_init))

# Method 3
t_init = time.time()
c_scipy = pearsonr(X[0, :], X[1, :])[0]
t_end = time.time()
print("Timing to evaluate correlation with pearsonr: " + "\n" +
      str(t_end - t_init))


# %timeit c_hat = np.corrcoef(X[0, :], X[1, :])[0, 1]  # N or N-1?
# %timeit c_scipy = pearsonr(X[0, :], X[1, :])[0]

print("Comapare three methods: " + "\n" +
      "With corrcoef: " + str(c_hat) + "\n" +
      "With algebra : " + str(c_scipy) + "\n" +
      "With pearsonr: " + str(c_manual) + "\n")


###############################################################################
# Q8

# Basic bootstrap IC
B = 500
c_mem = np.zeros([1, B])
for b in range(B):
    Xstar = X[:, np.random.randint(n, size=n)]
    c_mem[0, b] = pearsonr(Xstar[0, :], Xstar[1, :])[0]

# method 1:
basic_boot_root = c_mem - c_hat
ic_basic_boot = [c_hat - np.percentile(basic_boot_root, 97.5),
                 c_hat - np.percentile(basic_boot_root, 2.5)]

# method 2:
ic_basic_boot_2 = [2 * c_hat - np.percentile(c_mem, 97.5),
                   2 * c_hat - np.percentile(c_mem, 2.5)]

print("IC Basic Bootstrap    : [%.5f,%.5f]" % (ic_basic_boot[0],
                                               ic_basic_boot[1]))
print("IC Basic Bootstrap bis: [%.5f,%.5f]" % (ic_basic_boot_2[0],
                                               ic_basic_boot_2[1]))

###############################################################################
# Q9

# Percentile
ic_percentile_boot = [np.percentile(c_mem, 2.5),
                      np.percentile(c_mem, 97.5)]

print("IC Percentile Bootstrap : [%.5f,%.5f]" % (ic_percentile_boot[0],
                                                 ic_percentile_boot[1]))


###############################################################################
# Q10

# Asymptotic + Jacknife
c_jack = np.zeros([1, n])

for i in range(n):
    Xi = np.delete(X, (i), axis=1)
    c_jack[0, i] = np.corrcoef(Xi[0, :], Xi[1, :])[0, 1]

v_jack = (n - 1) * np.cov(c_jack, bias=True)

ic_asympt = [c_hat - np.sqrt(v_jack) * norm.ppf(0.975),
             c_hat - np.sqrt(v_jack) * norm.ppf(0.025)]

print("IC JACKNIFE : [%.5f,%.5f]" % (ic_asympt[0], ic_asympt[1]))


###############################################################################
# Q11

# The following function execute a loop on M (the number of Monte-Carlo
# simulation) and at each step of the loop it compute basic bootstrap,
# percentile bootstrap and Jackknife confidence interval. At the end of
# each step it registers in a vector whether c_0 is in each interval or not.
# Finally it computes the means over this vector.
# This is the estimated coverage probability by Monte-Carlo


def coverage(M, n, B, eps):
    """" Beware for n small some NaN can appears, e.g. n=3"""
    cover_basic = np.zeros([1, M])
    cover_percent = np.zeros([1, M])
    cover_asympt = np.zeros([1, M])
    for m in range(M):
        X = np.zeros([2, n])
        X[0, :] = np.random.uniform(0, 1, n)
        U = np.random.uniform(-eps, eps, n)
        X[1, :] = X[0, :] + U
        c_hat = pearsonr(X[0, :], X[1, :])[0]
        c_mem = np.zeros([1, B])
        for b in range(B):
            Xstar = X[:, np.random.randint(n, size=n)]
            c_mem[0, b] = pearsonr(Xstar[0, :], Xstar[1, :])[0]
        basic_boot_root = c_mem - c_hat
        ic_basic_boot = [c_hat - np.percentile(basic_boot_root, 97.5),
                         c_hat - np.percentile(basic_boot_root, 2.5)]
        ic_percentile_boot = [np.percentile(c_mem, 2.5),
                              np.percentile(c_mem, 97.5)]

        c_jack = np.zeros([1, n])
        for i in range(n):
            Xi = np.delete(X, (i), axis=1)
            c_jack[0, i] = np.corrcoef(Xi[0, :], Xi[1, :])[0, 1]

        v_jack = (n - 1) * np.cov(c_jack, bias=True)

        ic_asympt = [c_hat - np.sqrt(v_jack) * norm.ppf(0.975),
                     c_hat - np.sqrt(v_jack) * norm.ppf(0.025)]
        var_1 = 1. / 12
        var_2 = var_1 + (2 * eps) ** 2 / 12.
        cov_12 = var_1
        c_true = cov_12 / np.sqrt(var_1 * var_2)

        cover_basic[0, m] = (ic_basic_boot[0] <= c_true and
                             c_true <= ic_basic_boot[1])
        cover_percent[0, m] = (ic_percentile_boot[0] <= c_true and
                               c_true <= ic_percentile_boot[1])
        cover_asympt[0, m] = (ic_asympt[0] <= c_true and
                              c_true <= ic_asympt[1])
    return np.mean(cover_basic), np.mean(cover_percent), np.mean(cover_asympt)


# We shall give to the function the parameter of our experiment

[cover_basic, cover_percent, cover_asympt] = coverage(M=2000, n=300,
                                                      B=500, eps=.1)
print("Coverage Basic Bootstrap    : %.3f" % cover_basic)
print("Coverage Percentile Bootstrap : %.3f" % cover_percent)
print("Coverage jacknife : %.3f" % cover_asympt)


###############################################################################
# Q11
n_range = [30, 50, 100, 300]
mem_cov = np.zeros([3, np.size(n_range)])

for n_index, n_val in enumerate(n_range):
    B = 4 * n
    mem_cov[:, n_index] = coverage(M=2000, n=n_val, B=500, eps=.1)

fig = plt.figure()
plt.plot(n_range, mem_cov[0, :], 'b', label='basic')
plt.plot(n_range, mem_cov[1, :], 'r', label='percentile')
plt.plot(n_range, mem_cov[2, :], 'k', label='jacknife')
plt.plot(n_range, .95 * np.ones([np.size(n_range)]), 'g')
plt.title('Coverage probabilities')
plt.axis([0, np.max(n_range), np.min(mem_cov) - .1, 1])
plt.legend(numpoints=1, loc=4)
filename = 'fig_cover'
plt.show()
my_saving_display(fig, dirname, filename, imageformat)


###############################################################################
# Linear regresion
###############################################################################


###############################################################################
# Load data (from DM1)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/' +\
      'auto-mpg.data-original'

u_cols = ['mpg', 'cylinders', 'displacement', 'horsepower',
          'weight', 'acceleration', 'model year', 'origin', 'car name']

data = pd.read_csv(url, sep=r"\s+", names=u_cols, na_values='NA')
print(data.shape)

data = data.dropna(axis=0, how='any')
data.index = range(data.shape[0])
print(data.shape)

y = data['mpg']
X = data.drop(['origin', 'car name', 'mpg'], axis=1)
X['intercept'] = np.ones(X.shape[0])
X.shape


###############################################################################
# Q13 least-square estimator + residuals visualization

Xt = np.transpose(X)
Gram = Xt.dot(X)
theta_hat = np.linalg.solve(Gram, np.dot(Xt, y))

print("Least square estimator: ", theta_hat)

resid = y - np.dot(X, theta_hat)

dens_resid = gaussian_kde(resid)
xs = np.linspace(-20, 20)

plt.figure()
plt.plot(xs, dens_resid(xs))
plt.show

var_resid = np.var(resid)

# NB: One could also answer the question using the scikitlearn library.

# NB: The Gaussian assumption here is not unrealistic


###############################################################################
# Q14
# Student approach
# In the following we directly compute confidence intervals for all the
# coefficients of the regresssion model

n, p = X.shape
Gram_inv = np.linalg.inv(Gram)

sig_hat = np.sqrt((np.sum(resid ** 2) / (n - p)))
delta = sig_hat * np.sqrt(np.diag(Gram_inv)) * scipy.stats.t.ppf(0.975, n - p)
ic_aspt_min = theta_hat - delta
ic_aspt_max = theta_hat + delta
for j in range(p):
    print("IC for theta_%.d: [%.5f,%.5f]" % (j + 1, ic_aspt_min[j],
                                             ic_aspt_max[j]))

# NB: another approach is the asymptotic approach in which the Student law is
# replace by the Gaussian one.

ic_aspt_min = theta_hat - np.sqrt(np.diag(Gram_inv) * var_resid) *\
    scipy.stats.norm.ppf(0.975)
ic_aspt_max = theta_hat - np.sqrt(np.diag(Gram_inv) * var_resid) *\
    scipy.stats.norm.ppf(0.025)
ic_all = [ic_aspt_max, ic_aspt_max]

for j in range(p):
    print("IC for theta_%.d: [%.5f,%.5f]" % (j + 1, ic_aspt_min[j],
                                             ic_aspt_max[j]))


###############################################################################
# Q15

resid_star = resid[np.random.randint(n, size=n)]
y_star = np.dot(X, theta_hat) + resid_star

plt.rc('text', usetex=True)

#  Plot / visualisation
fig = plt.figure()
plt.plot(X.dot(theta_hat), y, 'wo', label='$y_i$ (obs)', markersize=7)
plt.plot(X.dot(theta_hat), y_star, 'bo', label='$y_i*$ (boot)', markersize=7)
plt.title(r'Observation and bootstrapped response versus $X \beta$')
plt.legend(numpoints=1, loc=2)
plt.xlabel(r'$X \hat{\theta}$')
filename = 'fig2'
plt.show()


###############################################################################
# Q16 Bootstrap confidence intervals

B = 500
theta_star_mem = np.zeros([B, p])
std_resid_star = np.zeros(B)   # for t-bootstrap, see below

for b in range(B):
    resid_star = resid[np.random.randint(n, size=n)]
    y_star = np.dot(X, theta_hat) + resid_star
    theta_star_mem[b, :] = np.linalg.solve(Gram, np.dot(Xt, y_star))
    resid_star_hat = y_star - np.dot(X, theta_star_mem[b, :])
    std_resid_star[b] = np.std(resid_star_hat)

# Basic bootstrap confidence interval
theta_star_perc = np.zeros([2, p])
for k in range(p):
    theta_star_perc[0, k] = np.percentile(theta_star_mem[:, k], 2.5)
    theta_star_perc[1, k] = np.percentile(theta_star_mem[:, k], 97.5)

ic_basic_min = 2 * theta_hat - theta_star_perc[1, :]
ic_basic_max = 2 * theta_hat - theta_star_perc[0, :]

# NB equivalent formulas are:
# ic_basic_min = theta_hat - (theta_star_perc[1, :] - theta_hat)
# ic_basic_max = theta_hat - (theta_star_perc[0, :] - theta_hat)

print('\n')
for j in range(p):
    print("IC basic boot., theta_%.d: [%.5f,%.5f]" % (j + 1, ic_basic_min[j],
          ic_basic_max[j]))

# N.B: For your information, the Percentile bootsrap confidence intervals
theta_star_perc = np.zeros([2, p])
for k in range(p):
    theta_star_perc[0, k] = np.percentile(theta_star_mem[:, k], 2.5)
    theta_star_perc[1, k] = np.percentile(theta_star_mem[:, k], 97.5)

ic_percent_min = theta_star_perc[0, :]
ic_percent_max = theta_star_perc[1, :]

for j in range(p):
    print("IC perc. for theta_%.d: [%.5f,%.5f]" % (j + 1, ic_percent_min[j],
                                                   ic_percent_max[j]))

# N.B: For your information, the Bootstrap-t confidence interval
root_t = np.transpose(theta_star_mem - np.squeeze(theta_hat)) / std_resid_star
theta_t_perc = np.zeros([2, p])
for k in range(p):
    theta_t_perc[0, k] = np.percentile(root_t[k, :], 2.5)
    theta_t_perc[1, k] = np.percentile(root_t[k, :], 97.5)


ic_boot_t_min = theta_hat - theta_t_perc[1, :] * np.sqrt(var_resid)
ic_boot_t_max = theta_hat - theta_t_perc[0, :] * np.sqrt(var_resid)

print('\n')
for j in range(p):
    print("IC student for theta_%.d: [%.5f,%.5f]" % (j + 1, ic_boot_t_min[j],
          ic_boot_t_max[j]))


# NB: For your information a plot of the estimated coefficients and the
# bootstrap coeficients
fig = plt.figure()
plt.plot(np.arange(p) + 1, theta_star_mem[0, :], 'bo', label='boot. coefs')
for b in range(B):
    plt.plot(np.arange(p) + 1, theta_star_mem[b, :], 'bx')
plt.plot(np.arange(p) + 1, theta_hat, 'ro', label='estimated coef',
         markersize=5)

plt.title(r'Estimated parameter and bootstrapped')
plt.legend(numpoints=1, loc=2)
plt.axis([.5, 7, -2, 1])
filename = 'fig2'
plt.show()


###############################################################################
# Q17

f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot([0, 0], [ic_percent_min[0], ic_percent_max[0]], '-b', label='95\% IC')
for k in range(p):
    ax1.plot([k, k], [ic_percent_min[k], ic_percent_max[k]], '+-b')
ax1.plot(theta_hat, 'ro', label='estimated', markersize=5)
ax1.set_title(r'Bootstrap-percentile')
ax1.legend(numpoints=1)
ax1.axis([-.5, 5.5, -1.2, 1.5])
filename = 'fig2'


ax2.plot([0, 0], [ic_basic_min[0], ic_basic_max[0]], '-b', label='95\% IC')
for k in range(p):
    ax2.plot([k, k], [ic_basic_min[k], ic_basic_max[k]], '+-b')
ax2.plot(theta_hat, 'ro', label='estimated', markersize=5)
ax2.set_title(r'Bootstrap-basic')
ax2.legend(numpoints=1)
ax2.axis([-.5, 5.5, -1.2, 1.5])
filename = 'fig2'

ax3.plot([0, 0], [ic_boot_t_min[0], ic_boot_t_max[0]], '-b', label='95\% IC')
for k in range(p):
    ax3.plot([k, k], [ic_boot_t_min[k], ic_boot_t_max[k]], '+-b')
ax3.plot(theta_hat, 'ro', label='estimated', markersize=5)
ax3.set_title(r'Bootstrap-t')
ax3.legend(numpoints=1)
ax3.axis([-.5, 5.5, -1.2, 1.5])

plt.show()

# For many coefficients, a test would reject that the coefficient is different
# from 0. It seems that variable selection here would be appropriate and could
# treated as below


###############################################################################
##

# X_name = ['cylinders', 'displacement', 'horsepower',
#           'weight', 'acceleration', 'model year']
# #
# #
# #
# # Student-forward
# #
# for k in range(6):

#     Xk = X[[X_name[k], 'intercept']]

#     Xtk = np.transpose(Xk)
#     Gramk = Xtk.dot(Xk)
#     theta_hatk = np.linalg.solve(Gramk, np.dot(Xtk, y))
#     residk = y - np.dot(Xk, theta_hatk)
#     sig_hatk = np.sqrt((np.sum(residk ** 2) / (n - 2)))
#     Gram_invk = np.linalg.inv(Gramk)

#     print("Least square estimator: ", theta_hatk)

#     root = np.abs(theta_hatk[0]) / (sig_hatk * np.sqrt((Gram_invk[0, 0])))
#     print(root)
#     print(2 * (1 - scipy.stats.t.cdf(root, n - 2)))

# # We keep the largest root. And iterate.
