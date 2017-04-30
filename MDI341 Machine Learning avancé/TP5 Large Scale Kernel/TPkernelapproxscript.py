# Authors: Bellet, Gramfort, Salmon

from time import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
plt.style.use('ggplot')

###############################################################################
# Requires file ijcnn1.dat to be present in the directory

dataset_path = 'ijcnn1.dat'
ijcnn1 = load_svmlight_file(dataset_path)
X = ijcnn1[0].todense()
y = ijcnn1[1]

###############################################################################
# Extract features

X_train, X_test, y_train, y_test = train_test_split(X[:60000, :], y[:60000],
                     train_size=20000, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

###############################################################################
# SVM classification (Question 1)

from sklearn.svm import SVC, LinearSVC

print("Fitting SVC rbf on %d samples..." % X_train.shape[0])
t0 = time()
# TODO
# clf =
# clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))

print("Predicting with SVC rbf on %d samples..." % X_test.shape[0])
t1 = time()
# TODO
print("done in %0.3fs" % (time() - t1))
timing_kernel = time() - t0
print("classification accuracy: %0.3f" % accuracy_kernel)

# TODO with LinearSVC

###############################################################################
# Gram approximation

from TP_kernel_approx_source import rank_trunc, nystrom, random_features

p = 200
r_noise = 100
r_signal = 20

intensity = 50

rng = np.random.RandomState(42)
X_noise = rng.randn(r_noise, p)
X_signal = rng.randn(r_signal, p)

gram_signal = np.dot(X_noise.T, X_noise) + intensity * np.dot(X_signal.T,
                                                              X_signal)
n_ranks = 100
ranks = np.arange(1, n_ranks + 1)
timing_fast = np.zeros(n_ranks)
timing_slow = np.zeros(n_ranks)
rel_error = np.zeros(n_ranks)

# TODO : Question 2 Implement rank_trunc function in source file

# TODO : Question 3 Evaluate accuracy with Frobenius norm as a function
# of the rank for both svd solvers

# Use linalg.norm(A, 'fro') to compute Frobenius norm of A

for k, rank in enumerate(ranks):
    pass

###############################################################################
# Display

fig, axes = plt.subplots(ncols=1, nrows=2)
ax1, ax2 = axes.ravel()

ax1.plot(ranks, timing_fast, '-')
ax1.plot(ranks, timing_slow, '-')

ax1.set_xlabel('Rank')
ax1.set_ylabel('Time')
ax2.plot(ranks, rel_error, '-')
ax2.set_xlabel('Rank')
ax2.set_ylabel('Relative Error')
plt.tight_layout()
plt.show()


###############################################################################
# Random Kernel Features:

n_samples, n_features = X_train.shape
n_samples_test, _ = X_test.shape
gamma = 1. / n_features

# TODO : Question 4 Implement random features in source file

Z_train, Z_test = random_features(X_train, X_test, gamma, c=300, seed=44)

# TODO : Question 5 Estimate training, testing time and accuracy


###############################################################################
# SVM Nystrom:

# TODO : Question 6 Implement Nystrom in source file

Z_train, Z_test = nystrom(X_train, X_test, gamma, c=500, k=200, seed=44)

print("Fitting SVC linear on %d samples..." % n_samples)
t0 = time()
clf = LinearSVC(dual=False)
clf.fit(Z_train, y_train)
print("done in %0.3fs" % (time() - t0))

print("Predicting with SVC linear on %d samples..." % n_samples_test)
t0 = time()
accuracy = clf.score(Z_test, y_test)
print("done in %0.3fs" % (time() - t0))
print("classification accuracy: %0.3f" % accuracy)


####################################################################
# Results / comparisons:

ranks = list(range(20, 750, 50))
n_ranks = len(ranks)
timing_rkf = np.zeros(n_ranks)
timing_nystrom = np.zeros(n_ranks)

accuracy_nystrom = np.zeros(n_ranks)
accuracy_rkf = np.zeros(n_ranks)

print("Training SVMs for various values of c...")

for i, c in enumerate(ranks):
    # t0 = time()
    # TODO Question 8

    # accuracy_rkf[i] = ...
    # timing_rkf[i] = time() - t0
    pass

###############################################################################
# Display bis

fig, axes = plt.subplots(ncols=1, nrows=2)
ax1, ax2 = axes.ravel()

ax1.plot(ranks, timing_nystrom, '-', label='Nystrom')
ax1.plot(ranks, timing_rkf, '-', label='RKF')
ax1.plot(ranks, timing_linear * np.ones(n_ranks), '-', label='LinearSVC')
ax1.plot(ranks, timing_kernel * np.ones(n_ranks), '-', label='RBF')

ax1.set_xlabel('Rank')
ax1.set_ylabel('Time')
ax1.legend(loc='lower right')

ax2.plot(ranks, accuracy_nystrom, '-', label='Nystrom')
ax2.plot(ranks, accuracy_rkf, '-', label='RKF')
ax2.plot(ranks, accuracy_linear * np.ones(n_ranks), '-', label='LinearSVC')
ax2.plot(ranks, accuracy_kernel * np.ones(n_ranks), '-', label='RBF')
ax2.set_xlabel('Rank')
ax2.set_ylabel('Accuracy')
ax2.legend(loc='lower right')
plt.tight_layout()
plt.show()