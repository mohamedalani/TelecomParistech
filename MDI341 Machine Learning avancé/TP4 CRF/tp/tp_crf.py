# -*- coding: utf-8 -*-
"""
Created on Thu Mar 05 14:38:23 2015

@author: essid
"""

import cPickle as pickle

import numpy as np
from pycrfsuite import Tagger
from flexcrf_tp.models.linear_chain import (_feat_fun_values,
                                            _compute_all_potentials,
                                            _forward_score,
                                            _backward_score,
                                            _partition_fun_value,
                                            _posterior_score)

from flexcrf_tp.crfsuite2flexcrf import convert_data_to_flexcrf

# -- Define vitrebi_decoder here:

def viterbi_decoder(m_xy, n=None, log_version=True):
    """
    Performs MAP inference, determining $y = \argmax_y P(y|x)$, using the
    Viterbi algorithm.

    Parameters
    ----------
    m_xy : ndarray, shape (n_obs, n_labels, n_labels)
        Values of log-potentials ($\log M_i(y_{i-1}, y_i, x)$)
        computed based on feature functions f_xy and/or user-defined potentials
        `psi_xy`. At t=0, m_xy[0, 0, :] contains values of $\log M_1(y_0, y_1)$
        with $y_0$ the fixed initial state.

    n : integer, default=None
        Time position up to which to decode the optimal sequence; if not
        specified (default) the score is computed for the whole sequence.

    Returns
    -------
    y_pred : ndarray, shape (n_obs,)
        Predicted optimal sequence of labels.

    """

    # YOUR CODE HERE .....

    pass

# -- Load data and crfsuite model and convert them-------------------------

RECREATE = True  # set to True to recreate flexcrf data with new model

CRFSUITE_MODEL_FILE = '../conll2002/conll2002-esp.crfsuite'
CRFSUITE_TEST_DATA_FILE = '../conll2002/conll2002-esp_crfsuite-test-data.dump'
FLEXCRF_TEST_DATA_FILE = '../conll2002/conll2002-esp_flexcrf-test-data.dump'

# crfsuite model
tagger = Tagger()
tagger.open(CRFSUITE_MODEL_FILE)
model = tagger.info()

data = pickle.load(open(CRFSUITE_TEST_DATA_FILE))
print "test data loaded."

if RECREATE:
    dataset, thetas = convert_data_to_flexcrf(data, model, n_seq=3)
    pickle.dump({'dataset': dataset, 'thetas': thetas},
                open(FLEXCRF_TEST_DATA_FILE, 'wb'))
else:
    dd = pickle.load(open(FLEXCRF_TEST_DATA_FILE))
    dataset = dd['dataset']
    thetas = dd['thetas']

# -- Start classification ------------------------------------------------

for seq in range(len(dataset)):
    # -- with crfsuite
    s_ = tagger.tag(data['X'][seq])
    y_ = np.array([int(model.labels[s]) for s in s_])
    prob_ = tagger.probability(s_)

    print "\n-- With crfsuite:"
    print "labels:\n", s_, "\n", y_
    print "probability:\t %f" % prob_

    # -- with flexcrf
    f_xy, y = dataset[seq]
    theta = thetas[seq]

    m_xy, f_m_xy = _compute_all_potentials(f_xy, theta)

    y_pred = viterbi_decoder(m_xy)

    # ADD CODE TO COMPUTE POSTERIOR PROBABILITY WITH FLEXCRF HERE ....

    print "-- With flexcrf:"
    print "labels:\n", y_pred
    print "equal predictions: ", all(y_pred == y_)
    #print "probability:\t %f" % prob
    #print "delta:\t %f" % abs(prob-prob_)

tagger.close()