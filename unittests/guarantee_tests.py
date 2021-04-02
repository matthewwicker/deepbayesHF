# Author: Matthew Wicker
import numpy as np
import sklearn
from sklearn import datasets

X_train, y_train = datasets.make_moons(n_samples=1000, noise=0.075)
X_test, y_test = datasets.make_moons(n_samples=1000, noise=0.075)

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

# Imports for Bayesian deep learning
import deepbayesHF
from deepbayesHF import optimizers
from deepbayesHF import PosteriorModel

chernoff_tests = 111
massart_tests = 111
problower_tests = 111
probupper_tests = 111
probexpect_tests = 111
probexpect_tests = 111

# Pre-condition: running of the classification test.
WIDTH = 64
# Load & test sample-based posterior
bayes_model = PosteriorModel("save_dir/HMC_MOONS_%s"%(WIDTH), deterministic=False)

# Load variational posterior
bayes_model = PosteriorModel("save_dir/VOGN_MOONS_%s"%(WIDTH), deterministic=False)

# Load deterministic network
bayes_model = PosteriorModel("save_dir/SGD_MOONS_%s"%(WIDTH), deterministic=True)
