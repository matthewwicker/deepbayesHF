#Author: Matthew Wicker
# Impliments the BayesByBackprop optimizer for BayesKeras

import os
import math
import logging
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from tqdm import tqdm
from tqdm import trange

from deepbayesHF.optimizers import optimizer 
from deepbayesHF.optimizers import losses
from deepbayesHF import analyzers
from abc import ABC, abstractmethod

# A dumb mistake on my part which needs to be factored out
def softplus(x):
     return tf.math.softplus(x)

class CyclicStochasticGradientLangevinDynamics(optimizer.Optimizer):
    def __init__(self):
        super().__init__()

    # I set default params for each sub-optimizer but none for the super class for
    # pretty obvious reasons
    def compile(self, keras_model, loss_fn, batch_size=64, learning_rate=0.35, decay=0.0,
                      epochs=10, prior_mean=-1, prior_var=-1, **kwargs):
        super().compile(keras_model, loss_fn, batch_size, learning_rate, decay,
                      epochs, prior_mean, prior_var, **kwargs)

        self.cycles =  kwargs.get('cycles', 3)
        self.sample_prop =  kwargs.get('sample_prop', 0.5)
        self.decay = 0.1
        self.posterior_samples = []
        self.num_rets = [] # This is the frequency array, but naming is currently consistent with HMC methods
        return self

    def step(self, features, labels, lrate):
        # Define the GradientTape context
        with tf.GradientTape(persistent=True) as tape:   # Below we add an extra variable for IBP
            tape.watch(self.posterior_mean) 
            predictions = self.model(features)
            if(self.robust_train == 0):
                worst_case = predictions 
                loss = self.loss_func(labels, predictions)

            elif(int(self.robust_train) == 1):
                predictions = self.model(features)
                logit_l, logit_u = analyzers.IBP(self, features, self.model.trainable_variables, eps=self.epsilon)
                v1 = tf.one_hot(labels, depth=10)
                v2 = 1 - tf.one_hot(labels, depth=10)
                worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
                worst_case = self.model.layers[-1].activation(worst_case)
                output = (self.robust_lambda * predictions) + ((1-self.robust_lambda) * worst_case)
                loss =  self.loss_func(labels, output)

            elif(int(self.robust_train) == 2):
                predictions = self.model(features)
                features_adv = analyzers.FGSM(self, features, self.attack_loss, eps=self.epsilon, num_models=-1)
                worst_case = self.model(features_adv)
                output = (self.robust_lambda * predictions) + ((1-self.robust_lambda) * worst_case)
                loss =  self.loss_func(labels, output)
                #self.train_rob(labels, worst_case)

            elif(int(self.robust_train) == 3):
                output = tf.zeros(predictions.shape)
                self.epsilon = max(0.0001, self.epsilon)
                self.eps_dist = tfp.distributions.Exponential(1.0/self.epsilon)
                for _mc_ in range(self.loss_monte_carlo):
                    eps = tfp.random.rayleigh([1], scale=self.epsilon/2.0)
                    logit_l, logit_u = analyzers.IBP(self, features, self.model.trainable_variables, eps=self.epsilon)
                    v1 = tf.one_hot(labels, depth=10)
                    v2 = 1 - tf.one_hot(labels, depth=10)
                    v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
                    worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
                    worst_case = self.model.layers[-1].activation(worst_case)
                    one_hot_cls = tf.one_hot(labels, depth=10)
                    output += (1.0/self.loss_monte_carlo) * worst_case
                loss = self.loss_func(labels, output)

            elif(int(self.robust_train) == 4):
                output = tf.zeros(predictions.shape)
                self.epsilon = max(0.0001, self.epsilon)
                self.eps_dist = tfp.distributions.Exponential(1.0/float(self.epsilon))
                for _mc_ in range(self.loss_monte_carlo):
                    #eps = tfp.random.rayleigh([1], scale=self.epsilon)
                    eps = self.eps_dist.sample()
                    features_adv = analyzers.FGSM(self, features, self.attack_loss, eps=self.epsilon, num_models=-1)
                    worst_case = self.model(features_adv)
                    output += (1.0/self.loss_monte_carlo) * worst_case
                loss = self.loss_func(labels, output)

        # Get the gradients
        weight_gradient = tape.gradient(loss, self.model.trainable_variables)
#        print(weight_gradient)
        weights = self.model.get_weights()
        new_weights = []
        for i in range(len(weight_gradient)):
            wg = tf.math.multiply(weight_gradient[i], lrate)
            eta = tf.random.normal(weight_gradient[i].shape, mean=0.0, stddev=lrate*2*weight_gradient[i])
            wg = tf.math.add(wg, eta)
            m = tf.math.subtract(weights[i], wg)
            new_weights.append(m)

        self.model.set_weights(new_weights)
        self.posterior_mean = new_weights

        self.train_loss(loss)
        if(self.mode == "regression"):
            labels = tf.reshape(labels, predictions.shape)
        self.train_metric(labels, predictions)
        #self.train_rob(labels, worst_case)
        return self.posterior_mean, self.posterior_var

    def cycle(self, X_train, y_train, X_test=None, y_test=None):
        self.num_batches = int(len(X_train)/self.batch_size)
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(100).batch(self.batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(self.batch_size)

        if(self.robust_linear):
            self.max_eps = self.epsilon
            self.epsilon = 0.0
            self.max_robust_lambda = self.robust_lambda

        lr = self.learning_rate; decay = self.decay
        for epoch in range(self.epochs):
            lrate = self.learning_rate * (1 / (1 + self.decay * epoch))

            # Run the model through train and test sets respectively
            for (features, labels) in tqdm(train_ds):
                features += np.random.normal(loc=0.0, scale=self.input_noise, size=features.shape)
                self.posterior_mean, _ = self.step(features, labels, lrate)
            if(epoch/self.epochs >= (1-self.sample_prop)):
                self.num_rets.append(1)
                self.posterior_samples.append(self.posterior_mean)

            for test_features, test_labels in test_ds:
                self.model_validate(test_features, test_labels)
                
            # Grab the results
            (loss, acc) = self.train_loss.result(), self.train_metric.result()
            (val_loss, val_acc) = self.valid_loss.result(), self.valid_metric.result()
            self.logging(loss, acc, val_loss, val_acc, epoch)
            
            # Clear the current state of the metrics
            self.train_loss.reset_states(), self.train_metric.reset_states()
            self.valid_loss.reset_states(), self.valid_metric.reset_states()
            self.extra_metric.reset_states()
            
            if(self.robust_linear):
                self.epsilon += self.max_eps/self.epochs


    def prior_sample(self):
        sampled_weights = []
        for i in range(len(self.prior_mean)):
            var = tf.math.sqrt(self.posterior_var[i])
            sampled_weights.append(tf.random.normal(shape=self.prior_mean[i].shape, mean=self.prior_mean[i], 
                                                    stddev=self.prior_var[i]))
        return sampled_weights

    def train(self, X_train, y_train, X_test=None, y_test=None):
        self.posterior_mean = self.prior_sample()
        self.model.set_weights(self.posterior_mean)
        for i in range(self.cycles):
            print("Cyclical Learning ~~~~~~~~~~~~ Starting Cycle %s"%(i))
            self.cycle(X_train, y_train, X_test, y_test)
            self.posterior_mean = self.prior_sample()
            self.model.set_weights(self.posterior_mean)

    def save(self, path):
        if(self.num_rets[0] == 0):
            self.num_rets = self.num_rets[1:]
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path+"/samples"):
            os.makedirs(path+"/samples")
        np.save(path+"/mean", np.asarray(self.posterior_mean))
        for i in range(len(self.posterior_samples)):
            np.save(path+"/samples/sample_%s"%(i), np.asarray(self.posterior_samples[i]))
        self.model.save(path+'/model.h5')
        np.save(path+"/freq",np.asarray(self.num_rets))
        model_json = self.model.to_json()
        with open(path+"/arch.json", "w") as json_file:
            json_file.write(model_json)
        super().save(path)
        

