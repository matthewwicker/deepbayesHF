# Author: Matthew Wicker


from statsmodels.stats.proportion import proportion_confint
import math
import numpy as np
import tensorflow as tf
from tqdm import trange
from . import attacks

def propagate_interval(W, b, x_l, x_u, marg=0):
    mu = tf.divide(tf.math.add(x_u, x_l), 2)
    r = tf.divide(tf.math.subtract(x_u, x_l), 2)
    mu_new = tf.math.add(tf.matmul(mu, W), b) 
    try:
        marg = tf.cast(marg, dtype=tf.float64)
        W = tf.cast(W, dtype=tf.float64)
        rad = tf.matmul(r, tf.math.abs(W)+marg) 
    except:
        marg = tf.cast(marg, dtype=tf.float32)
        W = tf.cast(W, dtype=tf.float32)
        rad = tf.matmul(r, tf.math.abs(W)+marg) 
    h_u = tf.math.add(mu_new, rad)
    h_l = tf.math.subtract(mu_new, rad)
    return h_l, h_u

def IBP_state(model, s0, s1, weights, weight_margin=0, logits=True):
    h_l = s0
    h_u = s1
    layers = model.model.layers
    offset = 0
    for i in range(len(layers)):
        if(len(layers[i].get_weights()) == 0):
            h_u = model.model.layers[i](h_u)
            h_l = model.model.layers[i](h_l)
            offset += 1
            continue
        w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
        sigma = model.posterior_var[2*(i-offset)]
        marg = weight_margin*sigma
        h_l, h_u = propagate_interval(w, b, h_l, h_u, marg=marg)
        if(i < len(layers)-1):
            h_l = model.model.layers[i].activation(h_l)
            h_u = model.model.layers[i].activation(h_u)
    return h_l, h_u

def IBP_conf(model, s0, s1, weights, weight_margin=0, logits=True):
    h_l = s0
    h_u = s1
    layers = model.model.layers
    offset = 0
    for i in range(len(layers)):
        if(len(layers[i].get_weights()) == 0):
            h_u = model.model.layers[i](h_u)
            h_l = model.model.layers[i](h_l)
            offset += 1
            continue
        if(i == len(layers)-1):
            # iterate over the number of classes:
            softmax_diffs = []
            num_classes = np.asarray(weights[(2*(i-offset))+1]).shape[-1]
            print("number of classes: ", num_classes)
            for k in range(num_classes):
                class_diffs = []
                for l in range(num_classes):
                    diff = weights[2*(i-offset)][:,k] - weights[2*(i-offset)][:,l]
                    max_diff = np.maximum(diff, 0)
                    min_diff = np.minimum(diff, 0)
                    bias_diff = weights[(2*(i-offset))+1][k] - weights[(2*(i-offset))+1][l]
                    #logit_diff = (max_diff * h_u) + (min_diff * h_l)  + bias_diff 
                    logit_diff = np.sum(max_diff * h_u) + np.sum(min_diff * h_l)  + bias_diff 
                    class_diff = logit_diff 
                    #print(class_diff)
                    #if(class_diff == 0.0):
                    #    print("These should be the same ", k,l)
                    class_diffs.append(class_diff)
                diff_val = -np.log(np.sum(np.exp(-1*np.asarray(class_diffs))))
                #diff_val = np.max(class_diffs)
                #diff_val = np.max(diff_val)
                print("Diff value for class ", k, " is ", diff_val)
                softmax_diffs.append(diff_val)
            return softmax_diffs
        w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
        sigma = model.posterior_var[2*(i-offset)]
        marg = weight_margin*sigma
        h_l, h_u = propagate_interval(w, b, h_l, h_u, marg=marg)
    return None #error state if we hit this!

def IBP(model, inp, weights, eps, predict=False):
    h_u = tf.clip_by_value(tf.math.add(inp, eps), 0.0, 1.0)
    h_l = tf.clip_by_value(tf.math.subtract(inp, eps), 0.0, 1.0)
    layers = model.model.layers
    offset = 0
    for i in range(len(layers)):
        if(len(layers[i].get_weights()) == 0):
            h_u = model.model.layers[i](h_u)
            h_l = model.model.layers[i](h_l)
            offset += 1
            continue
        w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
        h_l, h_u = propagate_interval(w, b, h_l, h_u)
        if(i < len(layers)-1):
            h_l = model.model.layers[i].activation(h_l)
            h_u = model.model.layers[i].activation(h_u)
    return h_l, h_u

# Code for merging overlapping intervals. Taken from here: 
# https://stackoverflow.com/questions/49071081/merging-overlapping-intervals-in-python
# This function simple takes in a list of intervals and merges them into all 
# continuous intervals and returns that list 
def merge_intervals(intervals):
    sorted_intervals = sorted(intervals)
    interval_index = 0
    intervals = np.asarray(intervals)
    for  i in sorted_intervals:
        if i[0] > sorted_intervals[interval_index][1]:
            interval_index += 1
            sorted_intervals[interval_index] = i
        else:
            sorted_intervals[interval_index] = [sorted_intervals[interval_index][0], i[1]]
    return sorted_intervals[:interval_index+1] 


"""
Given a set of disjoint intervals, compute the probability of a random
sample from a guassian falling in these intervals. (Taken from lemma)
of the document
"""
import math
from scipy.special import erf
def compute_erf_prob(intervals, mean, var):
    prob = 0.0
    for interval in intervals:
#        val1 = erf((mean-interval[0])/(math.sqrt(2)*(stddev)))
#        val2 = erf((mean-interval[1])/(math.sqrt(2)*(stddev)))
        val1 = erf((mean-interval[0])/(math.sqrt(2*(var))))
        val2 = erf((mean-interval[1])/(math.sqrt(2*(var))))
        prob += 0.5*(val1-val2)
    return prob

"""
Given a set of possibly overlapping intervals:
    - Merge all intervals into maximum continuous, disjoint intervals
    - compute probability of these disjoint intervals
    - do this for ALL values in a weight matrix
"""
def compute_interval_probs_weight(vector_intervals, marg, mean, var):
    means = mean; # vars = var
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in trange(len(vector_intervals[0])):
        for j in range(len(vector_intervals[0][0])):
            intervals = []
            for num_found in range(len(vector_intervals)):
                interval = [vector_intervals[num_found][i][j]-(var[i][j]*marg), vector_intervals[num_found][i][j]+(var[i][j]*marg)]
                intervals.append(interval)
            p = compute_erf_prob(merge_intervals(intervals), means[i][j], var[i][j])
            prob_vec[i][j] = p
    return np.asarray(prob_vec)

"""
Given a set of possibly overlapping intervals:
    - Merge all intervals into continuous, disjoint intervals
    - compute probability of these disjoint intervals
    - do this for ALL values in a *flat* bias matrix (vector)
"""
def compute_interval_probs_bias(vector_intervals, marg, mean, var):
    means = mean; #stds = var
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in range(len(vector_intervals[0])):
        intervals = []
        for num_found in range(len(vector_intervals)):
            #!*! Need to correct and make sure you scale margin
            interval = [vector_intervals[num_found][i]-(var[i]*marg), vector_intervals[num_found][i]+(var[i]*marg)]
            intervals.append(interval)
        p = compute_erf_prob(merge_intervals(intervals), means[i], var[i])
        prob_vec[i] = p
    return np.asarray(prob_vec)
        
def compute_probability(model, weight_intervals, margin, verbose=True):
    full_p = 1.0
    if(verbose == True):
        func = trange
    else:
        func = range
    # for every weight vector, get the intervals
    for i in func(len(model.posterior_mean)):
        if(i % 2 == 0): # then its a weight vector
#            p = compute_interval_probs_weight(weight_intervals[i], margin, model.posterior_mean[i], np.square(model.posterior_var[i]))
            p = compute_interval_probs_weight(weight_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i]))
        else:
#            p = compute_interval_probs_bias(weight_intervals[i], margin, model.posterior_mean[i], np.square(model.posterior_var[i]))
            p = compute_interval_probs_bias(weight_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i]))
        #print("Average weight prob: ", np.mean(p))
        p = np.prod(p)
        full_p *= p
    return full_p

def IBP_prob(model, s0, s1, w_marg, samples, predicate, i0=0, inflate=1.0):
    w_marg = w_marg**2
    safe_weights = []
    safe_outputs = []
    for i in range(samples):
        model.model.set_weights(model.sample(inflate=inflate))
        ol, ou = IBP_state(model, s0, s1, model.model.get_weights(), w_marg)
        if(predicate(np.squeeze(s0), np.squeeze(s1), np.squeeze(ol), np.squeeze(ou))):
            safe_weights.append(model.model.get_weights())
            ol = np.squeeze(ol); ou = np.squeeze(ou)
            #lower = np.squeeze(s0)[0:len(ol)] + ol; upper = np.squeeze(s1)[0:len(ou)] + ou
            safe_outputs.append([-1,1]) # This is used ONLY for control loops which needs its own verification section
    print("Found %s safe intervals"%(len(safe_weights)))
    if(len(safe_weights) < 2):
        return 0.0, -1
    p = compute_probability(model, np.swapaxes(np.asarray(safe_weights),1,0), w_marg)
    return p, np.squeeze(safe_outputs)


def IBP_upper(model, s0, s1, w_marg, samples, predicate, loss_fn, eps, inputs=[], inflate=1.0, mod_option=10):
    w_marg = w_marg**2
    safe_weights = []
    safe_outputs = []
    for i in trange(samples, desc="Checking Samples"):
        model.model.set_weights(model.sample(inflate=inflate))
        # Insert attacks here
        if(i%mod_option == 0):
            adv = attacks.FGSM(model, s0, loss_fn, eps, direction=-1, num_models=-1, order=1)
            ol, ou = IBP_state(model, adv, adv, model.model.get_weights(), w_marg)
            unsafe = predicate(np.squeeze(adv), np.squeeze(adv), np.squeeze(ol), np.squeeze(ou))
        else:
            unsafe = False
        for inp in inputs:
            if(unsafe == True):
                break
            ol, ou = IBP_state(model, inp, inp, model.model.get_weights(), w_marg)
            if(predicate(np.squeeze(inp), np.squeeze(inp), np.squeeze(ol), np.squeeze(ou))):
                unsafe = True
                break

        if(unsafe):
            safe_weights.append(model.model.get_weights())
            ol = np.squeeze(ol); ou = np.squeeze(ou)
            lower = np.squeeze(s0)[0:len(ol)] + ol; upper = np.squeeze(s1)[0:len(ou)] + ou
            safe_outputs.append([lower,upper])
    print("Found %s safe intervals"%(len(safe_weights)))
    if(len(safe_weights) < 2):
        return 0.0, -1
    p = compute_probability(model, np.swapaxes(np.asarray(safe_weights),1,0), w_marg)
    return p, np.squeeze(safe_outputs)


def IBP_uncert(model, s0, s1, w_marg, samples, predicate, i0=0, inflate=1.0):
    #def predicate_uncertain(iml, imu, ol, ou):
    # Step 1 : compute the max difference of the logit weights in the softmax layer (clip above 0)
    # Step 2 : compute the min difference of the logit weights in the softmax layer (clip above 0)
    # Step 3 : compute the difference in biases for the two classes
    # Step 4 : affine pass of upper and lower bounds through those values
    # Step 5 : profit
    w_marg = w_marg**2
    safe_weights = []
    safe_outputs = []
    for i in range(samples):
        model.model.set_weights(model.sample(inflate=inflate))
        checks = []
        softmax_diff = IBP_conf(model, s0, s1, model.model.get_weights(), w_marg)
        uncertain = predicate(np.squeeze(s0), np.squeeze(softmax_diff))	
        if(uncertain):
            safe_weights.append(model.model.get_weights())
    print("Found %s safe intervals"%(len(safe_weights)))
    if(len(safe_weights) < 2):
        return 0.0, -1
    p = compute_probability(model, np.swapaxes(np.asarray(safe_weights),1,0), w_marg)
    return p, np.squeeze(safe_outputs)


def IBP_prob_w(model, s0, s1, w_marg, w, predicate, i0=0):
    model.model.set_weights(model.sample())
    ol, ou = IBP_state(model, s0, s1, w, w_marg)
    if(predicate(np.squeeze(s0)[i0:i0+2], np.squeeze(s1)[i0:i0+2], np.squeeze(ol)[i0:i0+2], np.squeeze(ou)[i0:i0+2])):
        p = compute_probability(model, np.swapaxes(np.asarray([w]),1,0), w_marg)
        return p, -1
    else:
        return 0.0, -1



# also known as the chernoff bound
def okamoto_bound(epsilon, delta):
    return (-1*.5) * math.log(float(delta)/2) * (1.0/(epsilon**2))

# This is h_a in the paper
def absolute_massart_halting(succ, trials, I, epsilon, delta, alpha):
    gamma = float(succ)/trials
    if(I[0] < 0.5 and I[1] > 0.5):
        return -1
    elif(I[1] < 0.5):
        val = I[1]
        h = (9/2.0)*(((3*val + epsilon)*(3*(1-val)-epsilon))**(-1))
        return math.ceil((h*(epsilon**2))**(-1) * math.log((delta - alpha)**(-1)))
    elif(I[0] >= 0.5):
        val = I[0]
        h = (9/2.0)*(((3*(1-val) + epsilon)*((3*val)+epsilon))**(-1))
        return math.ceil((h*(epsilon**2))**(-1) * math.log((delta - alpha)**(-1)))

"""

"""
def chernoff_bound_verification(model, inp, eps, cls, **kwargs):
    from tqdm import trange
    delta = kwargs.get('delta', 0.3)
    alpha = kwargs.get('alpha', 0.05)
    confidence = kwargs.get('confidence', 0.95)
    verbose = kwargs.get('verbose', False)
    epsilon = 1-confidence
    chernoff_bound = math.ceil( (1/(2*epsilon**2)) * math.log(2/delta) )
    softmax = 0
    for i in trange(chernoff_bound, desc="Sampling for Chernoff Bound Satisfaction"):
        model.set_weights(model.sample())
        logit_l, logit_u = IBP(model, inp, model.model.get_weights(), eps, predict=False)
        v1 = tf.one_hot(cls, depth=10)
        v2 = 1 - tf.one_hot(cls, depth=10)
        v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
        worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
        if(type(softmax) == int):
            softmax = model.model.layers[-1].activation(worst_case)
        else:
            softmax += model.model.layers[-1].activation(worst_case)
    return softmax
    #print("Not yet implimented")

"""
property - a function that takes a vector and returns a boolean if it was successful
"""
def massart_bound_check(model, inp, eps, cls, **kwargs):
    delta = kwargs.get('delta', 0.3)
    alpha = kwargs.get('alpha', 0.05)
    confidence = kwargs.get('confidence', 0.95)
    verbose = kwargs.get('verbose', False)
    
    atk_locs = []
    epsilon = 1-confidence
    chernoff_bound = math.ceil( (1/(2*epsilon**2)) * math.log(2/delta) )
    print("BayesKeras. Maximum sample bound = %s"%(chernoff_bound))
    successes, iterations, misses = 0.0, 0.0, 0.0
    halting_bound = chernoff_bound
    I = [0,1]
    while(iterations <= halting_bound):
        if(iterations > 0 and verbose):
            print("Working on iteration: %s \t Bound: %s \t Param: %s"%(iterations, halting_bound, successes/iterations))  
        model.set_weights(model.sample())
        logit_l, logit_u = IBP(model, inp, model.model.get_weights(), eps, predict=False)
        v1 = tf.one_hot(cls, depth=10)
        v2 = 1 - tf.one_hot(cls, depth=10)
        worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
        if(np.argmax(np.squeeze(worst_case)) != cls):
            misses += 1
            result = 0
        else:
            result = 1
        successes += result
        iterations += 1
        # Final bounds computation below
        lb, ub = proportion_confint(successes, iterations, method='beta')
        if(math.isnan(lb)):
            lb = 0.0 # Setting lb to zero if it is Nans
        if(math.isnan(ub)):
            ub = 1.0 # Setting ub to one if it is Nans
        I = [lb, ub]
        hb = absolute_massart_halting(successes, iterations, I, epsilon, delta, alpha)
        if(hb == -1):
            halting_bound = chernoff_bound
        else:
            halting_bound = min(hb, chernoff_bound)
    if(verbose):
        print("Exited becuase %s >= %s"%(iterations, halting_bound))
    return successes/iterations
    #return None
